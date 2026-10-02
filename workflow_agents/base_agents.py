"""Prompt, evaluation, retrieval, routing, and planning agents."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Callable, Sequence

from openai import OpenAI


CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_BASE_URL = "https://openai.vocareum.com/v1"


def _make_client(api_key: str) -> OpenAI:
    if not api_key:
        raise ValueError("Pass an API key when initializing an agent.")
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
    )


def _chat(client: OpenAI, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content or ""


class DirectPromptAgent:
    """Send a prompt directly to the chat model without a system message."""

    def __init__(self, api_key: str):
        self.client = _make_client(api_key)

    def respond(self, prompt: str) -> str:
        return _chat(self.client, [{"role": "user", "content": prompt}])


class AugmentedPromptAgent:
    """Use a persona as a system instruction for each independent request."""

    def __init__(self, api_key: str, persona: str):
        self.client = _make_client(api_key)
        self.persona = persona

    def respond(self, prompt: str) -> str:
        system_prompt = (
            f"{self.persona}\nTreat this as a new conversation. Ignore any context "
            "from previous requests and answer only the current user message."
        )
        return _chat(
            self.client,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )


class KnowledgeAugmentedPromptAgent:
    """Answer using a persona and only the knowledge supplied at initialization."""

    def __init__(self, api_key: str, persona: str, knowledge: str):
        self.client = _make_client(api_key)
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, prompt: str) -> str:
        system_prompt = (
            f"{self.persona}\nTreat this as a new conversation and ignore prior context. "
            "Use only the knowledge below for factual claims. If the knowledge does not "
            "contain an answer, state what information is missing.\n\n"
            f"Knowledge:\n{self.knowledge}"
        )
        return _chat(
            self.client,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )


class RAGKnowledgePromptAgent:
    """Retrieve relevant knowledge chunks with embeddings before generating an answer."""

    def __init__(
        self,
        api_key: str,
        persona: str,
        knowledge: str,
        top_k: int = 3,
    ):
        self.client = _make_client(api_key)
        self.persona = persona
        self.knowledge = knowledge
        self.top_k = max(1, top_k)
        self._chunks = self._split_knowledge(knowledge)
        self._chunk_embeddings: list[list[float]] | None = None

    @staticmethod
    def _split_knowledge(knowledge: str, max_chars: int = 1200) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", knowledge) if part.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 1 > max_chars:
                chunks.append(current)
                current = ""
            if len(paragraph) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(
                    paragraph[index : index + max_chars]
                    for index in range(0, len(paragraph), max_chars)
                )
            else:
                current = f"{current}\n{paragraph}".strip()
        if current:
            chunks.append(current)
        return chunks or [knowledge]

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        result = self.client.embeddings.create(model=EMBEDDING_MODEL, input=list(texts))
        return [item.embedding for item in result.data]

    def _retrieve(self, prompt: str) -> list[str]:
        if self._chunk_embeddings is None:
            self._chunk_embeddings = self._embed(self._chunks)
        prompt_embedding = self._embed([prompt])[0]
        ranked = sorted(
            zip(self._chunks, self._chunk_embeddings),
            key=lambda pair: RoutingAgent.cosine_similarity(prompt_embedding, pair[1]),
            reverse=True,
        )
        return [chunk for chunk, _ in ranked[: self.top_k]]

    def respond(self, prompt: str) -> str:
        retrieved_knowledge = "\n\n".join(self._retrieve(prompt))
        system_prompt = (
            f"{self.persona}\nTreat this as a new conversation and ignore prior context. "
            "Use only the retrieved knowledge below for factual claims. If it does not "
            "contain an answer, state what information is missing.\n\n"
            f"Retrieved knowledge:\n{retrieved_knowledge}"
        )
        return _chat(
            self.client,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )


class EvaluationAgent:
    """Iteratively evaluate and refine a worker agent's response."""

    def __init__(
        self,
        api_key: str,
        persona: str,
        evaluation_criteria: str,
        agent_to_evaluate: Any,
        max_interactions: int = 3,
    ):
        self.client = _make_client(api_key)
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = agent_to_evaluate
        self.max_interactions = max(1, max_interactions)

    def evaluate(self, prompt: str) -> dict[str, Any]:
        worker_prompt = prompt
        final_response = ""
        evaluation = ""
        iterations = 0

        for iterations in range(1, self.max_interactions + 1):
            final_response = self.agent_to_evaluate.respond(worker_prompt)
            evaluation = _chat(
                self.client,
                [
                    {"role": "system", "content": self.persona},
                    {
                        "role": "user",
                        "content": (
                            "Assess the candidate against the criteria. Start with exactly "
                            "PASS if it satisfies them; otherwise start with REVISE and "
                            "explain the concrete changes needed.\n\n"
                            f"Criteria:\n{self.evaluation_criteria}\n\n"
                            f"Candidate response:\n{final_response}"
                        ),
                    },
                ],
                temperature=0,
            )
            if evaluation.strip().upper().startswith("PASS") or iterations == self.max_interactions:
                break

            correction = _chat(
                self.client,
                [
                    {"role": "system", "content": self.persona},
                    {
                        "role": "user",
                        "content": (
                            "Write concise correction instructions for the worker. Preserve "
                            "the original request and address the evaluation feedback.\n\n"
                            f"Original request:\n{prompt}\n\n"
                            f"Evaluation feedback:\n{evaluation}\n\n"
                            f"Current response:\n{final_response}"
                        ),
                    },
                ],
                temperature=0,
            )
            worker_prompt = (
                f"{prompt}\n\nRevise your previous response using these instructions:\n"
                f"{correction}\n\nPrevious response:\n{final_response}"
            )

        return {
            "final_response": final_response,
            "evaluation": evaluation,
            "iterations": iterations,
            "iteration_count": iterations,
        }


class RoutingAgent:
    """Choose a route by cosine similarity between task and route embeddings."""

    def __init__(self, api_key: str, agents: list[dict[str, Any]] | None = None):
        self.client = _make_client(api_key)
        self.agents = agents or []
        self._description_embeddings: dict[str, list[float]] = {}

    @staticmethod
    def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        dot_product = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot_product / (left_norm * right_norm)

    def get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding

    def route(self, prompt: str) -> Any:
        if not self.agents:
            raise ValueError("Add at least one route to RoutingAgent.agents before routing.")

        descriptions = [str(agent["description"]) for agent in self.agents]
        missing = [text for text in descriptions if text not in self._description_embeddings]
        if missing:
            embeddings = self.client.embeddings.create(model=EMBEDDING_MODEL, input=missing)
            self._description_embeddings.update(
                {text: item.embedding for text, item in zip(missing, embeddings.data)}
            )

        prompt_embedding = self.get_embedding(prompt)
        best_route = max(
            self.agents,
            key=lambda agent: self.cosine_similarity(
                prompt_embedding,
                self._description_embeddings[str(agent["description"])],
            ),
        )
        route_function: Callable[[str], Any] = best_route["func"]
        return route_function(prompt)


class ActionPlanningAgent:
    """Turn a high-level request into a clean list of actionable steps."""

    def __init__(self, api_key: str, knowledge: str):
        self.client = _make_client(api_key)
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt: str) -> list[str]:
        response = _chat(
            self.client,
            [
                {"role": "system", "content": self.knowledge},
                {"role": "user", "content": prompt},
            ],
        )
        steps = []
        for line in response.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                steps.append(cleaned)
        return steps