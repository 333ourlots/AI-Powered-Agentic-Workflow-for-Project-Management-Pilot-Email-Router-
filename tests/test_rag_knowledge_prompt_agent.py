from tests.test_support import get_api_key, show_and_save
from workflow_agents.base_agents import RAGKnowledgePromptAgent


def main() -> None:
    knowledge = (
        "A Product Manager defines user personas and user stories.\n\n"
        "A Program Manager defines product features and coordinates delivery.\n\n"
        "A Development Engineer turns approved stories and features into technical tasks."
    )
    prompt = "Who defines product features and coordinates delivery?"
    response = RAGKnowledgePromptAgent(
        api_key=get_api_key(),
        persona="You answer from retrieved project knowledge.",
        knowledge=knowledge,
    ).respond(prompt)
    show_and_save(
        "rag_knowledge_prompt_agent.txt",
        [
            "Agent: RAGKnowledgePromptAgent",
            f"Prompt: {prompt}",
            "Knowledge source: provided knowledge; relevant chunks retrieved with embeddings.",
            f"Response: {response}",
        ],
    )


if __name__ == "__main__":
    main()