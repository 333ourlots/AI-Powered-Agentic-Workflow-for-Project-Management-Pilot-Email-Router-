from tests.test_support import get_api_key, show_and_save
from workflow_agents.base_agents import AugmentedPromptAgent


def main() -> None:
    prompt = "Explain how a technical project manager can reduce scope ambiguity."
    persona = "You are a concise technical project management coach."
    response = AugmentedPromptAgent(api_key=get_api_key(), persona=persona).respond(prompt)
    show_and_save(
        "augmented_prompt_agent.txt",
        [
            "Agent: AugmentedPromptAgent",
            f"Prompt: {prompt}",
            "Knowledge source: general model knowledge; no reference document is supplied.",
            f"Persona impact: {persona}",
            f"Response: {response}",
        ],
    )


if __name__ == "__main__":
    main()