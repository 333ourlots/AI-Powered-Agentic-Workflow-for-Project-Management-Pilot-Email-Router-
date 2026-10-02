from tests.test_support import get_api_key, show_and_save
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent


def main() -> None:
    knowledge = (
        "InnovateNext's pilot workflow converts product specifications into user stories, "
        "product features, and engineering tasks."
    )
    prompt = "What three planning artifacts does the pilot workflow produce?"
    response = KnowledgeAugmentedPromptAgent(
        api_key=get_api_key(),
        persona="You are an assistant that answers only from the supplied knowledge.",
        knowledge=knowledge,
    ).respond(prompt)
    show_and_save(
        "knowledge_augmented_prompt_agent.txt",
        [
            "Agent: KnowledgeAugmentedPromptAgent",
            f"Prompt: {prompt}",
            "Knowledge source: provided knowledge supplied at initialization.",
            f"Response: {response}",
        ],
    )


if __name__ == "__main__":
    main()