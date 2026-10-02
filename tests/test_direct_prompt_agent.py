from tests.test_support import get_api_key, show_and_save
from workflow_agents.base_agents import DirectPromptAgent


def main() -> None:
    prompt = "In one sentence, explain why clear project requirements matter."
    response = DirectPromptAgent(api_key=get_api_key()).respond(prompt)
    show_and_save(
        "direct_prompt_agent.txt",
        [
            "Agent: DirectPromptAgent",
            f"Prompt: {prompt}",
            "Knowledge source: general model knowledge; no project knowledge is supplied.",
            f"Response: {response}",
        ],
    )


if __name__ == "__main__":
    main()