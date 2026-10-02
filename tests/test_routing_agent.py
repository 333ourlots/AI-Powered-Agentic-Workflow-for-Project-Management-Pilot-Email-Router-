from tests.test_support import get_api_key, show_and_save
from workflow_agents.base_agents import AugmentedPromptAgent, DirectPromptAgent, RoutingAgent


def main() -> None:
    api_key = get_api_key()
    prompt = "Create user stories for an email router used by project managers."
    routes = [
        {
            "name": "Product Manager",
            "description": "Defines product personas and user stories from user needs.",
            "func": AugmentedPromptAgent(
                api_key=api_key,
                persona="You are a product manager who writes concise user stories.",
            ).respond,
        },
        {
            "name": "General Assistant",
            "description": "Answers general questions that do not concern product planning.",
            "func": DirectPromptAgent(api_key=api_key).respond,
        },
    ]
    response = RoutingAgent(api_key=api_key, agents=routes).route(prompt)
    show_and_save(
        "routing_agent.txt",
        ["Agent: RoutingAgent", f"Prompt: {prompt}", f"Response: {response}"],
    )


if __name__ == "__main__":
    main()