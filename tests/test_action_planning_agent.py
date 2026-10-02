from tests.test_support import get_api_key, show_and_save
from workflow_agents.base_agents import ActionPlanningAgent


def main() -> None:
    prompt = "Create stories, define features from those stories, then write engineering tasks."
    knowledge = (
        "You are an action planning agent. Extract a short, ordered list of actionable "
        "steps from the request. Return only the numbered steps, one per line."
    )
    steps = ActionPlanningAgent(api_key=get_api_key(), knowledge=knowledge).extract_steps_from_prompt(
        prompt
    )
    show_and_save(
        "action_planning_agent.txt",
        ["Agent: ActionPlanningAgent", f"Prompt: {prompt}", "Steps:", *steps],
    )


if __name__ == "__main__":
    main()