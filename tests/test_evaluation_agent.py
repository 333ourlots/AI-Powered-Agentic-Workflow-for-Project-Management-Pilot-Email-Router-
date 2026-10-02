from tests.test_support import get_api_key, show_and_save
from workflow_agents.base_agents import DirectPromptAgent, EvaluationAgent


def main() -> None:
    prompt = "Write one user story about a project manager tracking project email."
    criteria = "The answer must follow: As a [user], I want [action] so that [benefit]."
    result = EvaluationAgent(
        api_key=get_api_key(),
        persona="You are an evaluation agent that checks the answers of other worker agents.",
        evaluation_criteria=criteria,
        agent_to_evaluate=DirectPromptAgent(api_key=get_api_key()),
        max_interactions=3,
    ).evaluate(prompt)
    show_and_save(
        "evaluation_agent.txt",
        [
            "Agent: EvaluationAgent",
            f"Prompt: {prompt}",
            f"Final response: {result['final_response']}",
            f"Evaluation: {result['evaluation']}",
            f"Iterations: {result['iteration_count']}",
        ],
    )


if __name__ == "__main__":
    main()