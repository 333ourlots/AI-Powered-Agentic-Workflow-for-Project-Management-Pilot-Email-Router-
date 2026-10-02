"""General-purpose TPM planning workflow, piloted with the Email Router spec."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

workflow_prompt = (
    "Create a complete project plan from the product specification. Plan in this order: "
    "(1) write user stories as the Product Manager, (2) define product features that "
    "support those stories as the Program Manager, and (3) create detailed engineering "
    "tasks mapped to the stories and features as the Development Engineer. Return exactly "
    "three actionable steps, one for each role, in that order."
)

knowledge_action_planning = (
    "You are an action planning agent for technical project management. Break the request "
    "into a small ordered list of actionable steps. Assign each step to one role: Product "
    "Manager for user stories, Program Manager for product features, or Development "
    "Engineer for implementation tasks. Return only the steps, one per line."
)

persona_product_manager = (
    "You are the Product Manager. Create user stories grounded in the supplied product "
    "specification. Do not define features or engineering tasks. Each story must follow "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
knowledge_product_manager = (
    "Use the supplied product specification to identify users, needs, and value. "
    "Do not invent capabilities that are not supported by the specification.\n\n"
)

persona_program_manager = (
    "You are the Program Manager. Define distinct product features that support the "
    "provided user stories and product specification. Do not write user stories or "
    "engineering implementation tasks."
)
knowledge_program_manager = (
    "For each feature, provide these fields: Feature Name, Description, Key Functionality, "
    "and User Benefit. Keep features traceable to the request and any provided stories."
)

persona_dev_engineer = (
    "You are a Development Engineer. Turn the provided stories and features into "
    "implementable, testable engineering tasks. Do not introduce unsupported product scope."
)
knowledge_dev_engineer = (
    "For every task, provide: Task ID, Task Title, Related User Story, Description, "
    "Acceptance Criteria, Estimated Effort, and Dependencies. Use unique task IDs and "
    "make acceptance criteria verifiable."
)

persona_program_manager_eval = (
    "You are an evaluation agent that checks product feature responses for completeness "
    "and adherence to the required structure."
)
persona_dev_engineer_eval = (
    "You are an evaluation agent that checks engineering task responses for completeness "
    "and adherence to the required structure."
)

program_manager_criteria = (
    "The answer should be product features that follow this structure:\n"
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
development_engineer_criteria = (
    "The answer should be tasks following this exact structure:\n"
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)


def _load_product_spec(spec_path: Path) -> str:
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"Product specification not found: {spec_path}. Place "
            "Product-Spec-Email-Router.txt in the project root or pass its path as an argument."
        )
    product_spec = spec_path.read_text(encoding="utf-8").strip()
    if not product_spec:
        raise ValueError(f"Product specification is empty: {spec_path}")
    return product_spec


def build_workflow(api_key: str, product_spec: str) -> tuple[ActionPlanningAgent, RoutingAgent]:
    """Instantiate planning, worker, evaluation, and routing agents."""
    action_planning_agent = ActionPlanningAgent(
        api_key=api_key,
        knowledge=knowledge_action_planning,
    )

    product_manager_knowledge = knowledge_product_manager + product_spec
    product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
        api_key=api_key,
        persona=persona_product_manager,
        knowledge=product_manager_knowledge,
    )
    product_manager_evaluation_agent = EvaluationAgent(
        api_key=api_key,
        persona="You are an evaluation agent that checks the answers of other worker agents",
        evaluation_criteria=(
            "The answer should be stories that follow the following structure: As a "
            "[type of user], I want [an action or feature] so that [benefit/value]."
        ),
        agent_to_evaluate=product_manager_knowledge_agent,
    )

    program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
        api_key=api_key,
        persona=persona_program_manager,
        knowledge=knowledge_program_manager,
    )
    program_manager_evaluation_agent = EvaluationAgent(
        api_key=api_key,
        persona=persona_program_manager_eval,
        evaluation_criteria=program_manager_criteria,
        agent_to_evaluate=program_manager_knowledge_agent,
    )

    development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
        api_key=api_key,
        persona=persona_dev_engineer,
        knowledge=knowledge_dev_engineer,
    )
    development_engineer_evaluation_agent = EvaluationAgent(
        api_key=api_key,
        persona=persona_dev_engineer_eval,
        evaluation_criteria=development_engineer_criteria,
        agent_to_evaluate=development_engineer_knowledge_agent,
    )

    def product_manager_support_function(query: str) -> str:
        result = product_manager_evaluation_agent.evaluate(query)
        return result["final_response"]

    def program_manager_support_function(query: str) -> str:
        result = program_manager_evaluation_agent.evaluate(query)
        return result["final_response"]

    def development_engineer_support_function(query: str) -> str:
        result = development_engineer_evaluation_agent.evaluate(query)
        return result["final_response"]

    routing_agent = RoutingAgent(api_key=api_key)
    routing_agent.agents = [
        {
            "name": "Product Manager",
            "description": (
                "Responsible for defining product personas and user stories only. "
                "Does not define features or engineering tasks."
            ),
            "func": product_manager_support_function,
        },
        {
            "name": "Program Manager",
            "description": (
                "Responsible for defining product features from user stories. "
                "Does not write user stories or detailed engineering tasks."
            ),
            "func": program_manager_support_function,
        },
        {
            "name": "Development Engineer",
            "description": (
                "Responsible for creating detailed, testable engineering tasks mapped "
                "to user stories and product features."
            ),
            "func": development_engineer_support_function,
        },
    ]
    return action_planning_agent, routing_agent


def run_workflow(
    api_key: str,
    product_spec: str,
    prompt: str = workflow_prompt,
) -> str:
    action_planning_agent, routing_agent = build_workflow(api_key, product_spec)
    workflow_steps = action_planning_agent.extract_steps_from_prompt(prompt)
    if not workflow_steps:
        raise RuntimeError("The ActionPlanningAgent returned no workflow steps.")

    completed_steps: list[str] = []
    completed_entries: list[dict[str, str]] = []
    output_lines = [f"Workflow prompt: {prompt}", ""]
    print(f"Workflow prompt: {prompt}\n")

    for index, current_step in enumerate(workflow_steps):
        print(f"Step {index + 1}: {current_step}")
        output_lines.append(f"Step {index + 1}: {current_step}")
        routed_query = current_step
        if completed_entries:
            prior_outputs = "\n\n".join(
                f"{entry['step']}\n{entry['response']}" for entry in completed_entries
            )
            routed_query += f"\n\nUse these prior workflow results as context:\n{prior_outputs}"

        result = routing_agent.route(routed_query)
        completed_steps.append(result)
        completed_entries.append({"step": current_step, "response": str(result)})
        print(f"Result:\n{result}\n")
        output_lines.extend([f"Result:\n{result}", ""])

    sections: dict[str, list[str]] = {
        "User Stories": [],
        "Product Features": [],
        "Engineering Tasks": [],
    }
    for entry in completed_entries:
        step = entry["step"].lower()
        if "story" in step or "product manager" in step:
            sections["User Stories"].append(entry["response"])
        elif "feature" in step or "program manager" in step:
            sections["Product Features"].append(entry["response"])
        elif "task" in step or "engineer" in step or "development" in step:
            sections["Engineering Tasks"].append(entry["response"])

    output_lines.append("CONSOLIDATED PROJECT PLAN")
    print("CONSOLIDATED PROJECT PLAN")
    for heading, content in sections.items():
        section_text = "\n\n".join(content) if content else "No result was routed to this section."
        section_heading = f"\n{heading}\n{'=' * len(heading)}"
        output_lines.extend([section_heading, section_text])
        print(section_heading)
        print(section_text)
    final_output = "\n".join(output_lines)
    return final_output


def main() -> None:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY in your environment or a local .env file. "
            "Do not commit the key or the .env file."
        )

    spec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "Product-Spec-Email-Router.txt"
    if not spec_path.is_absolute():
        spec_path = PROJECT_ROOT / spec_path
    product_spec = _load_product_spec(spec_path)
    prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else workflow_prompt

    try:
        output = run_workflow(openai_api_key, product_spec, prompt)
    except Exception as error:
        print(f"Workflow failed: {error}", file=sys.stderr)
        raise
    (PROJECT_ROOT / "workflow_output.txt").write_text(output + "\n", encoding="utf-8")
    print(f"\nSaved workflow output to {PROJECT_ROOT / 'workflow_output.txt'}")


if __name__ == "__main__":
    main()