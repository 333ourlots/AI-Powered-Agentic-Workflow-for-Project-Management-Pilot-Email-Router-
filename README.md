# AI-Powered Agentic Workflow for Project Management

Email Router pilot for the InnovateNext project-management workflow assignment.

## Setup

Use Python 3.10 or newer and install the required libraries:

```powershell
python -m pip install "openai>=1,<2" "python-dotenv>=1,<2"
```

Set `OPENAI_API_KEY` to the Vocareum key in your PowerShell session. The default API
endpoint is `https://openai.vocareum.com/v1`. Do not commit the key or a `.env` file.
Place `Product-Spec-Email-Router.txt` in the project root before running the workflow.

## Run

Run each standalone agent check from the project root. For example:

```powershell
python -m tests.test_direct_prompt_agent
python -m tests.test_augmented_prompt_agent
python -m tests.test_knowledge_augmented_prompt_agent
python -m tests.test_rag_knowledge_prompt_agent
python -m tests.test_evaluation_agent
python -m tests.test_routing_agent
python -m tests.test_action_planning_agent
```

Each successful test writes a transcript under `test_outputs/`. Run the full workflow
with `python agentic_workflow.py`; it writes `workflow_output.txt` after successful
completion. The script also accepts a product-spec path and a custom TPM prompt.

For the submission archive, follow the rubric's included-file list and exclude `.env`,
virtual environments, `__pycache__`, and other generated files not requested there.
