"""Web entrypoint for the Bayesian optimization agent."""

import subprocess

import logfire
from dotenv import load_dotenv
from langchain_experimental.tools.python.tool import PythonREPLTool
from prompts import BO_MAIN_AGENT_INSTRUCTION
from pydantic_ai import Tool
from pydantic_ai.ext.langchain import tool_from_langchain
from pydantic_deep import DeepAgentDeps, LocalBackend, create_deep_agent
from pydantic_deep.subagents import GENERAL_PURPOSE_SUBAGENT
from specialist import build_bo_specialist_subagent

load_dotenv()

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

deps = DeepAgentDeps(backend=LocalBackend(root_dir="."))


def bash(command: str) -> str:
    """Run a Bash command and return its combined output."""
    try:
        result = subprocess.run(  # noqa: S603 - This tool intentionally runs user-directed commands.
            ["/bin/bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=36_000,
        )
    except subprocess.TimeoutExpired:
        return "Command timed out after 36,000 seconds."
    return result.stdout + result.stderr


agent = create_deep_agent(
    "openai-responses:gpt-5.4",
    instructions="You are a helpful assistant. " + BO_MAIN_AGENT_INSTRUCTION,
    subagents=[build_bo_specialist_subagent(), dict(GENERAL_PURPOSE_SUBAGENT)],
    backend=deps.backend,
    tools=[tool_from_langchain(PythonREPLTool()), Tool(bash)],
    model_settings={"extra_body": {"text": {"verbosity": "low"}}},
    include_execute=True,
    include_todo=False,
    include_filesystem=False,
    include_skills=False,
    include_builtin_subagents=False,
    include_plan=False,
    include_memory=False,
    web_search=False,
    web_fetch=False,
)

# uv run uvicorn agent:app --reload
app = agent.to_web(models={"GPT-5.4": "openai-responses:gpt-5.4"}, deps=deps)
