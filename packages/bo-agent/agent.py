"""Web entrypoint for the Bayesian optimization agent."""

import os
from typing import cast

import logfire
from dotenv import load_dotenv
from langchain_experimental.tools.python.tool import PythonREPLTool
from prompts import BO_MAIN_AGENT_INSTRUCTION
from pydantic_ai.ext.langchain import tool_from_langchain
from pydantic_deep import DeepAgentDeps, LocalBackend, create_deep_agent
from pydantic_deep.subagents import GENERAL_PURPOSE_SUBAGENT
from specialist import build_bo_specialist_subagent
from subagents_pydantic_ai import SubAgentConfig

load_dotenv()

MEMORY_DIR = os.getenv("BO_AGENT_MEMORY_DIR", ".deep/memory")
AGENT_MODEL = "openai-responses:gpt-5.4"

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

backend = LocalBackend(root_dir=".")
deps = DeepAgentDeps(backend=backend)

agent = create_deep_agent(
    AGENT_MODEL,
    instructions="You are a helpful assistant. " + BO_MAIN_AGENT_INSTRUCTION,
    subagents=[
        cast(SubAgentConfig, build_bo_specialist_subagent(AGENT_MODEL)),
        GENERAL_PURPOSE_SUBAGENT,
    ],
    backend=backend,
    tools=[tool_from_langchain(PythonREPLTool())],
    model_settings={"extra_body": {"text": {"verbosity": "low"}}},
    include_execute=True,
    include_todo=False,
    include_filesystem=True,
    include_skills=False,
    include_builtin_subagents=False,
    include_plan=False,
    include_memory=True,
    memory_dir=MEMORY_DIR,
    web_search=False,
    web_fetch=False,
)

# uv run uvicorn agent:app --reload --port 8899
app = agent.to_web(models={"GPT-5.4": AGENT_MODEL}, deps=deps)
