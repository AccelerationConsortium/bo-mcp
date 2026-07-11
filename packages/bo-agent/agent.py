"""Web entrypoint for the Bayesian optimization agent."""

import os
from typing import cast

import logfire
from bo_mcp.tools import register_bo_mcp_tools
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

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

backend = LocalBackend(root_dir=".")
deps = DeepAgentDeps(backend=backend)

agent = create_deep_agent(
    "openai-responses:gpt-5.4",
    instructions="You are a helpful assistant. " + BO_MAIN_AGENT_INSTRUCTION,
    subagents=[
        cast(SubAgentConfig, build_bo_specialist_subagent()),
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
register_bo_mcp_tools(agent)

# uv run uvicorn agent:app --reload
app = agent.to_web(models={"GPT-5.4": "openai-responses:gpt-5.4"}, deps=deps)
