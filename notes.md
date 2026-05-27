# Notes

To quickstart a demo, host AI Agent via `langgraph dev` then use [Agent Chat UI](https://agentchat.vercel.app/) to interact. Customizable with on [GitHub Repo](https://github.com/langchain-ai/agent-chat-ui) too.

## model parameters

- temperature - [0-1] - [deterministic, creative]
- max token - llm output length
- timeout
- max retires

## Snippets

```py
from typing import Dict, Any, Callable
from dataclasses import dataclass
from pydantic import BaseModel
from langchain.messages import HumanMessage
from langchain.messages import RemoveMessage
from langchain.agents import create_agent
from langchain.agents import AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware import before_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware import wrap_model_call
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.memory import InMemorySaver


db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")
standard_model = init_chat_model("gpt-5-nano")
large_model = init_chat_model("claude-sonnet-4-5")

@dataclass
class ContextSchema:
  user_language: str = "English"
  user_role: str = "external"

class CustomState(AgentState):
  favorite_city: str

# Structured Output
class CapitalInfo(BaseModel):
  name: str
  population: int


# Tool
@tool
def update_favorite_city(favorite_city: str, runtime: ToolRuntime) -> Command:
  """Update ... to the state."""
  return Command(update={
    "favorite_city": favorite_city,
    "messages": [ToolMessage("Successfully updated favorite city", tool_call_id=runtime.tool_call_id)]}
    )

@tool
def read_favorite_city(runtime: ToolRuntime) -> str:
  """Read ... from the state."""
  try:
    return runtime.state["favorite_city"]
  except KeyError:
    return "No favorite city found in state"

@tool
def query_db(query: str) -> str:
  """Query the database for ... information"""
  try:
    return db.run(query)
  except Exception as e:
    return f"Error querying database: {e}"


# Sub-Agent
subagent_1 = create_agent(
  model='gpt-5-nano',
  tools=[...]
)

@tool
def call_subagent_1(runtime: ToolRuntime) -> float:
  """Call subagent 1 to ... """
  genre = runtime.state["genre"]
  response = subagent_1.invoke({"messages": [HumanMessage(content=f"Do ... with {x}")]})
  return response["messages"][-1].content


# Middleware
@before_agent
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
  """Remove all the tool messages from the state"""
  messages = state["messages"]
  tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
  return {"messages": [RemoveMessage(id=m.id) for m in tool_messages]}

@dynamic_prompt
def user_language_prompt(request: ModelRequest) -> str:
  """Generate system prompt based on user role."""
  user_language = request.runtime.context.user_language
  base_prompt = "..."
  return f"{base_prompt} only respond in {user_language}."

@wrap_model_call
async def state_based_model(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
  """Select model based on State conversation length."""
  message_count = len(request.messages)
  if message_count > 10:
    model = large_model
  else:
    model = standard_model
  request = request.override(model=model)
  return await handler(request)

@wrap_model_call
async def role_based_tool_set(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
  """Dynamically call tools based on the runtime context"""
  user_role = request.runtime.context.user_role
  if user_role == "internal":
    pass # internal users get access to all tools
  else:
    tools = [read_favorite_city] # external users only get access to web search
    request = request.override(tools=tools)
  return await handler(request)


summarizer = SummarizationMiddleware(
  model="gpt-4o-mini",
  trigger=("tokens", 100),  # token threshold to start summarization
  keep=("messages", 1)      # num of messages to keep after summarization
)


# System Prompt
system_prompt = """
You're a helpful assistant.
 Use your tools to calculate the square root and square of any number.
"""


# AI Agent
agent = create_agent(
  model="gpt-5-nano",
  tools=[update_favorite_city, call_subagent_1, read_favorite_city, query_db],
  system_prompt=system_prompt,
  checkpointer=InMemorySaver(),
  state_schema=CustomState,
  context_schema=ContextSchema,
  middleware=[summarizer, trim_messages, user_language_prompt, state_based_model, role_based_tool_set]
  response_format=CapitalInfo
)

# Short-Term Memory
config = {"configurable": {"thread_id": "1"}}

# Invoke for answer
response = agent.invoke(
  {"messages": [HumanMessage("What is the capital of UK?")]},
  context_schema = ContextSchema(user_language="Chinese"),
  config
)
print(response['messages'][-1].content)

# Stream answer
for token, metadata in agent.stream(
  {"messages": [HumanMessage("What is the capital of UK?")]},
  stream_mode="messages"
):
  if token.content:
    print(token.content, end="", flush=True)

# Test tool
update_favorite_city.invoke({"x": 4})

# Multimodal Question
multimodal_question = HumanMessage(content=[
  {"type": "text", "text": "Tell me about this capital"},
  {"type": "image", "base64": img_b64, "mime_type": "image/png"},
  {"type": "audio", "base64": aud_b64, "mime_type": "audio/wav"}
])

```

## Human in the Loop

```py
from langchain.messages import HumanMessage
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


class EmailState(AgentState):
  email: str

@tool
def read_email(runtime: ToolRuntime) -> str:
  """Read an email from the given address."""
  # take email from state
  return runtime.state["email"]

@tool
def send_email(body: str) -> str:
  """Send an email to the given address with the given subject and body."""
  # fake email sending
  return f"Email sent"


agent = create_agent(
  model="gpt-5-nano",
  tools=[read_email, send_email],
  state_schema=EmailState,
  checkpointer=InMemorySaver(),
  middleware=[
    HumanInTheLoopMiddleware(
      interrupt_on={
        "read_email": False,
        "send_email": True,
      },
      description_prefix="Tool execution requires approval",
    ),
  ],
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
  {
    "messages": [HumanMessage(content="Please read my email and send a response immediately. Send the reply now in the same thread.")],
    "email": "Hi Seán, I'm going to be late for our meeting tomorrow. Can we reschedule? Best, John."
  },
  config=config
)
# print(response['__interrupt__'][0].value['action_requests'][0]['args']['body'])


# Approve
response = agent.invoke(
  Command(resume={"decisions": [{"type": "approve"}]}),
  config=config # Same thread ID to resume the paused conversation
)

# Reject
response = agent.invoke(
  Command(resume={"decisions": [{"type": "reject", "message": "No please sign off"}]}),
  config=config # Same thread ID to resume the paused conversation
)

# Edit
response = agent.invoke(
  Command(resume={"decisions": [
    {
      "type": "edit",
      "edited_action": {
        "name": "send_email",
        "args": {"body": "This is the last straw, you're fired!"},
      }
    }
  ]}),
  config=config # Same thread ID to resume the paused conversation
)

```

## MCP Call

```py
from langchain_mcp_adapters.client import MultiServerMCPClient

# e.g. https://mcp.so/server/time/modelcontextprotocol
client = MultiServerMCPClient({
    "local_server": {
      "transport": "stdio",
      "command": "python",
      "args": [".../... .py"],
    },
    "time": {
      "transport": "stdio",
      "command": "uvx",
      "args": [
        "mcp-server-time",
        "--local-timezone=America/New_York"
      ]
    }
  }
)

# get tools
tools = await client.get_tools()

# get prompts
agent = create_agent(
  model="gpt-5-nano",
  tools=tools,
)
```

## MCP Server

```py
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("mcp_server")

# Tools
@mcp.tool()
def search_web(query: str) -> Dict[str, Any]:
    """Tool description"""
    return ...

# Resources
@mcp.resource("github://...")
def github_file():
    return ...

# Prompt
@mcp.prompt()
def prompt():
    return ...

if __name__ == "__main__":
    mcp.run(transport="stdio")
```
