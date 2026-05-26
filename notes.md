# Notes

## L1

- model parameters
  - temperature - [0-1] - [deterministic, creative]
  - max token - llm output length
  - timeout
  - max retires

- snippets

  ```py
  from pydantic import BaseModel
  from langchain.agents import create_agent
  from langchain.agents import AgentState
  from langchain.messages import HumanMessage
  from langchain.tools import tool, ToolRuntime
  from langgraph.checkpoint.memory import InMemorySaver
  from langchain_community.utilities import SQLDatabase


  db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

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
    response_format=CapitalInfo
  )

  # Short-Term Memory
  config = {"configurable": {"thread_id": "1"}}

  # Invoke for answer
  response = agent.invoke(
    {"messages": [HumanMessage("What is the capital of UK?")]},
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

- MCP Server

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

- MCP Call

  ```py
  from langchain_mcp_adapters.client import MultiServerMCPClient

  # e.g. https://mcp.so/server/time/modelcontextprotocol
  client = MultiServerMCPClient({
      "local_server": {
        "transport": "stdio",
        "command": "python",
        "args": ["resources/2.1_mcp_server.py"],
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
