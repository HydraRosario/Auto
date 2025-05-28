import os
import asyncio

from google.adk.agents import Agent
from google.adk.tools.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.models.lite_llm import LiteLlm

model = LiteLlm(
    model="gemini/gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

async def create_speaker_agent():
    """Creates the TTS Speaker agent by connecting to the Elevenlabs MCP server via uvx."""
    print("--- Attempting to start and connect to elevenlabs-mcp via uvx ---")

    tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command='uvx',
            args=['elevenlabs-mcp'],
            env={'ELEVENLABS_API_KEY': os.getenv('ELEVENLABS_API_KEY')}
        )
    )

    print(f"--- Connected to elevenlabs-mcp, Discovered {len(tools)} tool(s). ---")
    for tool in tools:
        print(f" -- Discovered tool: {tool.name}")
    
    agent_instance = Agent(
        name="speaker_agent",
        description=(
            "You are a Text-to-Speech agent. Take the text provided by the user or coordinator and "
            "use the available Elevenlabs TTS tool to convert it into audio. "
            "when calling the tool, set the parameter 'voice_name' to 'Will'. "
            "Return the result from the tool (expected to be a URL)."
        ),
        model=model,
        tools=tools,
    )

    return agent_instance, exit_stack

root_agent = create_speaker_agent()