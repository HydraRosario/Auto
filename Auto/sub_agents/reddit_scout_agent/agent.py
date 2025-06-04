import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

model = LiteLlm(
    model="gemini/gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

async def get_tools_async():
    """Conecta al servidor MCP de Reddit via uvx y retorna las herramientas."""
    print("--- Attempting to start and connect to mcp-reddit MCP server via uvx ---")
    
    try:
        # Importar correctamente MCPToolset
        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
        
        # Verificar si uvx está disponible
        await asyncio.create_subprocess_shell(
            'uvx --version', 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE
        )

        # Crear conexión usando el método correcto
        toolset = MCPToolset()
        tools, exit_stack = await toolset.connect_to_server(
            connection_params=StdioServerParameters(
                command='uvx',
                args=['--from', 'git+https://github.com/adhikasp/mcp-reddit.git', 'mcp-reddit'],
            )
        )
        
        print(f"--- Successfully connected to mcp-reddit server. Discovered {len(tools)} tool(s). ---")
        for tool in tools:
            print(f"  - Discovered tool: {tool.name}")
        return tools, exit_stack
        
    except ImportError as e:
        print(f"!!! ERROR: Cannot import MCPToolset: {e} !!!")
        return [], None
    except FileNotFoundError:
        print("!!! ERROR: 'uvx' command not found. Please install uvx: pip install uvx !!!")
        return [], None
    except AttributeError as e:
        print(f"!!! ERROR: MCPToolset API changed: {e} !!!")
        print("!!! Trying alternative connection method... !!!")
        
        # Método alternativo sin MCP
        return [], None
    except Exception as e:
        print(f"--- ERROR connecting to mcp-reddit server: {e} ---")
        return [], None

async def create_agent():
    """Crea la instancia del agente después de obtener herramientas del servidor MCP."""
    tools, exit_stack = await get_tools_async()
    
    if not tools:
        print("--- WARNING: No tools discovered from MCP server. Creating agent without Reddit functionality. ---")
        
        # Crear agente con funcionalidad simulada
        agent_instance = Agent(
            name="reddit_scout_agent",
            description="Reddit scout agent (MCP server unavailable - limited functionality)",
            model=model,
            instruction="""
            You are a Reddit News Scout, but unfortunately the Reddit MCP server is currently unavailable.
            
            When users ask for Reddit content, explain that:
            1. The Reddit functionality requires an external MCP server connection
            2. The connection is currently unavailable due to technical issues
            3. Suggest alternative approaches like:
               - Using the web search agent to find Reddit-related content online
               - Visiting Reddit directly in a browser
               - Checking back later when the service might be restored
            
            Always be helpful and suggest alternatives rather than just saying "no".
            """,
            tools=[],
        )
    else:
        # Crear agente con herramientas MCP
        agent_instance = Agent(
            name="reddit_scout_agent",
            description="Reddit scout agent that searches for hot posts in subreddits using MCP Reddit tools",
            model=model,
            instruction="""
            You are a Reddit News Scout. Your primary task is to fetch hot post titles from any subreddit using the connected Reddit MCP tool.
            
            WORKFLOW:
            1. **Identify Subreddit:** Determine which subreddit the user wants news from
            2. **Call Tool:** Use the available Reddit tool with the subreddit parameter
            3. **Present Results:** Format and present the results clearly
            4. **Handle Errors:** If the tool fails, explain the issue to the user
            
            IMPORTANT:
            - Always call the Reddit tool before providing any Reddit-specific information
            - Present results in a clear, organized format
            - Include subreddit context in your responses
            - If the tool returns errors, relay them accurately to the user
            - Never hallucinate Reddit posts or content
            """,
            tools=tools,
        )

    return agent_instance, exit_stack