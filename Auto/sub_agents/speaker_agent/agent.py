import os
import asyncio

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

model = LiteLlm(
    model="gemini/gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

async def create_speaker_agent():
    """Crea el agente TTS Speaker conectándose al servidor MCP de Elevenlabs via uvx."""
    print("--- Attempting to start and connect to elevenlabs-mcp via uvx ---")

    try:
        # Importar correctamente MCPToolset
        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
        
        # Verificar que la API key esté disponible
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            print("!!! WARNING: ELEVENLABS_API_KEY not found in environment !!!")
            raise ValueError("Missing ELEVENLABS_API_KEY")
        
        # Crear conexión usando el método correcto
        toolset = MCPToolset()
        tools, exit_stack = await toolset.connect_to_server(
            connection_params=StdioServerParameters(
                command='uvx',
                args=['elevenlabs-mcp'],
                env={'ELEVENLABS_API_KEY': api_key}
            )
        )

        print(f"--- Connected to elevenlabs-mcp, Discovered {len(tools)} tool(s). ---")
        for tool in tools:
            print(f" -- Discovered tool: {tool.name}")
            
        agent_instance = Agent(
            name="speaker_agent",
            description="Advanced Text-to-Speech agent using Elevenlabs API",
            model=model,
            instruction="""
            You are a Text-to-Speech specialist agent using Elevenlabs API.
            
            CORE FUNCTIONALITY:
            - Convert any text provided by users into high-quality speech
            - Use the 'Will' voice by default unless user specifies otherwise
            - Handle text preprocessing for better speech output
            - Provide audio URLs for generated speech
            
            WORKFLOW:
            1. Receive text input from user or coordinator
            2. Preprocess text if needed (remove special characters, etc.)
            3. Call Elevenlabs TTS tool with voice_name='Will'
            4. Return the audio URL or file path
            5. Handle any errors gracefully
            
            IMPORTANT:
            - Always set voice_name parameter to 'Will' unless specified otherwise
            - If the tool fails, explain the issue clearly
            - For long texts, warn users about processing time
            - Provide the audio URL/path in a clear format
            """,
            tools=tools,
        )
        
        return agent_instance, exit_stack
        
    except ImportError as e:
        print(f"!!! ERROR: Cannot import MCPToolset: {e} !!!")
        return create_fallback_speaker_agent(), None
    except FileNotFoundError:
        print("!!! ERROR: 'uvx' command not found. Please install uvx !!!")
        return create_fallback_speaker_agent(), None
    except AttributeError as e:
        print(f"!!! ERROR: MCPToolset API changed: {e} !!!")
        return create_fallback_speaker_agent(), None
    except ValueError as e:
        print(f"!!! ERROR: {e} !!!")
        return create_fallback_speaker_agent(), None
    except Exception as e:
        print(f"!!! ERROR connecting to elevenlabs-mcp: {e} !!!")
        return create_fallback_speaker_agent(), None

def create_fallback_speaker_agent():
    """Crea un agente de respaldo cuando el servicio TTS no está disponible."""
    return Agent(
        name="speaker_agent",
        description="Text-to-Speech agent (service unavailable - limited functionality)",
        model=model,
        instruction="""
        You are a Text-to-Speech agent, but unfortunately the Elevenlabs TTS service is currently unavailable.
        
        When users request text-to-speech conversion, explain that:
        1. The TTS functionality requires connection to Elevenlabs API
        2. The service is currently unavailable due to technical issues
        3. Suggest alternatives like:
           - Using built-in system TTS (Windows Speech, macOS Say command)
           - Online TTS services like Google Translate
           - Checking back later when the service might be restored
           - Using browser-based TTS extensions
        
        Always provide helpful alternatives rather than just declining the request.
        """,
        tools=[],
    )