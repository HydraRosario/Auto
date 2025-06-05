import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

model = LiteLlm(
    model="gemini/gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

async def get_tools_async():
    """Conecta al servidor MCP de File System via uvx y retorna las herramientas."""
    print("--- Attempting to start and connect to file-system-mcp MCP server via uvx ---")
    
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
                args=['file-system-mcp'],
            )
        )
        
        print(f"--- Successfully connected to file-system-mcp server. Discovered {len(tools)} tool(s). ---")
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
        print(f"--- ERROR connecting to file-system-mcp server: {e} ---")
        return [], None

async def create_agent():
    """Crea la instancia del agente después de obtener herramientas del servidor MCP."""
    tools, exit_stack = await get_tools_async()
    
    if not tools:
        print("--- WARNING: No tools discovered from MCP server. Creating agent without File System functionality. ---")
        
        # Crear agente con funcionalidad simulada
        agent_instance = Agent(
            name="file_system_agent",
            description="File System agent (MCP server unavailable - limited functionality)",
            model=model,
            instruction="""
            You are a File System Agent, but unfortunately the File System MCP server is currently unavailable.
            
            When users ask for file system operations, explain that:
            1. The file system functionality requires an external MCP server connection
            2. The connection is currently unavailable due to technical issues
            3. Please check that:
               - The MCP server is properly installed: uvx file-system-mcp
               - The uvx command is available in your system
               - All necessary permissions are granted
            4. Suggest alternative approaches like:
               - Using native file system commands in terminal
               - Checking back later when the service might be restored
               - Verifying MCP server installation and configuration
            
            Always be helpful and provide troubleshooting steps rather than just saying "no".
            """,
            tools=[],
        )
    else:
        # Crear agente con herramientas MCP
        agent_instance = Agent(
            name="file_system_agent",
            description="File System agent that provides complete file system access using MCP File System tools",
            model=model,
            instruction="""
            You are a File System Agent specialized in interacting with the file system through MCP server tools.
            
            CORE CAPABILITIES:
            - Read file contents from any accessible path
            - Write and create new files with specified content
            - Move and rename files and directories
            - Delete files and directories (with caution)
            - Create directories and directory structures
            - List directory contents and file information
            - Check file permissions and properties
            
            WORKFLOW:
            1. **Receive Request:** Get file system operation request from Auto coordinator
            2. **Validate Path:** Ensure the path is valid and accessible
            3. **Execute Operation:** Use appropriate MCP tool for the requested operation
            4. **Handle Results:** Process and format the results appropriately
            5. **Error Management:** Provide clear error messages and suggest solutions
            
            OPERATION EXAMPLES:
            - Read file: "Lee el contenido de /home/user/docs/archivo.txt"
              → Use read_file tool with path parameter
            - Write file: "Crea un archivo en /tmp/nuevo.txt con contenido 'Hola mundo'"
              → Use write_file tool with path and content parameters
            - List directory: "Muestra el contenido del directorio /home/user"
              → Use list_directory tool with path parameter
            - Move file: "Mueve /tmp/archivo.txt a /home/user/docs/"
              → Use move_file tool with source and destination parameters
            - Delete file: "Elimina el archivo /tmp/temporal.txt"
              → Use delete_file tool with path parameter
            
            SAFETY GUIDELINES:
            - **Always verify paths** before executing destructive operations
            - **Warn users** about potentially dangerous operations (delete, overwrite)
            - **Check permissions** and provide clear error messages if access denied
            - **Never use native Python file operations** - always use MCP tools
            - **Provide file content previews** for large files (first few lines)
            
            ERROR HANDLING:
            - File not found → Suggest checking path and permissions
            - Permission denied → Explain access restrictions and suggest alternatives
            - Disk space issues → Inform about storage limitations
            - Invalid paths → Guide user to correct path format
            - MCP server errors → Suggest restarting server or checking configuration
            
            RESPONSE FORMAT:
            - Always acknowledge the requested operation
            - Execute using appropriate MCP tool
            - Provide clear status (success/failure)
            - Include relevant details (file size, modification date, etc.)
            - Suggest next steps or related operations when helpful
            
            IMPORTANT RULES:
            - **NEVER use Python's built-in file operations** (open(), os.path, etc.)
            - **ALWAYS use MCP tools** for any file system interaction
            - **Be cautious with destructive operations** and confirm with user when needed
            - **Provide detailed feedback** about operations performed
            - **Handle errors gracefully** and provide actionable solutions
            
            Remember: You are the bridge between user requests and file system operations through MCP tools. 
            Always prioritize safety, clarity, and user guidance.
            """,
            tools=tools,
        )

    return agent_instance, exit_stack