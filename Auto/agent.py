import os
import asyncio
from contextlib import AsyncExitStack
from typing import Dict, Any, Optional, List

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm

# Importar agentes sincrónicos directamente
from .sub_agents.conversational_agent.agent import conversational_agent
from .sub_agents.summarizer_agent.agent import root_agent as summarizer_agent

# Importar factory functions para agentes que requieren async
from .sub_agents.reddit_scout_agent.agent import create_agent as create_reddit_scout_agent
from .sub_agents.speaker_agent.agent import create_speaker_agent
from .sub_agents.coordinator_agent.agent import create_coordinator_agent

# Importar web searcher (convertido a sincrono)
from .sub_agents.web_searcher_agent.agent import web_searcher_agent

model = LiteLlm(
    model="gemini/gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# Función para inicializar agentes async de forma sincrona
def initialize_async_agents():
    """Inicializa agentes async de forma sincrona para compatibilidad con ADK web."""
    
    async def _init_async_agents():
        agents = {}
        exit_stacks = []
        
        try:
            # Reddit Scout Agent
            print("🔧 Initializing Reddit Scout Agent...")
            try:
                reddit_agent, reddit_stack = await create_reddit_scout_agent()
                agents['reddit_scout'] = reddit_agent
                if reddit_stack and hasattr(reddit_stack, '__aenter__'):
                    exit_stacks.append(reddit_stack)
                print("✅ Reddit Scout Agent initialized")
            except Exception as e:
                print(f"⚠️ Reddit Scout Agent failed: {e}")
                # Crear agente dummy para evitar fallos
                agents['reddit_scout'] = Agent(
                    name="reddit_scout_dummy",
                    model=model,
                    description="Reddit scout agent (disabled - MCP server unavailable)",
                    instruction="Sorry, Reddit functionality is currently unavailable due to MCP server issues."
                )
            
            # Speaker Agent
            print("🔧 Initializing Speaker Agent...")
            try:
                speaker_agent, speaker_stack = await create_speaker_agent()
                agents['speaker'] = speaker_agent
                if speaker_stack and hasattr(speaker_stack, '__aenter__'):
                    exit_stacks.append(speaker_stack)
                print("✅ Speaker Agent initialized")
            except Exception as e:
                print(f"⚠️ Speaker Agent failed: {e}")
                # Crear agente dummy
                agents['speaker'] = Agent(
                    name="speaker_dummy",
                    model=model,
                    description="Text-to-speech agent (disabled - MCP server unavailable)",
                    instruction="Sorry, text-to-speech functionality is currently unavailable due to MCP server issues."
                )
            
            # Coordinator Agent
            print("🔧 Initializing Coordinator Agent...")
            try:
                coordinator_agent, coordinator_stack = await create_coordinator_agent()
                agents['coordinator'] = coordinator_agent
                if coordinator_stack and hasattr(coordinator_stack, '__aenter__'):
                    exit_stacks.append(coordinator_stack)
                print("✅ Coordinator Agent initialized")
            except Exception as e:
                print(f"⚠️ Coordinator Agent failed: {e}")
                # Crear agente dummy
                agents['coordinator'] = Agent(
                    name="coordinator_dummy",
                    model=model,
                    description="News pipeline coordinator (disabled - dependency issues)",
                    instruction="Sorry, news pipeline functionality is currently unavailable due to dependency issues."
                )
            
            return agents, exit_stacks
            
        except Exception as e:
            print(f"❌ Error initializing async agents: {e}")
            return {}, []
    
    # Ejecutar de forma sincrona
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Si ya hay un loop corriendo, usar asyncio.run en un thread
            import threading
            import concurrent.futures
            
            def run_in_thread():
                return asyncio.run(_init_async_agents())
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        else:
            return loop.run_until_complete(_init_async_agents())
    except RuntimeError:
        # No hay loop, crear uno nuevo
        return asyncio.run(_init_async_agents())

# Inicializar agentes
print("🚀 Initializing Auto Multi-Agent System...")
async_agents, exit_stacks = initialize_async_agents()

# Crear lista de todos los sub-agentes
all_sub_agents = [
    conversational_agent,
    web_searcher_agent,
    summarizer_agent,
    async_agents.get('reddit_scout'),
    async_agents.get('speaker'), 
    async_agents.get('coordinator')
]

# Filtrar agentes None
all_sub_agents = [agent for agent in all_sub_agents if agent is not None]

# Crear herramientas de agente para delegación
agent_tools = [AgentTool(agent) for agent in all_sub_agents]

# Crear el agente raíz
root_agent = Agent(
    name="Auto",
    model=model,
    description="Advanced Multi-Agent Orchestrator - Coordinates specialized agents for various tasks",
    instruction="""
    You are Auto, an intelligent multi-agent orchestrator that coordinates specialized sub-agents to handle user requests efficiently.
    
    AVAILABLE AGENTS:
    🗣️ conversational_agent - For general conversation, advice, casual chat, emotional support
    🔍 web_searcher_agent - For web searches, finding information online, research tasks
    📰 coordinator_agent - For complete Reddit news briefings (fetch → summarize → speak)
    📋 reddit_scout_agent - For direct Reddit post fetching (standalone use)
    🔊 speaker_agent - For text-to-speech conversion
    📝 summarizer_agent - For text summarization tasks
    
    DELEGATION STRATEGY:
    1. **Casual Conversation** → conversational_agent
       - General chat, advice, emotional support
       - Personal questions, casual discussions
       
    2. **Information Seeking** → web_searcher_agent
       - "Search for...", "Find information about...", "What's happening with..."
       - Research tasks, current events, facts
       
    3. **Reddit News** → coordinator_agent
       - "News briefing from r/...", "What's happening in r/..."
       - Complete news pipeline with audio output
       
    4. **Reddit Posts Only** → reddit_scout_agent
       - "Show me posts from r/...", "What are the hot posts in..."
       - Just fetching posts without summarization
       
    5. **Text-to-Speech** → speaker_agent
       - "Read this aloud", "Convert to speech", "Make audio of..."
       
    6. **Summarization** → summarizer_agent
       - "Summarize this text", "Give me a summary of..."
       - Text condensation tasks
    
    DECISION PROCESS:
    1. Analyze the user's request to identify the primary intent
    2. Select the most appropriate specialist agent
    3. Delegate the task with clear context
    4. Monitor the response and handle any issues
    5. If an agent suggests re-delegation, consider the recommendation
    
    IMPORTANT RULES:
    - Always delegate to the most appropriate specialist
    - Don't try to handle specialized tasks yourself
    - If multiple agents could handle a task, choose the most specific one
    - Provide clear context when delegating
    - Handle gracefully if an agent is unavailable
    
    FALLBACK BEHAVIOR:
    - If a specialized agent fails, inform the user and offer alternatives
    - For ambiguous requests, ask for clarification before delegating
    - Always maintain a helpful and professional tone
    """,
    tools=agent_tools,
    sub_agents=all_sub_agents,
)

print(f"✅ Auto orchestrator initialized with {len(all_sub_agents)} sub-agents")
print("🌐 Ready for ADK web interface")

# Función de limpieza para cuando se cierre la aplicación
def cleanup_resources():
    """Limpia recursos async si es necesario."""
    try:
        for stack in exit_stacks:
            if hasattr(stack, '__aexit__'):
                asyncio.run(stack.__aexit__(None, None, None))
    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")

import atexit
atexit.register(cleanup_resources)