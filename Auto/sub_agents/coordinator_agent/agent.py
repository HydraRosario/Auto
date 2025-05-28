import os
from contextlib import AsyncExitStack
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

model = LiteLlm(
    model="gemini/gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# Sub-agent factories
from reddit_scout_agent import create_agent as create_reddit_scout_agent
from summarizer.agent import create_summarizer_agent
from speaker_agent.agent import create_agent as create_speaker_agent

async def create_coordinator_agent():
    """Creates the coordinator agent and its sub-agents."""
    
    exit_Stack = AsyncExitStack()
    await exit_stack.__aenter__()
    
    reddit_agent, reddit_stack = await create_reddit_scout_agent()
    await exit_stack.enter_async_context(reddit_stack)

    summarizer_agent = create_summarizer_agent()

    speaker_agent, speaker_stack = await create_speaker_agent()
    await exit_stack.enter_async_context(speaker_stack)

    coordinator_agent = Agent(
        name="coordinator_agent",
        model=model,
        description="Coordinator Management Agent that coordinates between subagents",
        instruction="""
        You are a manager agent that is responsible for overseeing the work of the other agents.
        
        Always delegate the task to the appropiate agent. Use your best judgement to determine which agent is the best fit for the task.

        You are responsible for delegating task to the following agent:
        - conversational_agent (for general conversation) 
        - web_searcher_agent (for web search)
        """,
        sub_agents=[reddit_agent, summarizer_agent, speaker_agent],
    )

    return coordinator_agent, exit_stack
    
root_agent = create_coordinator_agent()