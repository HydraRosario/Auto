import os
from contextlib import AsyncExitStack
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# Import factory functions for sub-agents that coordinator_agent will manage
from ..reddit_scout_agent.agent import create_agent as create_reddit_scout_agent
#from ..summarizer_agent.agent import create_summarizer_agent 
from ..speaker_agent.agent import create_speaker_agent

# Model for the coordinator_agent itself
coordinator_model = LiteLlm(
    model="gemini/gemini-1.5-flash", 
    api_key=os.getenv("GOOGLE_API_KEY"),
)

async def create_coordinator_agent():
    """Creates the coordinator agent and its internally managed sub-agents (news pipeline)."""
    
    # This exit_stack is for resources managed by coordinator_agent's sub-agents
    internal_exit_stack = AsyncExitStack()
    await internal_exit_stack.__aenter__() 

    # 1. Create Reddit Scout Agent (asynchronous)
    reddit_agent, reddit_stack = await create_reddit_scout_agent()
    if hasattr(reddit_stack, '__aenter__'):
        await internal_exit_stack.enter_async_context(reddit_stack)

    # 2. Create Summarizer Agent (synchronous)
    #summarizer_agent_instance = create_summarizer_agent() 

    # 3. Create Speaker Agent (asynchronous)
    speaker_agent_instance, speaker_stack = await create_speaker_agent()
    if hasattr(speaker_stack, '__aenter__'):
        await internal_exit_stack.enter_async_context(speaker_stack)

    pipeline_sub_agents = {
        "reddit_scout": reddit_agent,
        #"summarizer": summarizer_agent_instance,
        "speaker": speaker_agent_instance
    }

    coordinator_agent_instance = Agent(
        name="news_pipeline_coordinator_agent", 
        model=coordinator_model,
        description="Coordinates a pipeline of agents (Reddit scout, summarizer, speaker) to generate and present news briefings.",
        instruction="""
        You are a News Pipeline Coordinator. Your goal is to produce a news briefing from Reddit.
        Follow these steps:
        1. Determine the subreddit from the user's request.
        2. Use the 'reddit_scout' agent to fetch hot posts from that subreddit.
        3. Take the output from 'reddit_scout' and pass it to the 'summarizer' agent to get a newscaster-style summary.
        4. Take the summary from 'summarizer' and pass it to the 'speaker' agent to convert it to speech (use voice 'Will').
        5. Present the final result, which might be the spoken audio URL or the text summary if speech fails.
        
        You have access to the following internal agents to achieve this:
        - reddit_scout: Fetches Reddit posts.
        - summarizer: Summarizes text in a newscaster style.
        - speaker: Converts text to speech.

        If any step fails, report the failure gracefully.
        """,
        sub_agents=[reddit_agent, speaker_agent_instance], 
    )

    return coordinator_agent_instance, internal_exit_stack
    
# root_agent = create_coordinator_agent() 