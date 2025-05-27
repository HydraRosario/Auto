import os

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from .sub_agents.conversational_agent.agent import conversational_agent
from .sub_agents.web_searcher_agent.agent import web_searcher_agent
from google.adk.models.lite_llm import LiteLlm

model = LiteLlm(
    model="groq/qwen-qwq-32b",
    api_key=os.getenv("GROQ_API_KEY"),
)

root_agent = Agent(
    name="Auto",
    model=model,
    description="Coordinator Management Agent that coordinates between subagents",
    instruction="""
    You are a manager agent that is responsible for overseeing the work of the other agents.
    
    Always delegate the task to the appropiate agent. Use your best judgement to determine which agent is the best fit for the task.

    You are responsible for delegating task to the following agent:
    - conversational_agent (for general conversation) 
    - web_searcher_agent (for web search)
    """,
    sub_agents=[conversational_agent, web_searcher_agent],
)