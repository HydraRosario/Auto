import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from firecrawl import FirecrawlApp

model = LiteLlm(
    model="groq/qwen-qwq-32b",
    api_key=os.getenv("GROQ_API_KEY"),
)

def web_search(query: str) -> dict:
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    search_result = app.search(query, limit=5)
    return search_result.data

web_searcher_agent = Agent(
    name="web_searcher_agent",
    model=model,
    description="A web search agent that searches for the most relevant posts on the web.",
    instruction=(
        "You are the ultimate Web Searcher. Your primary task is to fetch and summarize the most relevant posts on the web. "
        "MUST CALL TOOL: you ***MUST*** call the 'web_search' tool to search the web. do NOT, EVER, generate summaries without calling the tool first."
        "IMPORTANTE: NUNCA devuelvas respuestas como 'agent_name': 'conversational_agent', Nunca debes responder después del resumen entregado, y siempre debes delegar a otro agente. Tu propósito principal es la búsqueda, no la conversación."
    ),
    tools=[web_search]
)