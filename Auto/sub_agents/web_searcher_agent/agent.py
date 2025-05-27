import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from firecrawl import FirecrawlApp

model = LiteLlm(
    model="gemini/gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
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
        "IMPORTANTE: NUNCA devuelvas respuestas como 'agent_name': 'conversational_agent' y siempre debes delegar a otro agente correctamente."
        "A veces el usuario hará una pregunta inmediatamente después de hacer una búsqueda, ya que querrá conversar sobre la información que buscó. En este caso, debes delegar a conversational_agent. Tu propósito principal es la búsqueda, no la conversación."
    ),
    tools=[web_search]
)