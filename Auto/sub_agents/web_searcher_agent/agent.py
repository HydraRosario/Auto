import os
import asyncio
from typing import Dict, Any
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    print("⚠️ Firecrawl not available. Web search will be limited.")
    FIRECRAWL_AVAILABLE = False

def web_search_tool(query: str) -> Dict[str, Any]:
    """
    Enhanced web search function with error handling and fallback.
    """
    if not FIRECRAWL_AVAILABLE:
        return {
            "error": "Firecrawl library not available",
            "message": "Please install firecrawl: pip install firecrawl-py",
            "query": query
        }
    
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return {
            "error": "Missing FIRECRAWL_API_KEY",
            "message": "Please set FIRECRAWL_API_KEY environment variable",
            "query": query
        }
    
    try:
        app = FirecrawlApp(api_key=api_key)
        search_result = app.search(query, limit=5)
        
        if hasattr(search_result, 'data') and search_result.data:
            return {
                "success": True,
                "query": query,
                "results": search_result.data,
                "count": len(search_result.data)
            }
        else:
            return {
                "success": False,
                "query": query,
                "message": "No results found",
                "results": []
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "query": query,
            "message": "Failed to perform web search"
        }

def create_web_searcher_agent() -> Agent:
    """Create the web searcher agent."""
    
    model = LiteLlm(
        model="gemini/gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

    agent = Agent(
        name="web_searcher_agent",
        model=model,
        description="Advanced web search agent that finds and summarizes the most relevant information from the web.",
        instruction="""
        You are the Ultimate Web Searcher, specialized in finding and presenting relevant web information.
        
        CORE PROTOCOL:
        1. **MANDATORY TOOL CALL**: You MUST call the 'web_search_tool' for every search request
        2. **NO HALLUCINATION**: Never generate search results or summaries without calling the tool first
        3. **COMPREHENSIVE ANALYSIS**: After getting results, provide insightful analysis and synthesis
        4. **ERROR HANDLING**: If the tool fails, explain the issue and suggest alternatives
        
        SEARCH STRATEGY:
        - Use clear, focused search queries
        - Extract key information from multiple sources
        - Provide source attribution for all claims
        - Synthesize findings into coherent insights
        - Highlight conflicting information when present
        
        DELEGATION RULES:
        - If user wants to discuss search results → delegate to Auto for conversational_agent
        - If user needs follow-up searches → handle directly
        - If user requests Reddit-specific content → delegate to Auto for reddit_scout or coordinator
        
        RESPONSE FORMAT:
        1. Acknowledge the search query
        2. Call web_search_tool with optimized query
        3. Analyze and synthesize results
        4. Provide clear, well-structured summary
        5. Include relevant source links
        6. Suggest follow-up searches if helpful
        
        IMPORTANT: Your primary purpose is web search and information retrieval. Always use the search tool before providing any web-based information.
        """,
        tools=[web_search_tool]
    )
    
    return agent

web_searcher_agent = create_web_searcher_agent()

# For backward compatibility
async def get_web_searcher_agent():
    """Get the web searcher agent instance."""
    return web_searcher_agent