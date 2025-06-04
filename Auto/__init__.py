"""
Auto Multi-Agent Orchestrator

This module provides an advanced multi-agent system optimized for Google ADK web interface.
"""

# Import the root agent directly for ADK web compatibility
from .agent import root_agent

# Export for ADK
__all__ = ['root_agent']