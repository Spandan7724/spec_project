"""
Web scraping tools for AI agents
"""

from .web_scraper import SimpleWebScraper, GenericScrapingInterface, ScrapingResult
from .cache_manager import CacheManager
from .decision_engine import DecisionEngine
from .agent_interface import AgentScrapingInterface

__all__ = [
    'SimpleWebScraper',
    'GenericScrapingInterface', 
    'ScrapingResult',
    'CacheManager',
    'DecisionEngine',
    'AgentScrapingInterface'
]