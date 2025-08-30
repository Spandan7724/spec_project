#!/usr/bin/env python3
"""
Generic Web Scraping Tool for AI Agents
Simple, flexible tool that scrapes content without domain-specific filtering
"""
import asyncio
import sys
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from crawl4ai import AsyncWebCrawler

from .cache_manager import CacheManager

@dataclass
class ScrapingResult:
    """Result from web scraping operation"""
    url: str
    success: bool
    status_code: int
    content: str
    cached: bool = False
    cache_age_seconds: Optional[float] = None
    error: Optional[str] = None
    timestamp: Optional[float] = None

class SimpleWebScraper:
    """
    Generic web scraping tool for AI agents
    No hardcoded filtering - lets agents decide what to extract
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_manager = CacheManager(cache_dir)
        self.crawler = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.crawler = AsyncWebCrawler(verbose=False)
        await self.crawler.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
    
    async def scrape_url(self, url: str, bypass_cache: bool = False,
                        content_type: str = 'general') -> ScrapingResult:
        """
        Scrape a single URL with caching
        
        Args:
            url: Target URL to scrape
            bypass_cache: Force fresh scraping (ignore cache)
            content_type: Content type for cache TTL (optional)
            
        Returns:
            ScrapingResult with raw content - no filtering
        """
        # Check cache first (unless bypassed)
        if not bypass_cache:
            cached_entry = self.cache_manager.get(url, content_type)
            if cached_entry:
                return ScrapingResult(
                    url=url,
                    success=True,
                    status_code=200,
                    content=cached_entry.content,
                    cached=True,
                    cache_age_seconds=cached_entry.age_seconds(),
                    timestamp=cached_entry.timestamp
                )
        
        # Perform fresh scraping
        try:
            result = await self.crawler.arun(url=url, word_count_threshold=10)
            
            if result.status_code == 200:
                # Convert content to string if needed
                content_str = str(result.markdown) if hasattr(result.markdown, '__str__') else result.markdown
                
                # Cache the result with empty extracted_data (agent will extract)
                self.cache_manager.set(url, content_str, {}, content_type)
                
                return ScrapingResult(
                    url=url,
                    success=True,
                    status_code=result.status_code,
                    content=content_str,
                    cached=False,
                    timestamp=time.time()
                )
            else:
                return ScrapingResult(
                    url=url,
                    success=False,
                    status_code=result.status_code,
                    content="",
                    error=f"HTTP {result.status_code}",
                    timestamp=time.time()
                )
                
        except Exception as e:
            return ScrapingResult(
                url=url,
                success=False,
                status_code=0,
                content="",
                error=str(e),
                timestamp=time.time()
            )
    
    async def scrape_multiple_urls(self, urls: List[str], bypass_cache: bool = False,
                                 content_type: str = 'general', 
                                 max_concurrent: int = 3) -> List[ScrapingResult]:
        """
        Scrape multiple URLs concurrently
        
        Args:
            urls: List of URLs to scrape
            bypass_cache: Force fresh scraping for all URLs
            content_type: Content type for cache TTL
            max_concurrent: Maximum concurrent scraping operations
            
        Returns:
            List of ScrapingResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url):
            async with semaphore:
                return await self.scrape_url(url, bypass_cache, content_type)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed ScrapingResults
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ScrapingResult(
                    url=urls[i],
                    success=False,
                    status_code=0,
                    content="",
                    error=str(result),
                    timestamp=time.time()
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def should_bypass_cache(self, query_context: str = "") -> bool:
        """
        Simple cache bypass logic based on query keywords
        
        Args:
            query_context: User query or conversation context
            
        Returns:
            True if cache should be bypassed
        """
        return self.cache_manager.should_bypass_cache_for_query(query_context)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics"""
        return self.cache_manager.get_cache_stats()
    
    async def clear_cache(self):
        """Clear expired cache entries"""
        self.cache_manager.clear_expired()

# Simple agent interface
class GenericScrapingInterface:
    """
    Simplified interface for agents - minimal assumptions about content
    """
    
    @staticmethod
    async def scrape(url: str, query_context: str = "", 
                    bypass_cache: bool = False) -> Dict[str, Any]:
        """
        Simple scraping method - just get content, let agent process
        
        Args:
            url: URL to scrape
            query_context: Optional context for cache decisions
            bypass_cache: Force fresh data
            
        Returns:
            Simple result dict with content
        """
        async with SimpleWebScraper() as scraper:
            # Auto-bypass cache for time-sensitive queries
            if not bypass_cache:
                bypass_cache = scraper.should_bypass_cache(query_context)
            
            result = await scraper.scrape_url(url, bypass_cache)
            
            return {
                'success': result.success,
                'content': result.content,
                'url': result.url,
                'cached': result.cached,
                'cache_age_seconds': result.cache_age_seconds,
                'error': result.error,
                'timestamp': result.timestamp
            }
    
    @staticmethod
    async def scrape_multiple(urls: List[str], query_context: str = "",
                            bypass_cache: bool = False) -> Dict[str, Any]:
        """
        Scrape multiple URLs and return aggregated results
        
        Args:
            urls: List of URLs to scrape
            query_context: Optional context for cache decisions
            bypass_cache: Force fresh data
            
        Returns:
            Aggregated results from all URLs
        """
        async with SimpleWebScraper() as scraper:
            if not bypass_cache:
                bypass_cache = scraper.should_bypass_cache(query_context)
            
            results = await scraper.scrape_multiple_urls(urls, bypass_cache)
            
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            # Combine all content
            all_content = []
            sources = []
            errors = []
            
            for result in successful_results:
                all_content.append(f"=== {result.url} ===\n{result.content}")
                cache_info = f" (cached, {result.cache_age_seconds:.0f}s old)" if result.cached else ""
                sources.append(f"{result.url}{cache_info}")
            
            for result in failed_results:
                errors.append(f"{result.url}: {result.error}")
            
            return {
                'success': len(successful_results) > 0,
                'content': '\n\n'.join(all_content),
                'sources': sources,
                'total_sources': len(urls),
                'successful_sources': len(successful_results),
                'errors': errors,
                'timestamp': time.time()
            }

# CLI test functionality
async def cli_test():
    """Command-line test interface"""
    if len(sys.argv) < 2:
        print("Usage: python simple_web_scraper.py \"query\" [url1,url2,...]")
        print("\nExample:")
        print("  python simple_web_scraper.py \"latest USD EUR rate\"")
        print("  python simple_web_scraper.py \"test\" \"https://xe.com,https://wise.com\"")
        return
    
    query = sys.argv[1]
    
    if len(sys.argv) > 2:
        urls = [url.strip() for url in sys.argv[2].split(',')]
        print(f"Scraping custom URLs for: '{query}'")
        result = await GenericScrapingInterface.scrape_multiple(urls, query)
    else:
        # Just demonstrate single URL scraping
        test_url = "https://www.xe.com/currencyconverter/"
        print(f"Scraping test URL for: '{query}'")
        result = await GenericScrapingInterface.scrape(test_url, query)
    
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(cli_test())