#!/usr/bin/env python3
"""
Decision Engine for Web Scraping Tool
Determines when and how agents should scrape for information
"""
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ScrapingDecision:
    """Decision result for whether to scrape"""
    should_scrape: bool
    reason: str
    suggested_urls: List[str]
    content_type: str
    confidence: float  # 0.0 to 1.0

class DecisionEngine:
    """
    Autonomous decision-making for when agents should scrape web content
    """
    
    # Time-sensitive keywords that trigger immediate scraping
    TIME_SENSITIVE_KEYWORDS = [
        'today', 'latest', 'current', 'now', 'just announced', 'breaking',
        'recent', 'this week', 'yesterday', 'updated', 'new', 'fresh'
    ]
    
    # Provider-specific keywords
    PROVIDER_KEYWORDS = {
        'wise': ['wise', 'transferwise'],
        'remitly': ['remitly'],
        'xe': ['xe.com', 'xe money'],
        'revolut': ['revolut'],
        'paypal': ['paypal'],
        'western_union': ['western union', 'wu'],
    }
    
    # Economic/regulatory keywords
    ECONOMIC_KEYWORDS = [
        'fed', 'federal reserve', 'ecb', 'european central bank',
        'interest rate', 'monetary policy', 'inflation', 'gdp',
        'regulatory', 'regulation', 'policy change'
    ]
    
    # URL patterns for different content types
    URL_PATTERNS = {
        'exchange_rates': [
            'https://www.xe.com/currencyconverter/',
            'https://wise.com/gb/currency-converter/',
            'https://www.exchangerates.org.uk/',
        ],
        'economic_news': [
            'https://www.ft.com/search?q=',
            'https://www.reuters.com/search/news?blob=',
            'https://www.bloomberg.com/search?query=',
        ],
        'provider_policies': [
            'https://wise.com/help/',
            'https://www.remitly.com/us/en/help/',
            'https://www.xe.com/legal/',
        ],
        'regulatory_changes': [
            'https://www.federalreserve.gov/newsevents/',
            'https://www.ecb.europa.eu/press/',
            'https://www.bis.org/press/',
        ]
    }
    
    def analyze_query(self, user_query: str, conversation_context: str = "") -> ScrapingDecision:
        """
        Analyze user query to determine if scraping is needed
        
        Args:
            user_query: Current user question/request
            conversation_context: Previous conversation for context
            
        Returns:
            ScrapingDecision with recommendation
        """
        query_lower = user_query.lower()
        context_lower = conversation_context.lower()
        combined_text = f"{query_lower} {context_lower}"
        
        # Check for time-sensitive triggers
        time_sensitive_score = self._check_time_sensitivity(combined_text)
        
        # Check for provider-specific queries
        provider_score, detected_provider = self._check_provider_mentions(combined_text)
        
        # Check for economic/regulatory content
        economic_score = self._check_economic_content(combined_text)
        
        # Check for knowledge gaps (questions about unknown entities)
        gap_score = self._check_knowledge_gaps(user_query)
        
        # Calculate overall confidence and decision
        max_score = max(time_sensitive_score, provider_score, economic_score, gap_score)
        
        if max_score > 0.4:
            content_type, urls = self._determine_content_type_and_urls(
                combined_text, detected_provider
            )
            
            reason = self._generate_reason(
                time_sensitive_score, provider_score, 
                economic_score, gap_score, detected_provider
            )
            
            return ScrapingDecision(
                should_scrape=True,
                reason=reason,
                suggested_urls=urls,
                content_type=content_type,
                confidence=max_score
            )
        else:
            return ScrapingDecision(
                should_scrape=False,
                reason="Existing APIs and cached data should be sufficient",
                suggested_urls=[],
                content_type='general',
                confidence=1.0 - max_score
            )
    
    def _check_time_sensitivity(self, text: str) -> float:
        """Check for time-sensitive keywords"""
        matches = sum(1 for keyword in self.TIME_SENSITIVE_KEYWORDS if keyword in text)
        return min(matches * 0.5, 1.0)
    
    def _check_provider_mentions(self, text: str) -> Tuple[float, Optional[str]]:
        """Check for specific provider mentions"""
        for provider, keywords in self.PROVIDER_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return 0.8, provider
        return 0.0, None
    
    def _check_economic_content(self, text: str) -> float:
        """Check for economic/regulatory content"""
        matches = sum(1 for keyword in self.ECONOMIC_KEYWORDS if keyword in text)
        return min(matches * 0.25, 1.0)
    
    def _check_knowledge_gaps(self, query: str) -> float:
        """
        Check if query asks about something that might not be in our data
        """
        gap_indicators = [
            'what about', 'how about', 'but what if', 'i heard that',
            'is it true that', 'did you know', 'what if', 'but'
        ]
        
        query_lower = query.lower()
        matches = sum(1 for indicator in gap_indicators if indicator in query_lower)
        return min(matches * 0.4, 1.0)
    
    def _determine_content_type_and_urls(self, text: str, 
                                       detected_provider: Optional[str]) -> Tuple[str, List[str]]:
        """Determine content type and suggest URLs"""
        
        # Provider-specific scraping
        if detected_provider:
            return 'provider_policies', self._get_provider_urls(detected_provider)
        
        # Economic content
        if any(keyword in text for keyword in self.ECONOMIC_KEYWORDS):
            return 'economic_news', self.URL_PATTERNS['economic_news'][:2]
        
        # Exchange rate content
        if 'rate' in text or 'exchange' in text:
            return 'exchange_rates', self.URL_PATTERNS['exchange_rates'][:2]
        
        # Default to general news
        return 'economic_news', self.URL_PATTERNS['economic_news'][:1]
    
    def _get_provider_urls(self, provider: str) -> List[str]:
        """Get scraping URLs for specific provider"""
        provider_url_map = {
            'wise': [
                'https://wise.com/help/',
                'https://wise.com/gb/send-money/'
            ],
            'remitly': [
                'https://www.remitly.com/us/en/help/',
                'https://www.remitly.com/'
            ],
            'xe': [
                'https://www.xe.com/money-transfer/',
                'https://www.xe.com/legal/'
            ],
            'revolut': [
                'https://www.revolut.com/help/',
                'https://www.revolut.com/transfer-money/'
            ]
        }
        
        return provider_url_map.get(provider, [])
    
    def _generate_reason(self, time_score: float, provider_score: float,
                        economic_score: float, gap_score: float,
                        detected_provider: Optional[str]) -> str:
        """Generate human-readable reason for scraping decision"""
        reasons = []
        
        if time_score > 0.5:
            reasons.append("query requests current/latest information")
        
        if provider_score > 0.5:
            reasons.append(f"specific provider ({detected_provider}) mentioned")
        
        if economic_score > 0.5:
            reasons.append("economic/regulatory content detected")
        
        if gap_score > 0.5:
            reasons.append("potential knowledge gap identified")
        
        return "; ".join(reasons) if reasons else "general information request"
    
    def should_bypass_cache_for_query(self, query: str) -> bool:
        """
        Quick check if query indicates need for fresh data
        """
        return any(keyword in query.lower() for keyword in self.TIME_SENSITIVE_KEYWORDS)
    
    def classify_content_type(self, url: str, content: str) -> str:
        """
        Classify content type based on URL and content analysis
        
        Args:
            url: Source URL
            content: Scraped content
            
        Returns:
            Content type classification
        """
        url_lower = url.lower()
        content_lower = content.lower()
        
        # URL-based classification
        if 'currency' in url_lower or 'exchange' in url_lower:
            return 'exchange_rates'
        
        if 'news' in url_lower or 'press' in url_lower:
            return 'economic_news'
        
        if 'help' in url_lower or 'legal' in url_lower or 'terms' in url_lower:
            return 'provider_policies'
        
        if any(domain in url_lower for domain in ['federalreserve.gov', 'ecb.europa.eu']):
            return 'regulatory_changes'
        
        # Content-based classification
        economic_terms = ['interest rate', 'monetary policy', 'inflation', 'gdp']
        if any(term in content_lower for term in economic_terms):
            return 'economic_news'
        
        provider_terms = ['fee', 'transfer', 'send money', 'exchange rate']
        if any(term in content_lower for term in provider_terms):
            return 'provider_policies'
        
        return 'general'