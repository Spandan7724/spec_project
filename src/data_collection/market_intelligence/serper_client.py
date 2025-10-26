"""Serper.dev API client (news + general search)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import httpx

from src.utils.decorators import retry, timeout, log_execution
from src.utils.logging import get_logger
from src.config import get_config, load_config


logger = get_logger(__name__)


@dataclass
class SerperNewsResult:
    title: str
    url: str
    source: str
    snippet: str
    date: str
    position: int


@dataclass
class SerperSearchResult:
    title: str
    url: str
    snippet: str
    position: int


class SerperClient:
    """Client for Serper.dev API with news and search endpoints."""

    DEFAULT_WHITELISTED_DOMAINS = [
        "reuters.com",
        "bloomberg.com",
        "ft.com",
        "wsj.com",
        "apnews.com",
        "bbc.com",
        "cnbc.com",
        "marketwatch.com",
        "economist.com",
        "fxstreet.com",
        "forex.com",
        "investing.com",
        "tradingeconomics.com",
    ]

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        # Load config lazily for base_url/timeout; API key from argument or env
        try:
            cfg = get_config()
        except Exception:
            cfg = load_config()

        self.base_url = base_url or cfg.get("api.serper.base_url", "https://google.serper.dev")
        self.timeout = float(cfg.get("api.serper.timeout", 10))
        self.api_key = api_key or self._get_api_key_from_env()

        # Whitelist configuration (configurable + env override)
        cfg_whitelist = cfg.get("api.serper.domain_whitelist")
        self.whitelist_domains = (
            list(cfg_whitelist) if isinstance(cfg_whitelist, list) and cfg_whitelist else self.DEFAULT_WHITELISTED_DOMAINS
        )
        # Enable/disable via config or env SERPER_ENABLE_WHITELIST
        enable_from_cfg = cfg.get("api.serper.enable_whitelist", True)
        enable_from_env = self._get_bool_env("SERPER_ENABLE_WHITELIST")
        self.enable_whitelist = enable_from_env if enable_from_env is not None else bool(enable_from_cfg)

    def _get_api_key_from_env(self) -> str:
        import os
        key = os.getenv("SERPER_API_KEY")
        if not key:
            # For flexibility in unit tests, don't raise here if intentionally provided via __init__
            raise ValueError("SERPER_API_KEY not set and no api_key provided")
        return key

    @retry(max_attempts=3, delay=1.0, exceptions=(httpx.HTTPError, httpx.TimeoutException))
    @timeout(30.0)
    @log_execution(log_args=False, log_result=False)
    async def search_news(self, query: str, time_range: str = "qdr:d", num_results: int = 20) -> List[SerperNewsResult]:
        """Search news via /news endpoint and filter by whitelisted domains."""
        logger.info("Serper news search", extra={"query": query, "tbs": time_range, "num": num_results})
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/news",
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                json={"q": query, "tbs": time_range, "num": num_results},
            )
            resp.raise_for_status()
            data = resp.json() or {}

        items = data.get("news", [])
        results: List[SerperNewsResult] = []
        for idx, article in enumerate(items):
            url = article.get("link", "")
            allowed = True
            if self.enable_whitelist:
                allowed = url and any(domain in url for domain in self.whitelist_domains)
            if url and allowed:
                results.append(
                    SerperNewsResult(
                        title=article.get("title", ""),
                        url=url,
                        source=article.get("source", ""),
                        snippet=article.get("snippet", ""),
                        date=article.get("date", ""),
                        position=idx + 1,
                    )
                )
        return results

    @retry(max_attempts=3, delay=1.0, exceptions=(httpx.HTTPError, httpx.TimeoutException))
    @timeout(30.0)
    @log_execution(log_args=False, log_result=False)
    async def search(self, query: str, num_results: int = 10) -> List[SerperSearchResult]:
        """General search via /search endpoint (used for calendar URLs)."""
        logger.info("Serper general search", extra={"query": query, "num": num_results})
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/search",
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                json={"q": query, "num": num_results},
            )
            resp.raise_for_status()
            data = resp.json() or {}

        results: List[SerperSearchResult] = []
        for idx, r in enumerate(data.get("organic", [])):
            results.append(
                SerperSearchResult(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                    position=idx + 1,
                )
            )
        return results

    async def health_check(self) -> bool:
        try:
            _ = await self.search("test", num_results=1)
            return True
        except Exception as e:
            logger.error(f"Serper health check failed: {e}")
            return False

    @staticmethod
    def _get_bool_env(var_name: str) -> Optional[bool]:
        import os
        val = os.getenv(var_name)
        if val is None:
            return None
        truthy = {"1", "true", "yes", "on"}
        falsy = {"0", "false", "no", "off"}
        v = val.strip().lower()
        if v in truthy:
            return True
        if v in falsy:
            return False
        return None
