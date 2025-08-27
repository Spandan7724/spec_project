"""
Base Agent Framework for Multi-Agent Currency Decision System.

Provides the foundational agent class that all specialized agents inherit from,
with common functionality for LLM interaction, tool management, and state handling.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
import json

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from ..providers import ProviderManager, ProviderType, ChatResponse
from ..config import get_config_loader, AgentConfig

logger = logging.getLogger(__name__)


class AgentResult:
    """
    Standard result format for agent operations.
    
    Provides a consistent structure for agent outputs that can be
    easily consumed by other agents and the decision coordinator.
    """
    
    def __init__(self, 
                 agent_name: str,
                 success: bool = True,
                 data: Optional[Dict[str, Any]] = None,
                 reasoning: Optional[str] = None,
                 confidence: float = 0.0,
                 error_message: Optional[str] = None,
                 execution_time_ms: Optional[int] = None):
        """
        Initialize agent result.
        
        Args:
            agent_name: Name of the agent that produced this result
            success: Whether the operation was successful
            data: Structured data output from the agent
            reasoning: Natural language explanation of the agent's reasoning
            confidence: Confidence score (0.0 to 1.0)
            error_message: Error description if success=False
            execution_time_ms: Time taken to produce this result
        """
        self.agent_name = agent_name
        self.success = success
        self.data = data or {}
        self.reasoning = reasoning or ""
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
        self.error_message = error_message
        self.execution_time_ms = execution_time_ms
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "data": self.data,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"{status} {self.agent_name}(confidence={self.confidence:.2f}): {self.reasoning[:100]}..."


class BaseAgent(ABC):
    """
    Abstract base class for all currency decision agents.
    
    Provides common functionality for LLM interaction, tool management,
    configuration handling, and result formatting that all agents share.
    """
    
    def __init__(self, 
                 agent_name: str,
                 provider_manager: ProviderManager,
                 tools: Optional[List[BaseTool]] = None,
                 config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize base agent.
        
        Args:
            agent_name: Unique name for this agent
            provider_manager: LLM provider manager instance
            tools: List of tools this agent can use
            config_override: Override default configuration settings
        """
        self.agent_name = agent_name
        self.provider_manager = provider_manager
        self.tools = tools or []
        
        # Load configuration
        self._load_configuration(config_override)
        
        # Agent state
        self._conversation_history: List[Dict[str, Any]] = []
        self._execution_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "total_tokens_used": 0,
            "average_response_time_ms": 0
        }
        
        logger.debug(f"Initialized {self.agent_name} agent with {len(self.tools)} tools")
    
    def _load_configuration(self, config_override: Optional[Dict[str, Any]] = None) -> None:
        """Load agent configuration from config file and overrides."""
        try:
            config_loader = get_config_loader()
            config = config_loader.get_config()
            
            # Get agent-specific config
            agent_config = config.agents.get(self.agent_name)
            if agent_config is None:
                logger.warning(f"No configuration found for agent {self.agent_name}, using defaults")
                agent_config = AgentConfig(
                    name=self.agent_name,
                    temperature=0.3,
                    max_tokens=2000
                )
            
            # Store configuration
            self.config = agent_config
            
            # Apply any overrides
            if config_override:
                for key, value in config_override.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        logger.debug(f"Applied config override for {self.agent_name}: {key}={value}")
                    else:
                        logger.warning(f"Unknown config key for {self.agent_name}: {key}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration for {self.agent_name}: {e}")
            # Fallback to default configuration
            self.config = AgentConfig(
                name=self.agent_name,
                temperature=0.3,
                max_tokens=2000
            )
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        
        Each agent must define its role and capabilities in the system prompt.
        
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    async def process_request(self, request_data: Dict[str, Any]) -> AgentResult:
        """
        Process a request and return structured results.
        
        This is the main entry point for agent operations. Each agent
        implements its specific logic here.
        
        Args:
            request_data: Input data for the agent to process
            
        Returns:
            AgentResult with the agent's analysis and recommendations
        """
        pass
    
    async def chat_with_llm(self, 
                           user_message: str,
                           include_tools: bool = True,
                           conversation_context: Optional[List[Dict[str, Any]]] = None) -> ChatResponse:
        """
        Send a chat message to the LLM with agent context.
        
        Args:
            user_message: The user's message/request
            include_tools: Whether to include agent tools in the request
            conversation_context: Optional conversation history
            
        Returns:
            ChatResponse from the LLM
        """
        try:
            # Build message list
            messages = []
            
            # Add system prompt
            system_prompt = self.get_system_prompt()
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation context if provided
            if conversation_context:
                messages.extend(conversation_context)
            
            # Add user message
            messages.append({"role": "user", "content": user_message})
            
            # Prepare tools for the request
            tools = None
            if include_tools and self.tools:
                tools = self._format_tools_for_llm()
            
            # Get LLM parameters from configuration
            llm_params = {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            # Track request start time
            start_time = datetime.utcnow()
            
            # Make the request
            response = await self.provider_manager.chat(
                messages=messages,
                tools=tools,
                **llm_params
            )
            
            # Update execution stats
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_execution_stats(response, execution_time)
            
            # Store conversation history
            self._conversation_history.extend([
                {"role": "user", "content": user_message, "timestamp": start_time.isoformat()},
                {"role": "assistant", "content": response.content, "timestamp": datetime.utcnow().isoformat()}
            ])
            
            logger.debug(f"{self.agent_name} completed LLM request in {execution_time:.1f}ms")
            return response
            
        except Exception as e:
            logger.error(f"{self.agent_name} LLM request failed: {e}")
            self._execution_stats["total_requests"] += 1
            raise e
    
    def _format_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Format agent tools for LLM consumption."""
        formatted_tools = []
        
        for tool in self.tools:
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args if hasattr(tool, 'args') else {}
                }
            }
            formatted_tools.append(tool_def)
        
        return formatted_tools
    
    def _update_execution_stats(self, response: ChatResponse, execution_time_ms: float) -> None:
        """Update agent execution statistics."""
        self._execution_stats["total_requests"] += 1
        self._execution_stats["successful_requests"] += 1
        
        if response.usage:
            self._execution_stats["total_tokens_used"] += response.usage.get("total_tokens", 0)
        
        # Update average response time
        total_requests = self._execution_stats["total_requests"]
        current_avg = self._execution_stats["average_response_time_ms"]
        new_avg = ((current_avg * (total_requests - 1)) + execution_time_ms) / total_requests
        self._execution_stats["average_response_time_ms"] = new_avg
    
    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found
        """
        # Find the tool
        tool = None
        for t in self.tools:
            if t.name == tool_name:
                tool = t
                break
        
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in {self.agent_name} tools")
        
        try:
            # Execute the tool
            logger.debug(f"{self.agent_name} executing tool: {tool_name}")
            result = await tool.arun(**tool_args) if hasattr(tool, 'arun') else tool.run(**tool_args)
            
            return {
                "tool_name": tool_name,
                "success": True,
                "result": result,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"{self.agent_name} tool execution failed: {tool_name} - {e}")
            return {
                "tool_name": tool_name,
                "success": False,
                "result": None,
                "error": str(e)
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the agent's conversation history."""
        return self._conversation_history.copy()
    
    def clear_conversation_history(self) -> None:
        """Clear the agent's conversation history."""
        self._conversation_history.clear()
        logger.debug(f"Cleared conversation history for {self.agent_name}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get agent execution statistics."""
        return self._execution_stats.copy()
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools]
    
    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent's toolkit."""
        if tool.name not in self.get_available_tools():
            self.tools.append(tool)
            logger.debug(f"Added tool '{tool.name}' to {self.agent_name}")
        else:
            logger.warning(f"Tool '{tool.name}' already exists in {self.agent_name}")
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the agent's toolkit.
        
        Args:
            tool_name: Name of the tool to remove
            
        Returns:
            True if tool was removed, False if not found
        """
        for i, tool in enumerate(self.tools):
            if tool.name == tool_name:
                del self.tools[i]
                logger.debug(f"Removed tool '{tool_name}' from {self.agent_name}")
                return True
        
        logger.warning(f"Tool '{tool_name}' not found in {self.agent_name}")
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the agent.
        
        Returns:
            Health status dictionary
        """
        try:
            # Test basic LLM connectivity
            test_response = await self.chat_with_llm(
                "Health check - respond with 'OK'",
                include_tools=False
            )
            
            llm_healthy = "OK" in test_response.content.upper()
            
            return {
                "agent_name": self.agent_name,
                "llm_healthy": llm_healthy,
                "tools_count": len(self.tools),
                "configuration_loaded": self.config is not None,
                "total_requests": self._execution_stats["total_requests"],
                "success_rate": (
                    self._execution_stats["successful_requests"] / 
                    max(1, self._execution_stats["total_requests"])
                ),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed for {self.agent_name}: {e}")
            return {
                "agent_name": self.agent_name,
                "llm_healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def __str__(self) -> str:
        return f"{self.agent_name}(tools={len(self.tools)}, requests={self._execution_stats['total_requests']})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.agent_name}' tools={len(self.tools)}>"