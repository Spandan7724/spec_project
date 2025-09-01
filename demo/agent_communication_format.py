#!/usr/bin/env python3
"""
Demo: Agent Communication Format for Multi-Agent System
Shows the standardized message format for agent-to-agent communication
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

class MessageType(Enum):
    """Types of messages agents can send"""
    DATA_REQUEST = "data_request"
    DATA_RESPONSE = "data_response"  
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESPONSE = "analysis_response"
    DECISION_REQUEST = "decision_request"
    DECISION_RESPONSE = "decision_response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    COORDINATION = "coordination"

class Priority(Enum):
    """Message priority levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class AgentType(Enum):
    """Types of agents in the system"""
    DATA_COLLECTOR = "data_collector"
    ML_PREDICTOR = "ml_predictor"
    MARKET_ANALYZER = "market_analyzer"
    RISK_ASSESSOR = "risk_assessor"
    DECISION_MAKER = "decision_maker"
    COORDINATOR = "coordinator"
    WEB_SCRAPER = "web_scraper"
    NEWS_ANALYZER = "news_analyzer"

@dataclass
class AgentMessage:
    """Standard agent communication message format"""
    
    # Message metadata
    message_id: str
    message_type: MessageType
    priority: Priority
    timestamp: str
    
    # Agent identification
    sender_id: str
    sender_type: AgentType
    recipient_id: str
    recipient_type: AgentType
    
    # Message content
    content: Dict[str, Any]
    
    # Context and routing
    conversation_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Metadata
    ttl_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "sender_id": self.sender_id,
            "sender_type": self.sender_type.value,
            "recipient_id": self.recipient_id, 
            "recipient_type": self.recipient_type.value,
            "content": self.content,
            "conversation_id": self.conversation_id,
            "parent_message_id": self.parent_message_id,
            "correlation_id": self.correlation_id,
            "ttl_seconds": self.ttl_seconds,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

def create_sample_agent_communications() -> List[AgentMessage]:
    """
    Create sample agent communication messages demonstrating different scenarios
    """
    
    base_time = datetime.now()
    messages = []
    
    # 1. Coordinator requests data from data collector
    messages.append(AgentMessage(
        message_id="msg_001",
        message_type=MessageType.DATA_REQUEST,
        priority=Priority.HIGH,
        timestamp=base_time.isoformat(),
        sender_id="coordinator_001",
        sender_type=AgentType.COORDINATOR,
        recipient_id="data_collector_fx_001",
        recipient_type=AgentType.DATA_COLLECTOR,
        conversation_id="conv_currency_analysis_001",
        content={
            "request_type": "exchange_rates",
            "currency_pairs": ["USD/EUR", "USD/GBP"],
            "data_sources": ["alpha_vantage", "yahoo_finance", "exchangerate_host"],
            "urgency": "immediate",
            "use_cache": True,
            "max_age_minutes": 5
        },
        ttl_seconds=300
    ))
    
    # 2. Data collector responds with rate data
    messages.append(AgentMessage(
        message_id="msg_002",
        message_type=MessageType.DATA_RESPONSE,
        priority=Priority.HIGH,
        timestamp=(base_time.timestamp() + 15).__str__(),
        sender_id="data_collector_fx_001",
        sender_type=AgentType.DATA_COLLECTOR,
        recipient_id="coordinator_001",
        recipient_type=AgentType.COORDINATOR,
        conversation_id="conv_currency_analysis_001",
        parent_message_id="msg_001",
        content={
            "request_fulfilled": True,
            "data_type": "exchange_rates",
            "data": {
                "USD/EUR": {
                    "rate": 0.9245,
                    "timestamp": base_time.isoformat(),
                    "sources": ["alpha_vantage", "yahoo_finance"],
                    "confidence": "high"
                },
                "USD/GBP": {
                    "rate": 0.7856,
                    "timestamp": base_time.isoformat(),
                    "sources": ["yahoo_finance"],
                    "confidence": "medium"
                }
            },
            "metadata": {
                "collection_time_ms": 1234,
                "success_rate": 0.67,
                "cache_used": False
            }
        }
    ))
    
    # 3. Coordinator requests ML prediction
    messages.append(AgentMessage(
        message_id="msg_003",
        message_type=MessageType.ANALYSIS_REQUEST,
        priority=Priority.MEDIUM,
        timestamp=(base_time.timestamp() + 30).__str__(),
        sender_id="coordinator_001",
        sender_type=AgentType.COORDINATOR,
        recipient_id="ml_predictor_001",
        recipient_type=AgentType.ML_PREDICTOR,
        conversation_id="conv_currency_analysis_001",
        correlation_id="analysis_usd_eur_001",
        content={
            "analysis_type": "price_prediction",
            "currency_pair": "USD/EUR",
            "current_rate": 0.9245,
            "horizons": [1, 7, 30],
            "include_confidence": True,
            "context_data": {
                "recent_rates": [0.9245, 0.9238, 0.9241],
                "volatility": "moderate",
                "market_conditions": "stable"
            }
        },
        ttl_seconds=600
    ))
    
    # 4. ML predictor responds with forecast
    messages.append(AgentMessage(
        message_id="msg_004", 
        message_type=MessageType.ANALYSIS_RESPONSE,
        priority=Priority.MEDIUM,
        timestamp=(base_time.timestamp() + 45).__str__(),
        sender_id="ml_predictor_001",
        sender_type=AgentType.ML_PREDICTOR,
        recipient_id="coordinator_001",
        recipient_type=AgentType.COORDINATOR,
        conversation_id="conv_currency_analysis_001",
        parent_message_id="msg_003",
        correlation_id="analysis_usd_eur_001",
        content={
            "analysis_completed": True,
            "analysis_type": "price_prediction",
            "currency_pair": "USD/EUR",
            "model_id": "lstm_usdeur_v2.1",
            "model_confidence": 0.78,
            "predictions": {
                "1_day": {"mean": 0.9248, "p10": 0.9201, "p90": 0.9295},
                "7_day": {"mean": 0.9235, "p10": 0.9150, "p90": 0.9320},
                "30_day": {"mean": 0.9220, "p10": 0.9080, "p90": 0.9360}
            },
            "direction_probabilities": {
                "1_day": 0.52, "7_day": 0.48, "30_day": 0.42
            },
            "metadata": {
                "processing_time_ms": 2345,
                "features_used": 47,
                "data_quality": 0.95
            }
        }
    ))
    
    # 5. Risk assessor provides risk analysis
    messages.append(AgentMessage(
        message_id="msg_005",
        message_type=MessageType.ANALYSIS_RESPONSE,
        priority=Priority.MEDIUM,
        timestamp=(base_time.timestamp() + 60).__str__(),
        sender_id="risk_assessor_001",
        sender_type=AgentType.RISK_ASSESSOR,
        recipient_id="coordinator_001", 
        recipient_type=AgentType.COORDINATOR,
        conversation_id="conv_currency_analysis_001",
        content={
            "risk_analysis": {
                "overall_risk": "moderate",
                "volatility_risk": "low",
                "market_timing_risk": "medium",
                "liquidity_risk": "low",
                "event_risk": "high",
                "recommendation": "proceed_with_caution",
                "risk_factors": [
                    "Fed meeting next week",
                    "ECB rate decision pending",
                    "Geopolitical uncertainty"
                ],
                "suggested_actions": [
                    "Split large conversions",
                    "Monitor news closely",
                    "Set tight stop losses"
                ]
            }
        }
    ))
    
    # 6. Decision maker provides final recommendation
    messages.append(AgentMessage(
        message_id="msg_006",
        message_type=MessageType.DECISION_RESPONSE,
        priority=Priority.HIGH,
        timestamp=(base_time.timestamp() + 75).__str__(),
        sender_id="decision_maker_001",
        sender_type=AgentType.DECISION_MAKER,
        recipient_id="coordinator_001",
        recipient_type=AgentType.COORDINATOR,
        conversation_id="conv_currency_analysis_001",
        content={
            "decision": {
                "recommendation": "wait",
                "confidence": 0.72,
                "reasoning": "High event risk with Fed meeting approaching outweighs modest positive ML signals",
                "suggested_timing": "after_fed_meeting",
                "alternative_strategies": [
                    {
                        "strategy": "partial_conversion",
                        "amount_percent": 25,
                        "timing": "immediate"
                    },
                    {
                        "strategy": "dollar_cost_averaging",
                        "schedule": "weekly_25_percent",
                        "duration_weeks": 4
                    }
                ],
                "monitoring_triggers": [
                    "fed_rate_decision",
                    "rate_moves_below_0.9200",
                    "volatility_spike_above_20"
                ]
            }
        }
    ))
    
    # 7. Error message example
    messages.append(AgentMessage(
        message_id="msg_007",
        message_type=MessageType.ERROR,
        priority=Priority.CRITICAL,
        timestamp=(base_time.timestamp() + 90).__str__(),
        sender_id="data_collector_news_001",
        sender_type=AgentType.NEWS_ANALYZER,
        recipient_id="coordinator_001",
        recipient_type=AgentType.COORDINATOR,
        conversation_id="conv_currency_analysis_001",
        content={
            "error": {
                "error_code": "RATE_LIMIT_EXCEEDED",
                "error_message": "News API rate limit exceeded, falling back to cached data",
                "severity": "warning",
                "recoverable": True,
                "suggested_action": "retry_after_cooldown",
                "cooldown_seconds": 3600,
                "fallback_data_available": True,
                "fallback_data_age_minutes": 45
            }
        }
    ))
    
    return messages

def format_agent_communication_standard() -> Dict[str, Any]:
    """
    Define the standard format and rules for agent communication
    """
    
    return {
        "communication_protocol": {
            "version": "1.0",
            "message_format": "AgentMessage",
            "serialization": "JSON",
            "transport": "async_queue",
            "acknowledgment_required": True,
            "max_message_size_kb": 1024,
            "default_ttl_seconds": 300
        },
        
        "message_types": {
            message_type.value: {
                "description": f"Message type for {message_type.name}",
                "expected_content_fields": []  # Would define required fields per type
            }
            for message_type in MessageType
        },
        
        "priority_handling": {
            Priority.CRITICAL.value: {"max_queue_time_ms": 100, "retry_immediately": True},
            Priority.HIGH.value: {"max_queue_time_ms": 500, "retry_delay_ms": 1000},
            Priority.MEDIUM.value: {"max_queue_time_ms": 2000, "retry_delay_ms": 5000},
            Priority.LOW.value: {"max_queue_time_ms": 10000, "retry_delay_ms": 30000}
        },
        
        "agent_roles": {
            agent_type.value: {
                "description": f"{agent_type.name} agent responsibilities",
                "can_send_to": [t.value for t in AgentType],  # All agents can talk to all agents
                "typical_message_types": []  # Would define common message types per agent
            }
            for agent_type in AgentType
        },
        
        "conversation_management": {
            "conversation_id_required": True,
            "parent_message_tracking": True,
            "correlation_id_for_analysis": True,
            "max_conversation_duration_minutes": 60,
            "auto_cleanup_completed_conversations": True
        },
        
        "error_handling": {
            "max_retries_default": 3,
            "exponential_backoff": True,
            "dead_letter_queue": True,
            "error_escalation": True,
            "circuit_breaker_pattern": True
        },
        
        "monitoring_and_observability": {
            "message_tracing": True,
            "performance_metrics": True,
            "conversation_analytics": True,
            "agent_health_monitoring": True,
            "bottleneck_detection": True
        }
    }

def main():
    """Generate and display agent communication format examples"""
    
    print("=== Agent Communication Format Demo for Multi-Agent System ===\n")
    
    # Create sample messages
    messages = create_sample_agent_communications()
    
    print(f"Generated {len(messages)} sample agent messages")
    print("Message types:", set([msg.message_type.value for msg in messages]))
    print("Agent types involved:", set([msg.sender_type.value for msg in messages] + [msg.recipient_type.value for msg in messages]))
    print()
    
    print("=== Sample Agent Messages ===")
    for i, message in enumerate(messages, 1):
        print(f"\n{i}. {message.message_type.value.upper()}: {message.sender_type.value} → {message.recipient_type.value}")
        print(f"   ID: {message.message_id}")
        print(f"   Priority: {message.priority.value}")
        print(f"   Conversation: {message.conversation_id}")
        if message.parent_message_id:
            print(f"   Reply to: {message.parent_message_id}")
        print(f"   Content preview: {str(message.content)[:100]}...")
    
    print("\n=== JSON Message Format Examples ===")
    
    # Show a few messages in full JSON format
    for msg_type in [MessageType.DATA_REQUEST, MessageType.ANALYSIS_RESPONSE, MessageType.ERROR]:
        sample_msg = next((msg for msg in messages if msg.message_type == msg_type), None)
        if sample_msg:
            print(f"\n{msg_type.value.upper()} Message:")
            print(json.dumps(sample_msg.to_dict(), indent=2))
    
    print("\n=== Communication Protocol Standard ===")
    protocol = format_agent_communication_standard()
    print(json.dumps(protocol, indent=2))
    
    print("\n=== Key Points for Multi-Agent Implementation ===")
    print("• All agents use standardized AgentMessage format")
    print("• Messages include conversation tracking and correlation IDs")
    print("• Priority-based message handling with different queue times")
    print("• Built-in retry logic with exponential backoff")
    print("• Error messages include recovery suggestions")
    print("• TTL prevents stale messages from being processed")
    print("• Comprehensive metadata for monitoring and debugging")

if __name__ == "__main__":
    main()