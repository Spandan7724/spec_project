"""
State Management for Multi-Agent Currency Decision Workflows.

Defines the state structures used by LangGraph workflows to coordinate
multiple agents and track the progression of currency conversion decisions.
"""

from typing import Dict, Any, List, Optional, Union, Annotated
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

from ..models import ConversionRequest, DecisionRecommendation
from ..agents.base_agent import AgentResult


class WorkflowStatus(str, Enum):
    """Status of the workflow execution."""
    INITIALIZED = "initialized"
    MARKET_ANALYSIS = "market_analysis"
    RISK_ANALYSIS = "risk_analysis"
    COST_ANALYSIS = "cost_analysis"
    COORDINATION = "coordination"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MarketAnalysisResult:
    """Results from market intelligence analysis."""
    sentiment_score: float = 0.0  # -1.0 (very negative) to 1.0 (very positive)
    news_impact: float = 0.0  # 0.0 (no impact) to 1.0 (high impact)
    economic_events: List[Dict[str, Any]] = field(default_factory=list)
    market_regime: str = "unknown"  # "trending", "ranging", "volatile"
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "sentiment_score": self.sentiment_score,
            "news_impact": self.news_impact,
            "economic_events": self.economic_events,
            "market_regime": self.market_regime,
            "technical_indicators": self.technical_indicators,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


@dataclass
class RiskAnalysisResult:
    """Results from risk analysis."""
    volatility_score: float = 0.0  # 0.0 (stable) to 1.0 (very volatile)
    prediction_uncertainty: float = 0.0  # 0.0 (certain) to 1.0 (very uncertain)
    time_risk: float = 0.0  # 0.0 (no time pressure) to 1.0 (urgent)
    user_risk_alignment: float = 0.0  # How well recommendation aligns with user risk tolerance
    overall_risk: float = 0.0  # Combined risk score
    confidence: float = 0.0
    reasoning: str = ""
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "volatility_score": self.volatility_score,
            "prediction_uncertainty": self.prediction_uncertainty,
            "time_risk": self.time_risk,
            "user_risk_alignment": self.user_risk_alignment,
            "overall_risk": self.overall_risk,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "scenarios": self.scenarios
        }


@dataclass
class CostAnalysisResult:
    """Results from cost optimization analysis."""
    best_provider: str = ""
    estimated_cost: Decimal = Decimal('0')
    cost_percentage: float = 0.0
    potential_savings: Decimal = Decimal('0')
    timing_impact: float = 0.0  # How timing affects costs
    provider_comparison: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "best_provider": self.best_provider,
            "estimated_cost": float(self.estimated_cost),
            "cost_percentage": self.cost_percentage,
            "potential_savings": float(self.potential_savings),
            "timing_impact": self.timing_impact,
            "provider_comparison": self.provider_comparison,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


class CurrencyDecisionState:
    """
    Central state for the currency conversion decision workflow.
    
    This state is passed between all agents and tracks the complete
    decision-making process from input to final recommendation.
    """
    
    def __init__(self, 
                 request: ConversionRequest,
                 workflow_id: str = None):
        """
        Initialize the decision state.
        
        Args:
            request: Original conversion request from user
            workflow_id: Unique identifier for this workflow execution
        """
        # Core request data
        self.request = request
        self.workflow_id = workflow_id or f"workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Workflow tracking
        self.status = WorkflowStatus.INITIALIZED
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.current_step: Optional[str] = None
        self.steps_completed: List[str] = []
        self.errors: List[str] = []
        
        # Agent results
        self.market_analysis: Optional[MarketAnalysisResult] = None
        self.risk_analysis: Optional[RiskAnalysisResult] = None
        self.cost_analysis: Optional[CostAnalysisResult] = None
        
        # Agent execution tracking
        self.agent_results: Dict[str, AgentResult] = {}
        self.agent_execution_times: Dict[str, float] = {}
        
        # Final outputs
        self.recommendation: Optional[DecisionRecommendation] = None
        self.coordination_reasoning: str = ""
        self.confidence_score: float = 0.0
        
        # Conversation messages for LangGraph
        self.messages: List[BaseMessage] = []
        
        # Additional context
        self.context: Dict[str, Any] = {}
    
    def update_status(self, new_status: WorkflowStatus, step_name: str = None) -> None:
        """Update workflow status and tracking."""
        self.status = new_status
        if step_name:
            self.current_step = step_name
            if step_name not in self.steps_completed:
                self.steps_completed.append(step_name)
    
    def add_agent_result(self, agent_name: str, result: AgentResult) -> None:
        """Add result from a specific agent."""
        self.agent_results[agent_name] = result
        if result.execution_time_ms:
            self.agent_execution_times[agent_name] = result.execution_time_ms
        
        # Update specific analysis results based on agent type
        if agent_name == "market_intelligence" and result.success:
            self.market_analysis = self._parse_market_result(result.data)
        elif agent_name == "risk_analysis" and result.success:
            self.risk_analysis = self._parse_risk_result(result.data)
        elif agent_name == "cost_optimization" and result.success:
            self.cost_analysis = self._parse_cost_result(result.data)
    
    def _parse_market_result(self, data: Dict[str, Any]) -> MarketAnalysisResult:
        """Parse market intelligence data into structured result."""
        return MarketAnalysisResult(
            sentiment_score=data.get("sentiment_score", 0.0),
            news_impact=data.get("news_impact", 0.0),
            economic_events=data.get("economic_events", []),
            market_regime=data.get("market_regime", "unknown"),
            technical_indicators=data.get("technical_indicators", {}),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", "")
        )
    
    def _parse_risk_result(self, data: Dict[str, Any]) -> RiskAnalysisResult:
        """Parse risk analysis data into structured result."""
        return RiskAnalysisResult(
            volatility_score=data.get("volatility_score", 0.0),
            prediction_uncertainty=data.get("prediction_uncertainty", 0.0),
            time_risk=data.get("time_risk", 0.0),
            user_risk_alignment=data.get("user_risk_alignment", 0.0),
            overall_risk=data.get("overall_risk", 0.0),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            scenarios=data.get("scenarios", [])
        )
    
    def _parse_cost_result(self, data: Dict[str, Any]) -> CostAnalysisResult:
        """Parse cost analysis data into structured result."""
        return CostAnalysisResult(
            best_provider=data.get("best_provider", ""),
            estimated_cost=Decimal(str(data.get("estimated_cost", 0))),
            cost_percentage=data.get("cost_percentage", 0.0),
            potential_savings=Decimal(str(data.get("potential_savings", 0))),
            timing_impact=data.get("timing_impact", 0.0),
            provider_comparison=data.get("provider_comparison", {}),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", "")
        )
    
    def add_error(self, error_message: str, agent_name: str = None) -> None:
        """Add an error to the workflow state."""
        error_entry = f"[{datetime.utcnow().isoformat()}] "
        if agent_name:
            error_entry += f"{agent_name}: "
        error_entry += error_message
        self.errors.append(error_entry)
    
    def set_final_recommendation(self, recommendation: DecisionRecommendation, reasoning: str = "") -> None:
        """Set the final decision recommendation."""
        self.recommendation = recommendation
        self.coordination_reasoning = reasoning
        self.confidence_score = recommendation.confidence if recommendation else 0.0
        self.end_time = datetime.utcnow()
        self.update_status(WorkflowStatus.COMPLETED)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of workflow execution."""
        total_time = 0
        if self.end_time:
            total_time = (self.end_time - self.start_time).total_seconds() * 1000
        
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "total_execution_time_ms": total_time,
            "steps_completed": self.steps_completed,
            "agents_executed": list(self.agent_results.keys()),
            "successful_agents": [name for name, result in self.agent_results.items() if result.success],
            "failed_agents": [name for name, result in self.agent_results.items() if not result.success],
            "error_count": len(self.errors),
            "has_recommendation": self.recommendation is not None,
            "confidence_score": self.confidence_score,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None
        }
    
    def is_complete(self) -> bool:
        """Check if workflow has completed successfully."""
        return (self.status == WorkflowStatus.COMPLETED and 
                self.recommendation is not None)
    
    def has_critical_errors(self) -> bool:
        """Check if workflow has encountered critical errors."""
        return (self.status == WorkflowStatus.FAILED or
                len(self.errors) > 3 or  # Too many errors
                len([r for r in self.agent_results.values() if not r.success]) >= 2)  # Multiple agent failures
    
    def get_agent_consensus(self) -> Dict[str, Any]:
        """Analyze consensus between agents for coordination."""
        if not self.agent_results:
            return {"consensus_score": 0.0, "agreements": [], "conflicts": []}
        
        # Gather confidence scores
        confidences = [result.confidence for result in self.agent_results.values() if result.success]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Simple consensus analysis
        agreements = []
        conflicts = []
        
        # Check if market and risk analysis agree on urgency
        if self.market_analysis and self.risk_analysis:
            market_urgency = self.market_analysis.news_impact > 0.5
            risk_urgency = self.risk_analysis.time_risk > 0.5
            
            if market_urgency == risk_urgency:
                agreements.append("Market and risk analysis agree on timing urgency")
            else:
                conflicts.append("Market and risk analysis disagree on timing urgency")
        
        # Check if cost analysis aligns with other factors
        if self.cost_analysis and (self.market_analysis or self.risk_analysis):
            cost_favors_waiting = self.cost_analysis.potential_savings > Decimal('10')
            if cost_favors_waiting:
                agreements.append("Cost analysis suggests potential for savings")
        
        consensus_score = len(agreements) / (len(agreements) + len(conflicts)) if (agreements or conflicts) else 0.5
        
        return {
            "consensus_score": consensus_score,
            "average_confidence": avg_confidence,
            "agreements": agreements,
            "conflicts": conflicts,
            "participating_agents": len([r for r in self.agent_results.values() if r.success])
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "request": {
                "currency_pair": self.request.currency_pair,
                "amount": float(self.request.amount),
                "deadline": self.request.deadline.isoformat() if self.request.deadline else None,
                "user_id": self.request.user_profile.user_id,
                "risk_tolerance": self.request.user_profile.risk_tolerance.value
            },
            "market_analysis": self.market_analysis.to_dict() if self.market_analysis else None,
            "risk_analysis": self.risk_analysis.to_dict() if self.risk_analysis else None,
            "cost_analysis": self.cost_analysis.to_dict() if self.cost_analysis else None,
            "execution_summary": self.get_execution_summary(),
            "agent_consensus": self.get_agent_consensus(),
            "errors": self.errors,
            "recommendation": {
                "decision": self.recommendation.decision.value if self.recommendation else None,
                "confidence": self.recommendation.confidence if self.recommendation else None,
                "explanation": self.recommendation.explanation if self.recommendation else None
            } if self.recommendation else None
        }
    
    def __str__(self) -> str:
        status_icon = "✅" if self.is_complete() else "🔄" if self.status != WorkflowStatus.FAILED else "❌"
        return f"{status_icon} Workflow {self.workflow_id}: {self.status.value} ({len(self.agent_results)} agents)"


# Type annotations for LangGraph state management
StateMessages = Annotated[List[BaseMessage], add_messages]