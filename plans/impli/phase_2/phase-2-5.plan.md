<!-- f67714c1-a54f-4e8d-9617-16955f212afc 42d96858-77f8-4201-b81f-ba079d50e360 -->
# Phase 2.5: Prediction Agent & Caching

## Overview

Build the main MLPredictor that orchestrates the complete prediction pipeline: model loading, feature building, prediction, quality gates, caching, and fallback. Then create the LangGraph node that integrates predictions into the multi-agent workflow. This is the final piece that makes Phase 2 complete and connects Price Prediction to Layer 3 (Decision Engine).

## Implementation Steps

### Step 1: Main Predictor with Caching

**File**: `src/prediction/predictor.py`

Create the orchestrator that ties everything together:

```python
import time
import hashlib
import json
from typing import Dict, Optional
from datetime import datetime, timedelta
from src.utils.logging import get_logger

from .models import PredictionRequest, PredictionResponse, HorizonPrediction, PredictionQuality
from .data_loader import HistoricalDataLoader
from .feature_builder import FeatureBuilder
from .registry import ModelRegistry
from .config import PredictionConfig
from .backends.lightgbm_backend import LightGBMBackend
from .utils.fallback import FallbackPredictor

logger = get_logger(__name__)

class MLPredictor:
    """Main prediction service with caching and quality gates"""
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize ML predictor
        
        Args:
            config: Optional configuration (loads from yaml if not provided)
        """
        self.config = config or PredictionConfig.from_yaml()
        self.data_loader = HistoricalDataLoader()
        self.feature_builder = FeatureBuilder(self.config.technical_indicators)
        self.registry = ModelRegistry(
            self.config.model_registry_path,
            self.config.model_storage_dir
        )
        self.fallback = FallbackPredictor(self.config)
        self.cache = {}  # Simple in-memory cache: key -> (response, timestamp)
        
        logger.info("MLPredictor initialized")
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make prediction with caching and quality gates
        
        Args:
            request: Prediction request
        
        Returns:
            PredictionResponse with predictions or fallback
        """
        start_time = time.time()
        correlation_id = request.correlation_id or "unknown"
        
        logger.info(
            f"[{correlation_id}] Prediction request for {request.currency_pair}",
            extra={"correlation_id": correlation_id, "horizons": request.horizons}
        )
        
        # Check cache
        cache_key = self._get_cache_key(request)
        if cache_key in self.cache:
            cached_response, cached_time = self.cache[cache_key]
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600
            
            if age_hours <= request.max_age_hours:
                logger.info(
                    f"[{correlation_id}] Returning cached prediction (age: {age_hours:.1f}h)",
                    extra={"correlation_id": correlation_id, "cache_age_hours": age_hours}
                )
                cached_response.cached = True
                cached_response.processing_time_ms = (time.time() - start_time) * 1000
                return cached_response
        
        # Parse currency pair
        try:
            base, quote = request.currency_pair.split('/')
        except ValueError:
            logger.error(f"[{correlation_id}] Invalid currency pair: {request.currency_pair}")
            return self._create_error_response(
                request, 
                f"Invalid currency pair format: {request.currency_pair}"
            )
        
        # Try ML prediction
        try:
            response = await self._ml_predict(request, base, quote)
            
            # Check quality gate
            if self._passes_quality_gate(response):
                response.status = "success"
                logger.info(
                    f"[{correlation_id}] ML prediction successful",
                    extra={
                        "correlation_id": correlation_id,
                        "confidence": response.confidence,
                        "model_id": response.model_id
                    }
                )
            else:
                logger.warning(
                    f"[{correlation_id}] Prediction failed quality gate, using fallback",
                    extra={"correlation_id": correlation_id}
                )
                response = await self._fallback_predict(request, base, quote)
                response.status = "partial"
        
        except Exception as e:
            logger.error(
                f"[{correlation_id}] ML prediction failed: {str(e)}",
                extra={"correlation_id": correlation_id, "error": str(e)}
            )
            response = await self._fallback_predict(request, base, quote)
            response.status = "partial" if response.predictions else "error"
            response.warnings.append(f"ML prediction failed: {str(e)}")
        
        response.processing_time_ms = (time.time() - start_time) * 1000
        
        # Cache response
        self.cache[cache_key] = (response, datetime.now())
        
        logger.info(
            f"[{correlation_id}] Prediction complete: status={response.status}, "
            f"time={response.processing_time_ms:.0f}ms",
            extra={
                "correlation_id": correlation_id,
                "status": response.status,
                "processing_time_ms": response.processing_time_ms
            }
        )
        
        return response
    
    async def _ml_predict(
        self, 
        request: PredictionRequest,
        base: str,
        quote: str
    ) -> PredictionResponse:
        """
        ML-based prediction
        
        Args:
            request: Prediction request
            base: Base currency
            quote: Quote currency
        
        Returns:
            PredictionResponse from ML model
        """
        correlation_id = request.correlation_id or "unknown"
        
        # Load historical data
        logger.info(f"[{correlation_id}] Loading historical data")
        df = await self.data_loader.fetch_historical_data(
            base, quote, days=self.config.max_history_days
        )
        
        if df is None or len(df) < self.config.min_history_days:
            raise ValueError(
                f"Insufficient historical data: {len(df) if df is not None else 0} days "
                f"(minimum: {self.config.min_history_days})"
            )
        
        # Build features
        logger.info(f"[{correlation_id}] Building features")
        features_df = self.feature_builder.build_features(
            df,
            mode=request.features_mode,
            market_intel=None  # TODO: Pass market intel when available
        )
        
        if len(features_df) < 50:
            raise ValueError(f"Insufficient feature data: {len(features_df)} rows")
        
        # Get latest features (most recent row)
        latest_features = features_df.iloc[[-1]]
        latest_close = df['Close'].iloc[-1]
        
        # Load model from registry
        logger.info(f"[{correlation_id}] Loading model from registry")
        model_metadata = self.registry.get_model(
            request.currency_pair,
            model_type=self.config.predictor_backend
        )
        
        if not model_metadata:
            raise ValueError(f"No model found for {request.currency_pair}")
        
        # Load model objects
        model_obj, scaler_obj = self.registry.load_model_objects(model_metadata)
        
        # Create backend
        backend = self._create_backend(model_metadata['model_type'])
        backend.models = model_obj
        backend.scaler = scaler_obj
        backend.validation_metrics = model_metadata['validation_metrics']
        backend.feature_names = model_metadata.get('features_used', [])
        
        # Make predictions
        logger.info(f"[{correlation_id}] Generating predictions")
        raw_predictions = backend.predict(
            latest_features,
            request.horizons,
            include_quantiles=request.include_quantiles
        )
        
        # Build response
        predictions = {}
        for horizon, pred_data in raw_predictions.items():
            predictions[horizon] = HorizonPrediction(
                horizon=horizon,
                mean_change_pct=pred_data['mean_change'],
                quantiles=pred_data.get('quantiles'),
                direction_probability=pred_data.get('direction_prob')
            )
        
        # Quality metadata
        quality = PredictionQuality(
            model_confidence=backend.get_model_confidence(),
            calibrated=model_metadata['calibration_ok'],
            validation_metrics=model_metadata['validation_metrics'],
            notes=[]
        )
        
        response = PredictionResponse(
            status="success",
            confidence=quality.model_confidence,
            processing_time_ms=0,  # Will be set by caller
            currency_pair=request.currency_pair,
            horizons=request.horizons,
            predictions=predictions,
            latest_close=float(latest_close),
            features_used=model_metadata['features_used'],
            quality=quality,
            model_id=model_metadata['model_id'],
            cached=False,
            timestamp=datetime.now()
        )
        
        return response
    
    async def _fallback_predict(
        self, 
        request: PredictionRequest,
        base: str,
        quote: str
    ) -> PredictionResponse:
        """
        Fallback heuristic-based prediction
        
        Args:
            request: Prediction request
            base: Base currency
            quote: Quote currency
        
        Returns:
            PredictionResponse from fallback
        """
        correlation_id = request.correlation_id or "unknown"
        logger.info(f"[{correlation_id}] Using fallback heuristic predictor")
        
        return await self.fallback.predict(request, base, quote)
    
    def _create_backend(self, model_type: str):
        """
        Create backend instance
        
        Args:
            model_type: "lightgbm" or "lstm"
        
        Returns:
            Backend instance
        """
        if model_type == "lightgbm":
            return LightGBMBackend()
        elif model_type == "lstm":
            from .backends.lstm_backend import LSTMBackend
            return LSTMBackend()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _passes_quality_gate(self, response: PredictionResponse) -> bool:
        """
        Check if prediction passes quality gates
        
        Args:
            response: Prediction response
        
        Returns:
            True if quality gate passed
        """
        if not response.quality:
            logger.warning("No quality metadata available")
            return False
        
        # Check confidence threshold
        if response.confidence < 0.3:
            logger.warning(f"Low confidence: {response.confidence:.2f}")
            return False
        
        # Check calibration
        if not response.quality.calibrated:
            logger.warning("Model not calibrated")
            return False
        
        # Check sample size
        metrics = response.quality.validation_metrics
        if metrics:
            min_samples = min(
                m.get('n_samples', 0) 
                for m in metrics.values() 
                if isinstance(m, dict)
            )
            if min_samples < self.config.min_samples_required:
                logger.warning(f"Insufficient validation samples: {min_samples}")
                return False
        
        return True
    
    def _get_cache_key(self, request: PredictionRequest) -> str:
        """
        Generate cache key
        
        Args:
            request: Prediction request
        
        Returns:
            MD5 hash of request parameters
        """
        key_data = {
            'pair': request.currency_pair,
            'horizons': sorted(request.horizons),
            'mode': request.features_mode
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _create_error_response(
        self, 
        request: PredictionRequest, 
        error_msg: str
    ) -> PredictionResponse:
        """
        Create error response
        
        Args:
            request: Original request
            error_msg: Error message
        
        Returns:
            Error PredictionResponse
        """
        return PredictionResponse(
            status="error",
            confidence=0.0,
            processing_time_ms=0,
            warnings=[error_msg],
            currency_pair=request.currency_pair,
            horizons=request.horizons,
            predictions={},
            latest_close=0.0,
            features_used=[],
            quality=PredictionQuality(
                model_confidence=0.0,
                calibrated=False,
                validation_metrics={},
                notes=[error_msg]
            ),
            model_id="none",
            cached=False,
            timestamp=datetime.now()
        )
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.cache.clear()
        logger.info("Prediction cache cleared")
```

### Step 2: Prediction Agent Node

**File**: `src/agentic/nodes/prediction.py`

Create the LangGraph node for price prediction:

```python
from typing import Dict, Any
from src.agentic.state import AgentState
from src.prediction.predictor import MLPredictor
from src.prediction.models import PredictionRequest
from src.prediction.config import PredictionConfig
from src.utils.logging import get_logger
from src.utils.decorators import log_execution, timeout
import time

logger = get_logger(__name__)

@timeout(12.0)  # 12 second timeout for prediction
@log_execution(log_args=False, log_result=False)
async def prediction_node(state: AgentState) -> Dict[str, Any]:
    """
    Price Prediction agent node for LangGraph.
    
    Generates ML-based price forecasts using historical data and technical indicators.
    Updates state with predictions.
    
    Args:
        state: Current agent state
    
    Returns:
        Dictionary with price_forecast and prediction_status fields
    """
    start_time = time.time()
    correlation_id = state.get("correlation_id", "unknown")
    
    try:
        # Extract parameters from state
        currency_pair = f"{state.get('base_currency')}/{state.get('quote_currency')}"
        timeframe = state.get("timeframe", "1_day")
        
        # Map timeframe to days
        timeframe_mapping = {
            "immediate": 1,
            "1_day": 1,
            "1_week": 7,
            "1_month": 30
        }
        timeframe_days = timeframe_mapping.get(timeframe, 7)
        
        # Build prediction request
        pred_request = PredictionRequest(
            currency_pair=currency_pair,
            horizons=[1, 7, timeframe_days],  # Always include 1d, 7d, and user's timeframe
            include_quantiles=True,
            include_direction_probabilities=True,
            max_age_hours=1,
            features_mode="price_only",  # TODO: Switch to price_plus_intel when Market Intelligence integrated
            correlation_id=correlation_id
        )
        
        logger.info(
            f"[{correlation_id}] Starting price prediction for {currency_pair}",
            extra={
                "correlation_id": correlation_id,
                "currency_pair": currency_pair,
                "horizons": pred_request.horizons
            }
        )
        
        # Initialize predictor
        config = PredictionConfig.from_yaml()
        predictor = MLPredictor(config)
        
        # Get prediction
        pred_response = await predictor.predict(pred_request)
        
        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Format predictions for state
        predictions = {}
        for horizon, pred in pred_response.predictions.items():
            predictions[str(horizon)] = {
                "mean_change_pct": pred.mean_change_pct,
                "quantiles": pred.quantiles,
                "direction_prob": pred.direction_probability
            }
        
        # Build price forecast object
        price_forecast = {
            "status": pred_response.status,
            "confidence": pred_response.confidence,
            "predictions": predictions,
            "latest_close": pred_response.latest_close,
            "model_id": pred_response.model_id,
            "cached": pred_response.cached,
            "processing_time_ms": execution_time_ms,
            "features_used": pred_response.features_used,
            "warnings": pred_response.warnings
        }
        
        logger.info(
            f"[{correlation_id}] Prediction complete: status={pred_response.status}, "
            f"confidence={pred_response.confidence:.2f}, time={execution_time_ms}ms",
            extra={
                "correlation_id": correlation_id,
                "status": pred_response.status,
                "confidence": pred_response.confidence,
                "execution_time_ms": execution_time_ms
            }
        )
        
        # Return state updates (only fields this node is responsible for)
        return {
            "price_forecast": price_forecast,
            "prediction_status": pred_response.status,
            "prediction_error": None
        }
        
    except TimeoutError as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"[{correlation_id}] Price Prediction agent timed out",
            extra={"correlation_id": correlation_id, "execution_time_ms": execution_time_ms}
        )
        return {
            "prediction_status": "error",
            "prediction_error": f"Timeout after 12 seconds: {str(e)}",
            "price_forecast": None
        }
        
    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"[{correlation_id}] Price Prediction agent failed: {str(e)}",
            extra={
                "correlation_id": correlation_id,
                "error": str(e),
                "execution_time_ms": execution_time_ms
            }
        )
        
        return {
            "prediction_status": "error",
            "prediction_error": str(e),
            "price_forecast": None
        }
```

### Step 3: Update Graph Definition

**File**: `src/agentic/graph.py`

Update the graph to include the prediction node:

```python
# Add import at top
from src.agentic.nodes.prediction import prediction_node

# In create_graph() function:
# Replace placeholder prediction node with real one
workflow.add_node("price_prediction", prediction_node)

# Edge already exists from market_intelligence -> price_prediction
# Edge already exists from price_prediction -> decision_engine
```

### Step 4: Unit Tests for Predictor

**File**: `tests/prediction/test_predictor.py`

Test the main predictor orchestration:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.prediction.predictor import MLPredictor
from src.prediction.models import PredictionRequest, PredictionResponse
from src.prediction.config import PredictionConfig

@pytest.fixture
def mock_config():
    """Create mock configuration"""
    config = PredictionConfig()
    config.min_history_days = 100
    config.max_history_days = 365
    config.min_samples_required = 100
    return config


@pytest.mark.asyncio
async def test_predictor_cache_hit(mock_config):
    """Test that caching works correctly"""
    predictor = MLPredictor(mock_config)
    
    # Mock the _ml_predict method
    mock_response = MagicMock(spec=PredictionResponse)
    mock_response.status = "success"
    mock_response.confidence = 0.75
    mock_response.cached = False
    mock_response.processing_time_ms = 100
    
    predictor._ml_predict = AsyncMock(return_value=mock_response)
    predictor._passes_quality_gate = MagicMock(return_value=True)
    
    request = PredictionRequest(
        currency_pair="USD/EUR",
        horizons=[1, 7],
        max_age_hours=1
    )
    
    # First call - should call ML
    response1 = await predictor.predict(request)
    assert response1.cached == False
    assert predictor._ml_predict.call_count == 1
    
    # Second call - should use cache
    response2 = await predictor.predict(request)
    assert response2.cached == True
    assert predictor._ml_predict.call_count == 1  # Not called again


@pytest.mark.asyncio
async def test_predictor_fallback_on_ml_failure(mock_config):
    """Test fallback activates when ML fails"""
    predictor = MLPredictor(mock_config)
    
    # Mock ML to raise error
    predictor._ml_predict = AsyncMock(side_effect=ValueError("Model not found"))
    
    # Mock fallback to return valid response
    mock_fallback_response = MagicMock(spec=PredictionResponse)
    mock_fallback_response.status = "partial"
    mock_fallback_response.confidence = 0.2
    mock_fallback_response.predictions = {1: MagicMock()}
    mock_fallback_response.warnings = []
    
    predictor._fallback_predict = AsyncMock(return_value=mock_fallback_response)
    
    request = PredictionRequest(currency_pair="USD/EUR", horizons=[1])
    
    response = await predictor.predict(request)
    
    assert response.status == "partial"
    assert predictor._fallback_predict.called


@pytest.mark.asyncio
async def test_predictor_quality_gate_failure(mock_config):
    """Test fallback activates when quality gate fails"""
    predictor = MLPredictor(mock_config)
    
    # Mock ML prediction with low confidence
    mock_ml_response = MagicMock(spec=PredictionResponse)
    mock_ml_response.confidence = 0.1  # Too low
    mock_ml_response.quality = MagicMock(calibrated=True)
    
    predictor._ml_predict = AsyncMock(return_value=mock_ml_response)
    
    # Mock fallback
    mock_fallback_response = MagicMock(spec=PredictionResponse)
    mock_fallback_response.status = "partial"
    
    predictor._fallback_predict = AsyncMock(return_value=mock_fallback_response)
    
    request = PredictionRequest(currency_pair="USD/EUR", horizons=[1])
    
    response = await predictor.predict(request)
    
    assert predictor._fallback_predict.called


def test_cache_key_generation(mock_config):
    """Test cache key generation is consistent"""
    predictor = MLPredictor(mock_config)
    
    request1 = PredictionRequest(
        currency_pair="USD/EUR",
        horizons=[1, 7, 30]
    )
    
    request2 = PredictionRequest(
        currency_pair="USD/EUR",
        horizons=[30, 1, 7]  # Different order
    )
    
    key1 = predictor._get_cache_key(request1)
    key2 = predictor._get_cache_key(request2)
    
    assert key1 == key2  # Should be same despite order
```

### Step 5: Integration Tests

**File**: `tests/integration/test_agentic/test_prediction_integration.py`

Test the prediction node in the graph:

```python
import pytest
from src.agentic.graph import create_graph
from src.agentic.state import initialize_state

@pytest.mark.integration
@pytest.mark.asyncio
async def test_prediction_node_in_graph():
    """Test Prediction node within complete LangGraph workflow"""
    graph = create_graph()
    
    initial_state = initialize_state(
        "Should I convert 5000 USD to EUR now?",
        base_currency="USD",
        quote_currency="EUR",
        amount=5000,
        timeframe="1_week"
    )
    
    # Execute graph (will run all Layer 1 and Layer 2 agents)
    result = await graph.ainvoke(initial_state)
    
    # Verify Prediction executed
    assert result["prediction_status"] in ["success", "partial", "error"]
    
    if result["prediction_status"] == "success":
        assert result["price_forecast"] is not None
        assert "predictions" in result["price_forecast"]
        assert "confidence" in result["price_forecast"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prediction_after_market_data():
    """Test that prediction node receives market data correctly"""
    graph = create_graph()
    
    initial_state = initialize_state(
        "Convert USD to EUR",
        base_currency="USD",
        quote_currency="EUR"
    )
    
    result = await graph.ainvoke(initial_state)
    
    # Check that both market data and prediction ran
    assert result.get("market_snapshot") is not None
    assert result.get("price_forecast") is not None or result.get("prediction_error") is not None
```

## Key Design Decisions

1. **In-memory caching**: Simple dict-based cache with TTL (1 hour default)
2. **Quality gates**: Automated checks for confidence, calibration, sample size
3. **Graceful degradation**: Falls back to heuristics when ML fails
4. **Cache key includes mode**: Different predictions for price_only vs price_plus_intel
5. **12-second timeout**: Allows time for data loading + feature building + prediction
6. **Correlation ID propagation**: Tracks requests through entire pipeline
7. **Structured logging**: All operations logged with context
8. **State field isolation**: Node only returns its specific fields (no conflicts)
9. **Configuration from YAML**: Centralized config management
10. **Error handling**: Three levels (success, partial, error)

## Files to Create

- `src/prediction/__init__.py`
- `src/prediction/predictor.py`
- `src/agentic/nodes/prediction.py`
- `tests/prediction/test_predictor.py`
- `tests/integration/test_agentic/test_prediction_integration.py`

## Files to Update

- `src/agentic/graph.py` (replace placeholder prediction node)

## Dependencies

All dependencies already added in previous phases:

- Phase 2.1: Data loading and features
- Phase 2.2: Model registry
- Phase 2.3: LightGBM backend
- Phase 2.4: Fallback predictor

## Validation

Manual end-to-end testing:

```python
from src.prediction.predictor import MLPredictor
from src.prediction.models import PredictionRequest
from src.prediction.config import PredictionConfig

# Initialize
config = PredictionConfig.from_yaml()
predictor = MLPredictor(config)

# Test with model (if available)
request = PredictionRequest(
    currency_pair="USD/EUR",
    horizons=[1, 7, 30],
    include_quantiles=True,
    include_direction_probabilities=True,
    correlation_id="test-123"
)

response = await predictor.predict(request)

print(f"Status: {response.status}")
print(f"Confidence: {response.confidence:.2f}")
print(f"Model: {response.model_id}")
print(f"Cached: {response.cached}")
print(f"Time: {response.processing_time_ms:.0f}ms")

for horizon, pred in response.predictions.items():
    print(f"\n{horizon}d forecast:")
    print(f"  Mean: {pred.mean_change_pct:+.3f}%")
    print(f"  Direction prob: {pred.direction_probability:.2f}")
    if pred.quantiles:
        print(f"  90% CI: [{pred.quantiles['p10']:+.3f}%, {pred.quantiles['p90']:+.3f}%]")

# Test cache
response2 = await predictor.predict(request)
assert response2.cached == True
print("\n✓ Cache working correctly")

# Test in graph
from src.agentic.graph import create_graph
from src.agentic.state import initialize_state

graph = create_graph()
state = initialize_state(
    "Should I convert USD to EUR?",
    base_currency="USD",
    quote_currency="EUR",
    amount=1000
)

result = await graph.ainvoke(state)
print(f"\n✓ Graph execution: prediction_status={result['prediction_status']}")
```

## Success Criteria

- Predictor successfully loads models from registry
- Cache reduces repeated prediction time by >80%
- Quality gates correctly identify low-quality predictions
- Fallback activates when ML fails or quality gate fails
- Prediction node integrates into LangGraph without errors
- Timeout prevents hanging (12 seconds max)
- All state updates follow parallel execution pattern (no conflicts)
- Correlation IDs track requests through entire pipeline
- All unit tests pass with >80% coverage
- Integration test verifies full workflow
- Manual testing shows end-to-end prediction works

## Phase 2 Completion Checklist

After Phase 2.5, verify all Phase 2 components are complete:

- [x] Phase 2.1: Data pipeline (data loader + feature builder)
- [x] Phase 2.2: Model registry (JSON + pickle)
- [x] Phase 2.3: LightGBM backend + SHAP
- [x] Phase 2.4: Fallback heuristics
- [x] Phase 2.5: Main predictor + LangGraph node

**Phase 2 is now complete!** The Price Prediction Agent is fully functional and integrated into the multi-agent workflow.

## Next Phase

After Phase 2.5 completes, **Phase 2 (Price Prediction Agent) is DONE**!

Proceed to **Phase 3: Decision Engine Agent**, which will:

- Implement utility calculation model
- Create action scoring (convert_now, staged, wait)
- Add risk penalty calculation
- Implement staging algorithm
- Create decision engine node for LangGraph
- Integrate Market Data + Market Intelligence + Price Prediction into final recommendations

### To-dos

- [ ] Implement MLPredictor with caching and quality gates in src/prediction/predictor.py
- [ ] Create prediction_node for LangGraph in src/agentic/nodes/prediction.py
- [ ] Add in-memory caching with TTL and cache key generation
- [ ] Implement quality gate checks (confidence, calibration, sample size)
- [ ] Add graceful fallback integration when ML fails
- [ ] Update graph.py to use real prediction node instead of placeholder
- [ ] Write unit tests for MLPredictor (caching, quality gates, fallback)
- [ ] Write integration test for prediction node in full graph
- [ ] Manually validate end-to-end prediction pipeline with real data