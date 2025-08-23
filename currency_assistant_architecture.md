# Real-Time Currency Conversion Assistant - Technical Architecture

## Project Overview

A Python-based intelligent currency conversion assistant that helps users make optimal decisions about when to exchange currencies. The system combines real-time market data, machine learning predictions, and cost optimization to provide actionable advice like "convert now" or "wait" with clear reasoning.

## Core Value Proposition

- **Save Money**: Avoid bad conversion windows and hidden fees
- **Reduce Stress**: Simple binary recommendations instead of complex charts
- **Goal-Oriented**: Work backward from deadlines (tuition, travel, remittances)
- **Intelligent Timing**: ML-powered predictions for optimal conversion timing
- **Multi-Provider**: Compare costs across banks, fintechs, and services

---

## System Architecture

### Technology Stack (Python-Centric)

```python
# Core Framework
fastapi = "^0.104.0"      # API framework
sqlalchemy = "^2.0.0"     # Database ORM
redis = "^5.0.0"          # Caching & sessions
celery = "^5.3.0"         # Background tasks

# Machine Learning
torch = "^2.1.0"          # LSTM models
pandas = "^2.1.0"         # Data processing
scikit-learn = "^1.3.0"   # Traditional ML
numpy = "^1.25.0"         # Numerical computing

# Data & APIs
httpx = "^0.25.0"         # HTTP client
pydantic = "^2.4.0"       # Data validation
asyncpg = "^0.28.0"       # Async PostgreSQL
aioredis = "^2.0.0"       # Async Redis

# Monitoring & Deployment
prometheus-client = "^0.17.0"  # Metrics
structlog = "^23.1.0"          # Logging
pytest = "^7.4.0"              # Testing
```

### Database Architecture

```sql
-- PostgreSQL + TimescaleDB Extension
- users: User profiles and preferences
- goals: Conversion goals with deadlines  
- fx_rates: Time-series exchange rate data (TimescaleDB)
- providers: Bank/fintech fee structures
- conversions: Conversion history and outcomes
- notifications: Alert preferences and delivery log
```

---

## Detailed System Components

## 1. Data Ingestion & Processing Layer

### FX Data Collector (`data_collector.py`)
**Purpose**: Fetches real-time exchange rates from multiple data sources
**Technology**: AsyncIO + httpx for concurrent API calls
**Data Sources**:
- Alpha Vantage API (primary)
- Fixer.io API (backup)
- XE API (premium rates)
- Bank websites (web scraping)

**Implementation**:
```python
class FXDataCollector:
    def __init__(self):
        self.providers = ['alphavantage', 'fixer', 'xe']
        self.redis_client = aioredis.from_url("redis://localhost")
        self.db_session = async_session_maker()
    
    async def collect_rates(self, currency_pairs: List[str]):
        tasks = [self.fetch_from_provider(pair, provider) 
                for pair in currency_pairs 
                for provider in self.providers]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await self.store_rates(results)
        await self.cache_latest_rates(results)
```

**Features**:
- Real-time WebSocket connections for live data
- Rate limiting and error handling
- Data validation and outlier detection
- Multi-provider redundancy with failover
- Automatic retry logic with exponential backoff

**Update Frequency**: Every 30 seconds for major pairs, 5 minutes for exotic pairs

### Provider Fee Scraper (`fee_scraper.py`)
**Purpose**: Monitors exchange spreads and transfer fees across financial providers
**Technology**: Selenium + BeautifulSoup for web scraping

**Providers Monitored**:
- Traditional Banks: Chase, Wells Fargo, Bank of America
- Fintechs: Wise, Revolut, Remitly
- Crypto Exchanges: Coinbase, Kraken
- Money Transfer: Western Union, MoneyGram

**Implementation**:
```python
class ProviderFeeScraper:
    async def scrape_wise_fees(self, amount: float, from_curr: str, to_curr: str):
        # Selenium automation to get real-time quote
        driver = await self.get_driver()
        quote_data = await self.simulate_transfer(driver, amount, from_curr, to_curr)
        return {
            'provider': 'wise',
            'total_fee': quote_data['fee'],
            'exchange_rate': quote_data['rate'],
            'transfer_time': quote_data['delivery_time']
        }
```

**Update Frequency**: Every 4 hours, triggered when major rate changes detected

### Data Processor (`data_processor.py`)
**Purpose**: Cleans, validates, and enriches raw FX data with technical indicators
**Technology**: Pandas + NumPy for data manipulation

**Processing Pipeline**:
1. **Data Validation**: Remove outliers, handle missing values
2. **Technical Indicators**: RSI, moving averages, volatility metrics
3. **Feature Engineering**: Time-based features, lag variables
4. **Data Normalization**: Prepare features for ML model consumption

**Implementation**:
```python
class DataProcessor:
    def enrich_fx_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        df = raw_data.copy()
        
        # Technical indicators
        df['rsi'] = self.calculate_rsi(df['rate'], period=14)
        df['ma_20'] = df['rate'].rolling(window=20).mean()
        df['volatility'] = df['rate'].rolling(window=24).std()
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        return df
```

---

## 2. Machine Learning & Intelligence Layer

### LSTM Rate Forecaster (`forecaster.py`)
**Purpose**: Predicts currency exchange rates for next 1-24 hours with confidence intervals
**Technology**: PyTorch LSTM with attention mechanism

**Model Architecture**:
```python
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.fc_mean = nn.Linear(hidden_size, 1)  # Predicted rate
        self.fc_std = nn.Linear(hidden_size, 1)   # Uncertainty
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        mean = self.fc_mean(attn_out[:, -1, :])
        std = F.softplus(self.fc_std(attn_out[:, -1, :]))
        
        return mean, std
```

**Training Data**:
- 168 hours (7 days) of historical data as input sequence
- Multiple currency pairs: USD/EUR, USD/GBP, USD/JPY, etc.
- Features: OHLC rates, volume, spreads, technical indicators, time features

**Prediction Process**:
```python
class ForecastingService:
    async def predict_next_24h(self, currency_pair: str):
        # Get recent data
        recent_data = await self.get_recent_sequence(currency_pair, hours=168)
        features = self.processor.prepare_features(recent_data)
        
        # Model prediction
        with torch.no_grad():
            mean, std = self.model(features)
            
        return {
            'predicted_rate': float(mean),
            'confidence': float(1 / (1 + std)),  # Higher std = lower confidence
            'prediction_interval': {
                'lower': float(mean - 1.96 * std),
                'upper': float(mean + 1.96 * std)
            }
        }
```

**Model Performance**:
- Training: Walk-forward validation on 3+ years historical data
- Evaluation: MAPE (Mean Absolute Percentage Error) < 0.8% for 24h predictions
- Retraining: Daily with new data, A/B testing for model updates

### Cost Calculator (`cost_calculator.py`)
**Purpose**: Calculates true total cost across different providers including all fees
**Features**:
- Real-time spread analysis
- Fixed fee calculations
- Transfer speed considerations
- Hidden fee detection

**Implementation**:
```python
class CostCalculator:
    def calculate_total_cost(self, amount: float, from_curr: str, to_curr: str):
        providers = self.get_active_providers(from_curr, to_curr)
        cost_analysis = []
        
        for provider in providers:
            base_rate = self.get_mid_market_rate(from_curr, to_curr)
            provider_rate = self.get_provider_rate(provider, from_curr, to_curr)
            
            spread_cost = amount * (base_rate - provider_rate)
            fixed_fee = self.get_fixed_fee(provider, amount)
            
            total_cost = spread_cost + fixed_fee
            
            cost_analysis.append({
                'provider': provider,
                'exchange_rate': provider_rate,
                'spread_cost': spread_cost,
                'fixed_fee': fixed_fee,
                'total_cost': total_cost,
                'delivery_time': self.get_delivery_time(provider)
            })
        
        return sorted(cost_analysis, key=lambda x: x['total_cost'])
```

### Decision Engine (`decision_engine.py`)
**Purpose**: Makes "convert now" vs "wait" recommendations using rule-based logic
**Decision Matrix**:

| Predicted Savings | Confidence | Days to Deadline | Recommendation |
|-------------------|------------|------------------|----------------|
| > $50 | > 70% | > 7 days | WAIT |
| > $20 | > 60% | > 3 days | WAIT |
| < $10 | Any | Any | CONVERT_NOW |
| Any | < 50% | Any | CONVERT_NOW |
| Any | Any | < 2 days | CONVERT_NOW |

**Implementation**:
```python
class DecisionEngine:
    def make_recommendation(self, context: DecisionContext) -> Recommendation:
        current_cost = self.cost_calculator.calculate_best_cost(context.amount, 
                                                               context.from_currency, 
                                                               context.to_currency)
        
        prediction = await self.forecaster.predict_next_24h(context.currency_pair)
        predicted_cost = self.cost_calculator.calculate_future_cost(context.amount,
                                                                   prediction['predicted_rate'])
        
        expected_savings = current_cost - predicted_cost
        confidence = prediction['confidence']
        days_to_deadline = (context.deadline - datetime.now()).days if context.deadline else 30
        
        # Decision logic
        if expected_savings > 50 and confidence > 0.7 and days_to_deadline > 7:
            return Recommendation(
                action="WAIT",
                reason=f"Expected savings: ${expected_savings:.2f} (confidence: {confidence*100:.0f}%)",
                next_check=datetime.now() + timedelta(hours=24)
            )
        else:
            return Recommendation(
                action="CONVERT_NOW", 
                reason=self._generate_reason(expected_savings, confidence, days_to_deadline)
            )
```

---

## 3. Core Business Logic Layer

### Goal Manager (`goal_manager.py`)
**Purpose**: Tracks user conversion goals with deadlines and monitors progress
**Features**:
- Goal decomposition (split large amounts into smaller tranches)
- Deadline-based scheduling
- Progress tracking and milestone alerts
- Dynamic replanning based on market conditions

**Goal Types**:
- **One-time**: "I need €5000 by March for vacation"
- **Recurring**: "Convert $2000 to GBP every month"  
- **Threshold**: "Alert me when USD/EUR reaches 0.85"
- **Dollar-Cost Average**: "Convert $10000 over 3 months in equal parts"

**Implementation**:
```python
class GoalManager:
    async def create_goal(self, user_id: str, goal_request: GoalRequest) -> Goal:
        goal = Goal(
            user_id=user_id,
            source_amount=goal_request.amount,
            source_currency=goal_request.from_currency,
            target_currency=goal_request.to_currency,
            deadline=goal_request.deadline,
            risk_tolerance=goal_request.risk_tolerance
        )
        
        # Create conversion schedule
        schedule = await self.create_conversion_schedule(goal)
        goal.schedule = schedule
        
        # Get initial recommendation
        recommendation = await self.evaluate_goal(goal)
        goal.current_recommendation = recommendation
        
        return goal
    
    async def create_conversion_schedule(self, goal: Goal) -> List[ConversionWindow]:
        if goal.source_amount > 10000:  # Large amount - use tranching
            num_tranches = min(5, goal.source_amount // 2000)  # Max 5 tranches
            tranche_size = goal.source_amount / num_tranches
            
            windows = []
            for i in range(num_tranches):
                window = ConversionWindow(
                    amount=tranche_size,
                    earliest_date=datetime.now() + timedelta(days=i*7),
                    latest_date=goal.deadline - timedelta(days=(num_tranches-i-1)*2)
                )
                windows.append(window)
            
            return windows
        else:
            # Single conversion
            return [ConversionWindow(
                amount=goal.source_amount,
                earliest_date=datetime.now(),
                latest_date=goal.deadline
            )]
```

### Conversion Assistant (`conversion_assistant.py`)
**Purpose**: Main orchestrator that coordinates all components for user requests
**Workflow**:
1. Receive user query
2. Get current market data
3. Generate ML prediction
4. Calculate costs across providers
5. Make recommendation with reasoning
6. Log decision for learning

**Implementation**:
```python
class ConversionAssistant:
    def __init__(self):
        self.forecaster = ForecastingService()
        self.cost_calculator = CostCalculator()
        self.decision_engine = DecisionEngine()
        self.goal_manager = GoalManager()
    
    async def analyze_conversion(self, request: ConversionRequest) -> ConversionAnalysis:
        # Get current market state
        current_rate = await self.get_current_rate(request.currency_pair)
        provider_costs = await self.cost_calculator.calculate_all_providers(
            request.amount, request.from_currency, request.to_currency
        )
        
        # Get prediction
        prediction = await self.forecaster.predict_next_24h(request.currency_pair)
        
        # Make decision
        decision_context = DecisionContext(
            amount=request.amount,
            currency_pair=request.currency_pair,
            deadline=request.deadline,
            risk_tolerance=request.risk_tolerance,
            current_rate=current_rate,
            prediction=prediction
        )
        
        recommendation = await self.decision_engine.make_recommendation(decision_context)
        
        # Prepare response
        return ConversionAnalysis(
            recommendation=recommendation.action,
            reason=recommendation.reason,
            current_rate=current_rate,
            predicted_rate=prediction['predicted_rate'],
            confidence=f"{prediction['confidence']*100:.0f}%",
            best_provider=provider_costs[0],
            expected_savings=recommendation.expected_savings,
            next_check=recommendation.next_check
        )
```

### Notification Manager (`notification_manager.py`)
**Purpose**: Sends alerts when recommendations change or deadlines approach
**Channels**: Push notifications, email, SMS
**Triggers**:
- Rate alert thresholds hit
- Recommendation changes (WAIT → CONVERT_NOW)
- Goal deadlines approaching
- Significant market moves

**Implementation**:
```python
class NotificationManager:
    def __init__(self):
        self.fcm_client = FCMClient()
        self.email_client = SendGridClient()
        self.sms_client = TwilioClient()
    
    async def send_rate_alert(self, user: User, alert: RateAlert):
        message = f"💰 {alert.currency_pair} reached {alert.target_rate}! " \
                 f"Time to convert? Check your recommendations."
        
        # Send via user's preferred channels
        if user.preferences.push_enabled:
            await self.fcm_client.send(user.device_token, message)
        
        if user.preferences.email_enabled:
            await self.email_client.send_alert_email(user.email, alert)
        
        if user.preferences.sms_enabled and alert.priority == 'HIGH':
            await self.sms_client.send(user.phone, message)
```

---

## 4. API & Communication Layer

### FastAPI Web Server (`main.py`, `routes.py`)
**Purpose**: RESTful API endpoints for mobile and web applications
**Features**:
- Auto-generated OpenAPI documentation
- JWT authentication with refresh tokens
- Rate limiting and request validation
- CORS support for web clients

**Key Endpoints**:
```python
@app.post("/api/v1/analyze", response_model=ConversionAnalysis)
async def analyze_conversion(
    request: ConversionRequest,
    current_user: User = Depends(get_current_user)
):
    """Main endpoint: Should I convert currency now?"""
    return await conversion_assistant.analyze_conversion(request)

@app.post("/api/v1/goals", response_model=Goal)
async def create_goal(
    goal_request: GoalRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new conversion goal with deadline"""
    return await goal_manager.create_goal(current_user.id, goal_request)

@app.get("/api/v1/dashboard", response_model=UserDashboard) 
async def get_dashboard(
    current_user: User = Depends(get_current_user)
):
    """User's active goals and recommendations"""
    active_goals = await goal_manager.get_user_goals(current_user.id)
    recent_recommendations = await get_recent_recommendations(current_user.id)
    
    return UserDashboard(
        active_goals=active_goals,
        recommendations=recent_recommendations,
        savings_this_month=await calculate_savings(current_user.id)
    )

@app.get("/api/v1/providers", response_model=List[ProviderRate])
async def get_provider_rates(
    from_currency: str,
    to_currency: str,
    amount: float
):
    """Compare rates across all providers"""
    return await cost_calculator.calculate_all_providers(amount, from_currency, to_currency)
```

### WebSocket Manager (`websockets.py`)
**Purpose**: Real-time updates for connected clients
**Features**:
- Live rate updates
- Recommendation changes
- Goal progress notifications

**Implementation**:
```python
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
    
    async def broadcast_rate_update(self, currency_pair: str, new_rate: float):
        # Find users with active goals for this currency pair
        affected_users = await self.get_users_with_pair(currency_pair)
        
        for user_id in affected_users:
            if user_id in self.active_connections:
                await self.active_connections[user_id].send_json({
                    "type": "rate_update",
                    "currency_pair": currency_pair,
                    "rate": new_rate,
                    "timestamp": datetime.now().isoformat()
                })

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await connection_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client messages (heartbeat, subscription changes)
    except WebSocketDisconnect:
        connection_manager.disconnect(user_id)
```

---

## 5. Background Processing Layer

### Celery Task Processor (`tasks.py`)
**Purpose**: Handles scheduled and background jobs
**Redis**: Message broker and result backend
**Worker Configuration**: 4 workers for I/O bound tasks, 2 for CPU-intensive ML tasks

**Scheduled Tasks**:
```python
from celery import Celery
from celery.schedules import crontab

app = Celery('currency_assistant')

# Data collection - every 30 seconds
@app.task
def collect_fx_rates():
    asyncio.run(fx_collector.collect_all_pairs())

# Model predictions - every hour  
@app.task
def update_predictions():
    asyncio.run(forecasting_service.update_all_predictions())

# Goal evaluation - every 4 hours
@app.task  
def evaluate_user_goals():
    asyncio.run(goal_manager.evaluate_all_active_goals())

# Model retraining - daily at 2 AM
@app.task
def retrain_models():
    asyncio.run(model_trainer.retrain_lstm_models())

# Beat schedule
app.conf.beat_schedule = {
    'collect-fx-rates': {
        'task': 'tasks.collect_fx_rates',
        'schedule': 30.0,  # Every 30 seconds
    },
    'update-predictions': {
        'task': 'tasks.update_predictions', 
        'schedule': crontab(minute=0),  # Every hour
    },
    'evaluate-goals': {
        'task': 'tasks.evaluate_user_goals',
        'schedule': crontab(minute=0, hour='*/4'),  # Every 4 hours
    },
    'retrain-models': {
        'task': 'tasks.retrain_models',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    }
}
```

### Model Training Pipeline (`model_trainer.py`)
**Purpose**: Automated model training, validation, and deployment
**Features**:
- Walk-forward validation
- A/B testing framework
- Model versioning with MLflow
- Performance monitoring

**Training Process**:
```python
class ModelTrainer:
    async def retrain_lstm_models(self):
        currency_pairs = ['USD/EUR', 'USD/GBP', 'USD/JPY', 'EUR/GBP']
        
        for pair in currency_pairs:
            # Get training data (last 2 years)
            training_data = await self.get_training_data(pair, days=730)
            
            # Walk-forward validation
            metrics = await self.walk_forward_validation(training_data)
            
            if metrics['mape'] < self.current_models[pair].performance['mape']:
                # New model is better - deploy it
                await self.deploy_model(pair, metrics)
                await self.log_model_update(pair, metrics)
            
    async def walk_forward_validation(self, data: pd.DataFrame):
        """Time series cross-validation"""
        train_size = int(len(data) * 0.8)
        window_size = 168  # 7 days
        
        errors = []
        
        for i in range(train_size, len(data) - 24):  # 24h prediction horizon
            train_data = data[i-train_size:i]
            test_data = data[i:i+24]
            
            # Train model on this window
            model = self.train_lstm(train_data)
            
            # Predict next 24 hours
            prediction = model.predict(test_data.drop('rate', axis=1))
            actual = test_data['rate'].values
            
            error = np.abs((prediction - actual) / actual)
            errors.extend(error)
        
        return {
            'mape': np.mean(errors) * 100,
            'mae': np.mean(np.abs(np.array(errors))),
            'mse': np.mean(np.square(errors))
        }
```

---

## 6. Database & Storage Layer

### Database Schema (`models.py`)
**PostgreSQL + TimescaleDB** for time-series optimization

**Core Tables**:
```sql
-- Users and authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    risk_tolerance VARCHAR(20) DEFAULT 'moderate',
    preferred_providers JSONB,
    notification_preferences JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Currency conversion goals
CREATE TABLE goals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    source_amount DECIMAL(15,2) NOT NULL,
    source_currency VARCHAR(3) NOT NULL,
    target_amount DECIMAL(15,2),
    target_currency VARCHAR(3) NOT NULL,
    deadline TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    current_recommendation JSONB,
    conversion_schedule JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Time-series FX rates (TimescaleDB hypertable)
CREATE TABLE fx_rates (
    time TIMESTAMP NOT NULL,
    currency_pair VARCHAR(7) NOT NULL,
    rate DECIMAL(10,6) NOT NULL,
    bid DECIMAL(10,6),
    ask DECIMAL(10,6),
    volume BIGINT,
    provider VARCHAR(50),
    PRIMARY KEY (time, currency_pair)
);

SELECT create_hypertable('fx_rates', 'time');

-- Provider fees and spreads
CREATE TABLE provider_rates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider VARCHAR(50) NOT NULL,
    currency_pair VARCHAR(7) NOT NULL,
    rate DECIMAL(10,6) NOT NULL,
    spread_percent DECIMAL(5,4),
    fixed_fee DECIMAL(10,2),
    min_amount DECIMAL(15,2),
    max_amount DECIMAL(15,2),
    delivery_time_hours INTEGER,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Conversion history and outcomes
CREATE TABLE conversions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    goal_id UUID REFERENCES goals(id),
    amount DECIMAL(15,2) NOT NULL,
    from_currency VARCHAR(3) NOT NULL,
    to_currency VARCHAR(3) NOT NULL,
    rate_used DECIMAL(10,6) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    total_cost DECIMAL(15,2) NOT NULL,
    predicted_rate DECIMAL(10,6),
    actual_savings DECIMAL(10,2),
    executed_at TIMESTAMP DEFAULT NOW()
);
```

**SQLAlchemy Models**:
```python
from sqlalchemy import Column, String, Decimal, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    risk_tolerance = Column(String(20), default='moderate')
    preferred_providers = Column(JSON)
    notification_preferences = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Goal(Base):
    __tablename__ = 'goals'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    source_amount = Column(Decimal(15,2), nullable=False)
    source_currency = Column(String(3), nullable=False)
    target_currency = Column(String(3), nullable=False)
    deadline = Column(DateTime)
    status = Column(String(20), default='active')
    current_recommendation = Column(JSON)
    conversion_schedule = Column(JSON)
```

### Redis Cache Strategy (`cache.py`)
**Purpose**: High-speed caching for frequently accessed data
**Cache Patterns**:
- **Current Rates**: TTL 30 seconds
- **Provider Fees**: TTL 4 hours  
- **User Sessions**: TTL 24 hours
- **Predictions**: TTL 1 hour

**Implementation**:
```python
class CacheManager:
    def __init__(self):
        self.redis_client = aioredis.from_url("redis://localhost:6379")
    
    async def get_current_rate(self, currency_pair: str) -> Optional[float]:
        cached_rate = await self.redis_client.get(f"rate:{currency_pair}")
        if cached_rate:
            return float(cached_rate)
        return None
    
    async def set_current_rate(self, currency_pair: str, rate: float):
        await self.redis_client.setex(f"rate:{currency_pair}", 30, str(rate))
    
    async def cache_prediction(self, currency_pair: str, prediction: dict):
        key = f"prediction:{currency_pair}"
        await self.redis_client.setex(key, 3600, json.dumps(prediction))
```

---

## 7. Monitoring & Analytics Layer

### System Monitoring (`monitor.py`)
**Metrics Tracked**:
- API response times and error rates
- ML model prediction accuracy
- Data freshness and quality
- User engagement and conversion rates
- System resource utilization

**Implementation**:
```python
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Prometheus metrics
api_requests = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
prediction_accuracy = Gauge('ml_prediction_accuracy', 'MAPE of predictions', ['currency_pair'])
data_freshness = Gauge('data_freshness_seconds', 'Seconds since last data update', ['data_source'])

logger = structlog.get_logger()

class MonitoringService:
    @staticmethod
    def record_api_request(method: str, endpoint: str, status: int, duration: float):
        api_requests.labels(method=method, endpoint=endpoint, status=status).inc()
        
    @staticmethod
    def record_prediction_accuracy(currency_pair: str, mape: float):
        prediction_accuracy.labels(currency_pair=currency_pair).set(mape)
        
    @staticmethod
    async def check_system_health():
        health_checks = {
            'database': await self.check_database(),
            'redis': await self.check_redis(),
            'external_apis': await self.check_external_apis(),
            'ml_models': await self.check_ml_models()
        }
        
        logger.info("System health check", **health_checks)
        return health_checks
```

### Analytics Service (`analytics.py`)
**Purpose**: Track user behavior and system performance for insights
**Metrics**:
- Conversion success rates
- User engagement patterns
- Prediction accuracy by time horizon
- Cost savings achieved

**Implementation**:
```python
class AnalyticsService:
    async def track_conversion_outcome(self, conversion: Conversion):
        """Track actual vs predicted outcomes"""
        if conversion.predicted_rate:
            actual_accuracy = abs(conversion.rate_used - conversion.predicted_rate) / conversion.predicted_rate
            
            await self.record_metric('prediction_accuracy', {
                'currency_pair': f"{conversion.from_currency}/{conversion.to_currency}",
                'accuracy': actual_accuracy,
                'prediction_horizon': '24h'
            })
        
        await self.record_metric('user_conversion', {
            'user_id': conversion.user_id,
            'amount': float(conversion.amount),
            'savings': float(conversion.actual_savings or 0),
            'provider': conversion.provider
        })
    
    async def calculate_user_savings(self, user_id: str, days: int = 30) -> dict:
        """Calculate total savings for a user over time period"""
        conversions = await self.get_user_conversions(user_id, days)
        
        total_saved = sum(c.actual_savings for c in conversions if c.actual_savings)
        total_volume = sum(c.amount for c in conversions)
        
        return {
            'total_saved': total_saved,
            'total_volume': total_volume,
            'savings_rate': (total_saved / total_volume * 100) if total_volume > 0 else 0,
            'num_conversions': len(conversions)
        }
```

---

## 8. Project Structure

```
currency_assistant/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   └── dependencies.py         # Dependency injection
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── auth.py            # Authentication endpoints
│   │   ├── conversions.py     # Main conversion analysis
│   │   ├── goals.py           # Goal management
│   │   └── dashboard.py       # User dashboard
│   ├── websockets.py          # WebSocket connections
│   └── middleware.py          # Auth, CORS, monitoring
├── core/
│   ├── __init__.py
│   ├── conversion_assistant.py # Main orchestrator
│   ├── decision_engine.py      # Decision logic
│   └── goal_manager.py         # Goal tracking
├── data/
│   ├── __init__.py
│   ├── collector.py           # FX data collection
│   ├── processor.py           # Data processing
│   ├── models.py              # SQLAlchemy models
│   └── cache.py               # Redis operations
├── ml/
│   ├── __init__.py
│   ├── forecaster.py          # LSTM forecasting
│   ├── model_trainer.py       # Training pipeline  
│   └── cost_calculator.py     # Cost optimization
├── services/
│   ├── __init__.py
│   ├── providers.py           # External API clients
│   ├── notifications.py       # Alert system
│   ├── monitoring.py          # System monitoring
│   └── analytics.py           # User analytics
├── tasks/
│   ├── __init__.py
│   ├── celery_app.py          # Celery configuration
│   ├── data_tasks.py          # Data collection tasks
│   ├── ml_tasks.py            # ML training tasks
│   └── notification_tasks.py  # Notification tasks
├── utils/
│   ├── __init__.py
│   ├── calculations.py        # Financial calculations
│   ├── helpers.py             # Common utilities
│   └── exceptions.py          # Custom exceptions
├── tests/
│   ├── __init__.py
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── load/                  # Load testing
├── migrations/                 # Alembic migrations
├── docker/                    # Docker configuration
├── docs/                      # API documentation
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Poetry configuration
├── docker-compose.yml        # Local development
└── README.md                 # Project documentation
```

---

## Development & Deployment

### Local Development Setup
```bash
# Clone repository
git clone <repository-url>
cd currency_assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys and database URLs

# Start services with Docker Compose
docker-compose up -d postgres redis

# Run database migrations
alembic upgrade head

# Start Celery worker
celery -A tasks.celery_app worker --loglevel=info

# Start Celery beat scheduler
celery -A tasks.celery_app beat --loglevel=info

# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment
**Infrastructure**: AWS/GCP with Kubernetes
**Database**: RDS PostgreSQL with TimescaleDB
**Caching**: ElastiCache Redis
**Message Queue**: Amazon MQ (managed Celery)
**Monitoring**: Prometheus + Grafana + AlertManager

### Testing Strategy
```python
# Run all tests
pytest

# Test coverage
pytest --cov=app --cov-report=html

# Load testing
locust -f tests/load/locustfile.py --host=http://localhost:8000

# ML model validation
python -m tests.ml.test_forecaster
```

---

## Security & Compliance

### Data Security
- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **API Keys**: Stored in AWS Secrets Manager/HashiCorp Vault
- **User Data**: PII tokenization, GDPR compliance
- **Rate Limiting**: 100 requests/minute per user

### Financial Compliance
- **Data Sources**: Licensed financial data providers
- **Audit Trail**: All recommendations and outcomes logged
- **Disclaimers**: Clear risk warnings for users
- **Regulatory**: SOC 2 compliance for financial data handling

---

## Performance & Scaling

### Performance Targets
- **API Response Time**: < 200ms for 95% of requests
- **Prediction Latency**: < 500ms for LSTM inference
- **Data Freshness**: < 60 seconds for major currency pairs
- **Uptime**: 99.9% availability SLA

### Scaling Strategy
- **Horizontal Scaling**: Kubernetes pods with auto-scaling
- **Database**: Read replicas, connection pooling
- **Caching**: Redis Cluster with consistent hashing
- **ML Models**: Model serving with TorchServe/TensorFlow Serving

### Monitoring & Alerting
- **Prometheus**: Metrics collection
- **Grafana**: Real-time dashboards  
- **PagerDuty**: Critical alerts
- **Health Checks**: Kubernetes liveness/readiness probes

---

This comprehensive architecture provides a production-ready foundation for building an intelligent currency conversion assistant that can save users money through optimal timing recommendations while maintaining simplicity and reliability.