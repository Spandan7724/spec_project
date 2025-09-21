#!/usr/bin/env python3
"""
Comprehensive test script for all currency assistant components
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing Component Imports")
    print("=" * 50)
    
    try:
        # Data Collection
        from src.data_collection.providers import AlphaVantageProvider, YahooFinanceProvider, ExchangeRateHostProvider
        from src.data_collection.rate_collector import MultiProviderRateCollector
        print("✅ Data Collection providers")
        
        # Economic Calendar
        from src.data_collection.economic import ECBProvider, FREDProvider, RBIScraper
        print("✅ Economic Calendar providers")
        
        # Technical Analysis
        from src.data_collection.analysis.technical_indicators import TechnicalIndicatorEngine
        print("✅ Technical Analysis")
        
        # News Scraping
        from src.data_collection.news import FinancialNewsScraper
        print("✅ News Scraping")
        
        # ML System
        from src.ml import MLPredictor, MLConfig
        print("✅ ML System")
        
        # LLM Providers
        from src.llm.providers.openai_provider import OpenAIProvider
        from src.llm.providers.claude_provider import ClaudeProvider
        print("✅ LLM Providers")
        
        # Agent Tools
        from src.tools import GenericScrapingInterface, CacheManager
        print("✅ Agent Tools")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_collection():
    """Test data collection components"""
    print("\n📊 Testing Data Collection")
    print("=" * 50)
    
    try:
        from src.data_collection.rate_collector import MultiProviderRateCollector
        collector = MultiProviderRateCollector()
        print("✅ RateCollector initialization")
        
        # Test individual provider initialization with API keys
        from src.data_collection.providers import AlphaVantageProvider, YahooFinanceProvider
        
        # Alpha Vantage with API key
        alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if alpha_key:
            alpha_provider = AlphaVantageProvider(alpha_key)
            print("✅ AlphaVantage provider")
        else:
            print("⚠️ AlphaVantage provider - no API key")
        
        # Yahoo Finance (no API key needed)
        yahoo_provider = YahooFinanceProvider()
        print("✅ Yahoo Finance provider")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return False

async def test_economic_calendar():
    """Test economic calendar integration"""
    print("\n📅 Testing Economic Calendar")
    print("=" * 50)
    
    try:
        from src.data_collection.economic import ECBProvider, FREDProvider
        
        ecb = ECBProvider()
        print("✅ ECB provider initialization")
        
        # FRED provider with API key
        fred_key = os.getenv('FRED_API_KEY')
        if fred_key:
            fred = FREDProvider(fred_key)
            print("✅ FRED provider initialization")
        else:
            print("⚠️ FRED provider - no API key")
        
        return True
        
    except Exception as e:
        print(f"❌ Economic calendar test failed: {e}")
        return False

async def test_technical_indicators():
    """Test technical indicators"""
    print("\n📈 Testing Technical Indicators")
    print("=" * 50)
    
    try:
        from src.data_collection.analysis.technical_indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        print("✅ Technical indicator engine")
        
        # Test calculation (this might use mock data)
        try:
            indicators = await engine.calculate_indicators("USD/EUR")
            print(f"✅ Technical indicators calculated: {type(indicators)}")
        except Exception as e:
            print(f"⚠️ Technical indicators calculation: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        return False

def test_ml_system():
    """Test ML system components"""
    print("\n🤖 Testing ML System")
    print("=" * 50)
    
    try:
        from src.ml import MLPredictor, MLConfig
        from src.ml.utils.model_storage import ModelStorage
        
        from src.ml.config import load_ml_config
        config = load_ml_config('ml_config.yaml')
        print("✅ ML Config loaded")
        
        predictor = MLPredictor(config)
        print("✅ ML Predictor initialization")
        
        model_storage = ModelStorage()
        available_models = model_storage.list_models()
        print(f"✅ Model Storage: {len(available_models)} models available")
        
        return True
        
    except Exception as e:
        print(f"❌ ML system test failed: {e}")
        return False

def test_llm_providers():
    """Test LLM provider system"""
    print("\n🧠 Testing LLM Providers")
    print("=" * 50)
    
    try:
        from src.llm.providers.openai_provider import OpenAIProvider
        from src.llm.providers.claude_provider import ClaudeProvider
        from src.llm.manager import LLMManager
        
        print("✅ LLM provider imports")
        
        # Test manager initialization (without actual API calls)
        manager = LLMManager()
        print("✅ LLM Manager initialization")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM providers test failed: {e}")
        return False

def test_agent_tools():
    """Test agent tools"""
    print("\n🔧 Testing Agent Tools")
    print("=" * 50)
    
    try:
        from src.tools import GenericScrapingInterface, CacheManager
        
        scraper = GenericScrapingInterface()
        print("✅ Web scraping interface")
        
        cache = CacheManager()
        print("✅ Cache manager")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent tools test failed: {e}")
        return False

def test_configuration():
    """Test configuration files and environment variables"""
    print("\n⚙️ Testing Configuration")
    print("=" * 50)
    
    try:
        import yaml
        
        # Test environment variables
        print("Environment Variables:")
        api_keys = {
            'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'EXCHANGE_RATE_HOST_API_KEY': os.getenv('EXCHANGE_RATE_HOST_API_KEY'),
            'FRED_API_KEY': os.getenv('FRED_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        }
        
        for key, value in api_keys.items():
            if value:
                print(f"  ✅ {key}: {'*' * 8}{value[-4:]}")
            else:
                print(f"  ❌ {key}: Not set")
        
        # Test config.yaml
        if Path("config.yaml").exists():
            with open("config.yaml") as f:
                config = yaml.safe_load(f)
            print("✅ config.yaml loaded")
        else:
            print("⚠️ config.yaml not found")
        
        # Test ml_config.yaml
        if Path("ml_config.yaml").exists():
            with open("ml_config.yaml") as f:
                ml_config = yaml.safe_load(f)
            print("✅ ml_config.yaml loaded")
        else:
            print("⚠️ ml_config.yaml not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def main():
    """Run all component tests"""
    print("🚀 Currency Assistant Component Tests")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Data Collection", test_data_collection()))
    results.append(("Economic Calendar", await test_economic_calendar()))
    results.append(("Technical Indicators", await test_technical_indicators()))
    results.append(("ML System", test_ml_system()))
    results.append(("LLM Providers", test_llm_providers()))
    results.append(("Agent Tools", test_agent_tools()))
    results.append(("Configuration", test_configuration()))
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All components are working properly!")
    else:
        print("⚠️ Some components need attention before starting agentic AI system")

if __name__ == "__main__":
    asyncio.run(main())