#!/usr/bin/env python3
"""
Comprehensive test for provider cost analysis system.
"""

import asyncio
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection.providers.cost_analyzer import (
    ProviderCostAnalyzer,
    get_cost_comparison,
    get_best_provider
)

from data_collection.providers.wise_provider import WiseProvider
from data_collection.providers.revolut_provider import RevolutProvider  
from data_collection.providers.bank_provider import BankProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_individual_providers():
    """Test each provider individually."""
    
    logger.info("💰 Testing Individual Cost Providers")
    
    test_amount = 1000.0
    base_currency = "USD"
    quote_currency = "EUR"
    
    results = {}
    
    # Test Wise Provider
    logger.info("\n💳 Testing Wise Provider")
    try:
        async with WiseProvider() as wise:
            wise_cost = await wise.get_conversion_cost(base_currency, quote_currency, test_amount)
            
            if wise_cost:
                results['Wise'] = {
                    'success': True,
                    'cost_data': wise_cost.to_dict()
                }
                logger.info(f"✅ Wise: {test_amount} {base_currency} → {wise_cost.amount_received:.2f} {quote_currency}")
                logger.info(f"   • Exchange rate: {wise_cost.exchange_rate:.4f}")
                logger.info(f"   • Markup: {wise_cost.markup_percentage:.2f}%")
                logger.info(f"   • Transfer fee: {wise_cost.transfer_fee:.2f}")
                logger.info(f"   • Speed: {wise_cost.transfer_speed}")
                logger.info(f"   • Savings vs bank: ${wise_cost.savings_vs_bank:.2f}" if wise_cost.savings_vs_bank else "")
            else:
                results['Wise'] = {'success': False, 'error': 'No cost data returned'}
                logger.warning("❌ Wise: No cost data returned")
                
    except Exception as e:
        results['Wise'] = {'success': False, 'error': str(e)}
        logger.error(f"❌ Wise: Error - {e}")
    
    # Test Revolut Provider
    logger.info("\n🟣 Testing Revolut Provider")
    try:
        # Test different account tiers
        for tier in ['standard', 'premium']:
            logger.info(f"Testing Revolut {tier} account...")
            
            async with RevolutProvider(account_tier=tier) as revolut:
                revolut_cost = await revolut.get_conversion_cost(base_currency, quote_currency, test_amount)
                
                if revolut_cost:
                    results[f'Revolut_{tier}'] = {
                        'success': True,
                        'cost_data': revolut_cost.to_dict()
                    }
                    logger.info(f"✅ Revolut ({tier}): {test_amount} {base_currency} → {revolut_cost.amount_received:.2f} {quote_currency}")
                    logger.info(f"   • Exchange rate: {revolut_cost.exchange_rate:.4f}")
                    logger.info(f"   • Markup: {revolut_cost.markup_percentage:.2f}%")
                    logger.info(f"   • Speed: {revolut_cost.transfer_speed}")
                    logger.info(f"   • Percentage fee: {revolut_cost.percentage_fee:.2f}%")
                else:
                    results[f'Revolut_{tier}'] = {'success': False, 'error': 'No cost data returned'}
                    logger.warning(f"❌ Revolut ({tier}): No cost data returned")
                    
    except Exception as e:
        results['Revolut'] = {'success': False, 'error': str(e)}
        logger.error(f"❌ Revolut: Error - {e}")
    
    # Test Bank Provider
    logger.info("\n🏛️  Testing Bank Provider")
    try:
        # Test different banks
        banks = ['Chase', 'HSBC', 'Citibank']
        
        for bank_name in banks:
            logger.info(f"Testing {bank_name}...")
            
            async with BankProvider(bank_name=bank_name) as bank:
                bank_cost = await bank.get_conversion_cost(base_currency, quote_currency, test_amount)
                
                if bank_cost:
                    results[bank_name] = {
                        'success': True,
                        'cost_data': bank_cost.to_dict()
                    }
                    logger.info(f"✅ {bank_name}: {test_amount} {base_currency} → {bank_cost.amount_received:.2f} {quote_currency}")
                    logger.info(f"   • Exchange rate: {bank_cost.exchange_rate:.4f}")
                    logger.info(f"   • Total markup: {bank_cost.markup_percentage:.2f}%")
                    logger.info(f"   • Wire fee: ${bank_cost.transfer_fee:.2f}")
                    logger.info(f"   • Speed: {bank_cost.transfer_speed}")
                else:
                    results[bank_name] = {'success': False, 'error': 'No cost data returned'}
                    logger.warning(f"❌ {bank_name}: No cost data returned")
                    
    except Exception as e:
        results['Banks'] = {'success': False, 'error': str(e)}
        logger.error(f"❌ Banks: Error - {e}")
    
    return results


async def test_cost_comparison():
    """Test the integrated cost comparison system."""
    
    logger.info("\n📊 Testing Cost Comparison Engine")
    
    analyzer = ProviderCostAnalyzer()
    
    # Register providers
    analyzer.register_provider(WiseProvider())
    analyzer.register_provider(RevolutProvider(account_tier='premium'))
    analyzer.register_provider(BankProvider(bank_name='Chase'))
    analyzer.register_provider(BankProvider(bank_name='HSBC'))
    
    test_cases = [
        {'amount': 500, 'from': 'USD', 'to': 'EUR'},
        {'amount': 5000, 'from': 'GBP', 'to': 'USD'}, 
        {'amount': 10000, 'from': 'USD', 'to': 'INR'}
    ]
    
    results = {}
    
    for i, case in enumerate(test_cases, 1):
        logger.info(f"\n💸 Test Case {i}: {case['amount']} {case['from']} → {case['to']}")
        
        try:
            comparison = await analyzer.get_cost_comparison(
                base_currency=case['from'],
                quote_currency=case['to'],
                amount=case['amount']
            )
            
            if comparison:
                results[f'case_{i}'] = {
                    'success': True,
                    'comparison_data': comparison.to_dict()
                }
                
                logger.info("✅ Cost comparison completed")
                logger.info(f"   • Providers compared: {comparison.provider_count}")
                logger.info(f"   • Best provider: {comparison.best_provider.provider_name if comparison.best_provider else 'None'}")
                logger.info(f"   • Worst provider: {comparison.worst_provider.provider_name if comparison.worst_provider else 'None'}")
                logger.info(f"   • Potential savings: ${comparison.potential_savings:.2f}")
                
                # Show detailed ranking
                logger.info("\n🏆 Provider Ranking:")
                for rank, provider in comparison.get_ranking()[:5]:  # Top 5
                    logger.info(f"   {rank}. {provider.provider_name}: "
                              f"{provider.amount_received:.2f} {case['to']} "
                              f"(rate: {provider.exchange_rate:.4f}, "
                              f"total markup: {provider.markup_percentage:.2f}%)")
                
            else:
                results[f'case_{i}'] = {'success': False, 'error': 'No comparison returned'}
                logger.warning("❌ Cost comparison failed")
                
        except Exception as e:
            results[f'case_{i}'] = {'success': False, 'error': str(e)}
            logger.error(f"❌ Cost comparison error: {e}")
    
    return results


async def test_convenience_functions():
    """Test convenience functions."""
    
    logger.info("\n🛠️  Testing Convenience Functions")
    
    try:
        # Test get_cost_comparison
        logger.info("Testing get_cost_comparison()...")
        comparison = await get_cost_comparison('USD', 'EUR', 2000)
        
        if comparison:
            logger.info(f"✅ get_cost_comparison: {comparison.provider_count} providers compared")
            logger.info(f"   • Best: {comparison.best_provider.provider_name if comparison.best_provider else 'None'}")
            logger.info(f"   • Savings potential: ${comparison.potential_savings:.2f}")
        else:
            logger.warning("❌ get_cost_comparison: No comparison returned")
        
        # Test get_best_provider
        logger.info("Testing get_best_provider()...")
        best = await get_best_provider('USD', 'EUR', 2000)
        
        if best:
            logger.info(f"✅ get_best_provider: {best.provider_name}")
            logger.info(f"   • Rate: {best.exchange_rate:.4f}")
            logger.info(f"   • Amount received: {best.amount_received:.2f}")
            logger.info(f"   • Total costs: ${best.total_cost:.2f}")
        else:
            logger.warning("❌ get_best_provider: No provider returned")
            
    except Exception as e:
        logger.error(f"❌ Convenience function error: {e}")


async def test_real_world_scenarios():
    """Test real-world conversion scenarios."""
    
    logger.info("\n🌍 Testing Real-World Scenarios")
    
    scenarios = [
        {
            'name': 'Small tourist conversion',
            'amount': 200,
            'from': 'USD',
            'to': 'EUR',
            'priority': 'cost'
        },
        {
            'name': 'Business payment', 
            'amount': 50000,
            'from': 'USD',
            'to': 'GBP',
            'priority': 'speed'
        },
        {
            'name': 'Remittance to India',
            'amount': 3000,
            'from': 'USD', 
            'to': 'INR',
            'priority': 'cost'
        },
        {
            'name': 'Property purchase',
            'amount': 100000,
            'from': 'USD',
            'to': 'EUR',
            'priority': 'security'
        }
    ]
    
    for scenario in scenarios:
        logger.info(f"\n💼 Scenario: {scenario['name']}")
        logger.info(f"   Amount: {scenario['amount']} {scenario['from']} → {scenario['to']}")
        logger.info(f"   Priority: {scenario['priority']}")
        
        try:
            comparison = await get_cost_comparison(
                scenario['from'], 
                scenario['to'], 
                scenario['amount']
            )
            
            if comparison:
                best = comparison.best_provider
                
                if best:
                    logger.info(f"   💰 Best option: {best.provider_name}")
                    logger.info(f"      • You'll receive: {best.amount_received:.2f} {scenario['to']}")
                    logger.info(f"      • Total cost: ${best.total_cost:.2f}")
                    logger.info(f"      • Speed: {best.transfer_speed}")
                    logger.info(f"      • Savings vs bank: ${best.savings_vs_bank:.2f}" if best.savings_vs_bank else "")
                    
                    # Show alternative recommendations based on priority
                    if scenario['priority'] == 'speed':
                        fast_providers = [p for p in comparison.providers 
                                        if p.transfer_speed in ['instant', 'fast'] and p.is_quote_valid]
                        if fast_providers:
                            fastest = min(fast_providers, key=lambda x: ['instant', 'fast'].index(x.transfer_speed))
                            if fastest != best:
                                logger.info(f"      ⚡ Fastest option: {fastest.provider_name} "
                                          f"({fastest.amount_received:.2f} {scenario['to']})")
                    
                    elif scenario['priority'] == 'security':
                        bank_providers = comparison.get_providers_by_type(
                            comparison.providers[0].provider_type.__class__.TRADITIONAL_BANK
                        )
                        if bank_providers:
                            best_bank = min(bank_providers, key=lambda x: x.effective_rate)
                            logger.info(f"      🏛️  Safest option: {best_bank.provider_name} "
                                      f"({best_bank.amount_received:.2f} {scenario['to']})")
                
            else:
                logger.warning("   ❌ No comparison available for this scenario")
                
        except Exception as e:
            logger.error(f"   ❌ Scenario error: {e}")


async def main():
    """Run all provider cost tests."""
    
    logger.info("🚀 Currency Assistant - Provider Cost Analysis Test Suite")
    logger.info("=" * 70)
    
    try:
        # Run all tests
        provider_results = await test_individual_providers()
        comparison_results = await test_cost_comparison()
        await test_convenience_functions()
        await test_real_world_scenarios()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📋 PROVIDER COST ANALYSIS TEST SUMMARY")
        
        # Provider summary
        working_providers = sum(1 for r in provider_results.values() if r['success'])
        total_providers = len(provider_results)
        
        logger.info(f"💳 Individual Providers: {working_providers}/{total_providers} working")
        
        for provider, result in provider_results.items():
            if result['success']:
                cost_data = result['cost_data']
                logger.info(f"   ✅ {provider}: {cost_data['amount_received']:.2f} {cost_data['quote_currency']} "
                          f"(rate: {cost_data['exchange_rate']:.4f})")
            else:
                logger.error(f"   ❌ {provider}: {result['error']}")
        
        # Comparison summary
        working_comparisons = sum(1 for r in comparison_results.values() if r['success'])
        total_comparisons = len(comparison_results)
        
        logger.info(f"📊 Cost Comparisons: {working_comparisons}/{total_comparisons} working")
        
        for case, result in comparison_results.items():
            if result['success']:
                comp_data = result['comparison_data']
                logger.info(f"   ✅ {case}: {comp_data['provider_count']} providers, "
                          f"best: {comp_data['best_provider']}, "
                          f"savings: ${comp_data['potential_savings']:.2f}")
            else:
                logger.error(f"   ❌ {case}: {result['error']}")
        
        if working_providers > 0 and working_comparisons > 0:
            logger.info("🎉 PROVIDER COST ANALYSIS WORKING!")
        else:
            logger.warning("⚠️  Partial functionality - check provider integrations")
    
    except Exception as e:
        logger.error(f"❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())