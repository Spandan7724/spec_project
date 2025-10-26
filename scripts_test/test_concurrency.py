"""Test to visualize how semaphore controls concurrency."""

import asyncio
import time


async def mock_llm_call(article_num: int, delay: float = 2.0):
    """Simulate an LLM call that takes some time."""
    print(f"  📤 Starting article {article_num}")
    await asyncio.sleep(delay)
    print(f"  ✅ Finished article {article_num}")
    return f"result_{article_num}"


async def test_unlimited_concurrency(articles: int = 4):
    """Test with unlimited parallel calls."""
    print(f"\n{'='*60}")
    print(f"Test: Unlimited Concurrency (all {articles} articles at once)")
    print(f"{'='*60}")
    
    start = time.time()
    
    # Create tasks for all articles
    tasks = [mock_llm_call(i+1, delay=2.0) for i in range(articles)]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    
    print(f"\n⏱️  Total time: {elapsed:.2f} seconds")
    print(f"📊 Processed: {len(results)} articles")
    return elapsed


async def test_semaphore_concurrency(articles: int = 4, sem_limit: int = 2):
    """Test with semaphore limiting concurrency."""
    print(f"\n{'='*60}")
    print(f"Test: Semaphore Limited (max {sem_limit} concurrent)")
    print(f"{'='*60}")
    
    sem = asyncio.Semaphore(sem_limit)
    
    async def _one_with_sem(article_num: int):
        async with sem:
            return await mock_llm_call(article_num, delay=2.0)
    
    start = time.time()
    
    # Create tasks for all articles
    tasks = [asyncio.create_task(_one_with_sem(i+1)) for i in range(articles)]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    
    print(f"\n⏱️  Total time: {elapsed:.2f} seconds")
    print(f"📊 Processed: {len(results)} articles")
    return elapsed


async def main():
    print("\n🔬 Concurrency Control Demonstration")
    print("="*60)
    
    # Test 1: Unlimited (your current setup with 4 articles)
    unlimited_time = await test_unlimited_concurrency(articles=4)
    
    # Test 2: Limited to 2 concurrent (what happens if you have more articles)
    limited_time = await test_semaphore_concurrency(articles=8, sem_limit=2)
    
    # Test 3: Your actual setup with sem_limit=10, articles=4
    your_setup_time = await test_semaphore_concurrency(articles=4, sem_limit=10)
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"4 articles, unlimited:     {unlimited_time:.2f}s")
    print(f"4 articles, semaphore(10): {your_setup_time:.2f}s")
    print(f"8 articles, semaphore(2):  {limited_time:.2f}s")
    
    print("\n💡 Key Insight:")
    print("- With 4 articles and semaphore(10): All 4 run in parallel")
    print("- Total time ≈ time for 1 call (2s)")
    print("- Semaphore only matters when articles > semaphore limit")


if __name__ == "__main__":
    asyncio.run(main())
