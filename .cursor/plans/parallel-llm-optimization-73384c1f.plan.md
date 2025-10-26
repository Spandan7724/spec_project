<!-- 73384c1f-e418-4e20-b66f-6bb60fd826a4 b81dc4e0-057a-4df3-9c27-31802b4ee677 -->
# Parallel LLM Processing Optimization

## Goal

Reduce execution time from ~77 seconds to ~25-35 seconds by parallelizing major operations and increasing concurrency limits.

## Changes

### 1. Increase Concurrency Limits

**File**: `src/data_collection/market_intelligence/extractors/news_classifier.py`

Line 84: Change semaphore from 4 to 10 for more concurrent article classifications:

```python
sem = asyncio.Semaphore(10)  # Increased from 4
```

**File**: `src/data_collection/market_intelligence/extractors/calendar_extractor.py`

Line 142: Change semaphore from 3 to 8 for more concurrent event extractions:

```python
sem = asyncio.Semaphore(8)  # Increased from 3
```

### 2. Parallelize Major Operations

**File**: `src/data_collection/market_intelligence/intelligence_service.py`

Current sequential flow (lines 30-63):

```python
# Sequential - slow
news_snapshot = await self.aggregator.get_pair_snapshot(base, quote)
base_cal = await self.calendar.collect_calendar_urls(base, ...)
quote_cal = await self.calendar.collect_calendar_urls(quote, ...)
base_events = await self.cal_extractor.extract_events_batch(base_cal, base)
quote_events = await self.cal_extractor.extract_events_batch(quote_cal, quote)
```

Change to parallel execution using `asyncio.gather()`:

```python
import asyncio

# Run news aggregation, base calendar, and quote calendar in parallel
news_task = self.aggregator.get_pair_snapshot(base, quote)
base_cal_task = self.calendar.collect_calendar_urls(base, num_results=max(1, max_cal_sources * 2))
quote_cal_task = self.calendar.collect_calendar_urls(quote, num_results=max(1, max_cal_sources * 2))

news_snapshot, base_cal_full, quote_cal_full = await asyncio.gather(
    news_task, base_cal_task, quote_cal_task
)

# Limit results
base_cal = base_cal_full[:max_cal_sources]
quote_cal = quote_cal_full[:max_cal_sources]

# Run event extraction for both currencies in parallel
base_events, quote_events = await asyncio.gather(
    self.cal_extractor.extract_events_batch(base_cal, base),
    self.cal_extractor.extract_events_batch(quote_cal, quote)
)
```

### 3. Optional: Add Concurrency Config

**File**: `config.yaml`

Add configuration for concurrency limits (optional for future tuning):

```yaml
agents:
  market_intelligence:
    # ... existing config ...
    max_concurrent_llm_calls: 10  # Maximum concurrent LLM requests
```

## Expected Results

**Before**:

- News classification: Sequential per article (~15s × 4 articles = 60s)
- Calendar extraction: Sequential per currency (~10s × 4 sources = 40s)
- Total: ~77 seconds

**After**:

- News classification: Parallel 10 at a time (~15s for batch of 10)
- Calendar extraction: Parallel 8 at a time (~10s for batch of 8)
- Major operations: All running simultaneously
- Total: ~25-35 seconds (50-60% reduction)

## Testing

After changes, run:

```bash
uv run python test_graph.py
```

Expected output:

```
Execution time: 25-35 seconds (down from 77 seconds)
```

## Risks & Mitigations

**Risk**: API rate limiting with 10 concurrent calls

**Mitigation**: Semaphore prevents more than 10 simultaneous requests; can reduce to 8 if needed

**Risk**: Increased memory usage with parallel operations

**Mitigation**: Semaphores limit concurrency; operations still bounded

**Risk**: Harder to debug with parallel execution

**Mitigation**: Each operation has error handling; failures are captured individually

### To-dos

- [ ] Increase NewsClassifier semaphore from 4 to 10
- [ ] Increase CalendarExtractor semaphore from 3 to 8
- [ ] Refactor intelligence_service.py to use asyncio.gather() for parallel operations
- [ ] Run test_graph.py and verify execution time reduction