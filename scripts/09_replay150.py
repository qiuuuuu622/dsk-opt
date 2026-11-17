#!/usr/bin/env python3
"""Stream exactly 150 requests in schedule order, then drain all requests."""
import argparse
import asyncio
import gzip
import json
import math
import os
from pathlib import Path
import statistics
import time


def percentiles(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return {'n': 0, 'mean': None, **{f'p{q}': None for q in (50, 90, 95, 99)}}
    result = {'n': len(values), 'mean': statistics.mean(values)}
    for q in (50, 90, 95, 99):
        rank = (len(values) - 1) * q / 100
        lo, hi = math.floor(rank), math.ceil(rank)
        result[f'p{q}'] = values[lo] + (values[hi] - values[lo]) * (rank - lo)
    return result


def summarize(results, elapsed):
    ok = [r for r in results if r['status'] == 'ok']
    tokens = [r['completion_tokens'] for r in ok if r['completion_tokens'] is not None]
    return {
        'requested': 150, 'finished': len(results), 'success': len(ok),
        'failed': len(results) - len(ok), 'elapsed_sec': elapsed,
        'successful_qps': len(ok) / elapsed,
        'reported_output_tokens_per_sec': sum(tokens) / elapsed,
        'requests_with_token_usage': len(tokens),
        'ttft_ms': percentiles([r['ttft_ms'] for r in ok]),
        'tpot_ms_per_token': percentiles([r['tpot_ms_per_token'] for r in ok]),
        'e2e_ms': percentiles([r['e2e_ms'] for r in ok]),
        'definitions': {
            'ttft': 'HTTP request start to first nonempty content, reasoning, or tool-call delta; role-only chunks excluded',
            'e2e': 'HTTP request start to stream completion',
            'tpot': '(E2E - TTFT)/(server completion_tokens - 1); absent if usage missing or completion_tokens <= 1',
            'percentiles': 'linear interpolation across successful requests; missing samples excluded and n reported',
            'streaming': 'A streamed chunk can contain multiple tokens, especially with speculative decoding; TPOT is request-average, not per-token ITL',
        },
    }


async def run(args, plan):
    from openai import AsyncOpenAI
    events = plan['schedule']
    if len(events) != 150:
        raise ValueError('Prepared schedule must contain exactly 150 events')
    if [float(e['at_sec']) for e in events] != sorted(float(e['at_sec']) for e in events):
        raise ValueError('Schedule is not ordered')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally overwriting a previous run.
    handle = output.open('x', encoding='utf-8')
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.getenv('OPENAI_API_KEY', 'TEST_ONLY_UNUSED'),
                         timeout=args.timeout, max_retries=0)
    start = time.perf_counter()
    counts = {'ok': 0, 'error': 0}

    async def one(index, event):
        request = plan['resolved_requests'][event['event_id']]
        payload = {k: v for k, v in plan['config_registry'][request['config_id']].items() if v is not None}
        payload.update(model=args.model, messages=request['full_messages'], stream=True,
                       stream_options={'include_usage': True})
        t0 = time.perf_counter()
        first = last = usage = None
        finish_reasons = []
        output_deltas = []
        record = {'index': index + 1, 'event_id': event['event_id'],
                  'scheduled_at_sec': event['at_sec'], 'started_at_sec': t0 - start,
                  'dispatch_lag_ms': (t0 - start - float(event['at_sec'])) * 1000}
        async def consume():
                nonlocal first, last, usage
                stream = await client.chat.completions.create(
                    **payload, extra_headers={'x-request-id': str(event.get('x_request_id') or event['event_id'])})
                async with stream:
                    async for chunk in stream:
                        now = time.perf_counter()
                        for choice in chunk.choices or []:
                            delta = choice.delta
                            if choice.finish_reason is not None:
                                finish_reasons.append(choice.finish_reason)
                            if args.save_responses:
                                output_deltas.append(delta.model_dump(exclude_none=True))
                            nonempty = (delta.content or getattr(delta, 'reasoning_content', None)
                                        or getattr(delta, 'reasoning', None)
                                        or any(c.function and (c.function.name or c.function.arguments)
                                               for c in (delta.tool_calls or [])))
                            if nonempty:
                                first = now if first is None else first
                                last = now
                        if chunk.usage is not None:
                            usage = chunk.usage.model_dump()
        try:
            await asyncio.wait_for(consume(), timeout=args.timeout)
            end = time.perf_counter()
            tokens = usage.get('completion_tokens') if usage else None
            ttft = (first - t0) * 1000 if first is not None else None
            e2e = (end - t0) * 1000
            tpot = (e2e - ttft) / (tokens - 1) if ttft is not None and tokens is not None and tokens > 1 else None
            record.update(status='ok', ttft_ms=ttft, e2e_ms=e2e, tpot_ms_per_token=tpot,
                          last_output_ms=(last - t0) * 1000 if last is not None else None,
                          completion_tokens=tokens, usage=usage, finish_reasons=finish_reasons)
            if args.save_responses:
                record['output_deltas'] = output_deltas
        except Exception as exc:
            record.update(status='error', error=f'{type(exc).__name__}: {exc}',
                          elapsed_ms=(time.perf_counter() - t0) * 1000)
        counts[record['status']] += 1
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        handle.flush()
        print(json.dumps({'finished': sum(counts.values()), **counts, 'index': index + 1}), flush=True)
        return record

    tasks = []
    try:
        for index, event in enumerate(events):
            await asyncio.sleep(max(0, start + float(event['at_sec']) - time.perf_counter()))
            tasks.append(asyncio.create_task(one(index, event)))
            # Ensure creation and submission follow schedule order, even if behind schedule.
            await asyncio.sleep(0)
        print('All 150 requests submitted; waiting for in-flight requests.', flush=True)
        results = await asyncio.gather(*tasks)
        summary = summarize(results, time.perf_counter() - start)
        output.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()
        handle.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--model', default='sf-finance-dsk41-dynamo-vllm-alyeast-prod')
    ap.add_argument('--save-responses', action='store_true')
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    with gzip.open(args.plan, 'rt', encoding='utf-8') as f:
        plan = json.load(f)
    asyncio.run(run(args, plan))


if __name__ == '__main__':
    main()
