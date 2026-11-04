"""Load test for server/app.py: N streamed completions arriving at a chosen rate.

Prompts are exact-length slices of an offline-tokenized text, sent as token ids, and every
request generates exactly --max-tokens (ignore_eos), so runs are comparable. Reports client-side
TTFT / TPOT / E2E percentiles and throughput, plus the engine's average batch size and busy time
from the server's /metrics.

Usage:
    python bench/load_test.py --url http://127.0.0.1:8000 --num-requests 64 --rate 4 \\
        --prompt-len 128 --max-tokens 128
"""

import argparse
import array
import asyncio
import json
import random
import time

import httpx


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--rate", type=float, default=0.0,
                   help="mean arrivals per second (Poisson); 0 sends every request at once")
    p.add_argument("--max-concurrency", type=int, default=0, help="cap on requests in flight; 0 is none")
    p.add_argument("--prompt-len", type=int, default=128, help="prompt tokens, BOS included")
    p.add_argument("--max-tokens", type=int, default=128, help="tokens generated per request")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--ids", default="model/wikitext2_test.ids", help="flat little-endian int32 token ids")
    p.add_argument("--bos", type=int, default=1, help="BOS id put first in each prompt; -1 for none")
    p.add_argument("--seed", type=int, default=0, help="seeds prompt choice and arrival gaps")
    p.add_argument("--timeout", type=float, default=600, help="client-side seconds per request")
    p.add_argument("--out", help="write per-request rows and the summary as JSON here")
    return p.parse_args()


def load_prompts(args, rng):
    ids = array.array("i")
    with open(args.ids, "rb") as f:
        ids.frombytes(f.read())
    body = args.prompt_len - (1 if args.bos >= 0 else 0)
    if body <= 0 or len(ids) <= body:
        raise SystemExit(f"--prompt-len {args.prompt_len} does not fit {args.ids} ({len(ids)} ids)")
    prompts = []
    for _ in range(args.num_requests):
        start = rng.randrange(len(ids) - body)
        prompts.append(([args.bos] if args.bos >= 0 else []) + ids[start:start + body].tolist())
    return prompts


async def one_request(client, args, prompt):
    """Streams one completion; returns its timeline and outcome as a dict."""
    payload = {"prompt": prompt, "max_tokens": args.max_tokens, "temperature": args.temperature,
               "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
    row = {"status": None, "error": None, "ttft": None, "tpot": None, "e2e": None,
           "output_tokens": 0, "prompt_tokens": len(prompt)}
    start = time.perf_counter()
    chunks = []  # arrival time of each chunk that carried text
    try:
        async with client.stream("POST", f"{args.url}/v1/completions", json=payload) as r:
            row["status"] = r.status_code
            if r.status_code != 200:
                row["error"] = (await r.aread()).decode(errors="replace")[:200]
                return row
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if "error" in event:
                    row["error"] = event["error"]["message"]
                elif event.get("usage"):
                    row["output_tokens"] = event["usage"]["completion_tokens"]
                elif event["choices"] and event["choices"][0]["text"]:
                    chunks.append(time.perf_counter())
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    end = time.perf_counter()
    row["e2e"] = end - start
    if chunks:
        row["ttft"] = chunks[0] - start
        if row["output_tokens"] > 1:
            row["tpot"] = (chunks[-1] - chunks[0]) / (row["output_tokens"] - 1)
    return row


async def scrape(client, url):
    """Counters and gauges from /metrics as {name: value}; histogram lines are skipped."""
    try:
        r = await client.get(f"{url}/metrics")
    except httpx.HTTPError:
        return {}
    out = {}
    for line in r.text.splitlines():
        if line and not line.startswith("#") and "{" not in line:
            name, value = line.rsplit(" ", 1)
            out[name] = float(value)
    return out


def percentile(values, q):
    """Linear interpolation between closest ranks; q in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(args, rows, duration, before, after):
    ok = [r for r in rows if r["status"] == 200 and r["error"] is None]
    refused = sum(r["status"] == 429 for r in rows)
    out_tokens = sum(r["output_tokens"] for r in ok)
    in_tokens = sum(r["prompt_tokens"] for r in ok)
    summary = {
        "sent": len(rows), "ok": len(ok), "refused_429": refused, "failed": len(rows) - len(ok) - refused,
        "duration_s": duration, "request_rate": args.rate, "max_concurrency": args.max_concurrency,
        "prompt_len": args.prompt_len, "max_tokens": args.max_tokens,
        "req_per_s": len(ok) / duration, "output_tok_per_s": out_tokens / duration,
        "total_tok_per_s": (in_tokens + out_tokens) / duration,
    }
    for key in ("ttft", "tpot", "e2e"):
        values = [r[key] for r in ok if r[key] is not None]
        for q in (50, 90, 99):
            summary[f"{key}_p{q}"] = percentile(values, q)
        summary[f"{key}_mean"] = sum(values) / len(values) if values else float("nan")

    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    steps = delta.get("llm_engine_steps_total", 0)
    if steps > 0:
        summary["mean_batch"] = delta["llm_engine_decode_tokens_total"] / steps
        summary["engine_busy_pct"] = 100 * delta["llm_engine_busy_seconds_total"] / duration
    return summary


def print_summary(s):
    print(f"requests   {s['sent']} sent, {s['ok']} ok, {s['refused_429']} refused (429), {s['failed']} failed")
    rate = "all at once" if not s["request_rate"] else f"{s['request_rate']:g} req/s Poisson"
    print(f"load       {rate}, prompt {s['prompt_len']} + output {s['max_tokens']} tokens")
    print(f"duration   {s['duration_s']:.2f} s")
    print(f"throughput {s['req_per_s']:.2f} req/s, {s['output_tok_per_s']:.1f} output tok/s, "
          f"{s['total_tok_per_s']:.1f} total tok/s")
    if "mean_batch" in s:
        print(f"engine     mean decode batch {s['mean_batch']:.2f}, busy {s['engine_busy_pct']:.0f}% of the run")
    print(f"{'':12}{'p50':>10}{'p90':>10}{'p99':>10}{'mean':>10}")
    for key, label, scale in (("ttft", "TTFT ms", 1e3), ("tpot", "TPOT ms", 1e3), ("e2e", "E2E s", 1)):
        cells = "".join(f"{s[f'{key}_{q}'] * scale:>10.1f}" for q in ("p50", "p90", "p99", "mean"))
        print(f"{label:12}{cells}")


async def main():
    args = parse_args()
    rng = random.Random(args.seed)
    prompts = load_prompts(args, rng)
    limit = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency > 0 else None

    async with httpx.AsyncClient(timeout=args.timeout, limits=httpx.Limits(max_connections=None)) as client:
        before = await scrape(client, args.url)

        async def run(prompt):
            if limit is None:
                return await one_request(client, args, prompt)
            async with limit:
                return await one_request(client, args, prompt)

        tasks = []
        start = time.perf_counter()
        for prompt in prompts:
            tasks.append(asyncio.create_task(run(prompt)))
            if args.rate > 0:
                await asyncio.sleep(rng.expovariate(args.rate))
        rows = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start
        after = await scrape(client, args.url)

    summary = summarize(args, rows, duration, before, after)
    print_summary(summary)
    errors = [r["error"] for r in rows if r["error"] and r["status"] != 429]
    if errors:
        print(f"first error: {errors[0]}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "requests": rows}, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
