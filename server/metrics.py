"""Request latency histograms and counters, rendered in Prometheus text format for GET /metrics.

Times are measured in the server, from submit to the engine (after tokenization) to each token
reaching the request's coroutine.
"""

import threading
import time
from collections import defaultdict

TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
TPOT_BUCKETS = (0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1)
E2E_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300)


class Histogram:
    def __init__(self, name, help_text, buckets):
        self.name, self.help, self.buckets = name, help_text, buckets
        self.counts = [0] * len(buckets)  # per bucket, not cumulative
        self.sum = 0.0
        self.count = 0

    def observe(self, value):
        for i, upper in enumerate(self.buckets):
            if value <= upper:
                self.counts[i] += 1
                break
        self.sum += value
        self.count += 1

    def render(self):
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        cumulative = 0
        for upper, c in zip(self.buckets, self.counts):
            cumulative += c
            lines.append(f'{self.name}_bucket{{le="{upper}"}} {cumulative}')
        lines += [f'{self.name}_bucket{{le="+Inf"}} {self.count}',
                  f"{self.name}_sum {self.sum:.6f}", f"{self.name}_count {self.count}"]
        return lines


class RequestTimer:
    """One request's timeline: submit, first token, last token."""

    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens
        self.start = time.perf_counter()
        self.first = None
        self.last = None
        self.tokens = 0

    def token(self):
        self.last = time.perf_counter()
        if self.first is None:
            self.first = self.last
        self.tokens += 1


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.ttft = Histogram("llm_time_to_first_token_seconds",
                              "Submit to first generated token.", TTFT_BUCKETS)
        self.tpot = Histogram("llm_time_per_output_token_seconds",
                              "Mean gap between a request's tokens after the first.", TPOT_BUCKETS)
        self.e2e = Histogram("llm_request_latency_seconds", "Submit to the request ending.", E2E_BUCKETS)
        self.finished = defaultdict(int)  # by finish reason
        self.refused = defaultdict(int)   # by refusal status
        self.prompt_tokens = 0
        self.generation_tokens = 0

    def refuse(self, status):
        with self.lock:
            self.refused[status] += 1

    def finish(self, timer, reason):
        end = time.perf_counter()
        with self.lock:
            self.finished[reason] += 1
            self.prompt_tokens += timer.prompt_tokens
            self.generation_tokens += timer.tokens
            self.e2e.observe(end - timer.start)
            if timer.first is not None:
                self.ttft.observe(timer.first - timer.start)
            if timer.tokens > 1:
                self.tpot.observe((timer.last - timer.first) / (timer.tokens - 1))

    def render(self, engine_stats, healthy):
        """Prometheus text: server histograms and counters, then the engine's own stats."""
        def metric(name, kind, help_text, samples):
            out = [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]
            out += [f"{name}{labels} {value}" for labels, value in samples]
            return out

        with self.lock:
            lines = self.ttft.render() + self.tpot.render() + self.e2e.render()
            lines += metric("llm_requests_finished_total", "counter", "Requests ended, by finish reason.",
                            [(f'{{reason="{r}"}}', c) for r, c in sorted(self.finished.items())])
            lines += metric("llm_requests_refused_total", "counter", "Requests turned away, by status.",
                            [(f'{{status="{s}"}}', c) for s, c in sorted(self.refused.items())])
            lines += metric("llm_prompt_tokens_total", "counter", "Prompt tokens of finished requests.",
                            [("", self.prompt_tokens)])
            lines += metric("llm_generation_tokens_total", "counter",
                            "Tokens generated for finished requests.", [("", self.generation_tokens)])

        s = engine_stats
        gauges = [
            ("llm_requests_running", "Requests decoding now.", s["running"]),
            ("llm_requests_waiting", "Requests queued for a slot or KV pages.", s["waiting"]),
            ("llm_kv_pages_used", "KV pages held now.", s["kv_pages_used"]),
            ("llm_kv_pages_committed", "KV pages promised to running requests.", s["kv_pages_committed"]),
            ("llm_kv_pages_total", "KV pages in the pool.", s["kv_pages_total"]),
            ("llm_engine_healthy", "1 while the engine works, 0 after a failure.", int(healthy)),
        ]
        for name, help_text, value in gauges:
            lines += metric(name, "gauge", help_text, [("", value)])
        counters = [
            ("llm_engine_steps_total", "Decode passes.", s["steps"]),
            ("llm_engine_prefill_tokens_total", "Prompt tokens run through the model.", s["prefill_tokens"]),
            ("llm_engine_decode_tokens_total", "Rows decoded, one per running request per pass.",
             s["decode_tokens"]),
            ("llm_engine_busy_seconds_total", "Worker wall time spent stepping.", f'{s["busy_seconds"]:.6f}'),
        ]
        for name, help_text, value in counters:
            lines += metric(name, "counter", help_text, [("", value)])
        return "\n".join(lines) + "\n"
