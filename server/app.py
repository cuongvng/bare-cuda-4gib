"""OpenAI-compatible HTTP server over the bare-cuda-4gib engine.

Text goes through the Hugging Face tokenizer here; the C++ engine only sees token ids.
Sampling follows OpenAI's defaults (temperature 1, top_p 1); temperature 0 is greedy.
top_k is an extra, non-OpenAI field.

Usage:
    python server/app.py --model model/tinyllama_int4.llmbin \\
        --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import asyncio
import itertools
import json
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from metrics import Metrics, RequestTimer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Refused(Exception):
    """The engine turned a request away; status is "invalid", "busy" or "unavailable"."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class AsyncLLM:
    """Bridges the engine's worker thread to asyncio: one thread polls, one queue per request."""

    def __init__(self, engine):
        self.engine = engine
        self.loop = asyncio.get_running_loop()
        self.queues = {}
        self.timers = {}
        self.metrics = Metrics()
        self.ids = itertools.count()
        self.running = True
        self.thread = threading.Thread(target=self._dispatch, name="llm-dispatch", daemon=True)
        self.thread.start()

    def _dispatch(self):
        while self.running:
            for event in self.engine.poll(100):
                queue = self.queues.get(event[0])
                if queue is not None:
                    self.loop.call_soon_threadsafe(queue.put_nowait, event)

    def submit(self, prompt_ids, max_new_tokens, stop_ids, **options):
        """Queues a request and returns its id; raises Refused if the engine turns it away."""
        rid = next(self.ids)
        self.queues[rid] = asyncio.Queue()  # registered first, so no event can arrive unrouted
        status, message = self.engine.submit(rid, prompt_ids, max_new_tokens, stop_ids, **options)
        if status != "ok":
            del self.queues[rid]
            self.metrics.refuse(status)
            raise Refused(status, message)
        self.timers[rid] = RequestTimer(len(prompt_ids))
        return rid

    async def stream(self, rid):
        """Yields (token, finish) until the request ends; cancels it if the caller stops early."""
        finished = False
        reason = "cancelled"  # unless the engine says otherwise, the caller walked away
        timer = self.timers[rid]
        try:
            while True:
                _, token, _, finish = await self.queues[rid].get()
                if token >= 0:
                    timer.token()
                finished = bool(finish)
                if finished:
                    reason = finish
                yield token, finish
                if finished:
                    return
        finally:
            self.queues.pop(rid, None)
            self.metrics.finish(self.timers.pop(rid), reason)
            if not finished:
                self.engine.cancel(rid)

    def close(self):
        self.running = False
        self.thread.join()
        self.engine.stop()


class Detokenizer:
    """Turns a growing list of ids into text deltas, holding back half-finished UTF-8 characters."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = []
        self.text = ""

    def add(self, token):
        self.ids.append(token)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�"):
            return ""
        delta, self.text = text[len(self.text):], text
        return delta


def build_app(args):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from pydantic import BaseModel
    from transformers import AutoTokenizer

    sys.path.insert(0, args.engine_dir)
    import llm_engine_py

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    state = {}

    @asynccontextmanager
    async def lifespan(_app):
        engine = llm_engine_py.Engine(args.model, max_tokens=args.max_tokens, max_seqs=args.max_seqs,
                                      kv_pages=args.kv_pages, max_waiting=args.max_waiting)
        print(f"[server] engine ready: {engine.max_seqs} slots, {engine.kv_pages} KV pages, "
              f"{engine.max_tokens} tokens per sequence", flush=True)
        state["llm"] = AsyncLLM(engine)
        yield
        state["llm"].close()

    app = FastAPI(title="bare-cuda-4gib", lifespan=lifespan)

    class CompletionRequest(BaseModel):
        model: Optional[str] = None
        prompt: Union[str, List[int]]
        max_tokens: Optional[int] = 16
        stream: bool = False
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        top_k: Optional[int] = None
        seed: Optional[int] = None
        ignore_eos: bool = False
        stream_options: Optional[dict] = None
        ignore_eos: bool = False  # non-OpenAI: always generate max_tokens, for benchmarks
        stream_options: Optional[dict] = None

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: Optional[str] = None
        messages: List[ChatMessage]
        max_tokens: Optional[int] = None
        stream: bool = False
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        top_k: Optional[int] = None
        seed: Optional[int] = None

    # HTTP status and OpenAI-style error type for each way a request can fail.
    failures = {
        "invalid": (400, "invalid_request_error"),
        "busy": (429, "rate_limit_error"),
        "unavailable": (503, "service_unavailable"),
        "timeout": (504, "timeout"),
        "error": (500, "server_error"),
    }

    def error_body(message, kind):
        return {"error": {"message": message, "type": failures[kind][1]}}

    def error(message, kind="invalid"):
        headers = {"Retry-After": "1"} if kind == "busy" else None
        return JSONResponse(error_body(message, kind), status_code=failures[kind][0], headers=headers)

    def usage(prompt_ids, count):
        return {"prompt_tokens": len(prompt_ids), "completion_tokens": count,
                "total_tokens": len(prompt_ids) + count}

    def wants_usage(req):
        return bool(req.stream_options and req.stream_options.get("include_usage"))

    def failure_message(finish, count):
        if finish == "timeout":
            return f"request timed out after {args.request_timeout} s ({count} tokens generated)"
        return "engine failed: " + state["llm"].engine.failure

    def sse(payload):
        return f"data: {json.dumps(payload)}\n\n"

    def sampling(req):
        """Engine sampling kwargs from a request, with OpenAI's defaults for what it leaves out."""
        return {"temperature": 1.0 if req.temperature is None else req.temperature,
                "top_p": 1.0 if req.top_p is None else req.top_p,
                "top_k": req.top_k or 0,
                "seed": -1 if req.seed is None else req.seed,
                "timeout_ms": int(args.request_timeout * 1000)}

    def start(prompt_ids, req, kind):
        """Submits a request: (llm, id, response metadata, detokenizer), or an error response."""
        llm = state["llm"]
        budget = req.max_tokens if req.max_tokens is not None else llm.engine.max_tokens
        try:
            rid = llm.submit(prompt_ids, budget, [] if req.ignore_eos else stop_ids, **sampling(req))
        except Refused as e:
            return None, error(str(e), e.status)
        meta = {"id": f"{kind}-{uuid.uuid4().hex[:24]}", "created": int(time.time()),
                "model": args.model_name}
        return (llm, rid, meta, Detokenizer(tokenizer)), None

    async def collect(llm, rid, detok):
        """Runs a request to the end; returns (text, finish, tokens generated)."""
        text, finish, count = "", None, 0
        async for token, finish in llm.stream(rid):
            if token >= 0:
                count += 1
                text += detok.add(token)
        return text, finish, count

    @app.get("/health")
    async def health():
        engine = state["llm"].engine
        if not engine.healthy:
            return JSONResponse({"status": "failed", "error": engine.failure}, status_code=503)
        return {"status": "ok", "max_seqs": engine.max_seqs, "kv_pages": engine.kv_pages,
                "max_tokens": engine.max_tokens}

    @app.get("/metrics")
    async def metrics():
        llm = state["llm"]
        return PlainTextResponse(llm.metrics.render(llm.engine.stats(), llm.engine.healthy),
                                 media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": args.model_name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompt_ids = req.prompt if isinstance(req.prompt, list) else tokenizer(req.prompt).input_ids
        started, refused = start(prompt_ids, req, "cmpl")
        if refused:
            return refused
        llm, rid, meta, detok = started
        base = {**meta, "object": "text_completion"}

        if req.stream:
            async def events():
                count = 0
                async for token, finish in llm.stream(rid):
                    count += token >= 0
                    delta = detok.add(token) if token >= 0 else ""
                    if delta or finish:
                        choice = {"index": 0, "text": delta, "logprobs": None,
                                  "finish_reason": finish or None}
                        yield sse({**base, "choices": [choice]})
                    if finish in failures:
                        yield sse(error_body(failure_message(finish, count), finish))
                if wants_usage(req):
                    yield sse({**base, "choices": [], "usage": usage(prompt_ids, count)})
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")

        text, finish, count = await collect(llm, rid, detok)
        if finish in failures:
            return error(failure_message(finish, count), finish)
        return {**base,
                "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}],
                "usage": usage(prompt_ids, count)}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        if not getattr(tokenizer, "chat_template", None):
            return error("this tokenizer has no chat template; use /v1/completions")
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        has_bos = tokenizer.bos_token is not None and text.startswith(tokenizer.bos_token)
        prompt_ids = tokenizer(text, add_special_tokens=not has_bos).input_ids
        started, refused = start(prompt_ids, req, "chatcmpl")
        if refused:
            return refused
        llm, rid, meta, detok = started

        if req.stream:
            base = {**meta, "object": "chat.completion.chunk"}

            async def events():
                yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                                "finish_reason": None}]})
                count = 0
                async for token, finish in llm.stream(rid):
                    count += token >= 0
                    delta = detok.add(token) if token >= 0 else ""
                    if delta:
                        yield sse({**base, "choices": [{"index": 0, "delta": {"content": delta},
                                                        "finish_reason": None}]})
                    if finish:
                        yield sse({**base, "choices": [{"index": 0, "delta": {},
                                                        "finish_reason": finish}]})
                    if finish in failures:
                        yield sse(error_body(failure_message(finish, count), finish))
                if wants_usage(req):
                    yield sse({**base, "choices": [], "usage": usage(prompt_ids, count)})
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")

        text, finish, count = await collect(llm, rid, detok)
        if finish in failures:
            return error(failure_message(finish, count), finish)
        return {**meta, "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": finish}],
                "usage": usage(prompt_ids, count)}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="converted .llmbin model")
    parser.add_argument("--tokenizer", required=True, help="HF repo id or local dir of the model's tokenizer")
    parser.add_argument("--model-name", default="tinyllama", help="name reported by /v1/models")
    parser.add_argument("--engine-dir", default=os.path.join(REPO_ROOT, "build", "python"),
                        help="directory holding the built llm_engine_py module")
    parser.add_argument("--max-tokens", type=int, default=1024, help="prompt + generated tokens per sequence")
    parser.add_argument("--max-seqs", type=int, default=8, help="sequences decoded at once")
    parser.add_argument("--kv-pages", type=int, default=0, help="KV pool size; 0 fits it to free VRAM")
    parser.add_argument("--max-waiting", type=int, default=64,
                        help="requests queued before new ones get HTTP 429; 0 is unlimited")
    parser.add_argument("--request-timeout", type=float, default=300,
                        help="seconds a request may wait plus run before HTTP 504; 0 is none")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
