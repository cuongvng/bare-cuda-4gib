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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AsyncLLM:
    """Bridges the engine's worker thread to asyncio: one thread polls, one queue per request."""

    def __init__(self, engine):
        self.engine = engine
        self.loop = asyncio.get_running_loop()
        self.queues = {}
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

    def submit(self, prompt_ids, max_new_tokens, stop_ids, **sampling):
        """Queues a request and returns its id; raises ValueError if the engine refuses it."""
        rid = next(self.ids)
        self.queues[rid] = asyncio.Queue()  # registered first, so no event can arrive unrouted
        error = self.engine.submit(rid, prompt_ids, max_new_tokens, stop_ids, **sampling)
        if error:
            del self.queues[rid]
            raise ValueError(error)
        return rid

    async def stream(self, rid):
        """Yields (token, finish) until the request ends; cancels it if the caller stops early."""
        finished = False
        try:
            while True:
                _, token, _, finish = await self.queues[rid].get()
                finished = bool(finish)
                yield token, finish
                if finished:
                    return
        finally:
            self.queues.pop(rid, None)
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
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
    from transformers import AutoTokenizer

    sys.path.insert(0, args.engine_dir)
    import llm_engine_py

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    state = {}

    @asynccontextmanager
    async def lifespan(_app):
        engine = llm_engine_py.Engine(args.model, max_tokens=args.max_tokens,
                                      max_seqs=args.max_seqs, kv_pages=args.kv_pages)
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

    def error(message, status=400):
        return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}},
                            status_code=status)

    def sse(payload):
        return f"data: {json.dumps(payload)}\n\n"

    def sampling(req):
        """Engine sampling kwargs from a request, with OpenAI's defaults for what it leaves out."""
        return {"temperature": 1.0 if req.temperature is None else req.temperature,
                "top_p": 1.0 if req.top_p is None else req.top_p,
                "top_k": req.top_k or 0,
                "seed": -1 if req.seed is None else req.seed}

    def start(prompt_ids, req, kind):
        """Submits a request: (llm, id, response metadata, detokenizer), or an error response."""
        llm = state["llm"]
        budget = req.max_tokens if req.max_tokens is not None else llm.engine.max_tokens
        try:
            rid = llm.submit(prompt_ids, budget, stop_ids, **sampling(req))
        except ValueError as e:
            return None, error(str(e))
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
        return {"status": "ok", "max_seqs": engine.max_seqs, "kv_pages": engine.kv_pages,
                "max_tokens": engine.max_tokens}

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
                async for token, finish in llm.stream(rid):
                    delta = detok.add(token) if token >= 0 else ""
                    if delta or finish:
                        choice = {"index": 0, "text": delta, "logprobs": None,
                                  "finish_reason": finish or None}
                        yield sse({**base, "choices": [choice]})
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")

        text, finish, count = await collect(llm, rid, detok)
        return {**base,
                "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": count,
                          "total_tokens": len(prompt_ids) + count}}

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
                async for token, finish in llm.stream(rid):
                    delta = detok.add(token) if token >= 0 else ""
                    if delta:
                        yield sse({**base, "choices": [{"index": 0, "delta": {"content": delta},
                                                        "finish_reason": None}]})
                    if finish:
                        yield sse({**base, "choices": [{"index": 0, "delta": {},
                                                        "finish_reason": finish}]})
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")

        text, finish, count = await collect(llm, rid, detok)
        return {**meta, "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": count,
                          "total_tokens": len(prompt_ids) + count}}

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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
