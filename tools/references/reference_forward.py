"""Torch reference for the FP16 forward pass -- generates the fixtures ctest checks against.

Usage:
    python reference_forward.py <out_dir>
"""

import copy
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

# binformat.py stays with the offline converters in tools/convert -- this
# script is a consumer of the format, not part of its definition.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "convert"))
import binformat  # noqa: E402
import quantize  # noqa: E402

SEED = 20240501

# Small enough that ctest stays fast and the whole reference fits in host RAM,
# large enough that every kernel's reduction crosses at least one warp boundary.
REF_CONFIG = binformat.ModelConfig(
    vocab_size=64,
    hidden_size=32,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=8,
    intermediate_size=64,
    max_seq_len=256,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
)

DECODE_STEPS = 200

# The ref model's K is 32 or 64, so the smallest legal group size covers every matmul.
REF_GROUP = 32


# --- fixture plumbing -------------------------------------------------------

def fp32(t) -> bytes:
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f4").tobytes()


def fp16(t) -> bytes:
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f2").tobytes()


def as_fp16(t: torch.Tensor) -> torch.Tensor:
    """Round through FP16 and back, matching what the GPU reads from VRAM."""
    return t.to(torch.float16).to(torch.float32)


class Fixture:
    """Collects named tensors and writes them as one .llmbin file."""

    def __init__(self, config=REF_CONFIG):
        self.writer = binformat.ModelWriter(config=config)

    def add_fp32(self, name: str, t: torch.Tensor) -> None:
        self.writer.add_tensor(name, binformat.DTYPE_FP32, tuple(t.shape), fp32(t))

    def add_fp16(self, name: str, t: torch.Tensor) -> None:
        self.writer.add_tensor(name, binformat.DTYPE_FP16, tuple(t.shape), fp16(t))

    def add_raw(self, name: str, dtype: int, shape, data: np.ndarray) -> None:
        self.writer.add_tensor(name, dtype, tuple(shape), np.ascontiguousarray(data).tobytes())

    def write(self, path: str) -> None:
        self.writer.write(path)


def rand(*shape) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32)


# --- reference math (the definition the kernels must match) ------------------

def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * scale * weight


def rope_angles(head_dim: int, positions: torch.Tensor, theta: float):
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) * 2.0 / head_dim))
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq  # [T, half]
    return torch.cos(angles), torch.sin(angles)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """x: [T, heads, head_dim]. Llama's rotate_half pairing: i with i + head_dim/2."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    cos, sin = rope_angles(head_dim, positions, theta)
    cos = cos.unsqueeze(1)  # [T, 1, half]
    sin = sin.unsqueeze(1)
    lo, hi = x[..., :half], x[..., half:]
    return torch.cat([lo * cos - hi * sin, hi * cos + lo * sin], dim=-1)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """q: [T, NH, D]; k/v: [T, NKV, D]. Returns (masked scores, probs, out)."""
    t, n_heads, head_dim = q.shape
    n_kv = k.shape[1]
    group = n_heads // n_kv
    scale = 1.0 / (head_dim ** 0.5)

    kv_index = torch.div(torch.arange(n_heads), group, rounding_mode="floor")
    k_exp = k[:, kv_index, :]  # [T, NH, D]
    v_exp = v[:, kv_index, :]

    scores = torch.einsum("ihd,jhd->hij", q, k_exp) * scale
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hij,jhd->ihd", probs, v_exp)
    return scores, probs, out


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


# --- per-kernel fixtures ----------------------------------------------------

def write_embedding(path: str) -> None:
    vocab, hidden, n_tokens = 61, 48, 9
    table = as_fp16(rand(vocab, hidden))
    ids = torch.randint(0, vocab, (n_tokens,))

    f = Fixture()
    f.add_fp16("table", table)
    f.add_fp32("ids", ids.to(torch.float32))
    f.add_fp32("expected", table[ids])
    f.write(path)


def write_rmsnorm(path: str) -> None:
    # hidden is deliberately not a multiple of the warp size: the reduction's
    # tail is where a hand-written warp-shuffle version goes wrong.
    rows, hidden = 5, 300
    x = rand(rows, hidden)
    weight = as_fp16(rand(hidden))

    f = Fixture()
    f.add_fp32("input", x)
    f.add_fp16("weight", weight)
    f.add_fp32("expected", rmsnorm(x, weight, REF_CONFIG.rms_norm_eps))
    f.write(path)


def write_rope(path: str) -> None:
    n_tokens, n_heads, n_kv, head_dim = 7, 4, 2, 16
    q = rand(n_tokens, n_heads, head_dim)
    k = rand(n_tokens, n_kv, head_dim)
    # Not simply 0..T-1, so a kernel that ignores the position array fails.
    positions = torch.tensor([0, 1, 2, 5, 8, 13, 21], dtype=torch.float32)

    f = Fixture()
    f.add_fp32("q", q)
    f.add_fp32("k", k)
    f.add_fp32("positions", positions)
    f.add_fp32("q_expected", rope(q, positions, REF_CONFIG.rope_theta))
    f.add_fp32("k_expected", rope(k, positions, REF_CONFIG.rope_theta))
    f.write(path)


def write_gemv(path: str) -> None:
    n_out, k_in = 130, 300
    weight = as_fp16(rand(n_out, k_in))
    x = rand(k_in)

    f = Fixture()
    f.add_fp16("weight", weight)
    f.add_fp32("input", x)
    f.add_fp32("expected", weight @ x)
    f.write(path)


def write_gemm(path: str) -> None:
    m_rows, n_out, k_in = 17, 130, 300
    weight = as_fp16(rand(n_out, k_in))
    x = rand(m_rows, k_in)

    f = Fixture()
    f.add_fp16("weight", weight)
    f.add_fp32("input", x)
    f.add_fp32("expected", x @ weight.t())
    f.write(path)


# (group size, K): G=32 spans several groups per warp step, G=128 is the production size.
QUANT_CASES = [(32, 320), (128, 384)]
QUANT_SEED = 7


def quantize_weight(f: Fixture, base: str, w: np.ndarray, bits: int, group: int) -> np.ndarray:
    """Adds `base`.weight/.scales(/.mins) in the .llmbin layout; returns dequant(W), what the kernel sees."""
    if bits == 8:
        q, scales = quantize.quantize_int8(w, group)
        f.add_raw(base + ".weight", binformat.DTYPE_INT8, q.shape, q)
        deq = quantize.dequantize_int8(q, scales)
    else:
        q, scales, mins = quantize.quantize_int4(w, group)
        f.add_raw(base + ".weight", binformat.DTYPE_INT4, w.shape, q)
        f.add_raw(base + ".mins", binformat.DTYPE_FP16, mins.shape, mins.astype("<f2"))
        deq = quantize.dequantize_int4(q, scales, mins)
    f.add_raw(base + ".scales", binformat.DTYPE_FP16, scales.shape, scales.astype("<f2"))
    return deq


def random_weight(rng, n_out: int, k_in: int) -> np.ndarray:
    w = rng.standard_normal((n_out, k_in)).astype(np.float16).astype(np.float32)
    w[3] = 0.0  # all-zero groups: scale 0 must give exactly 0
    return w


def write_quant_matmul(path: str, bits: int, m_rows: int, seed: int) -> None:
    """m_rows = 1 is the GEMV fixture (1-D input); otherwise the GEMM fixture."""
    rng = np.random.default_rng(seed)
    f = Fixture()
    for group, k_in in QUANT_CASES:
        p = f"g{group}."
        w = torch.from_numpy(quantize_weight(f, p.rstrip("."), random_weight(rng, 130, k_in), bits, group))
        if m_rows == 1:
            x = torch.from_numpy(rng.standard_normal(k_in).astype(np.float32))
            f.add_fp32(p + "input", x)
            f.add_fp32(p + "expected", w @ x)
        else:
            x = torch.from_numpy(rng.standard_normal((m_rows, k_in)).astype(np.float32))
            f.add_fp32(p + "input", x)
            f.add_fp32(p + "expected", x @ w.t())
    f.write(path)


def write_attention(path: str) -> None:
    n_tokens, n_heads, n_kv, head_dim = 7, 4, 2, 8
    q = rand(n_tokens, n_heads, head_dim)
    k = rand(n_tokens, n_kv, head_dim)
    v = rand(n_tokens, n_kv, head_dim)
    scores, probs, out = attention(q, k, v)

    f = Fixture()
    f.add_fp32("q", q)
    f.add_fp32("k", k)
    f.add_fp32("v", v)
    f.add_fp32("scores_expected", scores)
    f.add_fp32("probs_expected", probs)
    f.add_fp32("expected", out)
    f.write(path)


def write_elementwise(path: str) -> None:
    n = 1000
    gate, up = rand(n), rand(n)
    res_x, res_y = rand(n), rand(n)
    logits = rand(REF_CONFIG.vocab_size * 7)

    f = Fixture()
    f.add_fp32("gate", gate)
    f.add_fp32("up", up)
    f.add_fp32("swiglu_expected", swiglu(gate, up))
    f.add_fp32("res_x", res_x)
    f.add_fp32("res_y", res_y)
    f.add_fp32("res_expected", res_x + res_y)
    f.add_fp32("logits", logits)
    f.add_fp32("argmax_expected", torch.tensor([float(int(logits.argmax()))]))
    f.write(path)


# --- whole-model fixture ----------------------------------------------------

LINEAR_NAMES = {
    "wq": "attn.wq", "wk": "attn.wk", "wv": "attn.wv", "wo": "attn.wo",
    "gate_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj", "down_proj": "mlp.down_proj",
}


class RefModel:
    """A random Llama-shaped model held as FP16-rounded FP32 tensors.

    Tensor names match tools/convert/convert_weights.py, so the C++ side loads
    this fixture through exactly the same code path as a real checkpoint.
    """

    def __init__(self, config=REF_CONFIG):
        self.config = config
        c = config
        q_dim = c.num_heads * c.head_dim
        kv_dim = c.num_kv_heads * c.head_dim

        # Scaled down so a 2-layer random stack doesn't saturate into a
        # constant argmax -- the decode check is only meaningful if the
        # predicted token actually moves between steps.
        def w(*shape):
            return as_fp16(torch.randn(*shape, dtype=torch.float32) * 0.05)

        self.tok_embeddings = w(c.vocab_size, c.hidden_size)
        self.output_norm = as_fp16(torch.ones(c.hidden_size) + 0.05 * rand(c.hidden_size))
        self.lm_head = w(c.vocab_size, c.hidden_size)
        self.bits = 16  # 8 or 4 after quantized(): matmul weights are then written quantized

        self.layers = []
        for _ in range(c.num_layers):
            self.layers.append({
                "attn_norm": as_fp16(torch.ones(c.hidden_size) + 0.05 * rand(c.hidden_size)),
                "wq": w(q_dim, c.hidden_size),
                "wk": w(kv_dim, c.hidden_size),
                "wv": w(kv_dim, c.hidden_size),
                "wo": w(c.hidden_size, q_dim),
                "ffn_norm": as_fp16(torch.ones(c.hidden_size) + 0.05 * rand(c.hidden_size)),
                "gate_proj": w(c.intermediate_size, c.hidden_size),
                "up_proj": w(c.intermediate_size, c.hidden_size),
                "down_proj": w(c.hidden_size, c.intermediate_size),
            })

    def write(self, path: str) -> None:
        f = Fixture(self.config)

        def add_linear(base: str, t: torch.Tensor) -> None:
            if self.bits == 16:
                return f.add_fp16(base + ".weight", t)
            # Quantizes the original weight, so the file holds exactly what quantized() dequantized.
            quantize_weight(f, base, self.source[base], self.bits, self.group)

        f.add_fp16("tok_embeddings.weight", self.tok_embeddings)
        f.add_fp16("output_norm.weight", self.output_norm)
        add_linear("lm_head", self.lm_head)
        for i, layer in enumerate(self.layers):
            p = f"layers.{i}."
            f.add_fp16(p + "attn_norm.weight", layer["attn_norm"])
            f.add_fp16(p + "ffn_norm.weight", layer["ffn_norm"])
            for key, name in LINEAR_NAMES.items():
                add_linear(p + name, layer[key])
        f.write(path)

    def quantized(self, bits: int, group: int) -> "RefModel":
        """A copy whose matmul weights are dequant(quant(W)): what the INT8/INT4 engine computes with."""
        out = copy.deepcopy(self)
        out.bits, out.group, out.source = bits, group, {}

        def conv(base: str, t: torch.Tensor) -> torch.Tensor:
            out.source[base] = t.numpy()
            return torch.from_numpy(quantize_weight(Fixture(), base, t.numpy(), bits, group))

        out.lm_head = conv("lm_head", self.lm_head)
        for i, layer in enumerate(out.layers):
            for key, name in LINEAR_NAMES.items():
                layer[key] = conv(f"layers.{i}.{name}", layer[key])
        return out

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward, no cache. Returns logits [T, vocab]."""
        c = self.config
        t = ids.shape[0]
        positions = torch.arange(t, dtype=torch.float32)
        x = self.tok_embeddings[ids]

        for layer in self.layers:
            h = rmsnorm(x, layer["attn_norm"], c.rms_norm_eps)
            q = (h @ layer["wq"].t()).view(t, c.num_heads, c.head_dim)
            k = (h @ layer["wk"].t()).view(t, c.num_kv_heads, c.head_dim)
            v = (h @ layer["wv"].t()).view(t, c.num_kv_heads, c.head_dim)
            q = rope(q, positions, c.rope_theta)
            k = rope(k, positions, c.rope_theta)
            _, _, attn_out = attention(q, k, v)
            x = x + (attn_out.reshape(t, c.num_heads * c.head_dim) @ layer["wo"].t())

            h = rmsnorm(x, layer["ffn_norm"], c.rms_norm_eps)
            gate = h @ layer["gate_proj"].t()
            up = h @ layer["up_proj"].t()
            x = x + (swiglu(gate, up) @ layer["down_proj"].t())

        return rmsnorm(x, self.output_norm, c.rms_norm_eps) @ self.lm_head.t()

    def greedy_decode(self, prompt: torch.Tensor, steps: int) -> torch.Tensor:
        """Argmax decode with no cache: every step re-runs the whole sequence."""
        ids = prompt.clone()
        generated = []
        for _ in range(steps):
            logits = self.forward(ids)
            next_id = int(logits[-1].argmax())
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([next_id], dtype=ids.dtype)])
        return torch.tensor(generated, dtype=torch.float32)


def write_forward(model_path: str, ref_path: str, model: RefModel = None, prompt=None):
    model = model or RefModel()
    model.write(model_path)

    prompt = torch.randint(0, REF_CONFIG.vocab_size, (5,)) if prompt is None else prompt
    logits = model.forward(prompt)

    f = Fixture()
    f.add_fp32("prompt_ids", prompt.to(torch.float32))
    f.add_fp32("prefill_logits", logits)
    f.add_fp32("generated_ids", model.greedy_decode(prompt, DECODE_STEPS))
    f.write(ref_path)
    return model, prompt


# --- entry point ------------------------------------------------------------

FIXTURES = {
    "ref_embedding.llmbin": write_embedding,
    "ref_rmsnorm.llmbin": write_rmsnorm,
    "ref_rope.llmbin": write_rope,
    "ref_gemv.llmbin": write_gemv,
    "ref_gemm.llmbin": write_gemm,
    "ref_attention.llmbin": write_attention,
    "ref_elementwise.llmbin": write_elementwise,
    # Own numpy seed, so adding these does not shift the torch stream above.
    "ref_gemv_int8.llmbin": lambda path: write_quant_matmul(path, 8, 1, QUANT_SEED),
    "ref_gemm_int8.llmbin": lambda path: write_quant_matmul(path, 8, 17, QUANT_SEED + 1),
    "ref_gemv_int4.llmbin": lambda path: write_quant_matmul(path, 4, 1, QUANT_SEED + 2),
    "ref_gemm_int4.llmbin": lambda path: write_quant_matmul(path, 4, 17, QUANT_SEED + 3),
}


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <out_dir>", file=sys.stderr)
        sys.exit(2)
    out_dir = sys.argv[1].rstrip("/")

    # One seed for the whole run: every fixture is reproducible from this file
    # alone, and regenerating is a no-op unless the reference math changed.
    torch.manual_seed(SEED)

    for name, fn in FIXTURES.items():
        fn(f"{out_dir}/{name}")
        print(f"wrote {out_dir}/{name}")

    model, prompt = write_forward(f"{out_dir}/ref_model.llmbin", f"{out_dir}/ref_forward.llmbin")
    print(f"wrote {out_dir}/ref_model.llmbin, {out_dir}/ref_forward.llmbin")

    # Same model and prompt, quantized weights: no extra randomness, so the FP16 fixtures above are unchanged.
    for bits in (8, 4):
        model_path, ref_path = f"{out_dir}/ref_model_int{bits}.llmbin", f"{out_dir}/ref_forward_int{bits}.llmbin"
        write_forward(model_path, ref_path, model.quantized(bits, REF_GROUP), prompt)
        print(f"wrote {model_path}, {ref_path}")


if __name__ == "__main__":
    main()
