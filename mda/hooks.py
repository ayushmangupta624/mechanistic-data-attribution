"""Forward hooks for caching QKVO activations (no LayerNorm; uses resid_pre)."""


class QKVOActivationCache:
    def __init__(self):
        self.X      = None   # [B, S, d_model]  residual stream pre-attention
        self.Q      = None   # [B, S, n_heads, d_head]
        self.K      = None   # [B, S, n_heads, d_head]
        self.V      = None   # [B, S, n_heads, d_head]
        self.Z      = None   # [B, S, n_heads, d_head]
        self.result = None   # [B, S, n_heads, d_model]

    def clear(self):
        self.X = self.Q = self.K = self.V = self.Z = self.result = None


def setup_qkvo_hooks(model, layer, cache):
    # No LayerNorm. We capture residual stream directly via hook_resid_pre
    def h_x(act, hook):      cache.X      = act; return act
    def h_q(act, hook):      cache.Q      = act; return act
    def h_k(act, hook):      cache.K      = act; return act
    def h_v(act, hook):      cache.V      = act; return act
    def h_z(act, hook):      cache.Z      = act; return act
    def h_result(act, hook): cache.result = act; return act

    hooks = [
        model.add_hook(f"blocks.{layer}.hook_resid_pre",      h_x,      dir="fwd"),
        model.add_hook(f"blocks.{layer}.attn.hook_q",         h_q,      dir="fwd"),
        model.add_hook(f"blocks.{layer}.attn.hook_k",         h_k,      dir="fwd"),
        model.add_hook(f"blocks.{layer}.attn.hook_v",         h_v,      dir="fwd"),
        model.add_hook(f"blocks.{layer}.attn.hook_z",         h_z,      dir="fwd"),
        model.add_hook(f"blocks.{layer}.attn.hook_result",    h_result, dir="fwd"),
    ]
    return hooks
