import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class RMSNorm(nn.Module):
    """Simple RMSNorm over last dim."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class SDPAAttention(nn.Module):
    """
    Cross-attention using torch SDPA, supports additive bias via attn_bias.

    Inputs:
      x:       (B, Lq, Dq)
      context: (B, Lk, Dk)

    Optional:
      attn_bias:
        - float additive bias to logits, shape broadcastable to (B, H, Lq, Lk)
          e.g. (B, 1, Lq, Lk) or (B, H, Lq, Lk)
        - OR bool mask (converted to -inf/0)
    """
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout_p: float = 0.0,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = n_heads * head_dim
        self.dropout_p = dropout_p

        self.q_proj = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, self.inner_dim, bias=False)

        if use_rmsnorm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.out_proj = nn.Linear(self.inner_dim, query_dim, bias=False)
        self.out_drop = nn.Dropout(dropout_p) if dropout_p and dropout_p > 1e-6 else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        # match your original Attention.init_weights() style
        # q_proj
        std = 1.0 / math.sqrt(self.query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)

        # k_proj / v_proj
        std = 1.0 / math.sqrt(self.context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        # out_proj
        std = 1.0 / math.sqrt(self.inner_dim)
        torch.nn.init.trunc_normal_(self.out_proj.weight, std=std, a=-3 * std, b=3 * std)

        # norms
        for layer in (self.q_norm, self.k_norm):
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        B, Lq, _ = x.shape
        _, Lk, _ = context.shape

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.n_heads, d=self.head_dim)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.n_heads, d=self.head_dim)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.n_heads, d=self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if attn_bias is not None:
            if attn_bias.dtype == torch.bool:
                attn_bias = attn_bias.to(q.device)
                attn_bias = torch.where(
                    attn_bias,
                    torch.zeros_like(attn_bias, dtype=q.dtype),
                    torch.full_like(attn_bias, float("-inf"), dtype=q.dtype),
                )
            else:
                attn_bias = attn_bias.to(device=q.device, dtype=q.dtype)

            if attn_bias.dim() == 3:          # (B, Lq, Lk)
                attn_bias = attn_bias.unsqueeze(1)  # (B, 1, Lq, Lk)

        scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(self.head_dim))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
            scale=scale,
        )

        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.out_drop(self.out_proj(out))
        return out


class MultiViewCrossAttentionSDPA(nn.Module):
    """
    x:       (B, V*Lq, D)
    context: (B, V*Lk, Dc)
    attn_bias (optional):
      - (B, V, Lq, Lk)
      - or (B, Lq, Lk)  -> broadcast to all V
    """
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        tokens_per_view_context: int = 1280,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tokens_per_view_context = tokens_per_view_context
        self.core = SDPAAttention(
            query_dim=query_dim,
            context_dim=context_dim,
            n_heads=num_heads,
            head_dim=head_dim,
            dropout_p=dropout,
            use_rmsnorm=True,
        )

    def init_weights(self) -> None:
        # just forward to core for consistency
        self.core.init_weights()

    def forward(
        self,
        x: torch.Tensor,                 # (B, V*Lq, D)
        context: torch.Tensor,           # (B, V*Lk, Dc)
        *,
        attn_bias: torch.Tensor | None = None,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        B, Ltot, D = x.shape
        assert context is not None

        V = context.shape[1] // self.tokens_per_view_context
        assert V >= 1 and context.shape[1] % self.tokens_per_view_context == 0, \
            f"context.shape[1]={context.shape[1]} not divisible by tokens_per_view_context={self.tokens_per_view_context}"

        assert Ltot % V == 0, f"x tokens {Ltot} not divisible by V={V}"
        Lq = Ltot // V
        Lk = self.tokens_per_view_context

        x_vb = rearrange(x, "b (v l) d -> (v b) l d", v=V, l=Lq)
        ctx_vb = rearrange(context, "b (v m) d -> (v b) m d", v=V, m=Lk)

        bias_vb = None
        if attn_bias is not None:
            if attn_bias.dim() == 4 and attn_bias.shape[:2] == (B, V):
                bias_vb = rearrange(attn_bias, "b v lq lk -> (v b) 1 lq lk")
            elif attn_bias.dim() == 3 and attn_bias.shape[0] == B:
                # (B,Lq,Lk) -> repeat over V to match (VB,1,Lq,Lk)
                bias_vb = attn_bias.unsqueeze(1).repeat_interleave(V, 0)
            else:
                raise ValueError(
                    f"Unsupported attn_bias shape {tuple(attn_bias.shape)}. "
                    f"Use (B,V,Lq,Lk) or (B,Lq,Lk)."
                )
        out_vb = self.core(x_vb, ctx_vb, attn_bias=bias_vb, softmax_scale=softmax_scale)
        out = rearrange(out_vb, "(v b) l d -> b (v l) d", v=V, b=B)
        return out
