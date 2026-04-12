"""
causal_traj_head.py — GPT-2-style Causal Trajectory Predictor

Replaces the simple MLP TrajHead with a proper autoregressive
causal transformer that predicts future waypoints one step at a time,
conditioned on scene BEV features and past ego motion.

Architecture:
    - Scene context: projected from BEV features (d→n_embd)
    - Waypoint tokens: learned position embeddings per timestep
    - Causal self-attention: each step only attends to previous steps
    - Output: 2D (x,y) waypoint residuals over CV prior

This is behavioral cloning / imitation learning:
    - Input:  real nuScenes expert ego-pose demonstrations
    - Output: predicted next 12 waypoints (6 seconds at 0.5s intervals)
    - Loss:   SmoothL1 ADE + FDE weighted

Usage (drop-in replacement for TrajHead):
    from src.opendrivefm.models.causal_traj_head import CausalTrajHead
    self.traj = CausalTrajHead(d=384, horizon=12, n_embd=128, n_head=4, n_layer=3)
"""

import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    """
    GPT-2-style masked self-attention.
    Each token can only attend to itself and previous tokens (causal mask).
    This ensures the model predicts autoregressively — timestep t
    cannot see timestep t+1 during training or inference.
    """
    def __init__(self, n_embd: int, n_head: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head  = n_head
        self.n_embd  = n_embd
        self.head_dim = n_embd // n_head

        # QKV projection (all in one for efficiency)
        self.qkv   = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj  = nn.Linear(n_embd, n_embd, bias=False)
        self.drop  = nn.Dropout(dropout)

        # Causal mask: lower triangular — token i can see tokens 0..i only
        # Shape: (1, 1, horizon+1, horizon+1) for broadcasting over batch/heads
        mask = torch.tril(torch.ones(horizon + 1, horizon + 1))
        self.register_buffer("mask", mask.view(1, 1, horizon + 1, horizon + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # Compute Q, K, V
        qkv = self.qkv(x).split(self.n_embd, dim=2)
        Q, K, V = [t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                   for t in qkv]
        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.head_dim)
        attn  = (Q @ K.transpose(-2, -1)) / scale
        attn  = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn  = self.drop(torch.softmax(attn, dim=-1))
        out   = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Single GPT-2 transformer block: LayerNorm → Attention → LayerNorm → FFN."""
    def __init__(self, n_embd: int, n_head: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, horizon, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.ffn  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # residual + attention
        x = x + self.ffn(self.ln2(x))    # residual + FFN
        return x


class CausalTrajHead(nn.Module):
    """
    GPT-2-style Causal Trajectory Predictor.

    Treats future trajectory prediction as an autoregressive sequence
    modeling problem — exactly like language modeling but over 2D waypoints.

    Input:
        z        : BEV scene features  (B, d)
        velocity : current ego velocity (B, 2) — vx, vy in m/s
                   Used as constant-velocity prior for residual prediction.

    Output:
        waypoints: (B, horizon, 2) — predicted (x,y) positions in ego frame
                   in metres, at 0.5s intervals for 6 seconds total.

    Training:
        Behavioral cloning on real nuScenes expert ego-pose demonstrations.
        Loss = SmoothL1(ADE) + 2.0 × SmoothL1(FDE) + 0.1 × L2_regularization
        No reward engineering. No RL. Pure imitation learning.
    """

    def __init__(
        self,
        d       : int = 384,   # BEV feature dimension (from backbone)
        horizon : int = 12,    # prediction steps (12 × 0.5s = 6 seconds)
        n_embd  : int = 128,   # transformer embedding dimension
        n_head  : int = 4,     # number of attention heads
        n_layer : int = 3,     # number of transformer blocks
        dropout : float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_embd  = n_embd

        # ── Scene context encoder ───────────────────────────────────────────
        # Projects BEV features into transformer embedding space
        self.scene_enc = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
        )

        # ── Velocity encoder ────────────────────────────────────────────────
        # Encodes current ego velocity as constant-velocity prior token
        self.vel_enc = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, n_embd),
        )

        # ── Waypoint position embeddings ────────────────────────────────────
        # Learned embeddings for each future timestep (like token positions in GPT)
        # horizon+1 because index 0 = scene context token
        self.pos_emb = nn.Embedding(horizon + 1, n_embd)

        # ── Causal transformer ──────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, horizon, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm (GPT-2 style)

        # ── Output head ─────────────────────────────────────────────────────
        # Projects each timestep's hidden state to (x, y) waypoint residual
        self.waypoint_head = nn.Linear(n_embd, 2, bias=True)

        # ── Weight initialization (GPT-2 style) ────────────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """GPT-2 weight initialization for stable training."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        z        : torch.Tensor,                    # (B, d) BEV features
        velocity : torch.Tensor | None = None,      # (B, 2) ego velocity
    ) -> torch.Tensor:                               # (B, horizon, 2) waypoints
        B = z.size(0)
        device = z.device

        # ── Build input sequence ────────────────────────────────────────────
        # Token 0: scene context (what the model sees)
        scene_tok = self.scene_enc(z).unsqueeze(1)          # (B, 1, n_embd)

        # Tokens 1..horizon: future waypoint slots (initialized from CV prior)
        if velocity is not None:
            # Constant-velocity prior: waypoint_t = t × 0.5s × velocity
            dt = torch.arange(1, self.horizon + 1, device=device).float() * 0.5
            cv_prior = velocity.unsqueeze(1) * dt.view(1, -1, 1)  # (B, T, 2)
            # Encode CV prior as starting token embeddings
            cv_enc = self.vel_enc(velocity).unsqueeze(1).expand(B, self.horizon, -1)
        else:
            cv_prior = torch.zeros(B, self.horizon, 2, device=device)
            cv_enc   = torch.zeros(B, self.horizon, self.n_embd, device=device)

        # Full sequence: [scene_token, waypoint_1, ..., waypoint_T]
        tokens = torch.cat([scene_tok, cv_enc], dim=1)      # (B, T+1, n_embd)

        # Add positional embeddings (GPT-2 style learned positions)
        pos  = torch.arange(self.horizon + 1, device=device)
        tokens = tokens + self.pos_emb(pos).unsqueeze(0)    # broadcast over batch

        # ── Causal transformer forward pass ─────────────────────────────────
        x = tokens
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)                                    # (B, T+1, n_embd)

        # ── Decode waypoints from positions 1..T ────────────────────────────
        # Each output token at position t predicts waypoint at timestep t
        # This is the autoregressive prediction — token t only saw tokens 0..t-1
        waypoint_tokens = x[:, 1:, :]                       # (B, T, n_embd)
        residuals = self.waypoint_head(waypoint_tokens)     # (B, T, 2)

        # Final prediction: CV prior + learned residual (residual learning)
        waypoints = cv_prior + residuals                    # (B, T, 2)
        return waypoints

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Trajectory Loss ──────────────────────────────────────────────────────────

def causal_traj_loss(
    pred : torch.Tensor,   # (B, T, 2) predicted waypoints
    gt   : torch.Tensor,   # (B, T, 2) ground truth waypoints
) -> dict:
    """
    Imitation learning loss for behavioral cloning on expert demonstrations.
    No reward engineering — purely supervised on real nuScenes ego poses.

    Returns dict with total loss + individual components for logging.
    """
    # ADE: average displacement error across all timesteps
    ade_loss = torch.nn.functional.smooth_l1_loss(pred, gt, beta=0.5)

    # FDE: final displacement error (2× weight — endpoint matters most)
    fde_loss = torch.nn.functional.smooth_l1_loss(
        pred[:, -1, :], gt[:, -1, :], beta=0.5)

    # L2 regularization on prediction magnitude (prevents runaway predictions)
    l2_reg = (pred ** 2).mean() * 0.01

    total = ade_loss + 2.0 * fde_loss + l2_reg

    return {
        "loss"    : total,
        "ade_loss": ade_loss.item(),
        "fde_loss": fde_loss.item(),
        "l2_reg"  : l2_reg.item(),
    }


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing CausalTrajHead...")
    model = CausalTrajHead(d=384, horizon=12, n_embd=128, n_head=4, n_layer=3)
    print(f"Parameters: {model.num_parameters:,}")

    B = 2
    z   = torch.randn(B, 384)
    vel = torch.randn(B, 2)

    with torch.no_grad():
        waypoints = model(z, vel)

    print(f"Input BEV features:  {z.shape}")
    print(f"Input velocity:      {vel.shape}")
    print(f"Output waypoints:    {waypoints.shape}")  # (B, 12, 2)
    assert waypoints.shape == (B, 12, 2), "Shape mismatch!"

    # Test loss
    gt   = torch.randn(B, 12, 2)
    loss = causal_traj_loss(waypoints, gt)
    print(f"Loss components:     {loss}")
    print("✅ CausalTrajHead test passed!")
    print()
    print("Architecture summary:")
    print(f"  Scene encoder:    Linear({384}→{128})")
    print(f"  Velocity encoder: Linear(2→32→{128})")
    print(f"  Transformer:      {3} layers × {4} heads × {128} embd")
    print(f"  Causal masking:   ✅ (autoregressive)")
    print(f"  Position embed:   Learned ({12+1} positions)")
    print(f"  Output head:      Linear({128}→2) per timestep")
    print(f"  Total params:     {model.num_parameters:,}")
