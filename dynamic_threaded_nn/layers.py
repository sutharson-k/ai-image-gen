import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

class DynamicThread(nn.Module):
    """
    Represents a dynamic directed synaptic thread between source layer i and target layer j.
    Applies linear projection, normalization, and optional activation to align feature dimensions.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if in_features != out_features:
            self.projector = nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.LayerNorm(out_features),
                nn.Dropout(dropout)
            )
        else:
            self.projector = nn.Identity()

    def forward(self, x: torch.Tensor, thread_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Source activations [batch_size, in_features]
            thread_weight: Dynamic scalar routing weight per sample [batch_size, 1]
        Returns:
            Weighted projected tensor [batch_size, out_features]
        """
        proj = self.projector(x)
        return proj * thread_weight


class DynamicThreadRouter(nn.Module):
    """
    Computes dynamic routing weights (threads) across layers based on the current layer representation.
    Supports soft gating, Gumbel-Softmax sparse gating, and Top-K thread routing.
    """
    def __init__(
        self, 
        current_dim: int, 
        num_target_layers: int, 
        routing_mode: str = "sparse_topk", 
        top_k: int = 2,
        temperature: float = 1.0
    ):
        super().__init__()
        self.num_target_layers = num_target_layers
        self.routing_mode = routing_mode
        self.top_k = min(top_k, num_target_layers)
        self.temperature = temperature
        
        # Gating network that decides connection strength to downstream layers
        self.gate_net = nn.Sequential(
            nn.Linear(current_dim, max(32, current_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(32, current_dim // 2), num_target_layers)
        )

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            thread_weights: [batch_size, num_target_layers] normalized thread routing weights
            thread_mask: [batch_size, num_target_layers] binary/sparse mask of active threads
        """
        logits = self.gate_net(x) / self.temperature # [B, num_targets]
        
        if self.routing_mode == "soft":
            weights = F.softmax(logits, dim=-1)
            mask = (weights > (1.0 / self.num_target_layers * 0.5)).float()
            return weights, mask

        elif self.routing_mode == "sparse_topk":
            # Select top-k downstream destination threads per sample
            topk_vals, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
            topk_probs = F.softmax(topk_vals, dim=-1)
            
            # Scatter into full target tensor
            weights = torch.zeros_like(logits)
            weights.scatter_(-1, topk_indices, topk_probs)
            
            mask = torch.zeros_like(logits)
            mask.scatter_(-1, topk_indices, 1.0)
            return weights, mask

        elif self.routing_mode == "gumbel":
            if training:
                weights = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
                mask = weights.detach().clone()
            else:
                weights = F.softmax(logits, dim=-1)
                mask = (weights == weights.max(dim=-1, keepdim=True)[0]).float()
            return weights, mask

        else:
            raise ValueError(f"Unknown routing_mode: {self.routing_mode}")
