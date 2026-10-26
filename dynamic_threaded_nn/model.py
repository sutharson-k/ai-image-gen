import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dynamic_threaded_nn.layers import DynamicThread, DynamicThreadRouter

class DynamicThreadedNetwork(nn.Module):
    """
    Neural Network with Dynamic Threads connecting each layer.
    
    Architecture:
      - Dynamic Directed Acyclic Graph (DAG) topology formed per sample.
      - Each layer dynamically determines which downstream layers to connect to (via DynamicThreadRouter).
      - Incoming dynamic threads are aggregated at each layer using dynamic gating weights.
      - Dynamic routing matrix can be retrieved for visualization and architectural inspection.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        routing_mode: str = "sparse_topk",
        top_k: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_layers = len(hidden_dims)
        self.routing_mode = routing_mode
        self.top_k = top_k
        self.temperature = temperature
        
        # All layer dimensions including input and output
        # Index 0: Input layer
        # Index 1..N: Hidden layers
        # Index N+1: Output layer
        self.all_dims = [input_dim] + hidden_dims + [output_dim]
        self.total_nodes = len(self.all_dims)
        
        # Internal layer transforms (computing local layer features)
        self.layer_transforms = nn.ModuleList()
        for idx in range(1, self.total_nodes):
            dim = self.all_dims[idx]
            self.layer_transforms.append(
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )

        # Routers for each layer i to all downstream layers j (where j > i)
        # Note: final output layer does not need a router
        self.routers = nn.ModuleList()
        for i in range(self.total_nodes - 1):
            num_targets = self.total_nodes - 1 - i  # from i+1 to total_nodes - 1
            router = DynamicThreadRouter(
                current_dim=self.all_dims[i],
                num_target_layers=num_targets,
                routing_mode=routing_mode,
                top_k=min(top_k, num_targets),
                temperature=temperature
            )
            self.routers.append(router)

        # Thread projectors between all valid (i, j) pairs where i < j
        # Keyed as ModuleDict with string key f"thread_{i}_to_{j}"
        self.threads = nn.ModuleDict()
        for i in range(self.total_nodes - 1):
            for j in range(i + 1, self.total_nodes):
                key = f"thread_{i}_to_{j}"
                self.threads[key] = DynamicThread(
                    in_features=self.all_dims[i],
                    out_features=self.all_dims[j],
                    dropout=dropout
                )

        # Output classifier / final prediction layer
        self.classifier = nn.Linear(output_dim, output_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        return_routing_graph: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with dynamic thread propagation.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            return_routing_graph: Whether to return adjacency & weight matrices of dynamic threads
            
        Returns:
            output: Prediction tensor [batch_size, output_dim]
            graph_info: Optional dictionary containing thread weights & active thread masks
        """
        batch_size = x.size(0)
        device = x.device
        
        # Store node activations: node_activations[i] has shape [batch_size, all_dims[i]]
        node_activations: List[Optional[torch.Tensor]] = [None] * self.total_nodes
        node_activations[0] = x
        
        # Accumulators for incoming dynamic thread signals for each layer j > 0
        incoming_signals: List[List[torch.Tensor]] = [[] for _ in range(self.total_nodes)]
        
        # Matrices to store graph connectivity: [batch_size, total_nodes, total_nodes]
        thread_weights_matrix = torch.zeros(
            (batch_size, self.total_nodes, self.total_nodes), 
            device=device
        )
        active_mask_matrix = torch.zeros(
            (batch_size, self.total_nodes, self.total_nodes), 
            device=device
        )
        
        # Propagate topologically from layer 0 to output layer
        for i in range(self.total_nodes - 1):
            # Compute / refine current node activation
            if i > 0:
                # Aggregate incoming threads from all prior layers k < i
                if len(incoming_signals[i]) == 0:
                    # Fallback if no thread reached this layer: default to zero
                    aggregated = torch.zeros(batch_size, self.all_dims[i], device=device)
                else:
                    # Sum of incoming dynamic thread signals
                    aggregated = torch.stack(incoming_signals[i], dim=0).sum(dim=0)
                
                # Apply layer transformation
                transformed = self.layer_transforms[i - 1](aggregated)
                node_activations[i] = transformed
            
            curr_act = node_activations[i]
            
            # Dynamic Router determines which downstream layers receive threads from layer i
            weights, mask = self.routers[i](curr_act, training=self.training)
            
            # Dispatch dynamic threads to downstream layers j in [i+1, total_nodes - 1]
            for target_offset in range(weights.size(-1)):
                j = i + 1 + target_offset
                w_ij = weights[:, target_offset:target_offset+1] # [batch_size, 1]
                m_ij = mask[:, target_offset]                    # [batch_size]
                
                thread_weights_matrix[:, i, j] = w_ij.squeeze(-1)
                active_mask_matrix[:, i, j] = m_ij
                
                # Propagate through thread projector
                key = f"thread_{i}_to_{j}"
                thread_output = self.threads[key](curr_act, w_ij)
                incoming_signals[j].append(thread_output)

        # Output Layer (total_nodes - 1)
        out_idx = self.total_nodes - 1
        if len(incoming_signals[out_idx]) == 0:
            out_aggregated = torch.zeros(batch_size, self.all_dims[out_idx], device=device)
        else:
            out_aggregated = torch.stack(incoming_signals[out_idx], dim=0).sum(dim=0)
            
        final_features = self.layer_transforms[out_idx - 1](out_aggregated)
        logits = self.classifier(final_features)
        
        if return_routing_graph:
            graph_info = {
                "thread_weights": thread_weights_matrix,  # [B, N, N]
                "active_mask": active_mask_matrix,        # [B, N, N]
                "layer_names": [f"Layer {idx} ({self.all_dims[idx]}d)" for idx in range(self.total_nodes)]
            }
            return logits, graph_info
            
        return logits, None
