# Dynamic Threaded Neural Network (DTNN)

A PyTorch architecture where **inter-layer connection threads** are formed dynamically per input sample rather than statically hardwired.

---

## 1. Architectural Concept

In conventional feedforward or residual networks, the inter-layer topology is fixed:
- Layer $i$ connects strictly to Layer $i+1$, or statically to all layers (like DenseNet).

In **Dynamic Threaded Neural Networks**:
1. **Latent Connection Threads**: For any source layer $i$ and downstream target layer $j > i$, a latent synaptic thread exists:
   $$\mathcal{T}_{i \to j}(\mathbf{h}_i) = \mathbf{W}_{i \to j}(\mathbf{h}_i)$$
2. **Dynamic Gated Routing**: A gating network at layer $i$ inspects current activations and dynamically computes continuous or sparse routing coefficients $\alpha_{i \to j}(\mathbf{x})$:
   $$\boldsymbol{\alpha}_i = \text{Route}(\mathbf{h}_i)$$
3. **Dynamic Aggregation at Target Layer $j$**:
   $$\mathbf{h}_j = \text{LayerTransform}\left(\sum_{i < j} \alpha_{i \to j}(\mathbf{x}) \cdot \mathcal{T}_{i \to j}(\mathbf{h}_i)\right)$$
4. **Adaptive Topological Routing**:
   - Simple inputs can use direct dynamic shortcut threads from early layers directly to the output.
   - Complex non-linear inputs dynamically route activations through multiple intermediate specialized layers.

---

## 2. Directory Structure

```
dynamic_threaded_nn/
├── __init__.py                       # Package exports
├── layers.py                         # DynamicThread & DynamicThreadRouter modules
├── model.py                          # DynamicThreadedNetwork (full DAG architecture)
├── visualizer.py                     # ASCII graph & Interactive SVG/HTML visualizer
├── train_demo.py                     # Training demo with synthetic spiral dataset
└── dynamic_threads_visualization.html # Generated interactive visualizer
```

---

## 3. How to Run the Demo

Run the training demo using your virtual environment:

```bash
.venv\Scripts\python.exe -m dynamic_threaded_nn.train_demo
```

### Supported Routing Modes:
- `"sparse_topk"`: Dynamically activates the top $k$ strongest threads per layer per sample.
- `"soft"`: Continuous softmax weighting across all forward threads with sparsity filtering.
- `"gumbel"`: Gumbel-Softmax discrete stochastic sampling with straight-through gradient estimator.

---

## 4. Interactive Visualization

After running the demo, open `dynamic_threaded_nn/dynamic_threads_visualization.html` in any web browser to interactively view the dynamic threads pulsing and routing between layers for each sample.
