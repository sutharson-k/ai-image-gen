import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from dynamic_threaded_nn.model import DynamicThreadedNetwork
from dynamic_threaded_nn.visualizer import render_ascii_threads, export_interactive_html

def generate_spiral_dataset(n_samples_per_class: int = 400, n_classes: int = 3):
    """Generates a 2D multi-class spiral dataset for non-linear classification."""
    X = []
    y = []
    for j in range(n_classes):
        r = np.linspace(0.0, 1.0, n_samples_per_class)
        t = np.linspace(j * 4.0, (j + 1) * 4.0, n_samples_per_class) + np.random.randn(n_samples_per_class) * 0.2
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_samples_per_class, j))
    
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return torch.tensor(X[indices]), torch.tensor(y[indices])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Dynamic Threaded Neural Network on: {device}")
    
    # 1. Dataset
    X, y = generate_spiral_dataset(n_samples_per_class=300, n_classes=3)
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size].to(device), y[:train_size].to(device)
    X_test, y_test = X[train_size:].to(device), y[train_size:].to(device)
    
    # 2. Build Model
    # Input (2D) -> Layer 1 (32D) -> Layer 2 (32D) -> Layer 3 (16D) -> Output (3D)
    model = DynamicThreadedNetwork(
        input_dim=2,
        hidden_dims=[32, 32, 16],
        output_dim=3,
        routing_mode="sparse_topk",
        top_k=2,
        dropout=0.05
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("\n[+] Training Dynamic Threaded Network...")
    model.train()
    epochs = 120
    batch_size = 64
    
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            logits, _ = model(batch_x, return_routing_graph=False)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        if epoch % 30 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                test_logits, _ = model(X_test, return_routing_graph=False)
                acc = (test_logits.argmax(dim=-1) == y_test).float().mean().item() * 100
                print(f"  Epoch [{epoch:03d}/{epochs}] - Loss: {epoch_loss / len(X_train):.4f} - Test Acc: {acc:.2f}%")
            model.train()

    # 3. Dynamic Thread Inspection on Unseen Test Samples
    print("\n[+] Evaluating Dynamic Threads on Test Samples...")
    model.eval()
    with torch.no_grad():
        sample_batch = X_test[:5]
        logits, graph_info = model(sample_batch, return_routing_graph=True)
        predictions = logits.argmax(dim=-1).cpu().numpy()
        ground_truth = y_test[:5].cpu().numpy()

    # Print ASCII dynamic connection threads for Sample 0 and Sample 1
    print(render_ascii_threads(graph_info, sample_idx=0))
    print(f"Sample #0 -> Ground Truth: {ground_truth[0]}, Predicted: {predictions[0]}")
    
    print(render_ascii_threads(graph_info, sample_idx=1))
    print(f"Sample #1 -> Ground Truth: {ground_truth[1]}, Predicted: {predictions[1]}")

    # 4. Export Interactive HTML Visualizer
    output_html = os.path.join(os.path.dirname(__file__), "dynamic_threads_visualization.html")
    export_interactive_html(graph_info, sample_indices=[0, 1, 2, 3, 4], output_path=output_html)
    print(f"\n[+] Interactive Dynamic Threads Visualizer exported to:\n    {output_html}")

if __name__ == "__main__":
    main()
