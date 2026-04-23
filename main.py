# -*- coding: utf-8 -*-
"""
Self-Pruning Neural Network for CIFAR-10
=========================================
This script implements a feed-forward neural network with learnable gate parameters
that allow the network to prune its own weights during training.

Key Components:
  - PrunableLinear: Custom linear layer with learnable gate_scores (sigmoid gates)
  - Sparsity Loss:  L1 penalty on gate values to encourage zeros
  - Training Loop:  Total Loss = CrossEntropy + lambda * SparsityLoss
  - Evaluation:     Reports test accuracy and sparsity level for 3 lambda values
  - Visualization:  Histogram of gate distributions for the best model
"""

import os
import sys

# Force UTF-8 output on Windows to avoid UnicodeEncodeError
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (saves PNG, no display needed)
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Part 1: PrunableLinear Layer
# ============================================================
class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable gate parameters.

    Each weight w_ij has a corresponding gate_score g_ij.
    The gate is computed as:  gate = sigmoid(gate_score)  ∈ (0, 1)
    The effective weight is:  w_ij * gate_ij

    When gate ≈ 0, the weight is effectively pruned from the network.
    Gradients flow through both `weight` and `gate_scores` automatically
    because all operations (sigmoid, element-wise multiply, matmul) are
    differentiable PyTorch ops.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias — same as nn.Linear
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores — one scalar per weight element
        # Initialized near 0 so sigmoid(gate_score) ≈ 0.5 initially
        self.gate_scores = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        # Kaiming init for weights; uniform init for gate scores
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.uniform_(self.gate_scores, -1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Map gate_scores → (0, 1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Step 2: Element-wise multiply — prune low-gate weights
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Step 3: Standard affine transform  y = x W^T + b
        #         Gradients flow through pruned_weights into both
        #         self.weight AND self.gate_scores automatically.
        return torch.nn.functional.linear(x, pruned_weights, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ============================================================
# Neural Network using PrunableLinear layers
# ============================================================
class SelfPruningNet(nn.Module):
    """
    3-layer feed-forward network for CIFAR-10 (10 classes).
    All linear layers are PrunableLinear to support self-pruning.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)   # 3072 → 512
        self.fc2 = PrunableLinear(512, 256)             # 512  → 256
        self.fc3 = PrunableLinear(256, 10)              # 256  → 10
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)           # flatten: (B, 3072)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)                     # logits: (B, 10)
        return x


# ============================================================
# Part 2: Sparsity Regularization Loss
# ============================================================
def sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    L1 penalty on all gate values across every PrunableLinear layer.

    L1 norm promotes sparsity because its gradient w.r.t. a small positive
    value is always +1 (constant), unlike L2 whose gradient shrinks to zero.
    This means the optimizer constantly pushes gate values toward 0 —
    once a gate reaches 0, its sigmoid is essentially stuck there.

    Returns:
        Scalar tensor (sum of all sigmoid(gate_scores) values)
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total = total + gates.sum()
    return total


# ============================================================
# Data Loading
# ============================================================
def get_dataloaders(batch_size: int = 128):
    """Download CIFAR-10 and return train/test DataLoaders."""
    # Normalize to [-1, 1] for better training stability
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


# ============================================================
# Part 3a: Training Loop
# ============================================================
def train_model(
    lambda_val: float,
    train_loader,
    device: torch.device,
    epochs: int = 8,
) -> nn.Module:
    """
    Train a SelfPruningNet with a given sparsity regularization coefficient λ.

    Total Loss = CrossEntropyLoss(logits, labels)
               + λ * Σ sigmoid(gate_scores)   [L1 sparsity on gates]

    Args:
        lambda_val: Weight of the sparsity penalty (controls prune aggressiveness)
        train_loader: PyTorch DataLoader for training data
        device: CPU or CUDA device
        epochs: Number of training epochs

    Returns:
        Trained model (on `device`)
    """
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*55}")
    print(f"  Training with lambda = {lambda_val}")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_cls_loss = 0.0
        running_sp_loss  = 0.0
        correct = 0
        total   = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # --- Classification loss ---
            cls_loss = criterion(outputs, labels)

            # --- Sparsity regularization ---
            sp_loss = sparsity_loss(model)

            # --- Combined loss ---
            loss = cls_loss + lambda_val * sp_loss

            loss.backward()
            optimizer.step()

            running_cls_loss += cls_loss.item()
            running_sp_loss  += sp_loss.item()

            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        train_acc  = 100.0 * correct / total
        avg_cls    = running_cls_loss / len(train_loader)
        avg_sp     = running_sp_loss  / len(train_loader)
        print(
            f"  Epoch [{epoch:2d}/{epochs}] "
            f"ClsLoss: {avg_cls:.4f}  "
            f"SpLoss: {avg_sp:.1f}  "
            f"TrainAcc: {train_acc:.2f}%"
        )

    return model


# ============================================================
# Part 3b: Evaluation
# ============================================================
def evaluate(model: nn.Module, test_loader, device: torch.device):
    """
    Evaluate the model on the test set and compute sparsity metrics.

    Sparsity = fraction of weights whose gate value < 0.01
    (a gate this small contributes < 1% of the original weight magnitude)

    Returns:
        accuracy  (float): Test accuracy in percent
        sparsity  (float): Sparsity level in percent
        all_gates (list):  Flat list of all gate values (for plotting)
    """
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total

    # --- Sparsity calculation ---
    total_weights  = 0
    pruned_weights = 0
    all_gates      = []

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten()
                total_weights  += len(gates)
                pruned_weights += int((gates < 0.01).sum())
                all_gates.extend(gates.tolist())

    sparsity = 100.0 * pruned_weights / total_weights if total_weights > 0 else 0.0
    return accuracy, sparsity, all_gates


# ============================================================
# Main experiment
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=128)

    # Three λ values: low / medium / high sparsity pressure
    lambdas = [0.0001, 0.001, 0.01]
    results = []
    best_model_gates  = None
    best_lambda       = None
    best_sparsity     = -1.0

    for lam in lambdas:
        model = train_model(lam, train_loader, device, epochs=8)
        acc, sparsity, gates = evaluate(model, test_loader, device)
        results.append((lam, acc, sparsity))

        print(f"\n  >> Test Accuracy : {acc:.2f}%")
        print(f"  >> Sparsity Level: {sparsity:.2f}%")

        # Track the model with highest sparsity for the plot
        if sparsity > best_sparsity:
            best_sparsity    = sparsity
            best_model_gates = gates
            best_lambda      = lam

    # -------------------------------------------------------
    # Print results table
    # -------------------------------------------------------
    print("\n")
    print("=" * 55)
    print(f"{'Lambda':>10} | {'Test Accuracy':>14} | {'Sparsity Level':>15}")
    print("-" * 55)
    for lam, acc, sp in results:
        print(f"{lam:>10.4f} | {acc:>13.2f}% | {sp:>14.2f}%")
    print("=" * 55)

    # -------------------------------------------------------
    # Plot gate distribution for the best (sparsest) model
    # -------------------------------------------------------
    out_path = "gate_distribution.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(best_model_gates, bins=80, color="#4C72B0", edgecolor="white",
            linewidth=0.4, alpha=0.85)
    ax.set_title(
        f"Gate Value Distribution  (lambda = {best_lambda}, Sparsity = {best_sparsity:.1f}%)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Gate Value  (sigmoid output)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.axvline(0.01, color="red", linestyle="--", linewidth=1.5,
               label="Prune threshold (0.01)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
