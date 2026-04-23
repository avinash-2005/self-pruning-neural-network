# Self-Pruning Neural Network for CIFAR-10

This repository contains a PyTorch implementation of a self-pruning neural network. The network uses a custom `PrunableLinear` layer with learnable gate parameters that allow the model to automatically prune unnecessary weights during training via L1 sparsity regularization.

## Key Features
- **PrunableLinear Layer**: A custom linear layer augmented with learnable gate parameters (sigmoid gates).
- **L1 Sparsity Loss**: Encourages the gate values to approach zero, effectively pruning the corresponding weights.
- **Dynamic Training**: The model balances standard Cross-Entropy loss with the sparsity loss, controlled by a hyperparameter `lambda`.
- **Visualization**: Automatically generates a histogram (`gate_distribution.png`) showing the distribution of gate values and the sparsity level achieved.

## Prerequisites & Dependencies

To run this project, you need Python 3.7+ and the following libraries installed:
- `torch`
- `torchvision`
- `matplotlib`
- `numpy`

You can install the required dependencies using pip:

```bash
pip install torch torchvision matplotlib numpy
```

## How to Run

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd self_pruning
   ```

2. **Run the training script**:
   ```bash
   python main.py
   ```

   The script will:
   - Automatically download the CIFAR-10 dataset into a `data/` folder.
   - Train the model using three different sparsity regularizations (`lambda = 0.0001`, `0.001`, and `0.01`).
   - Output the test accuracy and sparsity level for each lambda.
   - Save a visualization of the gate values for the sparsest model as `gate_distribution.png`.

## Output
Once the training is complete, a summary table will be printed in the console comparing the **Test Accuracy** and **Sparsity Level** across the different `lambda` configurations. 
Additionally, a plot showing the gate distribution will be saved to your directory as `gate_distribution.png`.
