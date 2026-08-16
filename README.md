# Word Predictor MLP:-

A lightweight, zero-dependency neural language model built from first principles. This repository contains both a **custom-built automatic differentiation (Autograd) engine built purely in standard Python** and an optimized modern **PyTorch implementation** that scales from bigram predictions to multi-token context language modeling.

---

## 🌟 Key Highlights

* **100% Custom Autograd Engine:** Implemented a scalar-valued backpropagation framework (`Value` class) from scratch with no external framework dependencies—supporting forward computation graphs and topological sorting for automatic reverse-mode differentiation.
* **Custom Neural Network Suite:** Manually designed `Neuron`, `Layer`, and `MLP` classes powered by the custom scalar engine, handling custom weight initializations, activations (`tanh`), and gradient updates.
* **Dual Architecture Approach:**
  * **Custom Python Engine:** Evaluates low-level autograd mechanics and custom loss tracking (NLL / Log-Likelihood).
  * **PyTorch Production Version:** Scales execution speed using vectorized tensors, `nn.Embedding`, and modern optimization (`Adam`).
* **Architectural Evolution:** Documents the transition from a single-word Bigram model (entropy floor at ~3.20 loss) to an $N$-gram context model achieving low loss (~0.20 NLL) for next-word prediction.

---

## 🏗 System Architecture

### 1. The Custom Autograd Engine (`Value`)
The engine tracks operations on scalar values, constructing a directed acyclic graph (DAG) dynamically:

* **Supported Operations:** Addition (`+`), Multiplication (`*`), Exponentiation (`exp()`), Logarithms (`log()`), Powers (`**`), and Hyperbolic Tangent (`tanh()`).
* **Reverse-Mode Autograd:** Uses topological sort on node dependency graphs to propagate chain-rule gradients backwards via `.backward()`.

### 2. Model Evolution & Loss Reduction

| Architecture Stage | Context Window | Key Layers | Training Loss (NLL) | Key Bottleneck / Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Bigram Model** | 1 Token | `Embedding(V, 16)` $\rightarrow$ `Linear(16, V)` | **~3.41 – 3.20** | High entropy limit; single-word context creates high ambiguity. |
| **Stage 2: Context N-Gram MLP** | $N$ Tokens ($N=3$) | `Embedding(V, 64)` $\rightarrow$ `Linear(N*64, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(128, V)` | **~0.20** | Solves single-word ambiguity by looking back $N$ words. |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch (optional, required only for the PyTorch implementation)
* NumPy & Matplotlib

```bash
pip install torch numpy matplotlib

```
🔮 Inference & Text Generation
Generate text using temperature-scaled sampling to prevent sampling loops:

Python
@torch.no_grad()
def generate_text(prompt, model, word_idx, idx_word, block_size=3, max_tokens=20, temperature=0.8):
    model.eval()
    tokens = prompt.lower().split()
    ```bash
    for _ in range(max_tokens):
        context = tokens[-block_size:]
        x = torch.tensor([[word_idx[w] for w in context]])
        logits = model(x) / temperature
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        tokens.append(idx_word[next_idx])
        
    return " ".join(tokens)

# Run Inference
print(generate_text("alice was beginning", model, word_idx, idx_word))
📜 Acknowledgments
Inspired by Andrej Karpathy's micrograd and MakeMore educational series, exploring language modeling foundations from raw scalar backpropagation up to modern neural networks.
