# IOI Circuit Discovery + Faithfulness Report

## Model + Environment

### Model
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Framework: TransformerLens
- Precision: `float16`

### Hardware
- Device: Apple M1 MacBook
- Backend: PyTorch MPS
- Approx VRAM / unified memory usage: ~9GB
- OS: macOS

### Runtime
- Activation patching runtime: ~1–2 minutes
- Stability analysis runtime: ~1 minute
- Robustness analysis runtime: ~30 seconds

---

# Methodology

## IOI Task

Clean example:

```text
When John and Mary went to the store, John gave a book to