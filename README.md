# IOI Circuit Discovery on LLaMA-3.2-1B-Instruct

This project performs mechanistic interpretability analysis on the Indirect Object Identification (IOI) task using a LLaMA-3.2-1B-Instruct model. The goal is to identify a sparse causal circuit responsible for correct indirect object prediction and evaluate its faithfulness, stability, and robustness.

---

## 1. Overview

We analyze how transformer layers contribute to IOI behavior using:

- Activation patching (clean ↔ corrupt interventions)
- Layer-wise causal effect measurement
- Sparse circuit extraction (layer-level approximation)
- Faithfulness evaluation against random baselines
- Stability analysis across perturbed discoveries
- Robustness testing under paraphrased prompts

Due to hardware constraints (Apple M1), this project uses **layer-level residual stream patching** instead of full edge-level EAP-IG circuit tracing.

---

## 2. Model & Setup

- **Model:** `meta-llama/Llama-3.2-1B-Instruct`
- **Framework:** TransformerLens
- **Precision:** float16
- **Device:** Apple M1 (MPS backend)
- **Memory constraints:** ~9GB unified memory usage

---

## 3. IOI Task

The model is evaluated on prompts of the form:

### Clean prompt:
When John and Mary went to the store, John gave a book to

### Corrupt prompt:
When John and Mary went to the store, Mary gave a book to


The correct behavior is predicting the indirect object ("Mary").

---

## 4. Circuit Discovery Method

We perform activation patching on the residual stream across transformer layers.

Key idea:
- Replace activations in corrupt runs with clean activations
- Measure how much each layer restores correct IOI behavior

---

## 5. Discovered Circuit

The following layers were identified as most causally important:
Layers: [10, 11, 12, 13, 14, 15]


These late layers dominate IOI behavior recovery.

---

## 6. Activation Patching Results

| Layer | Restoration |
|------|-------------|
| 10 | 0.5149 |
| 11 | 0.4772 |
| 12 | 0.7327 |
| 13 | 0.7406 |
| 14 | 0.7446 |
| 15 | 0.7366 |

---

## 7. Faithfulness Evaluation

### Full model IOI effect
- Clean score: 3.3125
- Corrupt score: -4.5781
- Effect size: 7.8906

### Circuit faithfulness
- Circuit recovers ~73.6% of full IOI effect

### Random baseline
- Random layer selection fails to recover meaningful IOI effect (~0.0)

---

## 8. Stability Analysis

We simulate circuit discovery under noise perturbations.

### Result:
- Top-k layer selection is perfectly stable

| Seed | Layers |
|------|--------|
| 0 | [10–15] |
| 1 | [10–15] |
| 2 | [10–15] |

- Mean Jaccard overlap: **1.0**

---

## 9. Robustness Analysis

We test circuit faithfulness under different corruption types:

| Condition | Faithfulness |
|----------|--------------|
| Token swap corruption | 0.7366 |
| Paraphrased corruption | 0.8534 |

The circuit generalizes well beyond exact prompt templates.

---

## 10. Key Findings

1. IOI behavior is strongly localized in late transformer layers (10–15).
2. A sparse layer-level circuit can recover ~70–85% of IOI behavior.
3. Random layer masks fail completely, confirming causal specificity.
4. The circuit is stable under small perturbations.
5. The circuit generalizes under paraphrased inputs.

---

## 11. Limitations

- This is a **layer-level approximation**, not a full edge-level circuit (e.g., EAP-IG).
- M1/MPS memory constraints prevented full-scale attention head decomposition.
- Results should be interpreted as **coarse causal structure**, not a complete mechanistic circuit.

---

## 12. Running the Code

### Main scripts:

```bash
python logit_diff.py
python activation_patching.py
python circuit_faithfulness.py
python stability.py
python robustness.py


