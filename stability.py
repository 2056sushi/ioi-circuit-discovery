import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import random
import numpy as np
import torch
torch.set_grad_enabled(False)

from transformer_lens import HookedTransformer
from transformer_lens.patching import get_act_patch_resid_pre

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)

model = HookedTransformer.from_pretrained_no_processing(
    "meta-llama/Llama-3.2-1B-Instruct",
    device=device,
    dtype=torch.float16
)

clean_prompt = (
    "When John and Mary went to the store, "
    "John gave a book to"
)

corrupt_prompt = (
    "When John and Mary went to the store, "
    "Mary gave a book to"
)

correct = " Mary"
incorrect = " John"

def logit_diff(logits):

    final_logits = logits[0, -1]

    correct_token = model.to_single_token(correct)
    incorrect_token = model.to_single_token(incorrect)

    return (
        final_logits[correct_token]
        - final_logits[incorrect_token]
    )

clean_tokens = model.to_tokens(clean_prompt)
corrupt_tokens = model.to_tokens(corrupt_prompt)

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

clean_score = logit_diff(clean_logits).item()
corrupt_score = logit_diff(corrupt_logits).item()

full_effect = clean_score - corrupt_score

results = get_act_patch_resid_pre(
    model=model,
    corrupted_tokens=corrupt_tokens,
    clean_cache=clean_cache,
    patching_metric=logit_diff,
)

final_pos = -1

layer_scores = []

for layer in range(results.shape[0]):

    score = results[layer, final_pos].item()

    restoration = (
        (score - corrupt_score)
        / (clean_score - corrupt_score)
    )

    layer_scores.append(restoration)

layer_scores = np.array(layer_scores)

print("\nBASE LAYER SCORES")
print(layer_scores)

discoveries = []

for seed in [0, 1, 2]:

    np.random.seed(seed)

    noisy_scores = (
        layer_scores
        + np.random.normal(0, 0.05, size=len(layer_scores))
    )

    top_layers = np.argsort(noisy_scores)[-6:]

    discoveries.append(set(top_layers.tolist()))

print("\nDISCOVERED CIRCUITS")

for i, d in enumerate(discoveries):

    print(f"Seed {i}: {sorted(d)}")

print("\nJACCARD OVERLAPS")

overlaps = []

for i in range(3):
    for j in range(i + 1, 3):

        a = discoveries[i]
        b = discoveries[j]

        jaccard = len(a & b) / len(a | b)

        overlaps.append(jaccard)

        print(
            f"Seeds {i} vs {j}: "
            f"{jaccard:.4f}"
        )

print("\nFINAL STABILITY")

print(
    f"Mean overlap: "
    f"{np.mean(overlaps):.4f}"
)

print(
    f"Std overlap:  "
    f"{np.std(overlaps):.4f}"
)