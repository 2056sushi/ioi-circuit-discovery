import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

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

clean_logits = model(clean_tokens)
corrupt_logits = model(corrupt_tokens)

clean_score = logit_diff(clean_logits).item()
corrupt_score = logit_diff(corrupt_logits).item()

print("\nBASELINE SCORES")
print("-" * 30)

print(f"Clean score:   {clean_score:.4f}")
print(f"Corrupt score: {corrupt_score:.4f}")

print("\nRUNNING PATCHING...")
print("-" * 30)

clean_logits, clean_cache = model.run_with_cache(clean_tokens)

results = get_act_patch_resid_pre(
    model=model,
    corrupted_tokens=corrupt_tokens,
    clean_cache=clean_cache,
    patching_metric=logit_diff,
)

print("\nPATCHING RESULTS")
print("-" * 30)
final_pos = -1

for layer in range(results.shape[0]):

    patched_score = results[layer, final_pos].item()

    restoration = (
        (patched_score - corrupt_score)
        / (clean_score - corrupt_score)
    )

    print(
        f"Layer {layer:2d} | "
        f"Patched Score: {patched_score:7.4f} | "
        f"Restoration: {restoration:.4f}"
    )