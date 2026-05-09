import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
torch.set_grad_enabled(False)

from transformer_lens import HookedTransformer

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)

model = HookedTransformer.from_pretrained_no_processing(
    "meta-llama/Llama-3.2-1B-Instruct",
    device=device,
    dtype=torch.float16
)

correct = " Mary"
incorrect = " John"

important_layers = [10, 11, 12, 13, 14, 15]

examples = [
    {
        "name": "token_swap",
        "clean":
            "When John and Mary went to the store, "
            "John gave a book to",

        "corrupt":
            "When John and Mary went to the store, "
            "Mary gave a book to"
    },

    {
        "name": "paraphrase",
        "clean":
            "John and Mary visited the market, "
            "and John handed a gift to",

        "corrupt":
            "John and Mary visited the market, "
            "and Mary handed a gift to"
    }
]

def logit_diff(logits):

    final_logits = logits[0, -1]

    correct_token = model.to_single_token(correct)
    incorrect_token = model.to_single_token(incorrect)

    return (
        final_logits[correct_token]
        - final_logits[incorrect_token]
    )

faithfulness_scores = []

for ex in examples:

    clean_tokens = model.to_tokens(ex["clean"])
    corrupt_tokens = model.to_tokens(ex["corrupt"])

    clean_logits, clean_cache = model.run_with_cache(
        clean_tokens
    )

    corrupt_logits, corrupt_cache = model.run_with_cache(
        corrupt_tokens
    )

    clean_score = logit_diff(clean_logits).item()
    corrupt_score = logit_diff(corrupt_logits).item()

    full_effect = clean_score - corrupt_score

    final_pos = -1

    def circuit_hook(activation, hook):

        layer_num = hook.layer()

        if layer_num in important_layers:

            activation[:, final_pos, :] = clean_cache[
                "resid_pre",
                layer_num
            ][:, final_pos, :]

        else:

            activation[:, final_pos, :] = corrupt_cache[
                "resid_pre",
                layer_num
            ][:, final_pos, :]

        return activation

    circuit_logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[
            (
                lambda name:
                name.endswith("hook_resid_pre"),
                circuit_hook
            )
        ]
    )

    circuit_score = logit_diff(circuit_logits).item()

    circuit_effect = (
        circuit_score - corrupt_score
    )

    faithfulness = (
        circuit_effect / full_effect
    )

    faithfulness_scores.append(faithfulness)

    print("\n" + "=" * 50)
    print(ex["name"].upper())
    print("=" * 50)

    print(f"Clean Score:      {clean_score:.4f}")
    print(f"Corrupt Score:    {corrupt_score:.4f}")
    print(f"Effect Size:      {full_effect:.4f}")

    print(f"Circuit Score:    {circuit_score:.4f}")
    print(f"Faithfulness:     {faithfulness:.4f}")

baseline = faithfulness_scores[0]

print("\nFINAL RETENTION")
print("-" * 30)

for name, score in zip(
    [x["name"] for x in examples],
    faithfulness_scores
):

    retention = score / baseline

    print(
        f"{name}: "
        f"{retention:.4f}"
    )