import torch
from transformer_lens import HookedTransformer
from ioi_dataset import DATASET

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)

model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device=device,
    dtype=torch.float16
)

def get_logit_diff(prompt, correct, incorrect):

    tokens = model.to_tokens(prompt)

    logits = model(tokens)

    final_logits = logits[0, -1]

    correct_token = model.to_single_token(correct)
    incorrect_token = model.to_single_token(incorrect)

    diff = (
        final_logits[correct_token]
        - final_logits[incorrect_token]
    )

    return diff.item()

clean_scores = []
corrupt_scores = []

print("\n=== IOI CLEAN vs CORRUPT ===\n")

for item in DATASET:

    clean_score = get_logit_diff(
        item["clean"],
        item["correct"],
        item["incorrect"]
    )

    corrupt_score = get_logit_diff(
        item["corrupt"],
        item["correct"],
        item["incorrect"]
    )

    clean_scores.append(clean_score)
    corrupt_scores.append(corrupt_score)

    print("CLEAN:")
    print(item["clean"])
    print(f"Score: {clean_score:.4f}")

    print()

    print("CORRUPT:")
    print(item["corrupt"])
    print(f"Score: {corrupt_score:.4f}")

    print("\n" + "=" * 60 + "\n")

avg_clean = sum(clean_scores) / len(clean_scores)
avg_corrupt = sum(corrupt_scores) / len(corrupt_scores)

print("FINAL RESULTS")
print("-" * 30)

print(f"Average CLEAN score:   {avg_clean:.4f}")
print(f"Average CORRUPT score: {avg_corrupt:.4f}")

effect = avg_clean - avg_corrupt

print(f"IOI EFFECT SIZE:       {effect:.4f}")