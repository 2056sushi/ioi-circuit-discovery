import torch
from transformer_lens import HookedTransformer

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)

model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device=device,
    dtype=torch.float16
)

prompt = "When John and Mary went to the store, John gave a book to"

tokens = model.to_tokens(prompt)

logits = model(tokens)

next_token = logits[0, -1].argmax(dim=-1)

print("Prediction:")
print(model.to_string(next_token))