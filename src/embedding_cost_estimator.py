import os
import json
import tiktoken

# === Config ===
DATASET_PATH = "ProseObjects-01/clauses.jsonl"   # path to your dataset
MODEL_NAME = "text-embedding-3-small"  # or "text-embedding-3-large"

# Pricing (as of Aug 2025)
PRICES = {
    "text-embedding-3-small": 0.02 / 1_000_000,   # $0.02 per 1M tokens
    "text-embedding-3-large": 0.13 / 1_000_000    # $0.13 per 1M tokens
}

def count_tokens(text: str, model: str = MODEL_NAME) -> int:
    """Count tokens in a text string using tiktoken."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def estimate_embedding_cost(dataset_path: str, model: str = MODEL_NAME):
    total_tokens = 0
    num_items = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            text = entry.get("text", "")
            tokens = count_tokens(text, model)
            total_tokens += tokens
            num_items += 1

    price_per_token = PRICES[model]
    estimated_cost = total_tokens * price_per_token

    print(f"📊 Embedding Cost Estimate ({model})")
    print(f"--------------------------------------")
    print(f"Number of documents: {num_items}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Price per 1M tokens: ${PRICES[model] * 1_000_000:.2f}")
    print(f"Estimated cost: ${estimated_cost:.4f}")

if __name__ == "__main__":
    estimate_embedding_cost(DATASET_PATH, MODEL_NAME)
