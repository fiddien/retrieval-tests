#!/bin/bash

# Make the API call with a comparison case
echo "Comparing logprobs for Chinese vs English generation..."

# First query - generate in Chinese
echo -e "\n=== GENERATION 1: Chinese Story ==="
curl -s http://localhost:5504/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Generate a brief story in Chinese about a rabbit."}
        ],
       "logprobs": 1,
       "max_tokens": 100
    }' > response_chinese.json

# Second query - generate in English
echo -e "\n=== GENERATION 2: English Story ==="
curl -s http://localhost:5504/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Generate a brief story in English about a rabbit."}
        ],
       "logprobs": 1,
       "max_tokens": 100
    }' > response_english.json

# Process both responses
python3 -c '
import json
import math
import numpy as np
import sys
from collections import defaultdict

def analyze_response(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    results = {}

    try:
        # Extract the message content
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            content = message.get("content", "")
            results["content"] = content

            # Extract logprobs
            if "logprobs" in data["choices"][0]:
                logprobs_data = data["choices"][0]["logprobs"]

                # Try to find token logprobs in different possible structures
                token_logprobs = None
                tokens = None

                if "content" in logprobs_data and isinstance(logprobs_data["content"], list):
                    token_logprobs = [item.get("logprob", 0) for item in logprobs_data["content"]]
                    tokens = [item.get("token", "") for item in logprobs_data["content"] if "token" in item]
                elif "token_logprobs" in logprobs_data:
                    token_logprobs = logprobs_data["token_logprobs"]
                    if "tokens" in logprobs_data:
                        tokens = logprobs_data["tokens"]

                if token_logprobs:
                    results["token_count"] = len(token_logprobs)
                    results["total_logprob"] = sum(token_logprobs)
                    results["normalized_logprob"] = sum(token_logprobs) / len(token_logprobs)
                    results["perplexity"] = math.exp(-results["normalized_logprob"])
                    results["min_logprob"] = min(token_logprobs)
                    results["max_logprob"] = max(token_logprobs)
                    results["std_dev"] = np.std(token_logprobs)

                    # Calculate confidence bands
                    bands = [(-float("inf"), -10), (-10, -5), (-5, -2), (-2, -1), (-1, -0.5), (-0.5, -0.1), (-0.1, 0)]
                    distribution = defaultdict(int)

                    for lp in token_logprobs:
                        for i, (lower, upper) in enumerate(bands):
                            if lower <= lp < upper:
                                distribution[f"band_{i}"] = distribution[f"band_{i}"] + 1
                                break
                            elif i == len(bands) - 1 and lp >= upper:  # Handle the last band upper bound
                                distribution[f"band_{i+1}"] = distribution[f"band_{i+1}"] + 1

                    results["distribution"] = {
                        f"{lower if lower != -float(\'inf\') else \'<\'}{upper}": count
                        for (lower, upper), count in zip(bands, [distribution[f"band_{i}"] for i in range(len(bands))])
                    }
                    results["distribution"][f">={bands[-1][1]}"] = distribution.get(f"band_{len(bands)}", 0)

                    # Token-level analysis
                    if tokens and len(tokens) == len(token_logprobs):
                        results["tokens"] = tokens

                        # Identify unusual tokens (very low probability)
                        threshold = np.mean(token_logprobs) - 1.5 * results["std_dev"]
                        unusual_tokens = [(i, tokens[i], token_logprobs[i])
                                        for i in range(len(tokens))
                                        if token_logprobs[i] < threshold]
                        unusual_tokens.sort(key=lambda x: x[2])  # Sort by logprob (ascending)

                        results["unusual_tokens"] = unusual_tokens[:5]  # Top 5 unusual tokens
    except Exception as e:
        results["error"] = str(e)

    return results

# Analyze both responses
chinese_results = analyze_response("response_chinese.json")
english_results = analyze_response("response_english.json")

# Print comparison
print("\n===== COMPARISON OF GENERATION METRICS =====")

# Content preview
print("\n--- Content Previews ---")
if "content" in chinese_results:
    print(f"Chinese: {chinese_results['content'][:100]}...")
if "content" in english_results:
    print(f"English: {english_results['content'][:100]}...")

# Key metrics comparison
metrics = [
    ("Token Count", "token_count", ""),
    ("Normalized Logprob", "normalized_logprob", ".4f"),
    ("Perplexity", "perplexity", ".2f"),
    ("Min Token Logprob", "min_logprob", ".4f"),
    ("Max Token Logprob", "max_logprob", ".4f"),
    ("Standard Deviation", "std_dev", ".4f")
]

print("\n--- Key Metrics ---")
print(f"{'Metric':<25} {'Chinese':<15} {'English':<15} {'Difference':<15}")
print("-" * 70)

for label, key, fmt in metrics:
    if key in chinese_results and key in english_results:
        chinese_val = chinese_results[key]
        english_val = english_results[key]
        diff = chinese_val - english_val

        if fmt:
            print(f"{label:<25} {chinese_val:{fmt}} {english_val:{fmt}} {diff:{fmt}}")
        else:
            print(f"{label:<25} {chinese_val} {english_val} {diff}")

# Distribution comparison
if "distribution" in chinese_results and "distribution" in english_results:
    print("\n--- Token Confidence Distribution ---")
    print(f"{'Logprob Range':<15} {'Chinese':<20} {'English':<20}")
    print("-" * 55)

    all_bands = sorted(set(list(chinese_results["distribution"].keys()) +
                           list(english_results["distribution"].keys())))

    for band in all_bands:
        chinese_count = chinese_results["distribution"].get(band, 0)
        english_count = english_results["distribution"].get(band, 0)

        chinese_pct = chinese_count / chinese_results["token_count"] * 100 if "token_count" in chinese_results else 0
        english_pct = english_count / english_results["token_count"] * 100 if "token_count" in english_results else 0

        print(f"{band:<15} {chinese_count:>3} ({chinese_pct:>5.1f}%) {english_count:>3} ({english_pct:>5.1f}%)")

# Unusual tokens
print("\n--- Most Unusual Tokens ---")

print("Chinese unusual tokens:")
if "unusual_tokens" in chinese_results:
    for i, token, logprob in chinese_results["unusual_tokens"]:
        print(f"  Position {i}: \"{token}\" (logprob: {logprob:.4f})")
else:
    print("  No unusual tokens data available")

print("\nEnglish unusual tokens:")
if "unusual_tokens" in english_results:
    for i, token, logprob in english_results["unusual_tokens"]:
        print(f"  Position {i}: \"{token}\" (logprob: {logprob:.4f})")
else:
    print("  No unusual tokens data available")

# Conclusion
print("\n--- Analysis Conclusion ---")
if "normalized_logprob" in chinese_results and "normalized_logprob" in english_results:
    if chinese_results["normalized_logprob"] > english_results["normalized_logprob"]:
        print("The model appears more confident generating in Chinese than English.")
    else:
        print("The model appears more confident generating in English than Chinese.")
else:
    print("Insufficient data to compare model confidence between languages.")

print("\nThis comparison shows how model confidence varies between different types of generations.")
'