#!/usr/bin/env python3
"""
Qwen API Logprob Analyzer

This script analyzes the logprobs from Qwen API completions to provide
meaningful metrics about the generation quality and model confidence.

Usage:
    python qwen_logprob_analyzer.py [--query "Your prompt here"] [--model "model_name"] [--save-csv]

Example:
    python qwen_logprob_analyzer.py --query "Generate a story in Chinese" --save-csv
"""

import argparse
import json
import math
import sys
import csv
from datetime import datetime
import requests
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

class QwenLogprobAnalyzer:
    def __init__(self,
                 api_url: str = "http://localhost:5504/v1/chat/completions",
                 model: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 system_prompt: str = "You are a helpful assistant."):
        """
        Initialize the analyzer with API parameters.

        Args:
            api_url: The Qwen API endpoint URL
            model: The model identifier to use
            system_prompt: The system prompt to use in the query
        """
        self.api_url = api_url
        self.model = model
        self.system_prompt = system_prompt

    def query_model(self, prompt: str, max_tokens: int = 300) -> Dict[str, Any]:
        """
        Query the Qwen API with the given prompt.

        Args:
            prompt: The user prompt to send to the model
            max_tokens: Maximum number of tokens to generate

        Returns:
            The API response as a dictionary
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "logprobs": 1,
            "max_tokens": max_tokens
        }

        try:
            print(f"Querying {self.model} with prompt: \"{prompt}\"")
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error querying API: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            sys.exit(1)

    def extract_logprobs(self, response: Dict[str, Any]) -> Tuple[List[str], List[float], str]:
        """
        Extract token logprobs from various potential response structures.

        Args:
            response: The API response dictionary

        Returns:
            Tuple of (tokens, logprobs, generated_text)
        """
        tokens = []
        logprobs = []
        generated_text = ""

        try:
            # Extract the generated text
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    generated_text = choice["message"]["content"]

                # Extract logprobs
                if "logprobs" in choice:
                    logprobs_data = choice["logprobs"]

                    # Try different possible structures based on API implementation
                    # Structure 1: OpenAI-like format with content array
                    if "content" in logprobs_data and isinstance(logprobs_data["content"], list):
                        for item in logprobs_data["content"]:
                            if "token" in item and "logprob" in item:
                                tokens.append(item["token"])
                                logprobs.append(item["logprob"])

                    # Structure 2: Array format with separate tokens and logprobs arrays
                    elif "tokens" in logprobs_data and "token_logprobs" in logprobs_data:
                        tokens = logprobs_data["tokens"]
                        logprobs = logprobs_data["token_logprobs"]

                        # Filter out None values that sometimes appear in logprobs
                        valid_indices = [i for i, lp in enumerate(logprobs) if lp is not None]
                        tokens = [tokens[i] for i in valid_indices]
                        logprobs = [logprobs[i] for i in valid_indices]
        except Exception as e:
            print(f"Error extracting logprobs: {e}")
            # Print the response structure to help with debugging
            print(f"Response structure: {json.dumps(response, indent=2)[:1000]}...")

        return tokens, logprobs, generated_text

    def analyze_logprobs(self, tokens: List[str], logprobs: List[float]) -> Dict[str, Any]:
        """
        Analyze the token logprobs to calculate meaningful metrics.

        Args:
            tokens: List of tokens
            logprobs: List of log probabilities for each token

        Returns:
            Dictionary of analysis results
        """
        if not tokens or not logprobs or len(tokens) != len(logprobs):
            return {"error": "Invalid tokens or logprobs data"}

        # Basic statistics
        total_logprob = sum(logprobs)
        token_count = len(tokens)
        normalized_logprob = total_logprob / token_count
        perplexity = math.exp(-normalized_logprob)

        # Statistical measures
        min_logprob = min(logprobs)
        max_logprob = max(logprobs)
        median_logprob = np.median(logprobs)
        std_dev = np.std(logprobs)

        # Calculate human-friendly confidence score (0-100)
        # Based on empirical observations of typical perplexity ranges
        # Lower perplexity = higher confidence
        max_reasonable_perplexity = 50  # Anything beyond this is considered very poor
        confidence_score = 100 * max(0, min(1, 1 - (perplexity / max_reasonable_perplexity)))

        # Confidence level determination based on various factors
        # 1. Base confidence from perplexity
        if perplexity < 5:
            confidence_level = "Very High"
            confidence_description = "The model is extremely confident in this generation."
        elif perplexity < 10:
            confidence_level = "High"
            confidence_description = "The model is confident in this generation."
        elif perplexity < 20:
            confidence_level = "Moderate"
            confidence_description = "The model has moderate confidence in this generation."
        elif perplexity < 40:
            confidence_level = "Low"
            confidence_description = "The model has low confidence in this generation."
        else:
            confidence_level = "Very Low"
            confidence_description = "The model has very low confidence in this generation."

        # 2. Factor in consistency (standard deviation)
        # High std_dev means inconsistent confidence across tokens
        consistency_ratio = std_dev / abs(normalized_logprob)
        if consistency_ratio > 1.0 and confidence_level != "Very Low":
            confidence_level = confidence_level + " (Inconsistent)"
            confidence_description += " However, confidence varies significantly across different parts."

        # 3. Check for unusually low probability tokens
        threshold = normalized_logprob - 1.5 * std_dev
        unusual_token_indices = [i for i, lp in enumerate(logprobs) if lp < threshold]
        unusual_tokens = [(i, tokens[i], logprobs[i]) for i in unusual_token_indices]
        unusual_tokens.sort(key=lambda x: x[2])  # Sort by logprob (ascending)

        unusual_token_ratio = len(unusual_token_indices) / token_count
        if unusual_token_ratio > 0.1 and confidence_level not in ["Low", "Very Low"]:
            confidence_description += f" {len(unusual_token_indices)} tokens ({unusual_token_ratio:.1%}) have unusually low confidence."

        # 4. Calculate "high confidence token percentage"
        # Tokens with logprob > -2 are considered "high confidence"
        high_conf_tokens = sum(1 for lp in logprobs if lp > -2)
        high_conf_percentage = (high_conf_tokens / token_count) * 100

        # Calculate histogram of logprobs
        bins = [-float("inf"), -10, -5, -3, -2, -1, -0.5, -0.1, 0]
        bin_labels = ["<-10", "-10 to -5", "-5 to -3", "-3 to -2",
                      "-2 to -1", "-1 to -0.5", "-0.5 to -0.1", "-0.1 to 0"]
        hist, _ = np.histogram(logprobs, bins=bins)
        distribution = {label: count for label, count in zip(bin_labels, hist)}

        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = np.percentile(logprobs, percentiles)
        percentile_data = {f"p{p}": v for p, v in zip(percentiles, percentile_values)}

        # Calculate rolling statistics
        window_size = min(10, token_count)
        rolling_avg = [sum(logprobs[i:i+window_size])/window_size
                       for i in range(0, token_count - window_size + 1)]

        return {
            "token_count": token_count,
            "total_logprob": total_logprob,
            "normalized_logprob": normalized_logprob,
            "perplexity": perplexity,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "confidence_description": confidence_description,
            "high_confidence_token_percentage": high_conf_percentage,
            "min_logprob": min_logprob,
            "max_logprob": max_logprob,
            "median_logprob": median_logprob,
            "std_dev": std_dev,
            "consistency_ratio": consistency_ratio,
            "unusual_tokens": unusual_tokens[:10],  # Limit to top 10
            "unusual_token_ratio": unusual_token_ratio,
            "distribution": distribution,
            "percentiles": percentile_data,
            "rolling_avg": rolling_avg,
            "raw_data": {
                "tokens": tokens,
                "logprobs": logprobs
            }
        }

    def save_token_data_to_csv(self, tokens: List[str], logprobs: List[float],
                               filename: Optional[str] = None) -> str:
        """
        Save token-level data to a CSV file.

        Args:
            tokens: List of tokens
            logprobs: List of log probabilities
            filename: Optional custom filename

        Returns:
            The filename of the created CSV file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"token_logprobs_{timestamp}.csv"

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["position", "token", "logprob", "probability", "cumulative_logprob"])

            cumulative_logprob = 0
            for i, (token, logprob) in enumerate(zip(tokens, logprobs)):
                cumulative_logprob += logprob
                writer.writerow([i, token, logprob, math.exp(logprob), cumulative_logprob])

        return filename

    def print_analysis_results(self, analysis: Dict[str, Any], generated_text: str,
                               show_tokens: int = 3) -> None:
        """
        Print the analysis results in a readable format.

        Args:
            analysis: The dictionary of analysis results
            generated_text: The generated text
            show_tokens: Number of sample tokens to show
        """
        if "error" in analysis:
            print(f"Analysis error: {analysis['error']}")
            return

        # Print generated text (truncated if too long)
        max_text_display = 2000
        print("\n=== GENERATED TEXT ===")
        if len(generated_text) > max_text_display:
            print(f"{generated_text[:max_text_display]}... (truncated, {len(generated_text)} chars total)")
        else:
            print(generated_text)

        # Print key metrics
        print("\n=== KEY METRICS ===")
        print(f"Token count: {analysis['token_count']}")
        print(f"Normalized logprob (per token): {analysis['normalized_logprob']:.4f}")
        print(f"Perplexity: {analysis['perplexity']:.4f}")
        print(f"Range: [{analysis['min_logprob']:.4f}, {analysis['max_logprob']:.4f}]")
        print(f"Standard deviation: {analysis['std_dev']:.4f}")

        # Print interpretation
        print("\n=== INTERPRETATION ===")
        if analysis['perplexity'] < 5:
            confidence = "Very high"
        elif analysis['perplexity'] < 10:
            confidence = "High"
        elif analysis['perplexity'] < 20:
            confidence = "Moderate"
        elif analysis['perplexity'] < 40:
            confidence = "Low"
        else:
            confidence = "Very low"

        print(f"Model confidence: {confidence} (based on perplexity of {analysis['perplexity']:.2f})")

        # Print distribution of token logprobs
        print("\n=== DISTRIBUTION OF TOKEN LOGPROBS ===")
        for label, count in analysis["distribution"].items():
            percentage = count / analysis["token_count"] * 100
            bar = "#" * int(percentage / 5)
            print(f"{label:>10}: {count:>3} tokens ({percentage:>5.1f}%) {bar}")

        # Print sample tokens from beginning
        if show_tokens > 0 and "raw_data" in analysis:
            tokens = analysis["raw_data"]["tokens"]
            logprobs = analysis["raw_data"]["logprobs"]

            print(f"\n=== SAMPLE TOKENS (FIRST {show_tokens}) ===")
            for i in range(min(show_tokens, len(tokens))):
                print(f"  {i}: \"{tokens[i]}\" (logprob: {logprobs[i]:.4f}, prob: {math.exp(logprobs[i]):.6f})")

        # Print unusual tokens
        if analysis["unusual_tokens"]:
            print("\n=== UNUSUAL TOKENS (LOW CONFIDENCE) ===")
            for i, token, logprob in analysis["unusual_tokens"]:
                print(f"  Position {i}: \"{token}\" (logprob: {logprob:.4f}, prob: {math.exp(logprob):.6e})")

        # Print original metrics
        print("\n=== ORIGINAL METRICS (LESS MEANINGFUL) ===")
        print(f"Total logprob: {analysis['total_logprob']:.4f}")
        print(f"Probability: {math.exp(analysis['total_logprob']):.6e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze logprobs from Qwen API completions")
    parser.add_argument("--query", type=str, default="Generate a story in Chinese with English translation.",
                        help="The prompt to send to the model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="The model to use")
    parser.add_argument("--max-tokens", type=int, default=300,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--api-url", type=str, default="http://localhost:5504/v1/chat/completions",
                        help="The API endpoint URL")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save token-level data to a CSV file")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.",
                        help="System prompt to use")
    args = parser.parse_args()

    # Initialize the analyzer
    analyzer = QwenLogprobAnalyzer(
        api_url=args.api_url,
        model=args.model,
        system_prompt=args.system_prompt
    )

    # Query the model
    response = analyzer.query_model(args.query, max_tokens=args.max_tokens)

    # Extract and analyze logprobs
    tokens, logprobs, generated_text = analyzer.extract_logprobs(response)

    if not tokens or not logprobs:
        print("Failed to extract tokens and logprobs from the response.")
        sys.exit(1)

    # Analyze the logprobs
    analysis_results = analyzer.analyze_logprobs(tokens, logprobs)

    # Print the analysis results
    analyzer.print_analysis_results(analysis_results, generated_text)

    # Save token data to CSV if requested
    if args.save_csv:
        csv_file = analyzer.save_token_data_to_csv(tokens, logprobs)
        print(f"\nToken-level data saved to: {csv_file}")
        print("You can use this CSV to perform custom analysis or visualizations.")

if __name__ == "__main__":
    main()