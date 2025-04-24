import re
from typing import Dict
import aiohttp
from rouge_score import rouge_scorer
import os

SYSTEM_PROMPT = """\
You are an intelligent assistant specializing in analyzing profiles and providing insights based on the Indonesian Air Force (TNI-AU) knowledge base. Your task is to extract and present relevant details about TNI-AU personnel in a concise yet comprehensive format."

Instructions:
Summarize the personnel's profile using the provided knowledge base.
Include the following details (if available):
1. Full Name & Rank
2. Current Position & Unit
3. Years of Service & Career Progression
4. Operational Experience (combat missions, joint operations, command roles)
5. Education & Training (military academies, advanced degrees, specialized courses)
6. Technical Expertise (aviation technology, weapon systems, strategic planning)
7. Notable Achievements & Recognitions

When answering:
If the required information is missing or irrelevant, include the sentence: "The answer you are looking for is not found in the knowledge base!"
Structure your responses for clarity, using bullet points, tables, or short paragraphs as needed to highlight key points.
Here is the knowledge base:
{knowledge}
The above is the knowledge base.
"""


class LLMEvaluator:
    def __init__(self, model_hosts: Dict[str, str]):
        """
        Initialize LLMEvaluator with a dictionary mapping model names to their vLLM hosts
        Args:
            model_hosts: Dict mapping model names to their vLLM host URLs
        """
        self.model_hosts = model_hosts
        self.system_prompt = SYSTEM_PROMPT
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        """Load stopwords from file"""
        stopwords_path = os.path.join(
            os.path.dirname(__file__), "nlp", "stopwords-id.txt"
        )
        try:
            with open(stopwords_path, "r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except FileNotFoundError:
            print(f"Warning: Stopwords file not found at {stopwords_path}")
            return set()

    async def generate(self, query: str, knowledge: str, model_name: str) -> Dict:
        """Generate response and evaluate it"""
        if model_name not in self.model_hosts:
            raise ValueError(f"No host configured for model: {model_name}")

        vllm_endpoint = f"{self.model_hosts[model_name]}/v1/chat/completions"
        messages = [
            {
                "role": "system",
                "content": self.system_prompt.format(knowledge=knowledge),
            },
            {"role": "user", "content": query},
        ]

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.1,
                "top_p": 0.3,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.7,
            }
            async with session.post(vllm_endpoint, json=payload) as response:
                result = await response.json()
                response_text = result["choices"][0]["message"]["content"]

                evaluation = self._evaluate_response(response_text, knowledge)
                return {"response": response_text, "evaluation": evaluation}

    def _evaluate_response(self, text: str, knowledge: str) -> Dict:
        """
        Evaluate response quality, check for Chinese characters, and calculate summarization metrics
        """
        evaluation = {
            "chinese_char_count": 0,
            "chinese_char_percentage": 0.0,
            "average_sentence_length": 0,
            "factual_consistency_score": 0.0,
            "summary_metrics": {
                "token_count": len(text.split()),
                "compression_ratio": len(text.split()) / len(knowledge.split())
                if knowledge
                else 0.0,
                "content_density": 0.0,
                "rouge_scores": {},
            },
        }

        # Calculate ROUGE scores
        scores = self.scorer.score(knowledge, text)
        evaluation["summary_metrics"]["rouge_scores"] = {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

        # Count Chinese characters
        chinese_chars = len([c for c in text if self._is_chinese_char(c)])
        total_chars = len(text)
        evaluation["chinese_char_count"] = chinese_chars
        evaluation["chinese_char_percentage"] = (
            (chinese_chars / total_chars) * 100 if total_chars > 0 else 0
        )

        # Calculate average sentence length and content density
        sentences = re.split(r"[.!?\n]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            evaluation["average_sentence_length"] = sum(
                len(s.split()) for s in sentences
            ) / len(sentences)

            # Calculate content density using loaded stopwords
            words = text.lower().split()
            content_words = [w for w in words if w not in self.stopwords]
            evaluation["summary_metrics"]["content_density"] = (
                len(content_words) / len(words) if words else 0
            )

        return evaluation

    def _is_chinese_char(self, char: str) -> bool:
        """
        Check if a character is Chinese
        """
        try:
            # Check if character is in CJK ranges
            return any(
                [
                    "\u4e00" <= char <= "\u9fff",  # CJK Unified Ideographs
                    "\u3400" <= char <= "\u4dbf",  # CJK Unified Ideographs Extension A
                    "\u20000"
                    <= char
                    <= "\u2a6df",  # CJK Unified Ideographs Extension B
                    "\u2a700"
                    <= char
                    <= "\u2b73f",  # CJK Unified Ideographs Extension C
                    "\u2b740"
                    <= char
                    <= "\u2b81f",  # CJK Unified Ideographs Extension D
                    "\u2b820"
                    <= char
                    <= "\u2ceaf",  # CJK Unified Ideographs Extension E
                    "\uf900" <= char <= "\ufaff",  # CJK Compatibility Ideographs
                ]
            )
        except TypeError:
            return False
