import re
import time
from typing import Dict
import aiohttp
from rouge_score import rouge_scorer
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """\
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
- If the required information is missing or irrelevant, include the sentence: "The answer you are looking for is not found in the knowledge base!"
- Structure your responses for clarity, using bullet points, tables, or short paragraphs as needed to highlight key points.
- Use Markdown headings for each section.
- Ensure the headings are in **Chinese**. For example: # 全名和职级
- Ensure the content is written in **{target_lang}**.

Here is the knowledge base:
{knowledge}
The above is the knowledge base.
"""

COMPARE_SUMMARY_SYSTEM_PROMPT = """\
You are an intelligent assistant specializing in analyzing personnel records and comparing service profiles based on the Indonesian Air Force (TNI-AU) knowledge base.

Your task is to generate a **concise and informative profile comparison summary** of TNI-AU personnel using the knowledge base provided.

---

### **Instructions:**

Extract and present the following information **if available** in the knowledge base:

1. **Nama & Pangkat Saat Ini**
   (Full Name and Current Rank)

2. **Korps atau Kejuruan**
   (Corps or Military Specialty)

3. **Riwayat Pendidikan Militer**
   - List of all formal and special military training education and programs attended
   - Include year and batch/angkatan if available

4. **Riwayat Jabatan**
   - List all the Previous and current positions held by the personnel with dates.
   - Include unit/base and date range if available

5. **Riwayat Kepangkatan**
   - History of all rank promotions with effective dates

6. **Riwayat Tanda Jasa**
   - Service medals, honors, or commendations received

---

### **Output Format:**

- Use **bullet points or short labeled sections** for clarity.
- Use Markdown headings for each section.
- Ensure the headings are in **Chinese**. For example: # 全名和职级
- Ensure the content is written in **{target_lang}**.
- If a section is not available in the knowledge base, include the sentence:
  **"Informasi ini tidak tersedia dalam basis data."**

---

### **Knowledge Base Input:**
{knowledge}

"""


class LLMEvaluator:
    def __init__(self, model_hosts: Dict[str, str], target_lang: str):
        """
        Initialize LLMEvaluator with a dictionary mapping model names to their vLLM hosts
        Args:
            model_hosts: Dict mapping model names to their vLLM host URLs
        """
        self.model_hosts = model_hosts
        self.SUMMARY_SYSTEM_PROMPTs = {
            "summary": SUMMARY_SYSTEM_PROMPT,
            "compare": COMPARE_SUMMARY_SYSTEM_PROMPT
        }
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.stopwords = self._load_stopwords()
        self.target_lang = target_lang

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

    async def generate(self, query: str, knowledge: str, model_name: str, mode: str = "summary") -> Dict:
        """
        Generate response and evaluate it
        Args:
            query: User query string
            knowledge: Knowledge base content
            model_name: Name of the model to use
            mode: Mode of operation - either "summary" or "compare" (default: "summary")
        """
        if model_name not in self.model_hosts:
            raise ValueError(f"No host configured for model: {model_name}")

        if mode not in self.SUMMARY_SYSTEM_PROMPTs:
            raise ValueError(f"Invalid mode: {mode}. Must be either 'summary' or 'compare'")

        vllm_endpoint = f"{self.model_hosts[model_name][0]}/chat/completions"
        api_key = self.model_hosts[model_name][1]

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        messages = [
            {
                "role": "system",
                "content": self.SUMMARY_SYSTEM_PROMPTs[mode].format(knowledge=knowledge, target_lang=self.target_lang),
            },
            {"role": "user", "content": query},
        ]

        generation_start_time = time.time()
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": 5000,
                "temperature": 0.1,
                "top_p": 0.3,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.7,
                "seed": 20250516,
            }
            try:
                async with session.post(vllm_endpoint, json=payload, headers=headers) as response:
                    response.raise_for_status()

                    content_type = response.headers.get('content-type', '')
                    if 'application/json' not in content_type.lower():
                        raise ValueError(f"Unexpected content type: {content_type}")

                    result = await response.json()
                    if not result.get("choices"):
                        raise ValueError("No choices in response")

                    response_text = result["choices"][0]["message"]["content"]
                    generation_time = time.time() - generation_start_time
                    logger.info(f"Generation time: {generation_time:.2f} seconds")

                    evaluation = await self._evaluate_response(response_text, knowledge)
                    evaluation["generation_time"] = generation_time
                    return {"response": response_text, "evaluation": evaluation}
            except (aiohttp.ClientError, ValueError) as e:
                raise RuntimeError(f"API request failed: {str(e)}") from e

    async def _evaluate_response(self, text: str, knowledge: str, ) -> Dict:
        """
        Evaluate response quality, check for Chinese characters, translate if needed, and calculate summarization metrics
        for both original and translated text when applicable.
        """
        logger.info("Starting response evaluation")

        def calculate_metrics(text_to_evaluate: str, prefix: str = "") -> Dict:
            """Helper function to calculate metrics for a given text"""
            # Count Chinese characters and collect them
            c_chars = self.filter_chinese_chars(text_to_evaluate)
            chinese_chars = len(c_chars[0])
            total_chars = len(text_to_evaluate)
            chinese_char_percentage = (chinese_chars / total_chars) * 100 if total_chars > 0 else 0

            # Find connected substrings of Chinese characters
            connected_substrings = []
            i = 0
            while i < len(c_chars[1]):  # c_chars[1] contains positions
                # Start a new substring
                start_pos = c_chars[1][i]
                substring = c_chars[0][i]
                j = i + 1
                
                # Extend substring while characters are adjacent
                while j < len(c_chars[1]) and c_chars[1][j] == c_chars[1][j-1] + 1:
                    substring += c_chars[0][j]
                    j += 1
                
                # Get context around the substring
                context_start = max(0, start_pos - 10)
                context_end = min(len(text_to_evaluate), start_pos + len(substring) + 10)
                context = text_to_evaluate[context_start:context_end]
                
                connected_substrings.append({
                    "substring": substring,
                    "position": start_pos,
                    "length": len(substring),
                    "context": context
                })
                i = j
            
            metrics = {
                f"{prefix}chinese_char_count": len(c_chars[0]),
                f"{prefix}chinese_char_percentage": chinese_char_percentage,
                f"{prefix}chinese_chars_found": c_chars[0],
                f"{prefix}chinese_substrings": connected_substrings,
                f"{prefix}average_sentence_length": 0,
                f"{prefix}summary_metrics": {
                    "token_count": len(text_to_evaluate.split()),
                    "compression_ratio": len(text_to_evaluate.split()) / len(knowledge.split()) if knowledge else 0.0,
                    "content_density": 0.0,
                    "rouge_scores": {},
                }
            }

            # Calculate ROUGE scores
            # logger.info(f"Calculating ROUGE scores for {prefix or 'original'} text")
            scores = self.scorer.score(knowledge, text_to_evaluate)
            metrics[f"{prefix}summary_metrics"]["rouge_scores"] = {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
            logger.info(f"{prefix or 'Original'} ROUGE scores: R1={scores['rouge1'].fmeasure:.3f}, R2={scores['rouge2'].fmeasure:.3f}, RL={scores['rougeL'].fmeasure:.3f}")

            # Calculate linguistic metrics
            # logger.info(f"Calculating linguistic metrics for {prefix or 'original'} text")
            sentences = re.split(r"[.!?\n]+", text_to_evaluate)
            sentences = [s.strip() for s in sentences if s.strip()]
            if sentences:
                metrics[f"{prefix}average_sentence_length"] = sum(
                    len(s.split()) for s in sentences
                ) / len(sentences)
                logger.info(f"{prefix or 'Original'} average sentence length: {metrics[f'{prefix}average_sentence_length']:.2f} words")

                # Calculate content density using loaded stopwords
                words = text_to_evaluate.lower().split()
                content_words = [w for w in words if w not in self.stopwords]
                metrics[f"{prefix}summary_metrics"]["content_density"] = (
                    len(content_words) / len(words) if words else 0
                )
                logger.info(f"{prefix or 'Original'} content density: {metrics[f'{prefix}summary_metrics']['content_density']:.3f}")

            if chinese_chars > 0:
                logger.info(f"Found Chinese characters in {prefix or 'original'} text:")
                logger.info("Connected substrings and their context:")
                for substring_info in metrics[f"{prefix}chinese_substrings"]:
                    logger.info(
                        f"  '{substring_info['substring']}' "
                        f"(length: {substring_info['length']}) "
                        f"at position {substring_info['position']}: "
                        f"...{substring_info['context']}..."
                    )

            return metrics

        # Calculate metrics for original text
        evaluation = calculate_metrics(text)

        # If Chinese characters detected, translate and calculate metrics for translated text
        if evaluation["chinese_char_count"] > 0:
            logger.info("Starting translation process")
            for model_name, (host, api_key) in self.model_hosts.items():
                if "BGE" in model_name.upper():
                    continue
                try:
                    logger.info(f"Attempting translation with model: {model_name}")
                    translated_text, translation_time = await self._translate_text(text, model_name)
                    evaluation["translated_text"] = translated_text
                    evaluation["translation_time"] = translation_time

                    # Calculate metrics for translated text
                    translated_metrics = calculate_metrics(translated_text, "translated_")
                    evaluation.update(translated_metrics)

                    logger.info("Translation successful")
                    break  # Use the first available model
                except Exception as e:
                    logger.error(f"Translation failed for model {model_name}: {str(e)}")
                    continue

        logger.info("Evaluation completed")
        return evaluation

    def filter_chinese_chars(self, text):
        """
        Efficiently filters Chinese characters from a string and returns both the characters
        and their positions.

        Args:
            text (str): The input string to filter.

        Returns:
            tuple: (str, list) containing:
                - A string of just the Chinese characters.
                - A list of positions where Chinese characters appeared in the original string.
        """
        chinese_chars = []
        positions = []

        for i, char in enumerate(text):
            code_point = ord(char)
            # Check if the character is within any of the Chinese character blocks
            is_chinese = (
                (0x4E00 <= code_point <= 0x9FFF) or    # CJK Unified Ideographs
                (0x3400 <= code_point <= 0x4DBF) or    # CJK Unified Ideographs Extension A
                (0x20000 <= code_point <= 0x2A6DF) or  # CJK Unified Ideographs Extension B
                (0x2A700 <= code_point <= 0x2B73F) or  # CJK Unified Ideographs Extension C
                (0x2B740 <= code_point <= 0x2B81F) or  # CJK Unified Ideographs Extension D
                (0x2B820 <= code_point <= 0x2CEAF) or  # CJK Unified Ideographs Extension E
                (0x2CEB0 <= code_point <= 0x2EBEF) or  # CJK Unified Ideographs Extension F
                (0xF900 <= code_point <= 0xFAFF)       # CJK Compatibility Ideographs
            )

            if is_chinese:
                chinese_chars.append(char)
                positions.append(i)

        return ''.join(chinese_chars), positions

    def _is_cjk_char(self, char: str) -> bool:
        """
        Check if a character is Chinese
        """
        try:
            # Check if character matches these character, ignore it
            if char in "–’ →":
                return False

            return any(
                [
                    "\u4e00" <= char <= "\u9fff",  # CJK Unified Ideographs
                    "\u3400" <= char <= "\u4dbf",  # CJK Unified Ideographs Extension A
                    "\u20000" <= char <= "\u2a6df",  # CJK Unified Ideographs Extension B
                    "\u2a700" <= char <= "\u2b73f",  # CJK Unified Ideographs Extension C
                    "\u2b740" <= char <= "\u2b81f",  # CJK Unified Ideographs Extension D
                    "\u2b820" <= char <= "\u2ceaf",  # CJK Unified Ideographs Extension E
                    "\uf900" <= char <= "\ufaff",  # CJK Compatibility Ideographs
                ]
            )
        except TypeError:
            return False

    async def _translate_text(self, text: str, model_name: str) -> str:
        """
        Translate text using the specified LLM model
        Args:
            text: Text to translate
            model_name: Name of the model to use for translation
        Returns:
            Translated text
        """
        vllm_endpoint = f"{self.model_hosts[model_name][0]}/chat/completions"
        api_key = self.model_hosts[model_name][1]

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        messages = [
            {
                "role": "system",
                "content": f"You are a translator. Translate the following text to {self.target_lang}. " \
                           f"If the text is partially in non {self.target_lang}, " \
                           "only translate those parts and keep the rest unchanged. "
                           "DO NOT add any additional information or use other langauge.",
            },
            {"role": "user", "content": text},
        ]

        translation_start_time = time.time()
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": 5000,
                "temperature": 0.1,
            }
            try:
                async with session.post(vllm_endpoint, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    result = await response.json()
                    if not result.get("choices"):
                        raise ValueError("No choices in response")
                    translated_text = result["choices"][0]["message"]["content"]
                    translation_time = time.time() - translation_start_time
                    logger.info(f"Translation time: {translation_time:.2f} seconds")
                    return translated_text, translation_time
            except (aiohttp.ClientError, ValueError) as e:
                logger.error(f"Translation request failed: {str(e)}")
                raise RuntimeError(f"Translation request failed: {str(e)}") from e
