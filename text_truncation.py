import os
from transformers import AutoTokenizer, PreTrainedTokenizer
from functools import lru_cache

# Set cache directory to local project directory
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.dirname(__file__), 'model_cache')
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'model_cache')

@lru_cache()
def get_bge_tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(
        'BAAI/bge-m3',
        cache_dir=os.path.join(os.path.dirname(__file__), 'model_cache'),
        local_files_only=False
    )

@lru_cache()
def get_qwen_tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(
        'Qwen/Qwen3-Embedding-0.6B',
        cache_dir=os.path.join(os.path.dirname(__file__), 'model_cache'),
        padding_side='left',
        local_files_only=False
    )

def truncate_text_bge(text: str, max_length: int = 8192) -> str:
    """Truncate text using XLM-RoBERTa tokenizer (used by BGE models)"""
    tokenizer = get_bge_tokenizer()
    # Use a more conservative limit of 8000 tokens to account for special tokens
    actual_max_length = min(max_length, 8000)

    # First encode without truncation to get the tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > actual_max_length:
        tokens = tokens[:actual_max_length]

    # Add special tokens back
    tokens = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
    return tokenizer.decode(tokens, skip_special_tokens=True)

def truncate_text_qwen(text: str, max_length: int = 8192) -> str:
    """Truncate text using Qwen tokenizer"""
    tokenizer = get_qwen_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)
