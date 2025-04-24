import pytest
from src.llm_evaluator import LLMEvaluator

@pytest.fixture
def evaluator():
    return LLMEvaluator("http://localhost:8000")

@pytest.mark.asyncio
async def test_chinese_character_detection(evaluator):
    # Test with mixed text containing Chinese characters
    test_text = "This is English 这是中文 More English"
    evaluation = evaluator._evaluate_response(test_text)

    assert evaluation["chinese_char_count"] == 4
    assert abs(evaluation["chinese_char_percentage"] - (4/len(test_text))*100) < 0.01
    assert not evaluation["has_citations"]

@pytest.mark.asyncio
async def test_citation_detection(evaluator):
    test_text = "According to Document: doc1, Fragment: 2, this is a cited response."
    evaluation = evaluator._evaluate_response(test_text)

    assert evaluation["has_citations"]
    assert evaluation["chinese_char_count"] == 0

@pytest.mark.asyncio
async def test_sentence_length(evaluator):
    test_text = "This is a short sentence. This is another sentence with more words."
    evaluation = evaluator._evaluate_response(test_text)

    expected_avg = ((len("This is a short sentence".split()) +
                    len("This is another sentence with more words".split())) / 2)
    assert abs(evaluation["average_sentence_length"] - expected_avg) < 0.01