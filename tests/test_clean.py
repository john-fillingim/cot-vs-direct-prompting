"""Unit tests for the answer extractor and scorer."""

import pandas as pd
import pytest

from src.clean import clean, extract_answer, is_correct


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("18", 18.0),
        ("The answer is 18.", 18.0),
        ("#### 18", 18.0),
        ("Step 1: 3 + 4 = 7\nStep 2: 7 * 2 = 14\n#### 14", 14.0),
        ("So she earns $1,234.50 in total.", 1234.5),
        ("#### $1,000", 1000.0),
        ("#### 42 dollars", 42.0),
        ("The temperature drops to -5 degrees.", -5.0),
        ("He reads pages 3-4 today", 4.0),
        ("#### 7\n\nNote: 99 was a distractor", 7.0),
        ("**Answer:** 25 apples", 25.0),
        ("\\boxed{72}", 72.0),
        ("50%", 50.0),
        ("Total: 3.", 3.0),
    ],
)
def test_extract_answer(text: str, expected: float) -> None:
    assert extract_answer(text) == expected


@pytest.mark.parametrize("text", ["", "I don't know.", "####", None])
def test_extract_answer_none(text) -> None:
    assert extract_answer(text) is None


def test_hash_without_number_falls_back_to_last_number() -> None:
    assert extract_answer("6 * 2 = 12\n#### twelve") == 12.0


def test_is_correct() -> None:
    assert is_correct(18.0, 18.0)
    assert is_correct(18.000000001, 18.0)
    assert not is_correct(18.5, 18.0)
    assert not is_correct(None, 18.0)


def test_clean_scores_and_flags_parse_failures() -> None:
    problems = pd.DataFrame({"problem_id": [1, 2], "gold": [10.0, 5.0], "n_steps": [2, 3]})
    base = {"model": "m", "model_digest": "d", "prompt_eval_count": 50, "latency_s": 1.0, "done_reason": "stop"}
    responses = pd.DataFrame([
        {**base, "problem_id": 1, "model_key": "small", "prompt_style": "cot", "response": "#### 10",
         "eval_count": 30},
        {**base, "problem_id": 2, "model_key": "small", "prompt_style": "cot", "response": "no idea",
         "eval_count": 3},
        {**base, "problem_id": 2, "model_key": "small", "prompt_style": "cot", "response": "5",
         "eval_count": 1},  # duplicate key: dropped
    ])
    out = clean(responses, problems).set_index("problem_id")
    assert len(out) == 2
    assert out.loc[1, "correct"] and not out.loc[1, "parse_failed"]
    assert not out.loc[2, "correct"] and out.loc[2, "parse_failed"]
