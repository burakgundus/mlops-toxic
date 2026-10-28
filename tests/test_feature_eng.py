import pandas as pd

from src.config import load_config
from src.pipelines.feature_eng_pipeline import preprocess, validate

LABELS = load_config("local")["labels"]


def make_df() -> pd.DataFrame:
    data = {
        "id": ["a", "b", "c", "d"],
        "comment_text": ["hello world", "you are bad", "nice work", "  spaced  "],
    }
    for label in LABELS:
        data[label] = [0, 1, 0, 0]
    return pd.DataFrame(data)


def test_valid_data_passes():
    assert validate(make_df(), LABELS) is True


def test_label_outside_zero_one_fails():
    df = make_df()
    df.loc[0, "toxic"] = 2
    assert validate(df, LABELS) is False


def test_null_comment_fails():
    df = make_df()
    df.loc[1, "comment_text"] = None
    assert validate(df, LABELS) is False


def test_missing_column_fails():
    df = make_df().drop(columns=["insult"])
    assert validate(df, LABELS) is False


def test_preprocess_strips_whitespace_and_keeps_input():
    df = make_df()
    clean = preprocess(df)
    assert clean.loc[3, "comment_text"] == "spaced"
    assert df.loc[3, "comment_text"] == "  spaced  "
