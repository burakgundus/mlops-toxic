import sys
from pathlib import Path
from typing import Any

import great_expectations as gx
import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config import load_config


def validate(df: pd.DataFrame, labels: list[str]) -> bool:
    required = ["id", "comment_text", *labels]

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("jigsaw_source")
    asset = data_source.add_dataframe_asset("train")
    batch_definition = asset.add_batch_definition_whole_dataframe("full_df")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="jigsaw_train_suite")
    for column in required:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=column))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=column))
    for label in labels:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=label, value_set=[0, 1])
        )

    result = batch.validate(suite)

    if not result.success:
        for r in result.results:
            if not r.success:
                print("FAILED:", r.expectation_config.type, r.expectation_config.kwargs)  # type: ignore[union-attr]
        return False

    print(f"All {len(result.results)} expectations passed.")
    return True


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["comment_text"] = clean["comment_text"].astype(str).str.strip()
    return clean


def build_features(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    labels = cfg["labels"]
    features_dir = Path(cfg["data"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        df["comment_text"],
        df[labels],
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=df["toxic"],
    )

    vectorizer = TfidfVectorizer(
        max_features=cfg["model"]["max_features"],
        ngram_range=tuple(cfg["model"]["ngram_range"]),
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    sparse.save_npz(features_dir / "X_train.npz", X_train)
    sparse.save_npz(features_dir / "X_val.npz", X_val)
    y_train.to_csv(features_dir / "y_train.csv", index=False)
    y_val.to_csv(features_dir / "y_val.csv", index=False)
    joblib.dump(vectorizer, features_dir / "vectorizer.joblib")
    print(f"features saved to {features_dir} (train {X_train.shape}, val {X_val.shape})")


def run(cfg: dict[str, Any]) -> None:
    df = pd.read_csv(cfg["data"]["raw_path"])
    if not validate(df, cfg["labels"]):
        raise SystemExit("data validation failed")

    clean = preprocess(df)
    out_path = Path(cfg["data"]["preprocessed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)
    print(f"preprocessed data saved to {out_path}")

    build_features(clean, cfg)


if __name__ == "__main__":
    run(load_config())
    sys.exit(0)
