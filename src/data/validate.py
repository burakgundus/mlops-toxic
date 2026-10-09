import sys

import great_expectations as gx
import pandas as pd

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
REQUIRED_COLUMNS = ["id", "comment_text", *LABEL_COLUMNS]


def validate_dataset(path: str = "data/raw/train.csv") -> bool:
    df = pd.read_csv(path)

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("jigsaw_source")
    asset = data_source.add_dataframe_asset("train")
    batch_definition = asset.add_batch_definition_whole_dataframe("full_train")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="jigsaw_train_suite")
    for column in REQUIRED_COLUMNS:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=column))

    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="comment_text"))
    for label in LABEL_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=label, value_set=[0, 1])
        )

    result = batch.validate(suite)

    if not result.success:
        for r in result.results:
            if not r.success:
                print("FAILED:", r.expectation_config.type, r.expectation_config.kwargs)
        return False

    print(f"All {len(result.results)} expectations passed.")
    return True


if __name__ == "__main__":
    sys.exit(0 if validate_dataset() else 1)
