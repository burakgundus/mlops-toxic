import argparse

from src.config import load_config
from src.pipelines import inference_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toxic comment inference.")
    parser.add_argument("--input", default=None, help="CSV with a comment_text column")
    parser.add_argument("--stage", default="Staging", help="Registry stage to load")
    args = parser.parse_args()
    inference_pipeline.run(load_config(), input_path=args.input, stage=args.stage)


if __name__ == "__main__":
    main()
