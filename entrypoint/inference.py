"""Batch inference entrypoint; HTTP serving lives in src.api.main."""

from src.config import load_config
from src.pipelines.inference_pipeline import run


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()
