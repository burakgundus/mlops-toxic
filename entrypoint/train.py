from src.config import load_config
from src.pipelines import feature_eng_pipeline, training_pipeline


def main() -> None:
    cfg = load_config()
    feature_eng_pipeline.run(cfg)
    training_pipeline.run(cfg)


if __name__ == "__main__":
    main()
