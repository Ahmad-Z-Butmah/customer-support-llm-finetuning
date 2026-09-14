from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

OUTPUT_DIR = Path("data")

TRAIN_SIZE = 4000
VAL_SIZE = 300
TEST_SIZE = 300

SEED = 42


def format_example(example):
    prompt = (
        "You are a helpful customer support assistant.\n\n"
        f"Customer: {example['instruction']}\n\n"
        "Assistant:"
    )

    completion = example["response"]

    return {
        "prompt": prompt,
        "completion": completion,
    }


def main():
    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    dataset = dataset.shuffle(seed=SEED)

    total_needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    dataset = dataset.select(
        range(min(total_needed, len(dataset)))
    )

    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
    )

    train = dataset.select(
        range(TRAIN_SIZE)
    )

    validation = dataset.select(
        range(
            TRAIN_SIZE,
            TRAIN_SIZE + VAL_SIZE,
        )
    )

    test = dataset.select(
        range(
            TRAIN_SIZE + VAL_SIZE,
            TRAIN_SIZE + VAL_SIZE + TEST_SIZE,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_json(
        OUTPUT_DIR / "train.jsonl"
    )

    validation.to_json(
        OUTPUT_DIR / "validation.jsonl"
    )

    test.to_json(
        OUTPUT_DIR / "test.jsonl"
    )

    print()
    print(f"Train: {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test: {len(test)}")
    print()
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()