import torch

from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import PeftModel


BASE_MODEL = (
    "Qwen/Qwen2.5-3B"
)

ADAPTER_PATH = (
    "models/customer-support-llama"
)

TEST_FILE = "data/test.jsonl"

TEST_SIZE = 10


def load_base_model():

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.float16
            ),
        )
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL
        )
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,
            quantization_config=(
                quantization_config
            ),
            device_map="auto",
        )
    )

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt,
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    text = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    return text[
        len(prompt):
    ].strip()


def main():

    test_dataset = load_dataset(
        "json",
        data_files=TEST_FILE,
        split="train",
    )

    test_dataset = test_dataset.select(
        range(
            min(
                TEST_SIZE,
                len(test_dataset),
            )
        )
    )

    print(
        "Loading base model..."
    )

    base_model, tokenizer = (
        load_base_model()
    )

    print(
        "Loading fine-tuned adapter..."
    )

    fine_tuned_model = (
        PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
        )
    )

    for index, item in enumerate(
        test_dataset
    ):

        prompt = item["prompt"]

        expected = item[
            "completion"
        ]

        base_model.disable_adapter_layers()

        base_response = generate(
            base_model,
            tokenizer,
            prompt,
        )

        fine_tuned_model.enable_adapter_layers()

        fine_tuned_response = generate(
            fine_tuned_model,
            tokenizer,
            prompt,
        )

        print()
        print("=" * 80)

        print(
            f"Example {index + 1}"
        )

        print()
        print("QUESTION:")
        print(prompt)

        print()
        print("EXPECTED:")
        print(expected)

        print()
        print("BASE LLAMA:")
        print(base_response)

        print()
        print("FINE-TUNED LLAMA:")
        print(fine_tuned_response)


if __name__ == "__main__":
    main()