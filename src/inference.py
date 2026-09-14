import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import PeftModel


BASE_MODEL = (
    "meta-llama/Llama-3.2-3B"
)

ADAPTER_PATH = (
    "models/customer-support-llama"
)


def load_model():

    use_bf16 = (
        torch.cuda.is_available()
        and
        torch.cuda.get_device_capability()[0]
        >= 8
    )

    dtype = (
        torch.bfloat16
        if use_bf16
        else torch.float16
    )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
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

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,
            quantization_config=(
                quantization_config
            ),
            device_map="auto",
        )
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model.eval()

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question,
):

    prompt = (
        "You are a helpful customer "
        "support assistant.\n\n"
        f"Customer: {question}\n\n"
        "Assistant:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )

    generated = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    response = generated[
        len(prompt):
    ].strip()

    return response


def main():

    model, tokenizer = load_model()

    print(
        "\nCustomer Support LLM"
    )

    print(
        "Type 'exit' to stop.\n"
    )

    while True:

        question = input(
            "Customer: "
        )

        if question.lower() == "exit":
            break

        response = generate_response(
            model,
            tokenizer,
            question,
        )

        print(
            f"\nAssistant: {response}\n"
        )


if __name__ == "__main__":
    main()