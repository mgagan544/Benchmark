import os
import sys
import time
import json
import torch
from PIL import Image

MODEL_ID = sys.argv[1]
IMAGE_PATH = sys.argv[2]

print("=" * 70)
print("MODEL:", MODEL_ID)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("=" * 70)

print("\nLoading image...")
image = Image.open(IMAGE_PATH).convert("RGB")
print("Image:", image.size)

print("\nGPU:")
print("CUDA:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

start_load = time.time()

# ---------------------------------------------------------
# QWEN FAMILY
# ---------------------------------------------------------

if MODEL_ID.startswith("Qwen/"):

    from transformers import AutoProcessor

    print("\nLoading Qwen processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        local_files_only=True
    )

    if "Qwen3.5" in MODEL_ID:
        from transformers import Qwen3_5ForConditionalGeneration

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        ).eval()

    elif "Qwen3-VL" in MODEL_ID:
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        ).eval()

    elif "Qwen2.5-VL" in MODEL_ID:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        ).eval()

    elif "Qwen2-VL" in MODEL_ID:
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        ).eval()

    else:
        raise ValueError(f"Unknown Qwen model: {MODEL_ID}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": "Describe the main visible text or sign in this image."
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

# ---------------------------------------------------------
# INTERNVL FAMILY
# ---------------------------------------------------------

elif MODEL_ID.startswith("OpenGVLab/InternVL"):

    from transformers import AutoModel, AutoTokenizer

    print("\nLoading InternVL tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True
    )

    print("Loading InternVL model...")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        local_files_only=True
    ).eval()

    # InternVL uses its own image preprocessing/chat API.
    # We'll test loading first and then use the native inference
    # interface after identifying the exact model version.

    load_time = time.time() - start_load

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("MODEL:", MODEL_ID)
    print("Load time:", round(load_time, 2), "seconds")
    print("=" * 70)

    result = {
        "model": MODEL_ID,
        "status": "LOADED",
        "load_time_sec": round(load_time, 2),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    print(json.dumps(result, indent=2))
    sys.exit(0)

else:
    raise ValueError(f"Unknown model family: {MODEL_ID}")


# ---------------------------------------------------------
# QWEN GENERATION
# ---------------------------------------------------------

load_time = time.time() - start_load

print("\nModel loaded.")
print("Load time:", round(load_time, 2), "seconds")

print("\nGenerating test answer...")

start_generation = time.time()

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

generation_time = time.time() - start_generation

input_len = inputs["input_ids"].shape[1]

generated_ids = generated_ids[:, input_len:]

answer = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
print("MODEL:", MODEL_ID)
print("GPU:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Load time:", round(load_time, 2), "sec")
print("Generation time:", round(generation_time, 2), "sec")
print("\nANSWER:")
print(answer)
print("=" * 70)

result = {
    "model": MODEL_ID,
    "status": "SUCCESS",
    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "load_time_sec": round(load_time, 2),
    "generation_time_sec": round(generation_time, 2),
    "answer": answer
}

output_name = MODEL_ID.replace("/", "_") + ".json"

output_path = os.path.join(
    "/home2/gagandeep/benchmark/results",
    output_name
)

with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print("\nSaved:", output_path)
