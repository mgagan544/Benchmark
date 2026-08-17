import os
import re
import json
import math
import time
import string
import argparse
from pathlib import Path

import torch
import pandas as pd

from transformers import AutoModel, AutoTokenizer

# Transformers 5.x compatibility for older InternVL custom model
from transformers import PreTrainedModel

from qwen_vl_utils import process_vision_info

# ============================================================
# CONFIGURATION
# ============================================================

# Your local project structure.
# Change VIDEO_ROOT if necessary.
VIDEO_ROOT = Path("videos/900")

# Gemini VQA files are here:
GT_ROOT = Path("results/results")

# Model predictions/evaluation results:
BENCHMARK_ROOT = Path("benchmarks_results/results")

START_VIDEO = 901
END_VIDEO = 1000

MAX_NEW_TOKENS = 128

# Qwen video sampling.
# Keep this fixed for the benchmark.
QWEN_FPS = 1.0

# InternVL number of sampled video segments.
INTERNVL_NUM_SEGMENTS = 8

# Local Hugging Face cache.
LOCAL_FILES_ONLY = True


# ============================================================
# MODEL REGISTRY
# ============================================================

MODELS = {

    # ---------------- QWEN 2 ----------------

    "Qwen2-VL-2B": {
        "hf_id": "Qwen/Qwen2-VL-2B-Instruct",
        "family": "qwen2vl",
    },

    "Qwen2-VL-7B": {
        "hf_id": "Qwen/Qwen2-VL-7B-Instruct",
        "family": "qwen2vl",
    },

    # ---------------- QWEN 2.5 ----------------

    "Qwen2.5-VL-3B": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "qwen25vl",
    },

    "Qwen2.5-VL-7B": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "qwen25vl",
    },

    # ---------------- QWEN 3 ----------------

    "Qwen3-VL-2B": {
        "hf_id": "Qwen/Qwen3-VL-2B-Instruct",
        "family": "qwen3vl",
    },

    "Qwen3-VL-4B": {
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "family": "qwen3vl",
    },

    "Qwen3-VL-8B": {
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "family": "qwen3vl",
    },

    # ---------------- QWEN 3.5 ----------------

    "Qwen3.5-4B": {
        "hf_id": "Qwen/Qwen3.5-4B",
        "family": "qwen35",
    },

    "Qwen3.5-9B": {
        "hf_id": "Qwen/Qwen3.5-9B",
        "family": "qwen35",
    },

    # ---------------- INTERNVL 2 ----------------

    "InternVL2-1B": {
        "hf_id": "OpenGVLab/InternVL2-1B",
        "family": "internvl",
    },

    "InternVL2-2B": {
        "hf_id": "OpenGVLab/InternVL2-2B",
        "family": "internvl",
    },

    "InternVL2-4B": {
        "hf_id": "OpenGVLab/InternVL2-4B",
        "family": "internvl",
    },

    "InternVL2-8B": {
        "hf_id": "OpenGVLab/InternVL2-8B",
        "family": "internvl",
    },

    # ---------------- INTERNVL 2.5 ----------------

    "InternVL2.5-1B": {
        "hf_id": "OpenGVLab/InternVL2_5-1B",
        "family": "internvl",
    },

    "InternVL2.5-2B": {
        "hf_id": "OpenGVLab/InternVL2_5-2B",
        "family": "internvl",
    },

    "InternVL2.5-4B": {
        "hf_id": "OpenGVLab/InternVL2_5-4B",
        "family": "internvl",
    },

    "InternVL2.5-8B": {
        "hf_id": "OpenGVLab/InternVL2_5-8B",
        "family": "internvl",
    },

    # ---------------- INTERNVL 3 ----------------

    "InternVL3-1B": {
        "hf_id": "OpenGVLab/InternVL3-1B",
        "family": "internvl",
    },

    "InternVL3-2B": {
        "hf_id": "OpenGVLab/InternVL3-2B",
        "family": "internvl",
    },

    "InternVL3-8B": {
        "hf_id": "OpenGVLab/InternVL3-8B",
        "family": "internvl",
    },

    # ---------------- INTERNVL 3.5 ----------------

    "InternVL3.5-4B": {
        "hf_id": "OpenGVLab/InternVL3_5-4B",
        "family": "internvl",
    },

    "InternVL3.5-8B": {
        "hf_id": "OpenGVLab/InternVL3_5-8B",
        "family": "internvl",
    },
}


# ============================================================
# TEXT NORMALIZATION / METRICS
# ============================================================

def norm(s):
    s = str(s).lower()

    s = re.sub(r"[.\*?]", "", s)

    s = re.sub(
        "[" + re.escape(string.punctuation) + "]",
        " ",
        s
    )

    return " ".join(s.split())


def levenshtein(a, b):

    m, n = len(a), len(b)

    dp = list(range(n + 1))

    for i in range(1, m + 1):

        prev = dp[:]
        dp[0] = i

        for j in range(1, n + 1):

            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[j] = min(
                prev[j] + 1,
                dp[j - 1] + 1,
                prev[j - 1] + cost
            )

    return dp[n]


def nls(pred, gt):

    pred = norm(pred)
    gt = norm(gt)

    maximum = max(len(pred), len(gt))

    if maximum == 0:
        return 1.0

    return max(
        0.0,
        1.0 - levenshtein(pred, gt) / maximum
    )


def calculate_anls_star(pred, gt):

    base = nls(pred, gt)

    pred_tokens = set(norm(pred).split())
    gt_tokens = set(norm(gt).split())

    overlap = (
        1.0
        if not gt_tokens
        else len(pred_tokens & gt_tokens) / len(gt_tokens)
    )

    return 0.7 * base + 0.3 * overlap


def bleu1(pred, gt):

    pred_tokens = norm(pred).split()
    gt_tokens = norm(gt).split()

    if not pred_tokens:
        return 0.0

    gt_counts = {}

    for token in gt_tokens:
        gt_counts[token] = gt_counts.get(token, 0) + 1

    pred_counts = {}

    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    overlap = 0

    for token, count in pred_counts.items():
        overlap += min(
            count,
            gt_counts.get(token, 0)
        )

    precision = overlap / len(pred_tokens)

    if len(pred_tokens) >= len(gt_tokens):
        bp = 1.0
    else:
        bp = math.exp(
            1 - len(gt_tokens) / max(1, len(pred_tokens))
        )

    return bp * precision


# ============================================================
# BERTSCORE
# ============================================================

_BERT_SCORER = None


def bertscore_f1(pred, gt):

    global _BERT_SCORER

    if _BERT_SCORER is None:

        from bert_score import BERTScorer

        print("\nLoading BERTScore model...")

        _BERT_SCORER = BERTScorer(
            lang="en",
            rescale_with_baseline=False,
        )

    _, _, f1 = _BERT_SCORER.score(
        [str(pred)],
        [str(gt)]
    )

    return float(f1[0])


# ============================================================
# VIDEO DISCOVERY
# ============================================================
def find_video(video_id):

    video_path = (
        VIDEO_ROOT / f"{video_id}.mp4"
    ).resolve()

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    return video_path

# ============================================================
# GEMINI VQA LOADING
# ============================================================

def load_ground_truth(video_id):

    path = (
        GT_ROOT
        / f"Video_{video_id}"
        / f"VQA_GEMINI_ALL_CATEGORIES_{video_id}.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Gemini VQA not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):

        if not data:
            raise ValueError(f"Empty JSON: {path}")

        data = data[0]

    if "qa_pairs" not in data:

        raise ValueError(
            f"'qa_pairs' not found in {path}"
        )

    return data["qa_pairs"]


# ============================================================
# QWEN VIDEO RUNNER
# ============================================================

class QwenRunner:

    def __init__(self, model_id, family):

        from transformers import AutoProcessor

        self.model_id = model_id
        self.family = family

        print(f"\nLoading {model_id}")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=LOCAL_FILES_ONLY
        )

        if family == "qwen25vl":

            from transformers import (
                Qwen2_5_VLForConditionalGeneration
            )

            self.model = (
                Qwen2_5_VLForConditionalGeneration
                .from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    local_files_only=LOCAL_FILES_ONLY,
                )
                .eval()
            )

        else:
            raise ValueError(
                f"Unsupported Qwen family: {family}"
            )

        print(f"Loaded: {model_id}")

    def answer(self, video_path, question):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "fps": QWEN_FPS,
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ]

        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = (
            process_vision_info(
                messages,
                return_video_kwargs=True,
            )
        )
        if "fps" in video_kwargs:
            if isinstance(video_kwargs["fps"], list):
                video_kwargs["fps"] = video_kwargs["fps"][0]

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        inputs = {
            k: v.to(self.model.device)
            if hasattr(v, "to")
            else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids
            in zip(
                inputs["input_ids"],
                output_ids
            )
        ]

        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return answer.strip()
# ============================================================
# INTERNVL VIDEO RUNNER
# ============================================================
class InternVLRunner:

    def __init__(self, model_id):

        import numpy as np

        from transformers import (
            AutoModel,
            AutoTokenizer,
            PreTrainedModel,
        )

        self.model_id = model_id

        print(f"\nLoading {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=LOCAL_FILES_ONLY,
        )

        # --------------------------------------------------------
        # Transformers 5.x compatibility
        # --------------------------------------------------------
        #
        # Older InternVL custom models expose:
        #     _tied_weights_keys
        #
        # Transformers 5.x expects:
        #     all_tied_weights_keys
        #
        # This must be patched BEFORE from_pretrained(),
        # because Transformers accesses it during loading.
        # --------------------------------------------------------

        if not hasattr(
            PreTrainedModel,
            "all_tied_weights_keys"
        ):
            PreTrainedModel.all_tied_weights_keys = {}

        # --------------------------------------------------------
        # Load InternVL
        # --------------------------------------------------------

        self.model = AutoModel.from_pretrained(
            model_id,
            dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=LOCAL_FILES_ONLY,
        ).eval().cuda()

        self.generation_config = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
        }

        print(f"Loaded: {model_id}")
        
    @staticmethod
    def get_index(
        bound,
        fps,
        max_frame,
        first_idx=0,
        num_segments=8
    ):

        if bound:

            start, end = bound

        else:

            start, end = -100000, 100000

        start_idx = max(
            first_idx,
            round(start * fps)
        )

        end_idx = min(
            round(end * fps),
            max_frame
        )

        seg_size = float(
            end_idx - start_idx
        ) / num_segments

        import numpy as np

        frame_indices = np.array([
            int(
                start_idx
                + (seg_size / 2)
                + np.round(seg_size * idx)
            )
            for idx in range(num_segments)
        ])

        return frame_indices

    def load_video(self, video_path):

        from decord import VideoReader, cpu
        from PIL import Image
        import torch
        import torchvision.transforms as T

        # InternVL's official video examples use
        # Decord + temporal segment sampling.
        #
        # The benchmark input remains the .mp4.
        # We are NOT creating a separate frame dataset.

        vr = VideoReader(
            str(video_path),
            ctx=cpu(0),
            num_threads=1
        )

        max_frame = len(vr) - 1

        fps = float(vr.get_avg_fps())

        frame_indices = self.get_index(
            None,
            fps,
            max_frame,
            first_idx=0,
            num_segments=INTERNVL_NUM_SEGMENTS
        )

        # Import the official preprocessing utilities
        # exposed by InternVL's remote-code implementation.
        try:
            import sys

            INTERNVL_ROOT = "/home2/gagandeep/InternVL/internvl_chat"

            if INTERNVL_ROOT not in sys.path:
                sys.path.insert(0, INTERNVL_ROOT)

            from internvl.train.dataset import (
                build_transform,
                dynamic_preprocess
            )

        except ImportError:

            # Fallback for installations where the helpers
            # are exposed through the remote model package.
            try:
                from dynamic_preprocess import (
                    build_transform,
                    dynamic_preprocess
                )
            except ImportError:

                raise RuntimeError(
                    "Could not import InternVL video "
                    "preprocessing utilities. "
                    "The model loaded, but its video "
                    "preprocessing package is missing."
                )

        transform = build_transform(
            is_train=False,
            input_size=448
        )

        pixel_values_list = []
        num_patches_list = []

        for frame_index in frame_indices:

            img = Image.fromarray(
                vr[frame_index].asnumpy()
            ).convert("RGB")

            img = dynamic_preprocess(
                img,
                image_size=448,
                use_thumbnail=True,
                max_num=1
            )

            pixel_values = [
                transform(tile)
                for tile in img
            ]

            pixel_values = torch.stack(
                pixel_values
            )

            num_patches_list.append(
                pixel_values.shape[0]
            )

            pixel_values_list.append(
                pixel_values
            )

        pixel_values = torch.cat(
            pixel_values_list
        )

        return pixel_values

    def answer(self, video_path, question):

        # This is the official InternVL-style video path:
        #
        # MP4 → temporal sampling → visual tensor
        # → model.chat()
        #
        # The user question remains unchanged.

        pixel_values = self.load_video(
            video_path
        )

        device = next(
            self.model.parameters()
        ).device

        pixel_values = pixel_values.to(
            device=device,
            dtype=torch.float16
        )

        video_prefix = "".join(
            [
                f"Frame{i + 1}: <image>\n"
                for i in range(INTERNVL_NUM_SEGMENTS)
            ]
        )

        prompt = (
            video_prefix
            + question
        )

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            self.generation_config,
            num_patches_list=[
                1
            ] * INTERNVL_NUM_SEGMENTS,
            history=None,
            return_history=False,
        )

        if isinstance(response, tuple):
            response = response[0]

        return str(response).strip()


# ============================================================
# MODEL FACTORY
# ============================================================

def create_runner(model_name):

    if model_name not in MODELS:

        raise ValueError(
            f"Unknown model: {model_name}\n\n"
            f"Available models:\n"
            + "\n".join(MODELS.keys())
        )

    config = MODELS[model_name]

    if config["family"].startswith("qwen"):

        return QwenRunner(
            config["hf_id"],
            config["family"]
        )

    elif config["family"] == "internvl":

        return InternVLRunner(
            config["hf_id"]
        )

    raise ValueError(
        f"Unsupported model family: "
        f"{config['family']}"
    )


# ============================================================
# EVALUATE ONE VIDEO
# ============================================================

def evaluate_video(
    runner,
    model_name,
    video_id,
    video_path,
    qa_pairs,
    model_output_dir,
):

    predictions = []

    metric_rows = []

    print(
        f"\n{'=' * 70}\n"
        f"VIDEO {video_id}\n"
        f"{video_path}\n"
        f"Questions: {len(qa_pairs)}\n"
        f"{'=' * 70}"
    )

    for index, qa in enumerate(qa_pairs, start=1):

        question = qa.get("question", "")
        ground_truth = qa.get("answer", "")

        category_id = qa.get(
            "category_id",
            ""
        )

        category_name = qa.get(
            "category_name",
            ""
        )

        print(
            f"\n[{index}/{len(qa_pairs)}]"
            f" Category {category_id}: "
            f"{category_name}"
        )

        print(
            f"Question: {question}"
        )

        start = time.time()

        try:

            prediction = runner.answer(
                video_path,
                question
            )

            inference_time = (
                time.time() - start
            )

            error = ""

        except Exception as e:

            prediction = ""
            inference_time = (
                time.time() - start
            )

            error = repr(e)

            print(
                f"ERROR: {error}"
            )

        print(
            f"Prediction: {prediction}"
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        if error:

            anls = 0.0
            anls_star = 0.0
            bleu = 0.0
            bert_f1 = 0.0

        else:

            anls = nls(
                prediction,
                ground_truth
            )

            anls_star_score = calculate_anls_star(
                prediction,
                ground_truth
            )

            bleu = bleu1(
                prediction,
                ground_truth
            )

            try:

                bert_f1 = bertscore_f1(
                    prediction,
                    ground_truth
                )

            except Exception as e:

                print(
                    f"BERTScore ERROR: {e}"
                )

                bert_f1 = float("nan")

        predictions.append({
            "category_id": category_id,
            "category_name": category_name,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "inference_time_sec": round(
                inference_time,
                3
            ),
            "error": error,
        })

        metric_rows.append({
            "video_id": video_id,
            "category_id": category_id,
            "category_name": category_name,
            "ANLS": anls,
            "ANLS_star": anls_star_score,
            "BLEU_1": bleu,
            "BERTScore_F1": bert_f1,
            "inference_time_sec": inference_time,
        })

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    output_json = (
        model_output_dir
        / f"VQA_{model_name}_{video_id}.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "video_id": video_id,
                "model": model_name,
                "video": str(video_path),
                "qa_pairs": predictions,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    df = pd.DataFrame(metric_rows)

    csv_path = (
        model_output_dir
        / f"eval_results_{video_id}.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nSaved predictions: {output_json}"
    )

    print(
        f"Saved metrics: {csv_path}"
    )

    return metric_rows


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Video VQA benchmark using fixed "
            "Gemini-generated questions."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help="Model to benchmark."
    )

    parser.add_argument(
        "--start",
        type=int,
        default=START_VIDEO
    )

    parser.add_argument(
        "--end",
        type=int,
        default=END_VIDEO
    )

    args = parser.parse_args()

    model_name = args.model

    model_output_dir = (
        BENCHMARK_ROOT
        / model_name
    )

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print("VIDEO VQA BENCHMARK")
    print("=" * 70)

    print(f"Model       : {model_name}")
    print(
        f"HF checkpoint: "
        f"{MODELS[model_name]['hf_id']}"
    )

    print(
        f"Video range : "
        f"{args.start} - {args.end}"
    )

    print(
        f"GT root     : {GT_ROOT}"
    )

    print(
        f"Video root  : {VIDEO_ROOT}"
    )

    print(
        f"Output      : {model_output_dir}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load model ONCE
    # --------------------------------------------------------

    runner = create_runner(
        model_name
    )

    all_metric_rows = []

    # --------------------------------------------------------
    # Process videos
    # --------------------------------------------------------

    for video_id in range(
        args.start,
        args.end + 1
    ):

        try:

            video_path = find_video(
                video_id
            )

            qa_pairs = load_ground_truth(
                video_id
            )

            rows = evaluate_video(
                runner=runner,
                model_name=model_name,
                video_id=video_id,
                video_path=video_path,
                qa_pairs=qa_pairs,
                model_output_dir=model_output_dir,
            )

            all_metric_rows.extend(rows)

        except Exception as e:

            print(
                f"\nFAILED Video_{video_id}: "
                f"{repr(e)}"
            )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    if all_metric_rows:

        all_df = pd.DataFrame(
            all_metric_rows
        )

        summary = (
            all_df[
                [
                    "ANLS",
                    "ANLS_star",
                    "BLEU_1",
                    "BERTScore_F1",
                    "inference_time_sec",
                ]
            ]
            .mean(numeric_only=True)
        )

        summary_path = (
            model_output_dir
            / "summary.csv"
        )

        summary_df = pd.DataFrame([
            {
                "model": model_name,
                "videos_evaluated": (
                    all_df["video_id"].nunique()
                ),
                "questions_evaluated": len(
                    all_df
                ),
                "ANLS": summary["ANLS"],
                "ANLS_star": summary["ANLS_star"],
                "BLEU_1": summary["BLEU_1"],
                "BERTScore_F1": (
                    summary["BERTScore_F1"]
                ),
                "mean_inference_time_sec": (
                    summary["inference_time_sec"]
                ),
            }
        ])

        summary_df.to_csv(
            summary_path,
            index=False
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "FINAL SUMMARY"
        )

        print(
            summary_df.to_string(
                index=False
            )
        )

        print(
            f"\nSaved: {summary_path}"
        )

    print(
        "\nBenchmark complete."
    )


if __name__ == "__main__":
    main()