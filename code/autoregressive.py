from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from airllm import AutoModel  # type: ignore[import-not-found]

    AIRLLM_AVAILABLE = True
except Exception:
    AIRLLM_AVAILABLE = False

try:
    from .data import load_dataset
    from .config import MODELS
    from .models import load_model_and_tokenizer
    from .paths import FIGS_DIR, RESULTS_DIR, resolve_data_path
except ImportError:
    from data import load_dataset
    from config import MODELS
    from models import load_model_and_tokenizer
    from paths import FIGS_DIR, RESULTS_DIR, resolve_data_path


HIDDEN_STATES_DIR = FIGS_DIR / "hidden_states_autoregressive"
OUTPUT_DIR = FIGS_DIR

HIDDEN_STATES_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def _generate_synthetic_dataset(n_samples: int, seed: int) -> Tuple[List[str], np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    n_fact = max(1, n_samples // 2)
    n_hall = max(1, n_samples - n_fact)

    factual_prompts = [f"Prompt {i}: Provide a factual answer." for i in range(n_fact)]
    hall_prompts = [f"Prompt {i}: Provide a speculative answer." for i in range(n_hall)]
    prompts = factual_prompts + hall_prompts

    texts = [f"Factual text {i}" for i in range(n_fact)] + [f"Hallucinated text {i}" for i in range(n_hall)]
    labels = np.array([0] * n_fact + [1] * n_hall, dtype=np.int64)

    order = rng.permutation(len(labels))
    texts = [texts[i] for i in order]
    prompts = [prompts[i] for i in order]
    labels = labels[order]
    return texts, labels, prompts


def _load_halueval_qa_autoregressive_inputs(
    n_samples: int,
    seed: int,
    data_dir: Path | None,
) -> Tuple[List[str], List[List[str]]]:
    data_path = resolve_data_path(
        "HaluEval/data/qa_data.json",
        data_root=Path(data_dir) if data_dir else None,
    )
    if not data_path.exists():
        raise FileNotFoundError(f"HaluEval QA data not found at {data_path}")

    rows: List[Tuple[str, List[str]]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = str(item.get("question", "")).strip()
            knowledge = str(item.get("knowledge", "")).strip()
            right_answer_raw = item.get("right_answer", "")
            if isinstance(right_answer_raw, list):
                right_answers = [str(x).strip() for x in right_answer_raw if str(x).strip()]
            else:
                right = str(right_answer_raw).strip()
                right_answers = [right] if right else []

            if not question or not right_answers:
                continue

            # Autoregressive validation prompt uses only the question to avoid teacher-forcing leakage.
            _ = knowledge
            prompt = f"Question: {question}\nAnswer:"
            rows.append((prompt, right_answers))

    if not rows:
        raise ValueError("No valid HaluEval QA rows found for autoregressive evaluation.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    take = min(int(n_samples), len(rows))

    prompts: List[str] = []
    references: List[List[str]] = []
    for idx in order[:take]:
        p, r = rows[idx]
        prompts.append(p)
        references.append(r)

    return prompts, references


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def _token_f1(prediction: str, reference: str) -> float:
    p_tokens = prediction.split()
    r_tokens = reference.split()
    if not p_tokens or not r_tokens:
        return 0.0

    p_set = set(p_tokens)
    r_set = set(r_tokens)
    overlap = len(p_set & r_set)
    if overlap == 0:
        return 0.0

    precision = overlap / max(1, len(p_set))
    recall = overlap / max(1, len(r_set))
    return float((2.0 * precision * recall) / (precision + recall + 1e-8))


def _is_halueval_prediction_correct(prediction: str, references: List[str], f1_threshold: float = 0.75) -> bool:
    pred_norm = _normalize_text(prediction)
    if not pred_norm or not references:
        return False

    for reference in references:
        ref_norm = _normalize_text(reference)
        if not ref_norm:
            continue
        if pred_norm == ref_norm:
            return True
        if ref_norm in pred_norm or pred_norm in ref_norm:
            return True
        if _token_f1(pred_norm, ref_norm) >= f1_threshold:
            return True

    return False


def _extract_hidden_from_generation_output(
    generation_output,
    sample_idx: int = 0,
    keep_full_trajectory: bool = False,
) -> np.ndarray:
    # Extract per-token layer hidden states as (n_tokens, n_layers, hidden_dim)
    hs_steps = getattr(generation_output, "hidden_states", None)
    if not hs_steps:
        return np.array([])

    collected_steps: List[np.ndarray] = []
    for step_hs in hs_steps:
        if step_hs is None:
            continue
        if isinstance(step_hs, tuple):
            layer_vectors = []
            for layer_tensor in step_hs:
                if layer_tensor is None:
                    continue
                # bfloat16 tensors cannot be converted to NumPy directly on some builds.
                layer_vectors.append(layer_tensor[sample_idx, -1, :].detach().float().cpu().numpy())
            if layer_vectors:
                collected_steps.append(np.stack(layer_vectors, axis=0))

    if not collected_steps:
        return np.array([])

    if keep_full_trajectory:
        return np.stack(collected_steps, axis=0)

    # Most downstream metrics only need the final generated token state.
    return np.expand_dims(collected_steps[-1], axis=0)


def extract_autoregressive_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    batch_size: int = 1,
    backend: str = "auto",
    prompt_max_length: int = 256,
    keep_full_trajectory: bool = False,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    # Extract hidden states during autoregressive generation
    # Returns per-sample arrays with shape (n_generated_tokens, n_layers, hidden_dim)
    hidden_states_list: List[np.ndarray] = []
    generated_texts: List[str] = []
    generated_completions: List[str] = []

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    if original_padding_side != "left":
        tokenizer.padding_side = "left"

    resolved_backend = backend
    if backend == "auto":
        resolved_backend = getattr(model, "_llmh_backend", "transformers")

    iterator = tqdm(prompts, desc="Autoregressive generation") if verbose else prompts

    if resolved_backend == "airllm":
        if not AIRLLM_AVAILABLE:
            raise RuntimeError("AirLLM backend requested but airllm is not installed.")

        for prompt in iterator:
            input_tokens = tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=False,
                truncation=True,
                max_length=prompt_max_length,
                padding=False,
            )
            try:
                generation_output = model.generate(
                    input_tokens["input_ids"],
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
                prompt_len = int(input_tokens["input_ids"].shape[1])
                sequence = generation_output.sequences[0]
                completion_ids = sequence[prompt_len:]
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                generated_completions.append(completion_text)
                generated_texts.append(tokenizer.decode(sequence, skip_special_tokens=True))
                hidden_states_list.append(
                    _extract_hidden_from_generation_output(
                        generation_output,
                        sample_idx=0,
                        keep_full_trajectory=keep_full_trajectory,
                    )
                )
            except Exception:
                generated_texts.append("")
                generated_completions.append("")
                hidden_states_list.append(np.array([]))
            finally:
                try:
                    del generation_output
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        tokenizer.padding_side = original_padding_side
        return hidden_states_list, generated_texts, generated_completions

    # Transformers path (CPU-first compatible)
    batch_size = max(1, int(batch_size))

    model.eval()
    device = next(model.parameters()).device

    base_iterator = range(0, len(prompts), batch_size)
    iterator = tqdm(base_iterator, desc="Autoregressive generation") if verbose else base_iterator

    try:
        with torch.inference_mode():
            for step_idx, start in enumerate(iterator):
                batch_prompts = prompts[start : start + batch_size]
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=prompt_max_length,
                ).to(device)

                try:
                    generation_output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        # Keep greedy decoding while neutralizing model-level sampling defaults.
                        temperature=1.0,
                        top_p=1.0,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    sequences = generation_output.sequences
                    for row_idx in range(len(batch_prompts)):
                        prompt_len = int(inputs["attention_mask"][row_idx].sum().item())
                        sequence = sequences[row_idx]
                        completion_ids = sequence[prompt_len:]
                        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                        generated_completions.append(completion_text)
                        generated_texts.append(tokenizer.decode(sequence, skip_special_tokens=True))
                        hidden_states_list.append(
                            _extract_hidden_from_generation_output(
                                generation_output,
                                sample_idx=row_idx,
                                keep_full_trajectory=keep_full_trajectory,
                            )
                        )
                except Exception:
                    # Retry one-by-one so a single failing sample does not drop the whole batch.
                    for prompt in batch_prompts:
                        single_inputs = tokenizer(
                            [prompt],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=prompt_max_length,
                        ).to(device)
                        try:
                            single_output = model.generate(
                                **single_inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                # Keep greedy decoding while neutralizing model-level sampling defaults.
                                temperature=1.0,
                                top_p=1.0,
                                use_cache=True,
                                return_dict_in_generate=True,
                                output_hidden_states=True,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                            sequence = single_output.sequences[0]
                            prompt_len = int(single_inputs["attention_mask"][0].sum().item())
                            completion_ids = sequence[prompt_len:]
                            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                            generated_completions.append(completion_text)
                            generated_texts.append(tokenizer.decode(sequence, skip_special_tokens=True))
                            hidden_states_list.append(
                                _extract_hidden_from_generation_output(
                                    single_output,
                                    sample_idx=0,
                                    keep_full_trajectory=keep_full_trajectory,
                                )
                            )
                        except Exception:
                            generated_texts.append("")
                            generated_completions.append("")
                            hidden_states_list.append(np.array([]))
                        finally:
                            try:
                                del single_output
                            except Exception:
                                pass
                            del single_inputs
                finally:
                    try:
                        del generation_output
                    except Exception:
                        pass
                    del inputs
                    if device.type == "cuda" and (step_idx + 1) % 2 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = original_padding_side

    return hidden_states_list, generated_texts, generated_completions


def run_autoregressive_experiments(
    model_name: str = "llama-3.2-1b",
    dataset_name: str = "halueval_qa",
    n_samples: int = 100,
    max_new_tokens: int = 30,
    batch_size: int = 8,
    prompt_max_length: int = 256,
    compression: str = "4bit",
    seed: int = 42,
    backend: str = "auto",
    dry_run: bool = False,
    data_dir: str | None = None,
    keep_full_trajectory: bool = False,
) -> Dict[str, object]:
    # Run autoregressive hidden-state extraction and save artifacts
    model_config = MODELS.get(model_name)
    if model_config is None:
        raise ValueError(f"Unknown model: {model_name}")

    label_source = "dataset_binary_labels"
    label_match_threshold = None
    references: List[List[str]] = []

    if dry_run:
        texts, labels, prompts = _generate_synthetic_dataset(n_samples, seed)
    elif dataset_name == "halueval_qa":
        prompts, references = _load_halueval_qa_autoregressive_inputs(
            n_samples=n_samples,
            seed=seed,
            data_dir=Path(data_dir) if data_dir else None,
        )
        texts = [ref_list[0] if ref_list else "" for ref_list in references]
        labels = np.zeros(len(prompts), dtype=np.int64)
        label_source = "generated_correctness_match"
        label_match_threshold = 0.75
    else:
        root_override = Path(data_dir) if data_dir else None
        texts, labels, prompts = load_dataset(
            dataset_name,
            n_samples=n_samples,
            seed=seed,
            data_dir=root_override,
        )

    if dry_run:
        rng = np.random.default_rng(seed)
        hidden_states_list = []
        generated_texts = []
        generated_completions = []
        n_layers = max(4, model_config.num_layers // 4)
        hidden_dim = min(512, model_config.hidden_dim)
        for _ in prompts:
            token_count = int(rng.integers(low=8, high=20))
            hidden_states_list.append(rng.standard_normal((token_count, n_layers, hidden_dim)).astype(np.float32))
            generated_texts.append("[DRY-RUN] synthetic generation output")
            generated_completions.append("[DRY-RUN] synthetic generation output")
        resolved_backend = "dry-run"
    else:
        model = None
        tokenizer = None
        # Optional compression hint for AirLLM path only.
        if backend == "airllm" and compression:
            import os

            os.environ["LLMH_USE_AIRLLM"] = "1"
        try:
            model, tokenizer = load_model_and_tokenizer(model_config, backend=backend)
            resolved_backend = getattr(model, "_llmh_backend", backend)
            hidden_states_list, generated_texts, generated_completions = extract_autoregressive_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                backend=resolved_backend,
                prompt_max_length=prompt_max_length,
                keep_full_trajectory=keep_full_trajectory,
                verbose=True,
            )
        finally:
            try:
                del model
            except Exception:
                pass
            try:
                del tokenizer
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    # Re-label based on model output correctness for HaluEval QA
    if (not dry_run) and dataset_name == "halueval_qa":
        labels = np.array(
            [
                0 if _is_halueval_prediction_correct(prediction, reference_candidates, f1_threshold=0.75) else 1
                for prediction, reference_candidates in zip(generated_completions, references)
            ],
            dtype=np.int64,
        )

    output_file = HIDDEN_STATES_DIR / f"{model_name}_{dataset_name}_autoregressive.npz"
    np.savez(
        output_file,
        hidden_states=np.array(hidden_states_list, dtype=object),
        labels=labels,
        prompts=np.array(prompts),
        generated_texts=np.array(generated_texts),
        generated_completions=np.array(generated_completions),
        original_texts=np.array(texts),
        reference_answers=np.array(references, dtype=object),
        label_source=np.array([label_source]),
    )

    n_valid_hidden = int(sum(1 for hs in hidden_states_list if hasattr(hs, "size") and hs.size > 0))

    metadata = {
        "model": model_config.name,
        "dataset": dataset_name,
        "requested_n_samples": int(n_samples),
        "n_samples": len(texts),
        "n_factual": int(np.sum(labels == 0)),
        "n_hallucinated": int(np.sum(labels == 1)),
        "n_valid_hidden_states": n_valid_hidden,
        "max_new_tokens": max_new_tokens,
        "batch_size": int(batch_size),
        "prompt_max_length": int(prompt_max_length),
        "compression": compression,
        "keep_full_trajectory": bool(keep_full_trajectory),
        "seed": seed,
        "backend": resolved_backend,
        "dry_run": dry_run,
        "label_source": label_source,
        "label_match_threshold": label_match_threshold,
        "reference_candidates_per_prompt_mean": float(np.mean([len(x) for x in references])) if references else 0.0,
    }

    metadata_file = RESULTS_DIR / f"{model_name}_{dataset_name}_autoregressive_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "hidden_states_file": str(output_file),
        "metadata_file": str(metadata_file),
        "metadata": metadata,
    }


def analyze_autoregressive_basins(
    model_name: str = "llama-3.2-1b",
    dataset_name: str = "halueval_qa",
) -> str | None:
    # Analyze basin structure in autoregressive hidden states
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    input_file = HIDDEN_STATES_DIR / f"{model_name}_{dataset_name}_autoregressive.npz"
    if not input_file.exists():
        return None

    data = np.load(input_file, allow_pickle=True)
    hidden_states_list = data["hidden_states"]
    labels = data["labels"]

    last_layer_states = []
    valid_indices = []
    for idx, hs in enumerate(hidden_states_list):
        if hasattr(hs, "shape") and hs.size > 0 and hs.shape[0] > 0 and hs.shape[1] > 0:
            last_layer = hs[:, -1, :]
            last_layer_states.append(np.mean(last_layer, axis=0))
            valid_indices.append(idx)

    if not last_layer_states:
        return None

    x = np.array(last_layer_states)
    y = labels[valid_indices]
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(x)

    factual_mask = y == 0
    hall_mask = y == 1

    plt.figure(figsize=(8, 6))
    plt.scatter(states_2d[factual_mask, 0], states_2d[factual_mask, 1], c="blue", alpha=0.55, s=45, label="Factual")
    plt.scatter(states_2d[hall_mask, 0], states_2d[hall_mask, 1], c="red", alpha=0.55, s=45, label="Hallucinated")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title(f"Autoregressive Basin Structure ({model_name}, {dataset_name})")
    plt.legend()
    plt.tight_layout()

    output_fig = OUTPUT_DIR / f"{model_name}_{dataset_name}_autoregressive_basin.png"
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close()
    return str(output_fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoregressive hidden-state extraction runner.")
    parser.add_argument("--model", default="llama-3.2-1b")
    parser.add_argument("--dataset", default="halueval_qa")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-max-length", type=int, default=256)
    parser.add_argument("--compression", default="4bit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="auto", choices=["auto", "transformers", "airllm"])
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional dataset root override (otherwise DATA_DIR and path fallbacks are used).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-full-trajectory", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_autoregressive_experiments(
        model_name=args.model,
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        prompt_max_length=args.prompt_max_length,
        compression=args.compression,
        seed=args.seed,
        backend=args.backend,
        dry_run=args.dry_run,
        data_dir=args.data_dir,
        keep_full_trajectory=args.keep_full_trajectory,
    )
    print(json.dumps(result, indent=2))

    if not args.skip_analysis:
        out_fig = analyze_autoregressive_basins(model_name=args.model, dataset_name=args.dataset)
        print(json.dumps({"analysis_figure": out_fig}, indent=2))
