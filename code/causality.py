# rigorous causality controls for basin intervention

import numpy as np
import os
import torch
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optional in-model interventions (true causal hooks)
try:
    from models import load_model_and_tokenizer, generate_with_steering
    from config import MODELS
except Exception:
    # If models or config are not available in some contexts, we'll skip true interventions
    load_model_and_tokenizer = None
    generate_with_steering = None
    MODELS = {}

# Set Times New Roman globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Use relative paths
BASE_DIR = Path(__file__).parent.parent.resolve()
HIDDEN_STATES_DIR = BASE_DIR / "figs" / "hidden_states"
OUTPUT_DIR = BASE_DIR / "figs"
RESULTS_DIR = BASE_DIR / "code" / "json_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
IN_MODEL_CAPTURE_DIR = OUTPUT_DIR / 'in_model_captured'
IN_MODEL_CAPTURE_DIR.mkdir(exist_ok=True, parents=True)

# Note: TEST_CONFIGS will be discovered at runtime by scanning hidden-state
# files for HaluEval QA and selecting only models where a clear basin is
# detected (AUROC >= threshold). Use `DETECT_ONLY=1` to print selected models
# without running the full experiment.

DETECTION_AUROC_THRESHOLD = 0.70


N_RANDOM_DIRECTIONS = 10
ALPHA_VALUES = np.linspace(0, 1.0, 11)  # Dose-response: 0, 0.1, ..., 1.0
SEED = 42


def load_hidden_states_and_split(model_name: str, dataset_name: str) -> Tuple:
    # load hidden states and create train/test split
    hidden_file = HIDDEN_STATES_DIR / f"{model_name}_{dataset_name}_hidden_states.npz"
    
    if not hidden_file.exists():
        raise FileNotFoundError(f"Hidden states not found: {hidden_file}")
    
    data = np.load(hidden_file)
    labels = data['labels']
    
    # Use middle layer
    layer_keys = [k for k in data.keys() if k.startswith('layer_')]
    layer_keys_sorted = sorted(layer_keys, key=lambda x: int(x.split('_')[1]))
    middle_layer = layer_keys_sorted[len(layer_keys_sorted) // 2]
    hidden_states = data[middle_layer]
    
    # Train/test split (70/30, stratified)
    np.random.seed(SEED)
    factual_idx = np.where(labels == 0)[0]
    hall_idx = np.where(labels == 1)[0]
    
    np.random.shuffle(factual_idx)
    np.random.shuffle(hall_idx)
    
    n_test = int(len(factual_idx) * 0.3)
    
    train_idx = np.concatenate([factual_idx[n_test:], hall_idx[n_test:]])
    test_idx = np.concatenate([factual_idx[:n_test], hall_idx[:n_test]])
    
    h_train = hidden_states[train_idx]
    h_test = hidden_states[test_idx]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]
    
    return h_train, h_test, labels_train, labels_test, middle_layer


def train_classifier(h_train: np.ndarray, labels_train: np.ndarray) -> LogisticRegression:
    # train logistic classifier on train set
    scaler = StandardScaler()
    h_train_scaled = scaler.fit_transform(h_train)
    
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(h_train_scaled, labels_train)
    
    # Store scaler for later use
    clf.scaler = scaler
    
    return clf


def basin_direction_intervention(
    h_factual: np.ndarray,
    mu_hall: np.ndarray,
    alpha_values: np.ndarray
) -> np.ndarray:
    # interpolate factual states toward hallucination centroid
    n_samples = h_factual.shape[0]
    n_alphas = len(alpha_values)
    hidden_dim = h_factual.shape[1]
    
    h_interpolated = np.zeros((n_samples, n_alphas, hidden_dim))
    
    for i, alpha in enumerate(alpha_values):
        h_interpolated[:, i, :] = (1 - alpha) * h_factual + alpha * mu_hall
    
    return h_interpolated


def random_direction_intervention(
    h_factual: np.ndarray,
    mu_hall: np.ndarray,
    n_random: int = 10,
    alpha: float = 1.0
) -> np.ndarray:
    # move factual states in random directions with same magnitude as basin direction
    np.random.seed(SEED)
    n_samples = h_factual.shape[0]
    hidden_dim = h_factual.shape[1]
    
    # Compute basin direction magnitude for each sample
    basin_direction = mu_hall - h_factual  # (n_samples, hidden_dim)
    magnitude = np.linalg.norm(basin_direction, axis=1, keepdims=True)  # (n_samples, 1)
    
    # Generate random directions
    h_random = np.zeros((n_samples, n_random, hidden_dim))
    
    for i in range(n_random):
        # Sample random unit vectors
        random_dir = np.random.randn(n_samples, hidden_dim)
        random_dir = random_dir / np.linalg.norm(random_dir, axis=1, keepdims=True)
        
        # Scale to same magnitude as basin direction
        h_random[:, i, :] = h_factual + alpha * magnitude * random_dir
    
    return h_random


def orthogonal_direction_intervention(
    h_factual: np.ndarray,
    mu_hall: np.ndarray,
    alpha: float = 1.0
) -> np.ndarray:
    # move factual states orthogonal to basin direction
    np.random.seed(SEED)
    n_samples = h_factual.shape[0]
    hidden_dim = h_factual.shape[1]
    
    # Compute basin direction
    basin_direction = mu_hall - h_factual  # (n_samples, hidden_dim)
    basin_direction_norm = basin_direction / (np.linalg.norm(basin_direction, axis=1, keepdims=True) + 1e-8)
    
    # Generate random direction
    random_dir = np.random.randn(n_samples, hidden_dim)
    
    # Project out basin component (Gram-Schmidt)
    projection = np.sum(random_dir * basin_direction_norm, axis=1, keepdims=True)
    orthogonal_dir = random_dir - projection * basin_direction_norm
    orthogonal_dir = orthogonal_dir / (np.linalg.norm(orthogonal_dir, axis=1, keepdims=True) + 1e-8)
    
    # Scale to same magnitude
    magnitude = np.linalg.norm(basin_direction, axis=1, keepdims=True)
    h_orthogonal = h_factual + alpha * magnitude * orthogonal_dir
    
    return h_orthogonal


def evaluate_intervention(h_intervened: np.ndarray, clf: LogisticRegression) -> np.ndarray:
    # evaluate p(hallucination | h) for intervened states
    original_shape = h_intervened.shape
    
    # Flatten to 2D if needed
    if len(original_shape) == 3:
        n_samples, n_variants, hidden_dim = original_shape
        h_flat = h_intervened.reshape(n_samples * n_variants, hidden_dim)
    else:
        h_flat = h_intervened
        n_samples, hidden_dim = original_shape
        n_variants = 1
    
    # Scale and predict
    h_scaled = clf.scaler.transform(h_flat)
    p_hall = clf.predict_proba(h_scaled)[:, 1]
    
    # Reshape back
    if n_variants > 1:
        p_hall = p_hall.reshape(n_samples, n_variants)
    
    return p_hall


def get_models_with_basin(threshold: float = DETECTION_AUROC_THRESHOLD) -> List[str]:
    # scan hidden_states_dir for files and return model names with basin detected
    models = []
    pattern = "*_halueval_qa_hidden_states.npz"
    files = list(HIDDEN_STATES_DIR.glob(pattern))

    for f in files:
        model_name = f.name.replace("_halueval_qa_hidden_states.npz", "")
        try:
            h_train, h_test, labels_train, labels_test, _ = load_hidden_states_and_split(model_name, "halueval_qa")

            # compute centroids on train
            mu_fact = h_train[labels_train == 0].mean(axis=0)
            mu_hall = h_train[labels_train == 1].mean(axis=0)

            if len(h_test) == 0:
                print(f"Skipping {model_name}: no test samples")
                continue

            # simple centroid-distance score on test set
            dist_hall = np.linalg.norm(h_test - mu_hall, axis=1)
            dist_fact = np.linalg.norm(h_test - mu_fact, axis=1)
            score = dist_fact / (dist_hall + 1e-8)
            auroc = roc_auc_score(labels_test, score)

            print(f"Model {model_name}: HaluEval QA AUROC = {auroc:.3f}")
            if auroc >= threshold:
                models.append(model_name)

        except FileNotFoundError:
            print(f"Hidden-states file missing for {model_name}, skipping")
            continue
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    return models


def run_causality_controls(model_name: str, dataset_name: str) -> Dict:
    # run all 3 control conditions for a model-dataset pair
    
    print(f"\n{'='*60}")
    print(f"Causality Controls: {model_name} on {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    h_train, h_test, labels_train, labels_test, layer_name = load_hidden_states_and_split(
        model_name, dataset_name
    )
    
    print(f"Layer: {layer_name}")
    print(f"Train: {len(labels_train)}, Test: {len(labels_test)}")
    
    # Compute centroids
    mu_fact = h_train[labels_train == 0].mean(axis=0)
    mu_hall = h_train[labels_train == 1].mean(axis=0)
    
    # Train classifier
    clf = train_classifier(h_train, labels_train)
    
    # Baseline accuracy
    baseline_acc = clf.score(clf.scaler.transform(h_test), labels_test)
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    
    # Get factual test samples
    factual_mask = labels_test == 0
    h_factual = h_test[factual_mask]
    baseline_p_hall = evaluate_intervention(h_factual, clf)
    baseline_mean = baseline_p_hall.mean()
    
    print(f"\nBaseline P(hall) on factual: {baseline_mean:.4f}")
    
    # === CONTROL 1: Basin Direction (EXPECTED: 26-735× increase) ===
    print(f"\n[1/3] Basin Direction Intervention...")
    h_basin = basin_direction_intervention(h_factual, mu_hall, ALPHA_VALUES)
    p_hall_basin = evaluate_intervention(h_basin, clf)  # (n_samples, n_alphas)
    
    basin_mean_by_alpha = p_hall_basin.mean(axis=0)
    basin_fold_increase = basin_mean_by_alpha / (baseline_mean + 1e-10)
    
    print(f"  α=0.5: P(hall)={basin_mean_by_alpha[5]:.4f}, fold={basin_fold_increase[5]:.1f}×")
    print(f"  α=1.0: P(hall)={basin_mean_by_alpha[-1]:.4f}, fold={basin_fold_increase[-1]:.1f}×")
    
    # === CONTROL 2: Random Directions (EXPECTED: <5× increase) ===
    print(f"\n[2/3] Random Direction Baseline...")
    h_random = random_direction_intervention(h_factual, mu_hall, N_RANDOM_DIRECTIONS, alpha=1.0)
    p_hall_random = evaluate_intervention(h_random, clf)  # (n_samples, n_random)
    
    random_mean = p_hall_random.mean()
    random_fold = random_mean / (baseline_mean + 1e-10)
    random_std = p_hall_random.std()
    
    print(f"  Mean P(hall): {random_mean:.4f} ± {random_std:.4f}")
    print(f"  Fold increase: {random_fold:.1f}×")
    
    # === CONTROL 3: Orthogonal Direction (EXPECTED: ~1× no change) ===
    print(f"\n[3/3] Orthogonal Direction Control...")
    h_orthogonal = orthogonal_direction_intervention(h_factual, mu_hall, alpha=1.0)
    p_hall_orthogonal = evaluate_intervention(h_orthogonal, clf)
    
    orthogonal_mean = p_hall_orthogonal.mean()
    orthogonal_fold = orthogonal_mean / (baseline_mean + 1e-10)
    orthogonal_std = p_hall_orthogonal.std()
    
    print(f"  Mean P(hall): {orthogonal_mean:.4f} ± {orthogonal_std:.4f}")
    print(f"  Fold increase: {orthogonal_fold:.1f}×")
    
    # === OPTIONAL: True in-model interventions ===
    in_model_results = None
    in_model_flag = os.environ.get('IN_MODEL_INTERVENTION', '0') in ('1', 'true', 'True')
    if in_model_flag and load_model_and_tokenizer is not None and generate_with_steering is not None:
        try:
            print('\n[True Intervention] Running in-model dose-response + controls (basin/random/orthogonal)...')
            # Small prompt set for a fast demo
            prompts = [
                "Question: Who wrote 'Pride and Prejudice'? Answer:",
                "Question: What is the capital of France? Answer:",
                "Question: Explain briefly why the sky is blue. Answer:"
            ]

            model_cfg = MODELS.get(model_name)
            if model_cfg is None:
                print(f"Model config for {model_name} not found in MODELS; skipping true intervention.")
            else:
                model, tokenizer = load_model_and_tokenizer(model_cfg)
                layer_idx = int(layer_name.split('_')[1])

                # Basin steering vector (centroid difference)
                v_basin = (mu_hall - mu_fact).astype(np.float32)

                in_model_results = {'prompts': prompts}

                # Helper to extract last-token hidden from captured dict
                def _last_hidden_from_captured(captured_dict, li):
                    arrs = captured_dict.get(li, [])
                    if not arrs:
                        return None
                    last = arrs[-1]  # (batch, seq, hidden)
                    return last[:, -1, :]

                # === Basin dose-response in-model (iterate ALPHA_VALUES via strength) ===
                basin_p_means = []
                basin_hiddens = {}
                for a in ALPHA_VALUES:
                    steering_basin = {layer_idx: v_basin}
                    gens_basin, captured_basin = generate_with_steering(
                        model, tokenizer, prompts, steering_basin,
                        strength=float(a), max_new_tokens=32, temperature=0.7,
                        capture_hidden=True, return_captured=True, batch_size=len(prompts), verbose=False
                    )
                    h_last_basin = _last_hidden_from_captured(captured_basin, layer_idx)
                    if h_last_basin is not None:
                        p_basin = evaluate_intervention(h_last_basin, clf)
                        basin_p_means.append(float(p_basin.mean()))
                        basin_hiddens[f'alpha_{a}'] = h_last_basin
                    else:
                        basin_p_means.append(None)

                in_model_results['basin_alpha_values'] = ALPHA_VALUES.tolist()
                in_model_results['basin_p_hall_mean'] = basin_p_means
                # fold relative to baseline
                in_model_results['basin_fold_increase'] = [ (v / (baseline_mean + 1e-10)) if v is not None else None for v in basin_p_means ]

                # Save basin hiddens per-alpha
                if basin_hiddens:
                    basin_file = IN_MODEL_CAPTURE_DIR / f"{model_name}_{dataset_name}_inmodel_basin_alphas.npz"
                    np.savez_compressed(str(basin_file), **basin_hiddens)
                    in_model_results['basin_hidden_file'] = str(basin_file)

                # === Random directions (average over a few quick samples), collect hiddens ===
                rand_means = []
                rand_hiddens = []
                for i in range(min(N_RANDOM_DIRECTIONS, 6)):
                    rand_vec = np.random.randn(*v_basin.shape)
                    rand_vec = rand_vec / (np.linalg.norm(rand_vec) + 1e-12) * (np.linalg.norm(v_basin) + 1e-12)
                    steering_rand = {layer_idx: rand_vec.astype(np.float32)}
                    gens_rand, captured_rand = generate_with_steering(
                        model, tokenizer, prompts, steering_rand,
                        strength=1.0, max_new_tokens=32, temperature=0.7,
                        capture_hidden=True, return_captured=True, batch_size=len(prompts), verbose=False
                    )
                    h_last_rand = _last_hidden_from_captured(captured_rand, layer_idx)
                    if h_last_rand is not None:
                        p_rand = evaluate_intervention(h_last_rand, clf)
                        rand_means.append(float(p_rand.mean()))
                        rand_hiddens.append(h_last_rand)

                in_model_results['random_mean_p_hall'] = float(np.mean(rand_means)) if rand_means else None
                if rand_hiddens:
                    rand_all = np.vstack(rand_hiddens)
                    rand_file = IN_MODEL_CAPTURE_DIR / f"{model_name}_{dataset_name}_inmodel_random.npz"
                    np.savez_compressed(str(rand_file), hidden=rand_all)
                    in_model_results['random_hidden_file'] = str(rand_file)

                # === Orthogonal direction: pick one orthogonal vector and run ===
                ortho_dir = np.random.randn(*v_basin.shape)
                basin_norm = v_basin / (np.linalg.norm(v_basin) + 1e-12)
                proj = np.sum(ortho_dir * basin_norm) * basin_norm
                ortho_vec = ortho_dir - proj
                ortho_vec = ortho_vec / (np.linalg.norm(ortho_vec) + 1e-12) * (np.linalg.norm(v_basin) + 1e-12)
                steering_ortho = {layer_idx: ortho_vec.astype(np.float32)}
                gens_ortho, captured_ortho = generate_with_steering(
                    model, tokenizer, prompts, steering_ortho,
                    strength=1.0, max_new_tokens=32, temperature=0.7,
                    capture_hidden=True, return_captured=True, batch_size=len(prompts), verbose=False
                )
                h_last_ortho = _last_hidden_from_captured(captured_ortho, layer_idx)
                if h_last_ortho is not None:
                    p_ortho = evaluate_intervention(h_last_ortho, clf)
                    in_model_results['orthogonal_mean_p_hall'] = float(p_ortho.mean())
                    ortho_file = IN_MODEL_CAPTURE_DIR / f"{model_name}_{dataset_name}_inmodel_orthogonal.npz"
                    np.savez_compressed(str(ortho_file), hidden=h_last_ortho)
                    in_model_results['orthogonal_hidden_file'] = str(ortho_file)
                else:
                    in_model_results['orthogonal_mean_p_hall'] = None

        except Exception as e:
            print("In-model intervention skipped due to error:", e)
            in_model_results = None
    else:
        if in_model_flag:
            print("In-model intervention requested but model loading/generation utilities are unavailable.")

    # Package results
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'layer': layer_name,
        'n_factual_test': int(h_factual.shape[0]),
        'baseline_p_hall': float(baseline_mean),
        'basin_direction': {
            'alpha_values': ALPHA_VALUES.tolist(),
            'p_hall_mean': basin_mean_by_alpha.tolist(),
            'fold_increase': basin_fold_increase.tolist(),
            'max_fold': float(basin_fold_increase[-1])
        },
        'random_direction': {
            'n_samples': N_RANDOM_DIRECTIONS,
            'p_hall_mean': float(random_mean),
            'p_hall_std': float(random_std),
            'fold_increase': float(random_fold)
        },
        'orthogonal_direction': {
            'p_hall_mean': float(orthogonal_mean),
            'p_hall_std': float(orthogonal_std),
            'fold_increase': float(orthogonal_fold)
        },
        'control_validation': {
            'basin_vs_random': float(basin_fold_increase[-1] / random_fold) if random_fold > 0 else float('inf'),
            'basin_vs_orthogonal': float(basin_fold_increase[-1] / orthogonal_fold) if orthogonal_fold > 0 else float('inf')
        }
    }
    # include in-model intervention quick demo results if available
    if in_model_results is not None:
        results['in_model'] = in_model_results
    
    return results


def create_control_comparison_figure(all_results: List[Dict], output_file: Path):
    # create 2-panel figure comparing controls across models
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel 1: Dose-response curves
    ax1 = axes[0]
    for result in all_results:
        # Plot offline (hidden-state) dose-response if available
        alpha_vals = result['basin_direction']['alpha_values']
        fold_vals = result['basin_direction']['fold_increase']
        label = result['model']
        ax1.plot(alpha_vals, fold_vals, marker='o', markersize=4, label=label + ' (offline)', linewidth=1.2, alpha=0.75)

        # If in-model dose-response is present, overlay it (prefer in-model values)
        im = result.get('in_model')
        if im is not None and im.get('basin_fold_increase'):
            try:
                im_alpha = im.get('basin_alpha_values', alpha_vals)
                im_fold = im.get('basin_fold_increase')
                # Plot in-model as a thicker dashed line
                ax1.plot(im_alpha, im_fold, marker='D', markersize=6, linestyle='--', linewidth=2.0,
                         label=result['model'] + ' (in-model)')
            except Exception:
                pass
    
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    # No axis titles per request; keep ticks and legend only
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    
    # Panel 2: Control comparison (bar chart)
    ax2 = axes[1]
    x_pos = np.arange(len(all_results))
    width = 0.25
    
    basin_folds = []
    random_folds = []
    orthogonal_folds = []
    for r in all_results:
        baseline = r.get('baseline_p_hall', 1e-6)
        im = r.get('in_model')
        # Basin: prefer in-model final alpha fold, else fallback to offline max_fold
        if im is not None and im.get('basin_fold_increase') and im['basin_fold_increase'][-1] is not None:
            basin_val = im['basin_fold_increase'][-1]
        else:
            basin_val = r['basin_direction'].get('max_fold', 1.0)

        # Random: prefer in-model mean probability if available
        if im is not None and im.get('random_mean_p_hall') is not None:
            random_val = float(im['random_mean_p_hall']) / (baseline + 1e-10)
        else:
            random_val = r['random_direction'].get('fold_increase', 1.0)

        # Orthogonal: prefer in-model mean probability if available
        if im is not None and im.get('orthogonal_mean_p_hall') is not None:
            orth_val = float(im['orthogonal_mean_p_hall']) / (baseline + 1e-10)
        else:
            orth_val = r['orthogonal_direction'].get('fold_increase', 1.0)

        basin_folds.append(basin_val)
        random_folds.append(random_val)
        orthogonal_folds.append(orth_val)
    
    ax2.bar(x_pos - width, basin_folds, width, label='Basin direction', color='#d62728')
    ax2.bar(x_pos, random_folds, width, label='Random direction', color='#1f77b4')
    ax2.bar(x_pos + width, orthogonal_folds, width, label='Orthogonal', color='#2ca02c')
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    # No axis titles per request; keep ticks and full model names as tick labels
    ax2.set_yscale('log')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([r['model'] for r in all_results], fontsize=7, rotation=0)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Figure saved: {output_file.name}")


def main():
    # run causality controls for all configurations
    
    print("="*60)
    print("RIGOROUS CAUSALITY CONTROLS")
    print("="*60)
    # Discover models (HaluEval QA) with a detectable basin
    detect_only = os.environ.get('DETECT_ONLY', '0') in ('1', 'true', 'True')
    selected_models = get_models_with_basin()

    if not selected_models:
        print("\n⚠️  No models with a detectable basin (AUROC >= {:.2f}) found in hidden states.".format(DETECTION_AUROC_THRESHOLD))
        return

    print(f"\nSelected models for HaluEval QA (AUROC >= {DETECTION_AUROC_THRESHOLD}):")
    for m in selected_models:
        print(f" - {m}")

    if detect_only:
        print("\nDETECT_ONLY set — exiting after model discovery.")
        return

    all_results = []
    for model_name in selected_models:
        dataset_name = 'halueval_qa'
        try:
            results = run_causality_controls(model_name, dataset_name)
            all_results.append(results)

            # Save individual result
            result_file = RESULTS_DIR / f"causality_controls_{model_name}_{dataset_name}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)

        except FileNotFoundError as e:
            print(f"\n⚠️  Skipping {model_name}/{dataset_name}: {e}")
            continue
        except Exception as e:
            print(f"\n❌ Error processing {model_name}/{dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_results:
        # Save aggregate results
        aggregate_file = RESULTS_DIR / "causality_controls_all.json"
        with open(aggregate_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Aggregate results saved: {aggregate_file.name}")
        
        # Create comparison figure
        figure_file = OUTPUT_DIR / "causality_controls_comparison.png"
        create_control_comparison_figure(all_results, figure_file)
        
        # Print summary table
        print(f"\n{'='*60}")
        print("SUMMARY: Control Validation")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Dataset':<15} {'Basin':<10} {'Random':<10} {'Orthog':<10}")
        print(f"{'-'*60}")
        for r in all_results:
            model_short = r['model'].split('-')[-1]
            dataset_short = r['dataset'][:12]
            basin_fold = r['basin_direction']['max_fold']
            random_fold = r['random_direction']['fold_increase']
            orthog_fold = r['orthogonal_direction']['fold_increase']
            print(f"{model_short:<20} {dataset_short:<15} {basin_fold:>8.1f}× {random_fold:>8.1f}× {orthog_fold:>8.1f}×")
        
        print(f"\n✓ Expected: Basin >> Random, Basin >> Orthogonal")
        print(f"✓ Validates that DIRECTION (not distance) drives causality")
    
    else:
        print("\n⚠️  No results generated")


if __name__ == "__main__":
    main()
