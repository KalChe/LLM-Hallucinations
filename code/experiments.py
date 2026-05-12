import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.decomposition import PCA

try:
    from .metrics_utils import load_autoregressive_last_token_hidden_states
except Exception:
    from metrics_utils import load_autoregressive_last_token_hidden_states

# Use relative paths
BASE_DIR = Path(__file__).parent.parent.resolve()
JSON_DIR = BASE_DIR / "code" / "json_results"
HIDDEN_STATES_DIR = BASE_DIR / "figs" / "hidden_states_autoregressive"
OUTPUT_DIR = BASE_DIR / "figs"

JSON_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# We'll process per-model JSON files produced by `causality_rigorous_controls.py`.

def visualize_causality_intervention(model_dataset):
    # create causality visualization using per-model json results
    # model_dataset looks like 'llama-3.2-1b_halueval_qa'
    model_name, dataset_name = model_dataset.rsplit('_', 1)
    result_file = JSON_DIR / f"causality_controls_{model_name}_{dataset_name}.json"
    if not result_file.exists():
        print(f"Skipping {model_dataset} (no results JSON: {result_file.name})")
        return None

    with open(result_file, 'r') as f:
        res = json.load(f)

    # Determine curve data: prefer in-model
    im = res.get('in_model')
    if im and im.get('basin_alpha_values') and im.get('basin_p_hall_mean'):
        alphas = im['basin_alpha_values']
        probs = im['basin_p_hall_mean']
        fold_increases = im.get('basin_fold_increase', [p / (res['baseline_p_hall'] + 1e-10) if p is not None else None for p in probs])
    else:
        # fallback to offline values saved under 'basin_direction'
        alphas = res['basin_direction']['alpha_values']
        probs = res['basin_direction'].get('p_hall_mean', [])
        fold_increases = res['basin_direction'].get('fold_increase', [])

    # Create figure with 2 subplots (no title per user request)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Intervention curve (probabilities)
    ax1.plot(alphas, probs, 'o-', linewidth=2, markersize=8, color='purple')
    # baseline as first point if available
    if probs:
        ax1.axhline(y=probs[0], color='blue', linestyle='--', alpha=0.5, label='Baseline (Factual)')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random Chance')
    if probs:
        ax1.fill_between(alphas, probs[0], probs, alpha=0.2, color='red', label='Induced Hallucination Risk')

    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)

    # Plot 2: Fold increase
    ax2.plot(alphas, fold_increases, 's-', linewidth=2, markersize=8, color='red')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No Change')
    if fold_increases:
        ax2.fill_between(alphas, 1, fold_increases, alpha=0.2, color='red')

    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_yscale('log')

    plt.tight_layout()
    return fig


def visualize_causality_3d_trajectory(model_dataset):
    # visualize intervention trajectory in 3D PCA from autoregressive hidden-state artifacts
    filepath = HIDDEN_STATES_DIR / f"{model_dataset}_autoregressive.npz"
    if not filepath.exists():
        print(f"Skipping {model_dataset} 3D (file not found)")
        return None

    layers, labels = load_autoregressive_last_token_hidden_states(filepath)
    layer_keys = sorted(list(layers.keys()))
    middle_layer = layer_keys[len(layer_keys) // 2]
    h = layers[middle_layer]
    
    # Split data
    np.random.seed(42)
    factual_idx = np.where(labels == 0)[0]
    hall_idx = np.where(labels == 1)[0]
    
    # Sample for visualization
    n_sample = min(1000, len(factual_idx))
    factual_sample_idx = np.random.choice(factual_idx, n_sample, replace=False)
    hall_sample_idx = np.random.choice(hall_idx, n_sample, replace=False)
    
    h_fact_samples = h[factual_sample_idx]
    h_hall_samples = h[hall_sample_idx]
    
    # Compute centroids
    mu_fact = h[factual_idx].mean(axis=0)
    mu_hall = h[hall_idx].mean(axis=0)
    
    # Create intervention trajectory
    alphas = np.linspace(0, 1, 11)
    trajectory = np.array([(1-a) * mu_fact + a * mu_hall for a in alphas])
    
    # PCA to 3D
    all_data = np.vstack([h_fact_samples, h_hall_samples, trajectory])
    pca = PCA(n_components=3)
    all_data_3d = pca.fit_transform(all_data)
    
    n_fact = len(h_fact_samples)
    n_hall = len(h_hall_samples)
    h_fact_3d = all_data_3d[:n_fact]
    h_hall_3d = all_data_3d[n_fact:n_fact+n_hall]
    trajectory_3d = all_data_3d[n_fact+n_hall:]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot samples
    ax.scatter(h_fact_3d[:, 0], h_fact_3d[:, 1], h_fact_3d[:, 2],
              c='blue', alpha=0.2, s=10, label='Factual Samples')
    ax.scatter(h_hall_3d[:, 0], h_hall_3d[:, 1], h_hall_3d[:, 2],
              c='red', alpha=0.2, s=10, label='Hallucination Samples')
    
    # Plot centroids
    ax.scatter(trajectory_3d[0, 0], trajectory_3d[0, 1], trajectory_3d[0, 2],
              c='blue', marker='*', s=500, edgecolors='black', linewidths=2, 
              label='μ_fact (Start)', zorder=10)
    ax.scatter(trajectory_3d[-1, 0], trajectory_3d[-1, 1], trajectory_3d[-1, 2],
              c='red', marker='*', s=500, edgecolors='black', linewidths=2, 
              label='μ_hall (End)', zorder=10)
    
    # Plot intervention trajectory
    ax.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2],
           'purple', linewidth=4, linestyle='--', alpha=0.8, label='Intervention Path')
    
    # Add arrows for direction
    for i in range(0, len(trajectory_3d)-1, 2):
        ax.quiver(trajectory_3d[i, 0], trajectory_3d[i, 1], trajectory_3d[i, 2],
                 trajectory_3d[i+1, 0] - trajectory_3d[i, 0],
                 trajectory_3d[i+1, 1] - trajectory_3d[i, 1],
                 trajectory_3d[i+1, 2] - trajectory_3d[i, 2],
                 color='purple', arrow_length_ratio=0.3, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_zlabel('PC3', fontsize=11)
    # No figure title per user request
    ax.legend(fontsize=10, loc='upper right')
    ax.view_init(elev=20, azim=45)
    
    # Overlay in-model captured hidden states if present
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    return fig


def generate_all_causality_figures():
    # generate all causality visualization figures
    print("="*60)
    print("GENERATING CAUSALITY INTERVENTION FIGURES")
    print("="*60)
    
    target_model_datasets = [
        "llama-3.2-3b_halueval_qa",
        "mistral-7b-v0.3_halueval_qa",
        "llama-3.1-8b_halueval_qa",
    ]

    for model_dataset in target_model_datasets:
        print(f"\nProcessing: {model_dataset}")

        # 3D trajectory with in-model overlays
        fig2 = visualize_causality_3d_trajectory(model_dataset)
        if fig2:
            output_file = OUTPUT_DIR / f'causality_3d_trajectory_{model_dataset}.png'
            fig2.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved 3D trajectory: {output_file.name}")
            plt.close(fig2)
    
    print("\n" + "="*60)
    print("All causality figures generated!")
    print("="*60)


if __name__ == '__main__':
    generate_all_causality_figures()
