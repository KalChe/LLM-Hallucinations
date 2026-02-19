# generate causality intervention visualizations showing factual to basin transformation

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.decomposition import PCA

# Use relative paths
BASE_DIR = Path(__file__).parent.parent.resolve()
JSON_DIR = BASE_DIR / "code" / "json_results"
HIDDEN_STATES_DIR = BASE_DIR / "figs" / "hidden_states"
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
    # visualize the intervention trajectory in 3d pca space and overlay in-model captures
    filepath = HIDDEN_STATES_DIR / f"{model_dataset}_hidden_states.npz"
    if not filepath.exists():
        print(f"Skipping {model_dataset} 3D (file not found)")
        return None
    
    data = np.load(filepath)
    labels = data['labels']
    layer_keys = sorted([k for k in data.keys() if k.startswith('layer_')],
                       key=lambda x: int(x.split('_')[1]))
    
    # Use middle layer
    middle_layer = layer_keys[len(layer_keys) // 2]
    h = data[middle_layer]
    
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
    model_name, dataset_name = model_dataset.rsplit('_', 1)
    in_model_dir = OUTPUT_DIR / 'in_model_captured'
    # Basin alphas file
    basin_file = in_model_dir / f"{model_name}_{dataset_name}_inmodel_basin_alphas.npz"
    if basin_file.exists():
        try:
            arr = np.load(str(basin_file))
            # arr contains keys like 'alpha_0.0', 'alpha_0.1', ...
            alpha_keys = sorted([k for k in arr.files if k.startswith('alpha_')], key=lambda s: float(s.split('_')[1]))
            means = []
            for k in alpha_keys:
                h = arr[k]
                if h is None:
                    continue
                mean_h = h.mean(axis=0, keepdims=True)
                means.append(mean_h)
            if means:
                means = np.vstack(means)
                means_3d = pca.transform(means)
                ax.plot(means_3d[:, 0], means_3d[:, 1], means_3d[:, 2], color='purple', linestyle='-', linewidth=2.5, marker='D', markersize=6, label='in-model basin (means)')
        except Exception:
            pass

    # Random captures
    rand_file = in_model_dir / f"{model_name}_{dataset_name}_inmodel_random.npz"
    if rand_file.exists():
        try:
            darr = np.load(str(rand_file))
            if 'hidden' in darr.files:
                hrand = darr['hidden']
                hrand_3d = pca.transform(hrand)
                ax.scatter(hrand_3d[:, 0], hrand_3d[:, 1], hrand_3d[:, 2], c='green', alpha=0.35, s=20, label='in-model random')
        except Exception:
            pass

    # Orthogonal captures
    ortho_file = in_model_dir / f"{model_name}_{dataset_name}_inmodel_orthogonal.npz"
    if ortho_file.exists():
        try:
            darr = np.load(str(ortho_file))
            if 'hidden' in darr.files:
                hortho = darr['hidden']
                hortho_3d = pca.transform(hortho)
                ax.scatter(hortho_3d[:, 0], hortho_3d[:, 1], hortho_3d[:, 2], c='orange', alpha=0.6, s=40, label='in-model orthogonal')
        except Exception:
            pass

    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    return fig


def generate_all_causality_figures():
    # generate all causality visualization figures
    print("="*60)
    print("GENERATING CAUSALITY INTERVENTION FIGURES")
    print("="*60)
    
    # Iterate over hidden-state files for halueval_qa and process each model
    pattern = '*_halueval_qa_hidden_states.npz'
    files = sorted(HIDDEN_STATES_DIR.glob(pattern))
    for f in files:
        base = f.name.replace('_hidden_states.npz', '')
        model_dataset = base
        print(f"\nProcessing: {model_dataset}")

        # Intervention curve (prefer in-model values saved in per-model JSON)
        fig1 = visualize_causality_intervention(model_dataset)
        if fig1:
            output_file = OUTPUT_DIR / f'causality_intervention_{model_dataset}.png'
            fig1.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved curve: {output_file.name}")
            plt.close(fig1)

        # 3D trajectory with in-model overlays
        fig2 = visualize_causality_3d_trajectory(model_dataset)
        if fig2:
            output_file = OUTPUT_DIR / f'causality_intervention_{model_dataset}.png'
            fig2.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved 3D (saved to same intervention filename): {output_file.name}")
            plt.close(fig2)
    
    print("\n" + "="*60)
    print("All causality figures generated!")
    print("="*60)


if __name__ == '__main__':
    generate_all_causality_figures()
