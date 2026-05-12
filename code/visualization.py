import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

try:
    from .metrics_utils import load_autoregressive_last_token_hidden_states
except Exception:
    from metrics_utils import load_autoregressive_last_token_hidden_states

# Use relative paths
BASE_DIR = Path(__file__).parent.parent.resolve()
HIDDEN_STATES_DIR = BASE_DIR / "figs" / "hidden_states_autoregressive"
OUTPUT_DIR = BASE_DIR / "figs"
SEED = 42

# Select representative examples
EXAMPLES = [
    ('llama-3.2-1b_halueval_qa_autoregressive.npz', 'Llama-1B QA (autoregressive)'),
    ('llama-3.2-3b_halueval_qa_autoregressive.npz', 'Llama-3B QA (autoregressive)'),
    ('mistral-7b-v0.3_halueval_qa_autoregressive.npz', 'Mistral-7B QA (autoregressive)'),
    ('llama-3.1-8b_halueval_qa_autoregressive.npz', 'Llama-8B QA (autoregressive)'),
]


def _load_layer_states(filepath: Path):
    # Supports both teacher-forced (layer_* arrays) and autoregressive object NPZ formats.
    data = np.load(filepath, allow_pickle=True)
    if any(k.startswith("layer_") for k in data.keys()):
        labels = data["labels"]
        layer_dict = {
            int(k.split("_")[1]): data[k]
            for k in data.keys()
            if k.startswith("layer_")
        }
        return layer_dict, labels

    layer_dict, labels = load_autoregressive_last_token_hidden_states(filepath)
    return layer_dict, labels


def visualize_layer_evolution_2d(filename, title, max_points=None):
    # generate 2d pca evolution across layers
    filepath = HIDDEN_STATES_DIR / filename
    if not filepath.exists():
        print(f"Skipping {filename} (not found)")
        return None
    
    print(f"Processing {filename} for 2D evolution...")
    layer_dict, labels = _load_layer_states(filepath)
    
    # Arial font, no bold
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'axes.titleweight': 'normal',
        'axes.labelweight': 'normal',
    })
    
    # Get all layers
    layer_keys = sorted(layer_dict.keys())
    
    # Plot all points by default; optional cap exists for very large artifacts.
    np.random.seed(SEED)
    sample_idx = np.arange(len(labels))
    if max_points is not None and len(labels) > int(max_points):
        sample_idx = np.random.choice(len(labels), int(max_points), replace=False)
    labels_sampled = labels[sample_idx]

    n_points = int(labels_sampled.shape[0])
    if n_points < 40:
        point_size = 36
        point_alpha = 0.75
    elif n_points < 100:
        point_size = 24
        point_alpha = 0.68
    elif n_points < 400:
        point_size = 16
        point_alpha = 0.58
    elif n_points < 2000:
        point_size = 10
        point_alpha = 0.45
    else:
        # Keep all points visible for full-dataset plots without oversaturating.
        point_size = 8
        point_alpha = 0.38
    
    # Select layer checkpoints: use all when small, otherwise evenly spaced.
    if len(layer_keys) <= 6:
        viz_layers = layer_keys
    else:
        step = max(1, len(layer_keys) // 6)
        viz_layers = layer_keys[::step]
        if viz_layers[-1] != layer_keys[-1]:
            viz_layers.append(layer_keys[-1])
    n_viz = len(viz_layers)
    
    # Create figure
    fig, axes = plt.subplots(2, (n_viz + 1) // 2, figsize=(18, 8))
    axes = axes.flatten()
    
    for idx, layer_idx in enumerate(viz_layers):
        h = layer_dict[layer_idx][sample_idx]
        
        # PCA to 2D
        pca = PCA(n_components=2)
        h_2d = pca.fit_transform(h)
        
        # Plot
        ax = axes[idx]
        fact_mask = labels_sampled == 0
        hall_mask = labels_sampled == 1
        
        ax.scatter(
            h_2d[fact_mask, 0],
            h_2d[fact_mask, 1],
            c='blue',
            alpha=point_alpha,
            s=point_size,
            label='Factual',
            edgecolors='none',
            rasterized=True,
        )
        ax.scatter(
            h_2d[hall_mask, 0],
            h_2d[hall_mask, 1],
            c='red',
            alpha=point_alpha,
            s=point_size,
            label='Hallucination',
            edgecolors='none',
            rasterized=True,
        )
        
        # Compute and plot centroids
        mu_fact = h_2d[fact_mask].mean(axis=0)
        mu_hall = h_2d[hall_mask].mean(axis=0)
        
        ax.scatter(*mu_fact, c='blue', marker='*', s=200, edgecolors='black', linewidths=1.5, zorder=10)
        ax.scatter(*mu_hall, c='red', marker='*', s=200, edgecolors='black', linewidths=1.5, zorder=10)
        
        # Basin separation line
        ax.plot([mu_fact[0], mu_hall[0]], [mu_fact[1], mu_hall[1]], 
               'k--', linewidth=1.5, alpha=0.5)
        
        separation = np.linalg.norm(mu_fact - mu_hall)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.set_title(f'Layer {layer_idx} (N={n_points})', fontsize=8)
        ax.tick_params(labelsize=7)
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    # Hide unused subplots
    for idx in range(n_viz, len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    
    return fig


def visualize_layer_evolution_3d(filename, title, max_points=None):
    # generate 3d pca evolution across layers
    filepath = HIDDEN_STATES_DIR / filename
    if not filepath.exists():
        print(f"Skipping {filename} (not found)")
        return None
    
    print(f"Processing {filename} for 3D evolution...")
    layer_dict, labels = _load_layer_states(filepath)
    
    # Get all layers
    layer_keys = sorted(layer_dict.keys())
    
    # Plot all points by default; optional cap exists for very large artifacts.
    np.random.seed(SEED)
    sample_idx = np.arange(len(labels))
    if max_points is not None and len(labels) > int(max_points):
        sample_idx = np.random.choice(len(labels), int(max_points), replace=False)
    labels_sampled = labels[sample_idx]

    n_points = int(labels_sampled.shape[0])
    if n_points < 40:
        point_size = 42
        point_alpha = 0.78
    elif n_points < 100:
        point_size = 28
        point_alpha = 0.70
    elif n_points < 400:
        point_size = 16
        point_alpha = 0.60
    elif n_points < 2000:
        point_size = 10
        point_alpha = 0.46
    else:
        point_size = 8
        point_alpha = 0.40
    
    # Select up to five evenly spaced checkpoint layers.
    n_panels = min(5, len(layer_keys))
    viz_indices = np.linspace(0, len(layer_keys) - 1, n_panels, dtype=int)
    viz_layers = [layer_keys[i] for i in viz_indices]
    
    # Create figure
    fig = plt.figure(figsize=(20, 4))
    
    for idx, layer_idx in enumerate(viz_layers):
        h = layer_dict[layer_idx][sample_idx]
        
        # PCA to 3D
        pca = PCA(n_components=3)
        h_3d = pca.fit_transform(h)
        
        # Plot
        ax = fig.add_subplot(1, 5, idx+1, projection='3d')
        fact_mask = labels_sampled == 0
        hall_mask = labels_sampled == 1
        
        ax.scatter(
            h_3d[fact_mask, 0],
            h_3d[fact_mask, 1],
            h_3d[fact_mask, 2],
            c='blue',
            alpha=point_alpha,
            s=point_size,
            label='Factual',
            edgecolors='none',
        )
        ax.scatter(
            h_3d[hall_mask, 0],
            h_3d[hall_mask, 1],
            h_3d[hall_mask, 2],
            c='red',
            alpha=point_alpha,
            s=point_size,
            label='Hallucination',
            edgecolors='none',
        )
        
        # Compute and plot centroids
        mu_fact = h_3d[fact_mask].mean(axis=0)
        mu_hall = h_3d[hall_mask].mean(axis=0)
        
        ax.scatter(*mu_fact, c='blue', marker='*', s=200, edgecolors='black', linewidths=1.5, zorder=10)
        ax.scatter(*mu_hall, c='red', marker='*', s=200, edgecolors='black', linewidths=1.5, zorder=10)
        
        # Basin separation line
        ax.plot([mu_fact[0], mu_hall[0]], 
               [mu_fact[1], mu_hall[1]],
               [mu_fact[2], mu_hall[2]], 
               'k--', linewidth=2, alpha=0.5)
        
        separation = np.linalg.norm(mu_fact - mu_hall)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.set_zlabel('PC3', fontsize=8)
        ax.set_title(f'Layer {layer_idx} (N={n_points})', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=20, azim=45)
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    
    return fig


def generate_all_evolution_figures():
    # generate all layer evolution figures
    print("="*60)
    print("GENERATING LAYER-WISE PCA EVOLUTION FIGURES")
    print("="*60)
    
    for filename, title in EXAMPLES:
        print(f"\nProcessing: {title}")
        
        # 2D evolution
        fig_2d = visualize_layer_evolution_2d(filename, title, max_points=None)
        if fig_2d:
            safe_name = filename.replace('_autoregressive.npz', '').replace('_hidden_states.npz', '')
            output_file = OUTPUT_DIR / f'layer_evolution_2d_{safe_name}.png'
            fig_2d.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved 2D: {output_file.name}")
            plt.close(fig_2d)
        
        # 3D evolution
        fig_3d = visualize_layer_evolution_3d(filename, title, max_points=None)
        if fig_3d:
            safe_name = filename.replace('_autoregressive.npz', '').replace('_hidden_states.npz', '')
            output_file = OUTPUT_DIR / f'layer_evolution_3d_{safe_name}.png'
            fig_3d.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved 3D: {output_file.name}")
            plt.close(fig_3d)
    
    print("\n" + "="*60)
    print("All evolution figures generated!")
    print("="*60)


if __name__ == '__main__':
    generate_all_evolution_figures()
