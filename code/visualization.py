"""
Generate layer-wise PCA evolution figures showing how basins form over time.
Creates small multiples visualization in 2D and 3D for all model-dataset pairs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Use relative paths
BASE_DIR = Path(__file__).parent.parent.resolve()
HIDDEN_STATES_DIR = BASE_DIR / "figs" / "hidden_states"
OUTPUT_DIR = BASE_DIR / "figs"
SEED = 42

# Select representative examples
EXAMPLES = [
    ('llama-3.2-1b_halueval_qa_hidden_states.npz', 'Llama-1B QA (Strong Basin)'),
    ('llama-3.2-1b_halueval_summarization_hidden_states.npz', 'Llama-1B Summ (No Basin)'),
    ('qwen-2.5-1.5b_halueval_qa_hidden_states.npz', 'Qwen-1.5B QA (Strong Basin)'),
    ('gemma-2-2b_halueval_summarization_hidden_states.npz', 'Gemma-2B Summ (No Basin)'),
]


def visualize_layer_evolution_2d(filename, title):
    """Generate 2D PCA evolution across layers"""
    filepath = HIDDEN_STATES_DIR / filename
    if not filepath.exists():
        print(f"Skipping {filename} (not found)")
        return None
    
    print(f"Processing {filename} for 2D evolution...")
    data = np.load(filepath)
    labels = data['labels']
    
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
    layer_keys = sorted([k for k in data.keys() if k.startswith('layer_')],
                       key=lambda x: int(x.split('_')[1]))
    
    # Sample for visualization (too many points slow down plotting)
    np.random.seed(SEED)
    n_samples = min(2000, len(labels))
    sample_idx = np.random.choice(len(labels), n_samples, replace=False)
    labels_sampled = labels[sample_idx]
    
    # Select layers to visualize (every 3rd layer to fit in grid)
    viz_layers = layer_keys[::3]
    n_viz = len(viz_layers)
    
    # Create figure
    fig, axes = plt.subplots(2, (n_viz + 1) // 2, figsize=(18, 8))
    axes = axes.flatten()
    
    for idx, layer_key in enumerate(viz_layers):
        layer_idx = int(layer_key.split('_')[1])
        h = data[layer_key][sample_idx]
        
        # PCA to 2D
        pca = PCA(n_components=2)
        h_2d = pca.fit_transform(h)
        
        # Plot
        ax = axes[idx]
        fact_mask = labels_sampled == 0
        hall_mask = labels_sampled == 1
        
        ax.scatter(h_2d[fact_mask, 0], h_2d[fact_mask, 1], 
                  c='blue', alpha=0.3, s=5, label='Factual')
        ax.scatter(h_2d[hall_mask, 0], h_2d[hall_mask, 1], 
                  c='red', alpha=0.3, s=5, label='Hallucination')
        
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
        ax.tick_params(labelsize=7)
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    # Hide unused subplots
    for idx in range(n_viz, len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    
    return fig


def visualize_layer_evolution_3d(filename, title):
    """Generate 3D PCA evolution across layers"""
    filepath = HIDDEN_STATES_DIR / filename
    if not filepath.exists():
        print(f"Skipping {filename} (not found)")
        return None
    
    print(f"Processing {filename} for 3D evolution...")
    data = np.load(filepath)
    labels = data['labels']
    
    # Get all layers
    layer_keys = sorted([k for k in data.keys() if k.startswith('layer_')],
                       key=lambda x: int(x.split('_')[1]))
    
    # Sample for visualization
    np.random.seed(SEED)
    n_samples = min(1500, len(labels))
    sample_idx = np.random.choice(len(labels), n_samples, replace=False)
    labels_sampled = labels[sample_idx]
    
    # Select key layers (beginning, middle, end)
    viz_indices = [0, len(layer_keys)//4, len(layer_keys)//2, 3*len(layer_keys)//4, -1]
    viz_layers = [layer_keys[i] for i in viz_indices]
    
    # Create figure
    fig = plt.figure(figsize=(20, 4))
    
    for idx, layer_key in enumerate(viz_layers):
        layer_idx = int(layer_key.split('_')[1])
        h = data[layer_key][sample_idx]
        
        # PCA to 3D
        pca = PCA(n_components=3)
        h_3d = pca.fit_transform(h)
        
        # Plot
        ax = fig.add_subplot(1, 5, idx+1, projection='3d')
        fact_mask = labels_sampled == 0
        hall_mask = labels_sampled == 1
        
        ax.scatter(h_3d[fact_mask, 0], h_3d[fact_mask, 1], h_3d[fact_mask, 2],
                  c='blue', alpha=0.3, s=5, label='Factual')
        ax.scatter(h_3d[hall_mask, 0], h_3d[hall_mask, 1], h_3d[hall_mask, 2],
                  c='red', alpha=0.3, s=5, label='Hallucination')
        
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
        ax.tick_params(labelsize=6)
        ax.view_init(elev=20, azim=45)
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    
    return fig


def generate_all_evolution_figures():
    """Generate all layer evolution figures"""
    print("="*60)
    print("GENERATING LAYER-WISE PCA EVOLUTION FIGURES")
    print("="*60)
    
    for filename, title in EXAMPLES:
        print(f"\nProcessing: {title}")
        
        # 2D evolution
        fig_2d = visualize_layer_evolution_2d(filename, title)
        if fig_2d:
            safe_name = filename.replace('_hidden_states.npz', '')
            output_file = OUTPUT_DIR / f'layer_evolution_2d_{safe_name}.png'
            fig_2d.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved 2D: {output_file.name}")
            plt.close(fig_2d)
        
        # 3D evolution
        fig_3d = visualize_layer_evolution_3d(filename, title)
        if fig_3d:
            safe_name = filename.replace('_hidden_states.npz', '')
            output_file = OUTPUT_DIR / f'layer_evolution_3d_{safe_name}.png'
            fig_3d.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved 3D: {output_file.name}")
            plt.close(fig_3d)
    
    print("\n" + "="*60)
    print("All evolution figures generated!")
    print("="*60)


if __name__ == '__main__':
    generate_all_evolution_figures()
