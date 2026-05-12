import numpy as np
import torch
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d

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

# Models to analyze
MODELS = [
    "llama-3.2-1b",
    "llama-3.2-3b",
    "gemma-2-2b",
    "qwen-2.5-1.5b",
    "microsoft-phi-1"  # New model
]

DATASET = "halueval_summarization"  # Focus on misconceptions
K_VALUES = [3, 5, 7, 10]  # Test different cluster counts
SEED = 42


def load_hidden_states(model_name: str, dataset_name: str = DATASET) -> Tuple:
    # load hidden states for middle layer
    hidden_file = HIDDEN_STATES_DIR / f"{model_name}_{dataset_name}_hidden_states.npz"
    
    if not hidden_file.exists():
        raise FileNotFoundError(f"Hidden states not found: {hidden_file}")
    
    data = np.load(hidden_file)
    labels = data['labels']
    
    # Use middle layer
    layer_keys = [k for k in data.keys() if k.startswith('layer_')]
    layer_keys_sorted = sorted(layer_keys, key=lambda x: int(x.split('_')[1]))
    middle_layer = layer_keys_sorted[len(layer_keys_sorted) -1]
    hidden_states = data[middle_layer]
    
    return hidden_states, labels, middle_layer


def find_optimal_k(h_hall: np.ndarray, k_values: List[int]) -> Tuple[int, Dict]:
    # find optimal number of clusters using silhouette score and elbow method
    
    scores = {}
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=20, max_iter=300)
        labels = kmeans.fit_predict(h_hall)
        
        silhouette = silhouette_score(h_hall, labels)
        calinski = calinski_harabasz_score(h_hall, labels)
        inertia = kmeans.inertia_
        
        scores[k] = {
            'silhouette': float(silhouette),
            'calinski_harabasz': float(calinski),
            'inertia': float(inertia),
            'labels': labels,
            'centers': kmeans.cluster_centers_
        }
    
    # Select K with best silhouette score
    optimal_k = max(k_values, key=lambda k: scores[k]['silhouette'])
    
    return optimal_k, scores


def _valid_k_values(n_hall_samples: int, k_values: List[int]) -> List[int]:
    # silhouette/calinski require at least 2 clusters and k < n_samples
    return [k for k in k_values if 2 <= k < n_hall_samples]


def multi_basin_detection(h_train: np.ndarray, labels_train: np.ndarray, k: int) -> Tuple:
    # Multi-basin detection.
    # Step 1: Compute reference centroid μ_0
    mu_ref = h_train[labels_train == 0].mean(axis=0)
    
    # Step 2: Extract hallucination samples
    h_hall = h_train[labels_train == 1]
    
    # Step 3: K-means clustering on hallucination samples
    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=20, max_iter=300)
    cluster_labels = kmeans.fit_predict(h_hall)
    
    # Step 4: Extract misconception centroids {μ_1, ..., μ_k}
    mu_misconceptions = kmeans.cluster_centers_
    
    return mu_ref, mu_misconceptions, cluster_labels, h_hall


def evaluate_multi_class_detection(
    h_test: np.ndarray,
    labels_test: np.ndarray,
    mu_ref: np.ndarray,
    mu_misconceptions: np.ndarray
) -> Dict:
    # evaluate multi-class classification: reference (0) vs. k misconceptions (1...k)
    k = len(mu_misconceptions)
    
    # Assign test samples to nearest centroid (Voronoi cells)
    all_centers = np.vstack([mu_ref[np.newaxis, :], mu_misconceptions])
    
    distances = np.linalg.norm(h_test[:, np.newaxis, :] - all_centers[np.newaxis, :, :], axis=2)
    predicted_cluster = np.argmin(distances, axis=1)
    
    # Binary ground truth
    is_hallucination = labels_test == 1
    
    # Multi-class prediction: 0 = factual, >0 = hallucination
    predicted_hallucination = predicted_cluster > 0
    
    # Metrics
    accuracy = (predicted_hallucination == is_hallucination).mean()
    
    # Confusion matrix
    tn = ((predicted_cluster == 0) & (labels_test == 0)).sum()
    fp = ((predicted_cluster > 0) & (labels_test == 0)).sum()
    fn = ((predicted_cluster == 0) & (labels_test == 1)).sum()
    tp = ((predicted_cluster > 0) & (labels_test == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'k_clusters': int(k),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }


def visualize_voronoi_cells(
    h_hall: np.ndarray,
    mu_ref: np.ndarray,
    mu_misconceptions: np.ndarray,
    cluster_labels: np.ndarray,
    model_name: str,
    output_file: Path
):
    # visualize voronoi partitioning in 2d pca space
    
    k = len(mu_misconceptions)
    
    # PCA to 2D
    all_points = np.vstack([mu_ref[np.newaxis, :], mu_misconceptions, h_hall])
    pca = PCA(n_components=2, random_state=SEED)
    all_points_2d = pca.fit_transform(all_points)
    
    mu_ref_2d = all_points_2d[0]
    mu_misconceptions_2d = all_points_2d[1:k+1]
    h_hall_2d = all_points_2d[k+1:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot hallucination samples colored by cluster
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    for cluster_id in range(k):
        mask = cluster_labels == cluster_id
        ax.scatter(
            h_hall_2d[mask, 0],
            h_hall_2d[mask, 1],
            c=[colors[cluster_id]],
            alpha=0.3,
            s=20,
            label=f'Misconception {cluster_id+1}'
        )
    
    # Plot centroids
    ax.scatter(mu_ref_2d[0], mu_ref_2d[1], c='green', marker='*', s=400, 
               edgecolors='black', linewidths=1.5, label='Reference', zorder=10)
    
    for i in range(k):
        ax.scatter(
            mu_misconceptions_2d[i, 0],
            mu_misconceptions_2d[i, 1],
            c=[colors[i]],
            marker='X',
            s=300,
            edgecolors='black',
            linewidths=1.5,
            zorder=10
        )
    
    # Voronoi diagram (approximate in 2D projection)
    all_centers_2d = np.vstack([mu_ref_2d[np.newaxis, :], mu_misconceptions_2d])
    
    # Only plot if centers are well-separated
    if len(all_centers_2d) >= 4:
        try:
            vor = Voronoi(all_centers_2d)
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', 
                            line_width=1, line_alpha=0.6, point_size=0)
        except:
            print(f"  Warning: Voronoi plot failed for {model_name}")
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def run_multi_basin_analysis(model_name: str) -> Dict:
    # run full multi-basin analysis for a single model
    
    print(f"\n{'='*60}")
    print(f"Multi-Basin Analysis: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load data
        hidden_states, labels, layer_name = load_hidden_states(model_name, DATASET)
        print(f"Layer: {layer_name}, Samples: {len(labels)}")
        
        # Train/test split
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
        
        print(f"Train: {len(labels_train)}, Test: {len(labels_test)}")
        
        # Find optimal K
        h_hall_train = h_train[labels_train == 1]
        print(f"\nHallucination samples: {len(h_hall_train)}")
        
        valid_k_values = _valid_k_values(len(h_hall_train), K_VALUES)
        if not valid_k_values:
            print(
                f"\nSkipping: insufficient hallucination samples ({len(h_hall_train)}) "
                f"for K choices {K_VALUES}"
            )
            return None

        optimal_k, clustering_scores = find_optimal_k(h_hall_train, valid_k_values)
        print(f"\nOptimal K: {optimal_k}")
        silhouette_str = ', '.join([f'{k}: {clustering_scores[k]["silhouette"]:.3f}' for k in valid_k_values])
        print(f"Silhouette scores: {silhouette_str}")
        
        # Run multi-basin detection with the selected K.
        mu_ref, mu_misconceptions, cluster_labels, h_hall = multi_basin_detection(
            h_train, labels_train, optimal_k
        )
        
        print(f"\nDetected {optimal_k} misconception basins + 1 reference basin")
        
        # Cluster sizes
        for i in range(optimal_k):
            count = (cluster_labels == i).sum()
            pct = count / len(cluster_labels) * 100
            print(f"  Basin {i+1}: {count} samples ({pct:.1f}%)")
        
        # Evaluate multi-class detection
        eval_results = evaluate_multi_class_detection(
            h_test, labels_test, mu_ref, mu_misconceptions
        )
        
        print(f"\nMulti-Class Detection Performance:")
        print(f"  Accuracy:  {eval_results['accuracy']:.3f}")
        print(f"  Precision: {eval_results['precision']:.3f}")
        print(f"  Recall:    {eval_results['recall']:.3f}")
        print(f"  F1:        {eval_results['f1']:.3f}")
        
        # Visualize
        viz_file = OUTPUT_DIR / f"multi_basin_{model_name}_voronoi.png"
        visualize_voronoi_cells(
            h_hall, mu_ref, mu_misconceptions, cluster_labels, model_name, viz_file
        )
        print(f"\nVisualization saved: {viz_file.name}")
        
        # Package results
        results = {
            'model': model_name,
            'dataset': DATASET,
            'layer': layer_name,
            'optimal_k': optimal_k,
            'clustering_scores': {k: {'silhouette': clustering_scores[k]['silhouette'],
                                      'calinski_harabasz': clustering_scores[k]['calinski_harabasz']}
                                  for k in valid_k_values},
            'cluster_sizes': [int((cluster_labels == i).sum()) for i in range(optimal_k)],
            'evaluation': eval_results
        }
        
        return results
        
    except FileNotFoundError as e:
        print(f"\nSkipping: {e}")
        return None
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_summary_figure(all_results: List[Dict], output_file: Path):
    # create summary figure showing optimal k and performance across models
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    models = [r['model'].split('-')[-1] for r in all_results]
    optimal_ks = [r['optimal_k'] for r in all_results]
    f1_scores = [r['evaluation']['f1'] for r in all_results]
    
    # Panel 1: Optimal K
    ax1 = axes[0]
    ax1.bar(range(len(models)), optimal_ks, color='#1f77b4', alpha=0.7)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Optimal K (misconception basins)')
    ax1.set_ylim([0, max(optimal_ks) + 2])
    ax1.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    
    # Panel 2: F1 scores
    ax2 = axes[1]
    ax2.bar(range(len(models)), f1_scores, color='#2ca02c', alpha=0.7)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Multi-class F1 score')
    ax2.set_ylim([0, 1.0])
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSummary figure saved: {output_file.name}")


def main():
    # run multi-basin analysis for all models
    
    print("="*60)
    print("MULTI-BASIN CLUSTERING")
    print("="*60)
    
    all_results = []
    
    for model in MODELS:
        result = run_multi_basin_analysis(model)
        if result:
            all_results.append(result)
            
            # Save individual result
            result_file = RESULTS_DIR / f"multi_basin_{model}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
    
    if all_results:
        # Save aggregate
        aggregate_file = RESULTS_DIR / "multi_basin_all.json"
        with open(aggregate_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nAggregate results: {aggregate_file.name}")
        
        # Summary figure
        summary_file = OUTPUT_DIR / "multi_basin_summary.pdf"
        create_summary_figure(all_results, summary_file)
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY: Multi-Basin Detection")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Optimal K':<12} {'F1 Score':<10}")
        print(f"{'-'*60}")
        for r in all_results:
            model_short = r['model'].split('-')[-1]
            print(f"{model_short:<20} {r['optimal_k']:<12} {r['evaluation']['f1']:.3f}")
        
        avg_k = np.mean([r['optimal_k'] for r in all_results])
        avg_f1 = np.mean([r['evaluation']['f1'] for r in all_results])
        print(f"\nAverage K: {avg_k:.1f} misconception basins")
        print(f"Average F1: {avg_f1:.3f}")
        print("Multiple competing attractors detected in the latent space")
    
    else:
        print("\nNo results generated")


if __name__ == "__main__":
    main()
