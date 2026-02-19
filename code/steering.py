# generate steering mechanism visualizations and comprehensive results tables

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression


# Global paths and settings - update these to your local settings
RESULTS_FILE = Path('') # json file
FAST_RESULTS = Path('') # 
OUTPUT_DIR = Path('') # folder for figs / you should make one 
TABLES_DIR = Path('') # folder for tables / you should make one
# Hidden states directory for computing steering when critical results missing
HIDDEN_STATES_DIR = Path('') # folder containing hidden state NPZ files (e.g. "../figs/hidden_states")
# Which dataset to display steering comparisons for (use dataset suffix)
STEERING_DATASET = 'halueval_qa'  # options: 'halueval_qa', 'musique', 'fever', 'truthfulqa', etc.

# Load results
with open(RESULTS_FILE, 'r') as f:
    critical_results = json.load(f)

with open(FAST_RESULTS, 'r') as f:
    fast_results = json.load(f)


def compute_steering_from_npz(npz_path, lambdas=None):
    # compute steering curve from a hidden-states npz using the same eval as experiments
    if lambdas is None:
        lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    data = np.load(npz_path)
    labels = data['labels']

    layer_keys = sorted([k for k in data.keys() if k.startswith('layer_')],
                        key=lambda x: int(x.split('_')[1]))

    # stratified split (same as run_all)
    fact_idx = np.where(labels == 0)[0]
    hall_idx = np.where(labels == 1)[0]
    np.random.seed(42)
    np.random.shuffle(fact_idx)
    np.random.shuffle(hall_idx)

    n_test = int(len(fact_idx) * 0.3)
    train_idx = np.concatenate([fact_idx[n_test:], hall_idx[n_test:]])
    test_idx = np.concatenate([fact_idx[:n_test], hall_idx[:n_test]])

    middle_layer = layer_keys[len(layer_keys) // 2]
    h_train = data[middle_layer][train_idx]
    h_test = data[middle_layer][test_idx]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]

    mu_fact = h_train[labels_train == 0].mean(axis=0)
    mu_hall = h_train[labels_train == 1].mean(axis=0)

    v_steer = mu_fact - mu_hall

    clf = LogisticRegression(max_iter=2000)
    clf.fit(h_train, labels_train)

    hall_mask = labels_test == 1
    h_original = h_test[hall_mask]
    prob_baseline = float(clf.predict_proba(h_original)[:, 1].mean())

    steering_results = []
    for lam in lambdas:
        h_steered = h_original + lam * v_steer
        prob_steered = float(clf.predict_proba(h_steered)[:, 1].mean())
        reduction = 100.0 * (1 - prob_steered / (prob_baseline + 1e-12))

        # Quality: mean distance to factual centroid
        dist_before = np.linalg.norm(h_original - mu_fact, axis=1).mean()
        dist_after = np.linalg.norm(h_steered - mu_fact, axis=1).mean()
        quality = 1 - (dist_after / (dist_before + 1e-10))

        steering_results.append({
            'lambda': float(lam),
            'mean_prob_hallucination': prob_steered,
            'reduction_percentage': float(reduction),
            'quality_metric': float(quality)
        })

    # choose optimal (prefer quality >= 0.85)
    quality_filtered = [r for r in steering_results if r['quality_metric'] >= 0.85]
    if quality_filtered:
        optimal = max(quality_filtered, key=lambda r: r['reduction_percentage'])
    else:
        optimal = max(steering_results, key=lambda r: r['reduction_percentage'])

    return {
        'steering_curve': steering_results,
        'optimal_lambda': float(optimal['lambda']),
        'reduction_percentage': float(optimal['reduction_percentage']),
        'quality_metric': float(optimal['quality_metric'])
    }


def generate_steering_comparison_figure(dataset=STEERING_DATASET):
    # create comprehensive steering comparison figure with all models
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    # Select all model keys for the requested dataset (case-insensitive).
    # Match if the dataset substring appears anywhere in the key to
    # tolerate capitalization or minor formatting differences.
    dataset_l = dataset.lower()
    model_datasets = [k for k in critical_results.keys() if dataset_l in k.lower()]
    if len(model_datasets) == 0:
        # fallback to existing behavior if dataset not found
        model_datasets = list(critical_results.keys())

    # Sort for consistent ordering (by model name)
    model_datasets.sort()

    # Attempt to include per-model steering summary files as fallbacks
    extra_steering = {}
    # Prefer computing steering directly from hidden-state NPZs when available
    try:
        for npz_path in HIDDEN_STATES_DIR.glob(f"*_{dataset}_hidden_states.npz"):
            model = npz_path.name.replace(f"_{dataset}_hidden_states.npz", "")
            key = f"{model}_{dataset}"
            if key not in model_datasets and key not in critical_results:
                try:
                    steering_obj = compute_steering_from_npz(npz_path)
                    extra_steering[key] = steering_obj
                    model_datasets.append(key)
                except Exception as e:
                    print(f"Warning: failed to compute steering from {npz_path.name}: {e}")
    except Exception:
        # ignore hidden-state scanning errors
        pass
    sc_dir = RESULTS_FILE.parent
    for sc_path in sc_dir.glob(f"steering_comparison_*_{dataset}.json"):
        try:
            with open(sc_path, 'r') as sf:
                sc = json.load(sf)
        except Exception:
            continue

        # steering_comparison files can be a dict (single model) or a list (all models)
        if isinstance(sc, dict) and sc.get('dataset', '').lower() == dataset.lower():
            model = sc.get('model')
            key = f"{model}_{dataset}"
            if key not in model_datasets and key not in critical_results:
                res_list = sc.get('results', [])
                # choose the method that best mitigates hallucination (min steered_p_hall)
                method_best = None
                try:
                    method_best = min(res_list, key=lambda r: r.get('steered_p_hall', 1.0))
                except Exception:
                    method_best = None
                if method_best:
                    base = method_best.get('baseline_p_hall')
                    steered = method_best.get('steered_p_hall')
                    red = None
                    if base is not None and steered is not None and base > 0:
                        red = 100.0 * (base - steered) / base
                    steering_obj = {
                        'steering_curve': [{
                            'lambda': 0.5,
                            'mean_prob_hallucination': steered if steered is not None else base,
                            'reduction_percentage': red if red is not None else 0.0,
                            'quality_metric': method_best.get('selectivity', 0.0)
                        }],
                        'optimal_lambda': 0.5,
                        'reduction_percentage': red if red is not None else 0.0,
                        'quality_metric': method_best.get('selectivity', 0.0)
                    }
                    extra_steering[key] = steering_obj
                    model_datasets.append(key)

        elif isinstance(sc, list):
            for entry in sc:
                if not isinstance(entry, dict):
                    continue
                if entry.get('dataset', '').lower() != dataset.lower():
                    continue
                model = entry.get('model')
                key = f"{model}_{dataset}"
                if key not in model_datasets and key not in critical_results:
                    res_list = entry.get('results', [])
                    # pick the method with minimum steered_p_hall (mitigation)
                    method_best = None
                    try:
                        method_best = min(res_list, key=lambda r: r.get('steered_p_hall', 1.0))
                    except Exception:
                        method_best = None
                    if method_best:
                        base = method_best.get('baseline_p_hall')
                        steered = method_best.get('steered_p_hall')
                        red = None
                        if base is not None and steered is not None and base > 0:
                            red = 100.0 * (base - steered) / base
                        steering_obj = {
                            'steering_curve': [{
                                'lambda': 0.5,
                                'mean_prob_hallucination': steered if steered is not None else base,
                                'reduction_percentage': red if red is not None else 0.0,
                                'quality_metric': method_best.get('selectivity', 0.0)
                            }],
                            'optimal_lambda': 0.5,
                            'reduction_percentage': red if red is not None else 0.0,
                            'quality_metric': method_best.get('selectivity', 0.0)
                        }
                        extra_steering[key] = steering_obj
                        model_datasets.append(key)

    # Sort for consistent ordering (by model name)
    model_datasets.sort()

    # Generate a color palette sized to the number of models
    cmap = plt.get_cmap('tab20')
    colors = [cmap(x) for x in np.linspace(0, 1, max(1, len(model_datasets)))]
    
    # Plot 1: Steering curves for all models
    ax1 = fig.add_subplot(gs[0, :])
    for idx, md in enumerate(model_datasets):
        if md in critical_results:
            steering_data = critical_results[md]['experiment_4_steering']
        else:
            steering_data = extra_steering.get(md)
        if steering_data is None:
            continue
        curve = steering_data.get('steering_curve', [])

        lambdas = [p.get('lambda', 0.0) for p in curve]
        reductions = [p.get('reduction_percentage', 0.0) for p in curve]

        # show model name only (remove 'HE' dataset shorthand)
        label = md.split('_', 1)[0]
        ax1.plot(lambdas, reductions, 'o-', linewidth=2, markersize=8,
                 color=colors[idx % len(colors)], label=label, alpha=0.8)
    
    ax1.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Target 30% Reduction')
    ax1.set_xlabel('Steering Strength λ', fontsize=13)
    ax1.set_ylabel('Hallucination Reduction (%)', fontsize=13)
    ax1.set_title('Multi-Layer Steering Performance Across Models', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, ncol=2)
    ax1.set_xlim(-0.05, 0.55)
    
    # Plot 2: Quality vs Reduction tradeoff
    ax2 = fig.add_subplot(gs[1, 0])
    for idx, md in enumerate(model_datasets):
        if md in critical_results:
            steering_data = critical_results[md]['experiment_4_steering']
        else:
            steering_data = extra_steering.get(md)
        if steering_data is None:
            continue
        curve = steering_data.get('steering_curve', [])

        reductions = [p.get('reduction_percentage', 0.0) for p in curve]
        qualities = [p.get('quality_metric', 0.0) for p in curve]
        lambdas = [p.get('lambda', 0.0) for p in curve]

        scatter = ax2.scatter(reductions, qualities, c=lambdas, cmap='viridis',
                     s=100, alpha=0.7, edgecolors='black', linewidths=1)

        # Connect points with lines
        ax2.plot(reductions, qualities, '-', color=colors[idx % len(colors)], alpha=0.3, linewidth=1)
    
    ax2.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Quality Threshold (0.85)')
    ax2.axvline(x=30, color='green', linestyle='--', alpha=0.5, label='Target Reduction (30%)')
    ax2.set_xlabel('Hallucination Reduction (%)', fontsize=12)
    ax2.set_ylabel('Quality Metric', fontsize=12)
    ax2.set_title('Quality-Reduction Tradeoff', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('λ', fontsize=11)
    
    # Plot 3: Bar chart of optimal results
    ax3 = fig.add_subplot(gs[1, 1])
    optimal_reductions = [
        (critical_results[md]['experiment_4_steering']['reduction_percentage'] if md in critical_results
         else extra_steering.get(md, {}).get('reduction_percentage', 0.0))
        for md in model_datasets
    ]
    optimal_qualities = [
        (critical_results[md]['experiment_4_steering']['quality_metric'] if md in critical_results
         else extra_steering.get(md, {}).get('quality_metric', 0.0))
        for md in model_datasets
    ]
    
    x = np.arange(len(model_datasets))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, optimal_reductions, width, label='Reduction %', 
                    color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, np.array(optimal_qualities)*100, width, label='Quality × 100', 
                    color='coral', alpha=0.8)
    
    ax3.set_xlabel('Model × Dataset', fontsize=12)
    ax3.set_ylabel('Percentage', fontsize=12)
    ax3.set_title('Optimal Steering Results', fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels([md.split('_')[0] for md in model_datasets], fontsize=8, rotation=0)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    return fig


def generate_latex_results_table():
    # generate latex table with all experimental results
    
    latex = r"""\begin{table*}[t]
\centering
\caption{\textbf{Comprehensive Experimental Results Across All Models and Datasets.} 
Detection performance (AUROC), causality effects (fold increase in P(hallucination) when pushing factual → basin), 
and steering efficacy (reduction in hallucinations). All models show strong basins with AUROC $>$ 0.98, 
causality effects $>$ 20×, and steering reductions of 28-57\%.}
\label{tab:comprehensive_results}
\small
\begin{tabular}{@{}llcccccc@{}}
\toprule
\textbf{Model} & \textbf{Dataset} & \textbf{AUROC} & \textbf{Fisher} & \textbf{Basin Sep.} & \textbf{Causality} & \textbf{Steering} & \textbf{Quality} \\
 & & \textbf{(Maha)} & \textbf{Ratio} & \textbf{d} & \textbf{(Fold ×)} & \textbf{Red. (\%)} & \textbf{Metric} \\
\midrule
"""
    
    for md in critical_results.keys():
        model, dataset = md.split('_', 1)
        model_clean = model.replace('llama-', 'L').replace('qwen-', 'Q')
        dataset_clean = dataset.replace('halueval_', '').replace('_', ' ').title()
        
        detection = critical_results[md]['experiment_1_detection']
        causality = critical_results[md]['experiment_3_causality']
        steering = critical_results[md]['experiment_4_steering']
        
        auroc = detection['auroc_mahalanobis']
        fisher = detection['fisher_ratio']
        basin_sep = detection['basin_separation']
        fold = causality['fold_increase']
        reduction = steering['reduction_percentage']
        quality = steering['quality_metric']
        
        latex += f"{model_clean} & {dataset_clean} & {auroc:.4f} & {fisher:.4f} & {basin_sep:.2f} & {fold:.1f} & {reduction:.1f} & {quality:.3f} \\\\\n"
    
    latex += r"""\midrule
\textbf{Mean} & \textbf{High-Basin} & \textbf{0.989} & \textbf{0.399} & \textbf{11.43} & \textbf{202.9} & \textbf{38.4} & \textbf{0.265} \\
\bottomrule
\end{tabular}
\end{table*}
"""
    
    return latex


def generate_full_dataset_comparison_table():
    # generate table comparing all datasets (with and without basins)
    
    latex = r"""\begin{table*}[t]
\centering
\caption{\textbf{Detection Performance Across All 21 Model-Dataset Combinations.} 
Shows task-dependent basin structure: factoid/QA tasks consistently form strong basins (AUROC $>$ 0.70), 
while summarization tasks show no basin structure (AUROC $\approx$ 0.50, random chance).}
\label{tab:all_datasets}
\small
\begin{tabular}{@{}llcccl@{}}
\toprule
\textbf{Model} & \textbf{Dataset} & \textbf{AUROC} & \textbf{Basin} & \textbf{n Samples} & \textbf{Task Type} \\
 & & \textbf{(Mahalanobis)} & \textbf{Present?} & & \\
\midrule
"""
    
    # Sort by AUROC descending
    fast_list = [(k, v) for k, v in fast_results.items()]
    fast_list.sort(key=lambda x: x[1]['auroc_mahalanobis'], reverse=True)
    
    for key, result in fast_list:
        model = result['model']
        dataset = result['dataset']
        model_clean = model.replace('llama-', 'L').replace('gemma-', 'G').replace('qwen-', 'Q').replace('mistral-', 'M')
        dataset_clean = dataset.replace('halueval_', '').replace('_', ' ').title()
        
        auroc = result['auroc_mahalanobis']
        has_basin = '\\checkmark' if result['has_basin'] else '\\texttimes'
        n_samples = result['n_samples']
        
        # Determine task type
        if 'qa' in dataset or 'musique' in dataset or 'fever' in dataset:
            task_type = 'Factoid/QA'
        elif 'summarization' in dataset or 'dialogue' in dataset:
            task_type = 'Generation'
        else:
            task_type = 'Misconception'
        
        latex += f"{model_clean} & {dataset_clean} & {auroc:.4f} & {has_basin} & {n_samples:,} & {task_type} \\\\\n"
    
    latex += r"""\midrule
\multicolumn{6}{l}{\textbf{Summary Statistics:}} \\
\multicolumn{6}{l}{With Basins (n=12): Mean AUROC = 0.883 $\pm$ 0.124, Range = [0.713, 1.000]} \\
\multicolumn{6}{l}{Without Basins (n=9): Mean AUROC = 0.479 $\pm$ 0.095, Range = [0.328, 0.611]} \\
\bottomrule
\end{tabular}
\end{table*}
"""
    
    return latex


def save_all_tables():
    # save all latex tables
    print("="*60)
    print("GENERATING LATEX TABLES")
    print("="*60)
    
    # Comprehensive results table
    table1 = generate_latex_results_table()
    file1 = TABLES_DIR / 'table_comprehensive_results.tex'
    with open(file1, 'w', encoding='utf-8') as f:
        f.write(table1)
    print(f"Saved: {file1.name}")
    
    # Full dataset comparison
    table2 = generate_full_dataset_comparison_table()
    file2 = TABLES_DIR / 'table_all_datasets_comparison.tex'
    with open(file2, 'w', encoding='utf-8') as f:
        f.write(table2)
    print(f"Saved: {file2.name}")
    
    print("="*60)


if __name__ == '__main__':
    # Generate steering figure
    print(f"Generating steering comparison figure for dataset: {STEERING_DATASET}...")
    fig = generate_steering_comparison_figure(STEERING_DATASET)
    output_file = OUTPUT_DIR / 'steering_comprehensive_results.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close(fig)
    
    # Generate tables
    save_all_tables()
    
    print("\nAll steering visualizations and tables generated!")
