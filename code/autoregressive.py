# autoregressive generation experiments using airllm

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

try:
    from airllm import AutoModel
    AIRLLM_AVAILABLE = True
except ImportError:
    AIRLLM_AVAILABLE = False
    print("Warning: airllm not installed. Install with: pip install airllm")

try:
    from .data import load_dataset
    from .config import MODELS
except ImportError:
    from data import load_dataset
    from config import MODELS


# Use relative paths
BASE_DIR = Path(__file__).parent.parent.resolve()
HIDDEN_STATES_DIR = BASE_DIR / "figs" / "hidden_states_autoregressive"
OUTPUT_DIR = BASE_DIR / "figs"
RESULTS_DIR = BASE_DIR / "code" / "json_results"

HIDDEN_STATES_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def extract_autoregressive_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    batch_size: int = 1,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[str]]:
    # extract hidden states during autoregressive generation
    if not AIRLLM_AVAILABLE:
        raise ImportError("airllm not installed")
    
    hidden_states_list = []
    generated_texts = []
    
    iterator = tqdm(prompts, desc="Generating") if verbose else prompts
    
    for prompt in iterator:
        # Tokenize input
        input_tokens = tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=256,
            padding=False
        )
        
        # Generate with AirLLM
        try:
            # Note: AirLLM handles device placement automatically
            generation_output = model.generate(
                input_tokens['input_ids'],  # Let AirLLM handle device
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_hidden_states=True,  # Request hidden states
            )
            
            # Decode generated text
            generated_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
            
            # Extract hidden states from generation
            # Note: Structure depends on AirLLM's output format
            if hasattr(generation_output, 'hidden_states') and generation_output.hidden_states:
                # hidden_states is tuple of (n_generated_tokens, n_layers, batch=1, seq_len, hidden_dim)
                all_hidden_states = []
                
                for token_hidden_states in generation_output.hidden_states:
                    # token_hidden_states is tuple of tensors (n_layers,)
                    # Each is (batch=1, seq_len, hidden_dim)
                    layer_states = []
                    for layer_tensor in token_hidden_states:
                        # Take last position (the newly generated token)
                        state = layer_tensor[0, -1, :].cpu().numpy()
                        layer_states.append(state)
                    all_hidden_states.append(np.array(layer_states))
                
                # Stack: (n_tokens, n_layers, hidden_dim)
                hidden_states_array = np.array(all_hidden_states)
                hidden_states_list.append(hidden_states_array)
            else:
                # Fallback: empty array if hidden states not available
                hidden_states_list.append(np.array([]))
                
        except Exception as e:
            print(f"Error generating for prompt: {e}")
            generated_texts.append("")
            hidden_states_list.append(np.array([]))
    
    return hidden_states_list, generated_texts


def run_autoregressive_experiments(
    model_name: str = "llama-3.2-1b",
    dataset_name: str = "halueval_qa",
    n_samples: int = 100,
    max_new_tokens: int = 50,
    compression: str = "4bit",  # '4bit', '8bit', or None
    seed: int = 42,
):
    # run autoregressive generation experiment
    if not AIRLLM_AVAILABLE:
        print("Error: airllm not installed. Install with: pip install airllm")
        return
    
    # Get model config
    model_config = MODELS.get(model_name)
    if not model_config:
        print(f"Unknown model: {model_name}")
        return
    
    print(f"Loading model {model_config.name} with AirLLM...")
    print(f"Compression: {compression}")
    
    # Load model with AirLLM
    model = AutoModel.from_pretrained(
        model_config.hf_name,
        compression=compression if compression else None,
    )
    tokenizer = model.tokenizer
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    texts, labels, prompts = load_dataset(dataset_name, n_samples=n_samples, seed=seed)
    
    print(f"Loaded {len(texts)} samples ({np.sum(labels==0)} factual, {np.sum(labels==1)} hallucinated)")
    
    # Run autoregressive generation
    print("Extracting hidden states during autoregressive generation...")
    hidden_states_list, generated_texts = extract_autoregressive_hidden_states(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=1,
        verbose=True,
    )
    
    # Save results
    output_file = HIDDEN_STATES_DIR / f"{model_name}_{dataset_name}_autoregressive.npz"
    print(f"Saving to {output_file}")
    
    # Convert list of arrays to save format
    # Since each sample can have different number of generated tokens, save as object arrays
    np.savez(
        output_file,
        hidden_states=np.array(hidden_states_list, dtype=object),
        labels=labels,
        prompts=np.array(prompts),
        generated_texts=np.array(generated_texts),
        original_texts=np.array(texts),
    )
    
    # Save metadata
    metadata = {
        "model": model_config.name,
        "dataset": dataset_name,
        "n_samples": len(texts),
        "n_factual": int(np.sum(labels == 0)),
        "n_hallucinated": int(np.sum(labels == 1)),
        "max_new_tokens": max_new_tokens,
        "compression": compression,
        "seed": seed,
    }
    
    metadata_file = RESULTS_DIR / f"{model_name}_{dataset_name}_autoregressive_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Experiment complete!")
    print(f"  Hidden states: {output_file}")
    print(f"  Metadata: {metadata_file}")


def analyze_autoregressive_basins(
    model_name: str = "llama-3.2-1b",
    dataset_name: str = "halueval_qa",
):
    # analyze basin structure in autoregressive hidden states
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    # Load autoregressive hidden states
    input_file = HIDDEN_STATES_DIR / f"{model_name}_{dataset_name}_autoregressive.npz"
    
    if not input_file.exists():
        print(f"File not found: {input_file}")
        print("Run autoregressive experiment first.")
        return
    
    print(f"Loading {input_file}")
    data = np.load(input_file, allow_pickle=True)
    
    hidden_states_list = data['hidden_states']
    labels = data['labels']
    
    print(f"Loaded {len(hidden_states_list)} samples")
    
    # Analyze basin structure at each generation step
    # Take last layer, average across generation steps
    
    last_layer_states = []
    valid_indices = []
    
    for i, hs in enumerate(hidden_states_list):
        if len(hs) > 0 and hs.shape[0] > 0:
            # hs shape: (n_tokens, n_layers, hidden_dim)
            # Take last layer, average over tokens
            last_layer = hs[:, -1, :]  # (n_tokens, hidden_dim)
            mean_state = np.mean(last_layer, axis=0)  # (hidden_dim,)
            last_layer_states.append(mean_state)
            valid_indices.append(i)
    
    if len(last_layer_states) == 0:
        print("No valid hidden states found")
        return
    
    last_layer_states = np.array(last_layer_states)
    labels_valid = labels[valid_indices]
    
    print(f"Valid samples: {len(last_layer_states)}")
    
    # PCA visualization
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(last_layer_states)
    
    # Plot
    plt.figure(figsize=(8, 6))
    
    factual_mask = labels_valid == 0
    hall_mask = labels_valid == 1
    
    plt.scatter(states_2d[factual_mask, 0], states_2d[factual_mask, 1], 
                c='blue', alpha=0.6, label='Factual', s=50)
    plt.scatter(states_2d[hall_mask, 0], states_2d[hall_mask, 1], 
                c='red', alpha=0.6, label='Hallucinated', s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'Autoregressive Basin Structure\n{model_name} - {dataset_name}')
    plt.legend()
    plt.tight_layout()
    
    output_fig = OUTPUT_DIR / f"{model_name}_{dataset_name}_autoregressive_basin.png"
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {output_fig}")
    plt.show()


if __name__ == "__main__":
    # Example: Run autoregressive experiment on Llama 3.2 1B
    run_autoregressive_experiments(
        model_name="llama-3.2-1b",
        dataset_name="halueval_qa",
        n_samples=100,
        max_new_tokens=50,
        compression="4bit",  # Use 4-bit compression to fit in memory
        seed=42,
    )
    
    # Analyze results
    analyze_autoregressive_basins(
        model_name="llama-3.2-1b",
        dataset_name="halueval_qa",
    )
