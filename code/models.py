import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings

# Handle both package and direct imports
try:
    from .config import ModelConfig, MODELS
except ImportError:
    from config import ModelConfig, MODELS


def get_device() -> torch.device:
    # Get the best available device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_config: ModelConfig,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    # Load a model and tokenizer from HuggingFace
    if device is None:
        device = get_device()
    
    print(f"Loading {model_config.name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.hf_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.hf_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    if device.type != "cuda":
        model = model.to(device)
    
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    return model, tokenizer


def extract_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layers: List[int],
    batch_size: int = 4,
    max_length: int = 256,
    position: str = 'last',
    verbose: bool = True,
) -> Dict[int, np.ndarray]:
    # Extract hidden states from specified layers
    # Returns dict mapping layer_idx -> np.ndarray (n_samples, hidden_dim)
    device = next(model.parameters()).device
    model.eval()
    
    all_hidden_states = {l: [] for l in layers}
    
    iterator = range(0, len(texts), batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Extracting hidden states")
    
    with torch.inference_mode():
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            
            outputs = model(**inputs, output_hidden_states=True)
            
            for layer_idx in layers:
                if layer_idx >= len(outputs.hidden_states):
                    continue
                
                hidden = outputs.hidden_states[layer_idx]  # (batch, seq, hidden)
                
                for seq_idx in range(len(batch_texts)):
                    seq_len = inputs.attention_mask[seq_idx].sum().item()
                    
                    if position == 'last':
                        h = hidden[seq_idx, int(seq_len) - 1, :]
                    elif position == 'first':
                        h = hidden[seq_idx, 0, :]
                    elif position == 'mean':
                        h = hidden[seq_idx, :int(seq_len), :].mean(dim=0)
                    else:
                        h = hidden[seq_idx, int(seq_len) - 1, :]
                    
                    h_np = h.cpu().float().numpy()
                    h_np = np.clip(h_np, -1e4, 1e4)
                    
                    if np.isnan(h_np).any() or np.isinf(h_np).any():
                        h_np = np.nan_to_num(h_np, nan=0.0)
                    
                    all_hidden_states[layer_idx].append(h_np)
            
            del outputs, inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    for layer_idx in all_hidden_states:
        all_hidden_states[layer_idx] = np.array(all_hidden_states[layer_idx], dtype=np.float32)
    
    return all_hidden_states


def extract_all_layer_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 4,
    max_length: int = 256,
    verbose: bool = True,
) -> Dict[int, np.ndarray]:
    # Extract hidden states from ALL layers
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    layers = list(range(num_layers))
    return extract_hidden_states(model, tokenizer, texts, layers, batch_size, max_length, 'last', verbose)


class SteeringHook:
    # Apply steering vectors during generation: h_steered = h + λ(Φ(x)) · v_steer
    
    def __init__(
        self,
        steering_vectors: Dict[int, np.ndarray],
        strength: float = 1.0,
        adaptive_fn: Optional[Callable] = None,
    ):
        # Initialize steering hook with vectors and strength configuration
        self.steering_vectors = {
            k: torch.tensor(v, dtype=torch.float16) 
            for k, v in steering_vectors.items()
        }
        self.strength = strength
        self.adaptive_fn = adaptive_fn
        self.hooks = []
    
    def _create_hook(self, layer_idx: int):
        # Create a forward hook for a specific layer
        steering_vec = self.steering_vectors[layer_idx]
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            device = hidden.device
            vec = steering_vec.to(device)
            
            # Get adaptive strength if provided
            if self.adaptive_fn is not None:
                multiplier = self.adaptive_fn(hidden.detach().cpu().numpy())
            else:
                multiplier = 1.0
            
            # Apply steering
            steered = hidden + self.strength * multiplier * vec
            
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered
        
        return hook
    
    def attach(self, model: AutoModelForCausalLM):
        # Attach steering hooks to model
        for layer_idx, vec in self.steering_vectors.items():
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                # Llama-style architecture
                if layer_idx < len(model.model.layers):
                    layer = model.model.layers[layer_idx]
                    hook = layer.register_forward_hook(self._create_hook(layer_idx))
                    self.hooks.append(hook)
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                # GPT-style architecture
                if layer_idx < len(model.transformer.h):
                    layer = model.transformer.h[layer_idx]
                    hook = layer.register_forward_hook(self._create_hook(layer_idx))
                    self.hooks.append(hook)
    
    def detach(self):
        # Remove all hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def generate_with_steering(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    steering_vectors: Dict[int, np.ndarray],
    strength: float = 1.0,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 1,
    verbose: bool = True,
) -> List[str]:
    # Generate text with steering applied
    hook = SteeringHook(steering_vectors, strength)
    hook.attach(model)
    
    generations = []
    
    iterator = range(0, len(prompts), batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Generating with steering")
    
    try:
        with torch.inference_mode():
            for i in iterator:
                batch_prompts = prompts[i:i+batch_size]
                
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(next(model.parameters()).device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                for output in outputs:
                    text = tokenizer.decode(output, skip_special_tokens=True)
                    generations.append(text)
    finally:
        hook.detach()
    
    return generations


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 4,
    max_length: int = 256,
) -> List[float]:
    # Compute perplexity for each text
    device = next(model.parameters()).device
    model.eval()
    
    perplexities = []
    
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            
            outputs = model(**inputs, labels=inputs.input_ids)
            
            for j in range(len(batch_texts)):
                # Compute per-sample loss
                loss = outputs.loss.item()  # Simplified - ideally per-sample
                ppl = np.exp(loss)
                perplexities.append(min(ppl, 1e6))  # Clip extreme values
            
            del outputs, inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    return perplexities


def get_attention_entropy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layers: List[int],
    batch_size: int = 4,
    max_length: int = 256,
) -> Dict[int, np.ndarray]:
    # Compute attention entropy at each layer
    # High entropy = uniform attention = potential basin trapping
    device = next(model.parameters()).device
    model.eval()
    
    entropies = {l: [] for l in layers}
    
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            
            outputs = model(**inputs, output_attentions=True)
            
            for layer_idx in layers:
                if layer_idx >= len(outputs.attentions):
                    continue
                
                # Attention shape: (batch, heads, seq, seq)
                attn = outputs.attentions[layer_idx]
                
                for seq_idx in range(len(batch_texts)):
                    seq_len = inputs.attention_mask[seq_idx].sum().item()
                    
                    # Get attention for last token
                    last_attn = attn[seq_idx, :, int(seq_len)-1, :int(seq_len)]  # (heads, seq_len)
                    
                    # Compute entropy per head and average
                    eps = 1e-10
                    entropy = -(last_attn * torch.log(last_attn + eps)).sum(dim=-1).mean()
                    entropies[layer_idx].append(entropy.item())
            
            del outputs, inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    for layer_idx in entropies:
        entropies[layer_idx] = np.array(entropies[layer_idx])
    
    return entropies
