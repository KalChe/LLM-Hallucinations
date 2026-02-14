"""
Data loading utilities for hallucination detection experiments.

Supports:
- HaluEval (QA, dialogue, summarization)
- MuSiQue (multi-hop reasoning)
- HalluDial (dialogue hallucination)
- TruthfulQA
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class HallucinationSample:
    """A single sample with factual and hallucinated response."""
    prompt: str
    factual_response: str
    hallucinated_response: str
    metadata: Dict = None


def load_halueval_qa(
    data_path: Path = Path("c:/Users/cheru/Downloads/llm-hallucinations/HaluEval/data/qa_data.json"),
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Load HaluEval QA data.
    
    Args:
        data_path: Path to qa_data.json
        n_samples: If set, subsample to this many total samples
        seed: Random seed for subsampling
    
    Returns:
        texts: List of full texts (prompt + response)
        labels: Array of labels (0=factual, 1=hallucinated)
        prompts: List of prompts only
    """
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    samples.append(item)
                except json.JSONDecodeError:
                    continue
    
    factual_texts = []
    hallucinated_texts = []
    prompts = []
    
    for item in samples:
        question = item.get('question', '')
        knowledge = item.get('knowledge', '')
        right_answer = item.get('right_answer', '')
        hall_answer = item.get('hallucinated_answer', '')
        
        prompt = f"Question: {question}\nContext: {knowledge}\nAnswer:"
        
        if right_answer and hall_answer:
            factual_texts.append(f"{prompt} {right_answer}")
            hallucinated_texts.append(f"{prompt} {hall_answer}")
            prompts.append(prompt)
    
    # Balance classes
    np.random.seed(seed)
    min_samples = min(len(factual_texts), len(hallucinated_texts))
    
    if n_samples is not None:
        min_samples = min(min_samples, n_samples // 2)
    
    indices = np.random.permutation(min_samples)
    
    factual_subset = [factual_texts[i] for i in indices]
    hall_subset = [hallucinated_texts[i] for i in indices]
    prompts_subset = [prompts[i] for i in indices]
    
    texts = factual_subset + hall_subset
    labels = np.array([0] * len(factual_subset) + [1] * len(hall_subset))
    all_prompts = prompts_subset + prompts_subset  # Same prompts for both classes
    
    return texts, labels, all_prompts


def load_halueval_dialogue(
    data_path: Path = Path("c:/Users/cheru/Downloads/llm-hallucinations/HaluEval/data/dialogue_data.json"),
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Load HaluEval dialogue data."""
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    samples.append(item)
                except json.JSONDecodeError:
                    continue
    
    factual_texts = []
    hallucinated_texts = []
    prompts = []
    
    for item in samples:
        dialogue_history = item.get('dialogue_history', '')
        knowledge = item.get('knowledge', '')
        right_response = item.get('right_response', '')
        hall_response = item.get('hallucinated_response', '')
        
        prompt = f"Dialogue: {dialogue_history}\nKnowledge: {knowledge}\nResponse:"
        
        if right_response and hall_response:
            factual_texts.append(f"{prompt} {right_response}")
            hallucinated_texts.append(f"{prompt} {hall_response}")
            prompts.append(prompt)
    
    np.random.seed(seed)
    min_samples = min(len(factual_texts), len(hallucinated_texts))
    
    if n_samples is not None:
        min_samples = min(min_samples, n_samples // 2)
    
    indices = np.random.permutation(min_samples)
    
    factual_subset = [factual_texts[i] for i in indices]
    hall_subset = [hallucinated_texts[i] for i in indices]
    prompts_subset = [prompts[i] for i in indices]
    
    texts = factual_subset + hall_subset
    labels = np.array([0] * len(factual_subset) + [1] * len(hall_subset))
    all_prompts = prompts_subset + prompts_subset
    
    return texts, labels, all_prompts


def load_halueval_summarization(
    data_path: Path = Path("c:/Users/cheru/Downloads/llm-hallucinations/HaluEval/data/summarization_data.json"),
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Load HaluEval summarization data."""
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    samples.append(item)
                except json.JSONDecodeError:
                    continue
    
    factual_texts = []
    hallucinated_texts = []
    prompts = []
    
    for item in samples:
        document = item.get('document', '')
        right_summary = item.get('right_summary', '')
        hall_summary = item.get('hallucinated_summary', '')
        
        # Truncate document for efficiency
        prompt = f"Document: {document[:500]}...\nSummary:"
        
        if right_summary and hall_summary:
            factual_texts.append(f"{prompt} {right_summary}")
            hallucinated_texts.append(f"{prompt} {hall_summary}")
            prompts.append(prompt)
    
    np.random.seed(seed)
    min_samples = min(len(factual_texts), len(hallucinated_texts))
    
    if n_samples is not None:
        min_samples = min(min_samples, n_samples // 2)
    
    indices = np.random.permutation(min_samples)
    
    factual_subset = [factual_texts[i] for i in indices]
    hall_subset = [hallucinated_texts[i] for i in indices]
    prompts_subset = [prompts[i] for i in indices]
    
    texts = factual_subset + hall_subset
    labels = np.array([0] * len(factual_subset) + [1] * len(hall_subset))
    all_prompts = prompts_subset + prompts_subset
    
    return texts, labels, all_prompts


def load_truthfulqa(
    data_path: Path = Path("c:/Users/cheru/Downloads/llm-hallucinations/TruthfulQA/TruthfulQA.csv"),
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Load TruthfulQA data."""
    import csv
    
    prompts = []
    factual_answers = []
    hallucinated_answers = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row.get('Question', '').strip()
                best_answer = row.get('Best Answer', '').strip()
                incorrect_answer = row.get('Incorrect Answers', '').strip()
                
                if question and best_answer and incorrect_answer:
                    incorrect = incorrect_answer.split(';')[0].strip()
                    if incorrect:
                        prompts.append(question)
                        factual_answers.append(best_answer)
                        hallucinated_answers.append(incorrect)
    except FileNotFoundError:
        print(f"TruthfulQA not found at {data_path}, creating synthetic samples")
        for i in range(100):
            prompts.append(f"Question {i}: What is true about topic X?")
            factual_answers.append(f"Factual answer {i} based on evidence.")
            hallucinated_answers.append(f"Hallucinated answer {i} with false claims.")
    
    np.random.seed(seed)
    n_available = len(prompts)
    
    if n_samples and n_samples < n_available * 2:
        n_per_class = n_samples // 2
        indices = np.random.choice(n_available, n_per_class, replace=False)
    else:
        indices = np.arange(n_available)
    
    texts = []
    labels = []
    prompts_out = []
    
    for idx in indices:
        texts.append(prompts[idx] + " " + factual_answers[idx])
        labels.append(0)
        prompts_out.append(prompts[idx])
        
        texts.append(prompts[idx] + " " + hallucinated_answers[idx])
        labels.append(1)
        prompts_out.append(prompts[idx])
    
    return texts, np.array(labels), prompts_out


def load_halludial(
    data_path: Path = Path("c:/Users/cheru/Downloads/llm-hallucinations/HalluDial"),
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Load HalluDial dialogue hallucination data."""
    
    texts = []
    labels = []
    prompts = []
    
    # Try to load from actual dataset files first
    spontaneous_file = data_path / "data" / "spontaneous" / "spontaneous_train.json"
    induced_file = data_path / "data" / "induced" / "induced_train.json"
    
    # Also try meta-evaluation results
    detect_file = data_path / "meta-evaluation_result" / "Llama-2-13b-chat-hf_detect.json"
    
    loaded = False
    
    # Try spontaneous/induced data
    for data_file in [spontaneous_file, induced_file]:
        if data_file.exists():
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"      Loaded {len(data)} samples from {data_file.name}")
                
                for item in data:
                    if 'response' in item and 'label' in item:
                        response_text = item['response']
                        is_hallucinated = (item['label'] == 1 or item['label'] == 'hallucinated')
                        
                        texts.append(response_text)
                        labels.append(1 if is_hallucinated else 0)
                        
                        prompt = item.get('dialogue_history', item.get('context', ''))
                        if isinstance(prompt, str) and len(prompt) > 200:
                            prompt = prompt[:200] + '...'
                        prompts.append(prompt)
                
                loaded = True
                break
            except Exception as e:
                continue
    
    # Try meta-evaluation results if dataset not found
    if not loaded and detect_file.exists():
        try:
            with open(detect_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"      Loaded {len(data)} samples from meta-evaluation results")
            
            # This file only has hallucinated samples, so we need to create balanced data
            for item in data:
                if 'response' in item:
                    response_text = item['response']
                    # All in this file are hallucinated
                    texts.append(response_text)
                    labels.append(1)
                    
                    prompt = item.get('dialogue_history', '')
                    if isinstance(prompt, str) and len(prompt) > 200:
                        prompt = prompt[:200] + '...'
                    prompts.append(prompt)
            
            # Also add the knowledge as non-hallucinated samples for balance
            for item in data:
                if 'knowledge' in item:
                    knowledge_text = item['knowledge']
                    texts.append(knowledge_text)
                    labels.append(0)  # Knowledge is factual
                    
                    prompt = item.get('dialogue_history', '')
                    if isinstance(prompt, str) and len(prompt) > 200:
                        prompt = prompt[:200] + '...'
                    prompts.append(prompt)
            
            loaded = True
            
        except Exception as e:
            print(f"      Failed to load meta-evaluation: {e}")
    
    # If nothing worked, create synthetic balanced data
    if not loaded or len(texts) == 0:
        print(f"      WARNING: HalluDial not found at {data_path}")
        print(f"      Download from: https://huggingface.co/datasets/FlagEval/HalluDial")
        print(f"      Creating synthetic samples for testing")
        
        # Create balanced synthetic data
        for i in range(100):
            if i < 50:
                # Factual samples
                texts.append(f"Factual dialogue response {i}: This is based on known information.")
                labels.append(0)
            else:
                # Hallucinated samples
                texts.append(f"Hallucinated dialogue response {i}: This contains made-up information.")
                labels.append(1)
            prompts.append(f"Dialogue context {i}")
    
    print(f"      Total samples: {len(texts)} ({sum(labels)} hallucinated, {len(labels)-sum(labels)} factual)")
    
    texts = np.array(texts)
    labels = np.array(labels)
    prompts = np.array(prompts)
    
    np.random.seed(seed)
    factual_idx = np.where(labels == 0)[0]
    hall_idx = np.where(labels == 1)[0]
    
    if n_samples:
        n_per_class = n_samples // 2
        factual_idx = np.random.choice(factual_idx, min(n_per_class, len(factual_idx)), replace=False)
        hall_idx = np.random.choice(hall_idx, min(n_per_class, len(hall_idx)), replace=False)
    
    selected_idx = np.concatenate([factual_idx, hall_idx])
    np.random.shuffle(selected_idx)
    
    return texts[selected_idx].tolist(), labels[selected_idx], prompts[selected_idx].tolist()


def load_musique(
    data_dir: Path = Path("c:/Users/cheru/Downloads/llm-hallucinations/musique_data/data"),
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Load MuSiQue multi-hop QA data.
    
    Creates hallucinated responses by providing wrong answers from other questions.
    """
    # Find available data files
    data_files = list(data_dir.glob("*.json")) + list(data_dir.glob("*.jsonl"))
    
    if not data_files:
        print(f"Warning: No MuSiQue data found in {data_dir}")
        return [], np.array([]), []
    
    samples = []
    for data_file in data_files:
        with open(data_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    samples.extend(data)
                else:
                    samples.append(data)
            except json.JSONDecodeError:
                # Try JSONL format
                f.seek(0)
                for line in f:
                    if line.strip():
                        try:
                            samples.append(json.loads(line))
                        except:
                            continue
    
    if not samples:
        return [], np.array([]), []
    
    # Extract QA pairs
    qa_pairs = []
    for item in samples:
        question = item.get('question', '')
        answer = item.get('answer', '') or item.get('answer_text', '')
        paragraphs = item.get('paragraphs', [])
        context = ' '.join([p.get('paragraph_text', '') for p in paragraphs[:2]])[:500]
        
        if question and answer:
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'context': context,
            })
    
    np.random.seed(seed)
    
    # Create factual and hallucinated samples
    factual_texts = []
    hallucinated_texts = []
    prompts = []
    
    all_answers = [q['answer'] for q in qa_pairs]
    
    for i, qa in enumerate(qa_pairs):
        prompt = f"Question: {qa['question']}\nContext: {qa['context']}\nAnswer:"
        
        # Factual: correct answer
        factual_texts.append(f"{prompt} {qa['answer']}")
        
        # Hallucinated: random wrong answer
        wrong_idx = np.random.choice([j for j in range(len(all_answers)) if j != i])
        hallucinated_texts.append(f"{prompt} {all_answers[wrong_idx]}")
        
        prompts.append(prompt)
    
    min_samples = min(len(factual_texts), len(hallucinated_texts))
    if n_samples is not None:
        min_samples = min(min_samples, n_samples // 2)
    
    indices = np.random.permutation(min_samples)
    
    factual_subset = [factual_texts[i] for i in indices]
    hall_subset = [hallucinated_texts[i] for i in indices]
    prompts_subset = [prompts[i] for i in indices]
    
    texts = factual_subset + hall_subset
    labels = np.array([0] * len(factual_subset) + [1] * len(hall_subset))
    all_prompts = prompts_subset + prompts_subset
    
    return texts, labels, all_prompts


def load_halludial(
    data_path: Path = Path("c:/Users/cheru/Downloads/llm-hallucinations/HalluDial_data"),
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Load HalluDial dataset (conversational hallucination).
    
    HalluDial contains multi-turn dialogues with hallucinated responses.
    Unlike HaluEval, this is focused on conversational contexts.
    
    Args:
        data_path: Path to HalluDial directory (may contain zipped files)
        n_samples: If set, subsample to this many total samples
        seed: Random seed for subsampling
    
    Returns:
        texts: List of full texts (dialogue context + response)
        labels: Array of labels (0=factual, 1=hallucinated)
        prompts: List of dialogue contexts only
    """
    import zipfile
    import os
    
    factual_texts = []
    hallucinated_texts = []
    prompts = []
    
    # Try to find data files (could be .json, .jsonl, or .zip)
    if not data_path.exists():
        print(f"Warning: HalluDial path {data_path} does not exist. Returning empty dataset.")
        return [], np.array([]), []
    
    json_files = []
    
    # Check for zip files and extract
    for file in data_path.glob("*.zip"):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            # Extract to temporary location
            extract_path = data_path / "extracted"
            extract_path.mkdir(exist_ok=True)
            zip_ref.extractall(extract_path)
            
            # Find JSON files in extracted content
            for extracted_file in extract_path.rglob("*.json*"):
                json_files.append(extracted_file)
    
    # Also check for direct JSON files
    json_files.extend(list(data_path.glob("*.json")))
    json_files.extend(list(data_path.glob("*.jsonl")))
    
    if not json_files:
        print(f"Warning: No JSON files found in {data_path}. Returning empty dataset.")
        return [], np.array([]), []
    
    # Load data from all JSON files
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                # Try loading as JSON array first
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = data
                    else:
                        samples = [data]
                except json.JSONDecodeError:
                    # If that fails, try JSONL format
                    f.seek(0)
                    samples = []
                    for line in f:
                        if line.strip():
                            try:
                                samples.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            
            # Process samples
            for item in samples:
                # HalluDial format may vary, try multiple keys
                dialogue = item.get('dialogue', item.get('context', item.get('conversation', '')))
                response_true = item.get('response', item.get('true_response', item.get('correct_response', '')))
                response_hall = item.get('hallucinated_response', item.get('false_response', item.get('incorrect_response', '')))
                
                # Also check for explicit labels
                if not response_hall and 'label' in item:
                    if item['label'] in ['hallucinated', 'false', 'incorrect', 1]:
                        response_hall = item.get('response', '')
                        response_true = ''
                    elif item['label'] in ['factual', 'true', 'correct', 0]:
                        response_true = item.get('response', '')
                        response_hall = ''
                
                prompt = f"Dialogue: {dialogue}\nResponse:"
                
                if response_true:
                    factual_texts.append(f"{prompt} {response_true}")
                    prompts.append(prompt)
                    
                if response_hall:
                    hallucinated_texts.append(f"{prompt} {response_hall}")
                    if response_hall and not response_true:
                        prompts.append(prompt)
        
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    if not factual_texts and not hallucinated_texts:
        print("Warning: No valid samples loaded from HalluDial. Returning empty dataset.")
        return [], np.array([]), []
    
    # Balance classes
    np.random.seed(seed)
    min_samples = min(len(factual_texts), len(hallucinated_texts))
    
    if min_samples == 0:
        # If we only have one class, duplicate prompts with different responses
        print(f"Warning: HalluDial has imbalanced classes (factual={len(factual_texts)}, hall={len(hallucinated_texts)})")
        if len(factual_texts) == 0:
            factual_texts = hallucinated_texts[:len(hallucinated_texts)]
        if len(hallucinated_texts) == 0:
            hallucinated_texts = factual_texts[:len(factual_texts)]
        min_samples = min(len(factual_texts), len(hallucinated_texts))
    
    if n_samples is not None:
        min_samples = min(min_samples, n_samples // 2)
    
    indices = np.random.permutation(min_samples)
    
    factual_subset = [factual_texts[i] for i in indices[:min_samples]]
    hall_subset = [hallucinated_texts[i] for i in indices[:min_samples]]
    prompts_subset = [prompts[i] if i < len(prompts) else prompts[0] for i in indices[:min_samples]]
    
    texts = factual_subset + hall_subset
    labels = np.array([0] * len(factual_subset) + [1] * len(hall_subset))
    all_prompts = prompts_subset + prompts_subset
    
    print(f"Loaded HalluDial: {len(factual_subset)} factual + {len(hall_subset)} hallucinated = {len(texts)} total")
    
    return texts, labels, all_prompts


def load_dataset(
    dataset_name: str,
    n_samples: Optional[int] = None,
    seed: int = 42,
    data_dir: Path = Path("c:/Users/cheru/Downloads/llm-hallucinations"),
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Unified dataset loading interface.
    
    Args:
        dataset_name: One of 'halueval_qa', 'halueval_dialogue', 'halueval_summarization', 'musique'
        n_samples: Total number of samples (half factual, half hallucinated)
        seed: Random seed
        data_dir: Base directory containing datasets
    
    Returns:
        texts, labels, prompts
    """
    loaders = {
        'halueval_qa': lambda: load_halueval_qa(
            data_dir / "HaluEval/data/qa_data.json", n_samples, seed
        ),
        'halueval_dialogue': lambda: load_halueval_dialogue(
            data_dir / "HaluEval/data/dialogue_data.json", n_samples, seed
        ),
        'halueval_summarization': lambda: load_halueval_summarization(
            data_dir / "HaluEval/data/summarization_data.json", n_samples, seed
        ),
        'musique': lambda: load_musique(
            data_dir / "musique_data/data", n_samples, seed
        ),
        'truthfulqa': lambda: load_truthfulqa(
            data_dir / "TruthfulQA/TruthfulQA.csv", n_samples, seed
        ),
        'halludial': lambda: load_halludial(
            data_dir / "HalluDial", n_samples, seed
        ),
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name]()


def train_test_split_data(
    texts: List[str],
    labels: np.ndarray,
    prompts: List[str],
    test_size: float = 0.3,
    seed: int = 42,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Split data into train and test sets, stratified by label.
    
    Returns:
        train_texts, test_texts, train_labels, test_labels, train_prompts, test_prompts
    """
    np.random.seed(seed)
    
    n = len(texts)
    n_test = int(n * test_size)
    n_train = n - n_test
    
    # Stratified split
    factual_idx = np.where(labels == 0)[0]
    hall_idx = np.where(labels == 1)[0]
    
    np.random.shuffle(factual_idx)
    np.random.shuffle(hall_idx)
    
    n_test_per_class = n_test // 2
    
    test_idx = np.concatenate([
        factual_idx[:n_test_per_class],
        hall_idx[:n_test_per_class]
    ])
    train_idx = np.concatenate([
        factual_idx[n_test_per_class:],
        hall_idx[n_test_per_class:]
    ])
    
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    train_prompts = [prompts[i] for i in train_idx]
    test_prompts = [prompts[i] for i in test_idx]
    
    return train_texts, test_texts, train_labels, test_labels, train_prompts, test_prompts
