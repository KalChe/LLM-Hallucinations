"""
Configuration for all experiments.

Centralized configuration management for reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    hf_name: str
    num_layers: int
    hidden_dim: int
    steering_layers: List[int]
    
    @property
    def all_layers(self) -> List[int]:
        return list(range(self.num_layers))


# Model registry
MODELS: Dict[str, ModelConfig] = {
    "llama-3.2-1b": ModelConfig(
        name="llama-3.2-1b",
        hf_name="meta-llama/Llama-3.2-1B",
        num_layers=16,
        hidden_dim=2048,
        steering_layers=[8, 9, 10, 11, 12, 13, 14, 15],
    ),
    "llama-3.2-3b": ModelConfig(
        name="llama-3.2-3b",
        hf_name="meta-llama/Llama-3.2-3B",
        num_layers=28,
        hidden_dim=3072,
        steering_layers=[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    ),
    "gemma-2-2b": ModelConfig(
        name="gemma-2-2b",
        hf_name="google/gemma-2-2b",
        num_layers=26,
        hidden_dim=2304,
        steering_layers=[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    ),
    "qwen-2.5-1.5b": ModelConfig(
        name="qwen-2.5-1.5b",
        hf_name="Qwen/Qwen2.5-1.5B",
        num_layers=28,
        hidden_dim=1536,
        steering_layers=[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    ),
    "phi-2": ModelConfig(
        name="phi-2",
        hf_name="microsoft/phi-2",
        num_layers=32,
        hidden_dim=2560,
        steering_layers=[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    )
}


@dataclass
class ExperimentConfig:
    """Global experiment configuration."""
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.resolve())
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.resolve())
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.resolve() / "figs")
    
    # Data settings
    n_samples: int = 5000  # Total samples (2500 factual + 2500 hallucinated)
    test_split: float = 0.3
    
    # Model settings
    batch_size: int = 4
    max_length: int = 256
    
    # Computation
    seed: int = 42
    device: str = "cuda"
    
    # Ablation settings
    ablation_layers: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 15])
    steering_magnitudes: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    basin_radii: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0])
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class FigureConfig:
    """Publication figure settings."""
    
    # Dimensions (single column: 3.5", double column: 7")
    single_col_width: float = 3.5
    double_col_width: float = 7.0
    
    # Style
    dpi: int = 300
    format: str = "pdf"
    
    # Colors (academic style - gray + burgundy)
    colors: Dict[str, str] = field(default_factory=lambda: {
        'factual': '#4a90d9',      # Blue
        'hallucination': '#d9534f', # Red
        'primary': '#2c3e50',       # Dark blue-gray
        'secondary': '#8b1a1a',     # Burgundy
        'tertiary': '#27ae60',      # Green
        'gray': '#7f8c8d',          # Gray
        'light_gray': '#bdc3c7',    # Light gray
    })
    
    # Font settings
    font_family: str = 'serif'
    font_size: int = 10
    title_size: int = 11
    label_size: int = 10
    legend_size: int = 9
    
    @property
    def style_dict(self) -> Dict:
        return {
            'font.family': self.font_family,
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.label_size,
            'legend.fontsize': self.legend_size,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
        }


# Default configurations
DEFAULT_CONFIG = ExperimentConfig()
DEFAULT_FIGURE_CONFIG = FigureConfig()
