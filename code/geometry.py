# geometric metrics for hallucination basin analysis

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.covariance import LedoitWolf
from scipy.stats import norm
import warnings
from scipy.linalg import svdvals


@dataclass
class BasinGeometry:
    # computed basin geometry for a single layer
    layer_idx: int
    mu_factual: np.ndarray
    mu_hallucinated: np.ndarray
    steering_vector: np.ndarray
    centroid_separation: float
    mahalanobis_distance_sq: float
    corrected_mahalanobis_distance: float
    hallucination_risk_probability: float
    fisher_ratio: float
    effective_dim_factual: float
    effective_dim_hallucinated: float
    ledoit_wolf_shrinkage: float
    # New universal metrics
    effective_rank_factual: float = 0.0
    effective_rank_hallucinated: float = 0.0
    effective_rank_combined: float = 0.0
    flow_magnitude: float = 0.0


def compute_effective_dimensionality(X: np.ndarray, method: str = 'participation_ratio') -> float:
    # compute effective dimensionality of data distribution
    if X.shape[0] < 2:
        return 0.0
    
    # Center data
    X_centered = X - X.mean(axis=0)
    
    # Compute covariance eigenvalues
    try:
        # Use SVD for numerical stability
        _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
        eigenvalues = (s ** 2) / (X.shape[0] - 1)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
        
        if len(eigenvalues) == 0:
            return 0.0
        
        if method == 'participation_ratio':
            # Participation ratio: (sum λ_i)^2 / sum λ_i^2
            total = eigenvalues.sum()
            if total < 1e-10:
                return 0.0
            eff_dim = (total ** 2) / (eigenvalues ** 2).sum()
        else:
            # Eigenvalue decay: number of eigenvalues explaining 90% variance
            explained_var = np.cumsum(eigenvalues) / eigenvalues.sum()
            eff_dim = np.searchsorted(explained_var, 0.9) + 1
        
        return float(eff_dim)
        
    except np.linalg.LinAlgError:
        return 0.0


def compute_effective_rank(X: np.ndarray, epsilon: float = 1e-10) -> float:
    # compute effective rank via entropy of singular values (manifold collapse metric)
    if X.shape[0] < 2:
        return 1.0
    
    # Center the data
    X_centered = X - X.mean(axis=0)
    
    # Compute singular values (more stable than eigenvalues of covariance)
    try:
        singular_values = svdvals(X_centered)
        
        # Remove near-zero singular values
        singular_values = singular_values[singular_values > epsilon]
        
        if len(singular_values) == 0:
            return 1.0
        
        # Normalize to probability distribution
        total = singular_values.sum()
        if total < epsilon:
            return 1.0
            
        p = singular_values / total
        
        # Compute Shannon entropy
        # Use -p*log(p) with safe handling of zeros
        p_safe = np.maximum(p, epsilon)
        entropy = -np.sum(p * np.log(p_safe))
        
        # Effective rank is exp(entropy)
        r_eff = np.exp(entropy)
        
        return float(r_eff)
        
    except (np.linalg.LinAlgError, ValueError):
        return 1.0


def compute_layer_flow_magnitude(
    hidden_states_current: np.ndarray,
    hidden_states_previous: np.ndarray
) -> float:
    # compute flow magnitude between consecutive layers (reasoning collapse metric)
    if hidden_states_current.shape != hidden_states_previous.shape:
        return 0.0
    
    # Compute layer-to-layer differences
    delta_h = hidden_states_current - hidden_states_previous
    
    # L2 norm for each sample
    norms = np.linalg.norm(delta_h, axis=1)
    
    # Return mean flow magnitude
    return float(np.mean(norms))


def compute_unified_risk_score(
    distance_to_basin: float,
    effective_rank: float,
    flow_magnitude: float,
    hidden_dim: int,
    basin_threshold: float = 0.5,
    flow_scale: float = 1.0
) -> float:
    # compute unified risk score combining all three collapse types
    # Type 1: Basin Attraction (inverted - closer = higher risk)
    basin_risk = 1.0 - min(distance_to_basin / basin_threshold, 1.0)
    
    # Type 2: Manifold Trapping (normalized effective rank drop)
    manifold_risk = 1.0 - (effective_rank / max(hidden_dim, 1.0))
    
    # Type 3: Flow Stagnation (exponential decay)
    flow_risk = np.exp(-flow_magnitude / max(flow_scale, 0.1))
    
    # Take maximum of all three risks
    unified_risk = max(basin_risk, manifold_risk, flow_risk)
    
    return float(np.clip(unified_risk, 0.0, 1.0))


def compute_fisher_ratio(
    factual_states: np.ndarray,
    hallucinated_states: np.ndarray,
) -> float:
    # compute fisher discriminant ratio
    if factual_states.shape[0] < 2 or hallucinated_states.shape[0] < 2:
        return 0.0
    
    mu_f = factual_states.mean(axis=0)
    mu_h = hallucinated_states.mean(axis=0)
    
    between_class = np.linalg.norm(mu_f - mu_h) ** 2
    
    var_f = np.var(factual_states, axis=0, ddof=1).sum()
    var_h = np.var(hallucinated_states, axis=0, ddof=1).sum()
    within_class = var_f + var_h
    
    if within_class < 1e-10:
        return 0.0
    
    return between_class / within_class


def compute_mahalanobis_distance(
    factual_states: np.ndarray,
    hallucinated_states: np.ndarray,
    apply_correction: bool = True,
) -> Tuple[float, float, float, float]:
    # compute regularized mahalanobis distance between class centers
    mu_f = factual_states.mean(axis=0)
    mu_h = hallucinated_states.mean(axis=0)
    delta_mu = mu_f - mu_h
    
    # Combined data for covariance estimation
    X = np.vstack([factual_states, hallucinated_states])
    n, d = X.shape
    
    # Clean data
    X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
    delta_mu = np.nan_to_num(delta_mu, nan=0.0)
    
    try:
        # Ledoit-Wolf shrinkage estimator
        lw = LedoitWolf().fit(X)
        Sigma_inv = lw.precision_
        shrinkage = lw.shrinkage_
        
        # Squared Mahalanobis distance
        delta_sq = float(delta_mu @ Sigma_inv @ delta_mu)
        
        # Finite-sample correction (when n > d)
        if apply_correction and delta_sq > 1e-10 and n > d + 10:
            correction = (1 + (4*d)/(n * delta_sq)) * (n / (n - d))
            delta_corrected = np.sqrt(delta_sq) / np.sqrt(correction)
        else:
            delta_corrected = np.sqrt(max(0, delta_sq))
        
        # Risk probability
        risk_prob = float(norm.cdf(-delta_corrected / 2))
        
        return delta_sq, delta_corrected, risk_prob, shrinkage
        
    except Exception as e:
        warnings.warn(f"Mahalanobis computation failed: {e}")
        return 0.0, 0.0, 0.5, 0.0


def compute_basin_geometry(
    hidden_states: Dict[int, np.ndarray],
    labels: np.ndarray,
    verbose: bool = True,
) -> Dict[int, BasinGeometry]:
    # compute full basin geometry for each layer
    results = {}
    
    factual_mask = labels == 0
    hall_mask = labels == 1
    
    for layer_idx in sorted(hidden_states.keys()):
        if verbose:
            print(f"  Layer {layer_idx}...", end=" ", flush=True)
        
        H = hidden_states[layer_idx].astype(np.float32)
        
        # Clean NaN/Inf
        H = np.nan_to_num(H, nan=0.0, posinf=1e4, neginf=-1e4)
        
        factual_states = H[factual_mask]
        hall_states = H[hall_mask]
        
        if len(factual_states) < 10 or len(hall_states) < 10:
            if verbose:
                print("SKIP (insufficient samples)")
            continue
        
        # Centroids
        mu_factual = factual_states.mean(axis=0)
        mu_hall = hall_states.mean(axis=0)
        
        # Steering vector (from hallucination to factual)
        steering_vec = mu_factual - mu_hall
        centroid_sep = np.linalg.norm(steering_vec)
        
        # Fisher ratio
        fisher = compute_fisher_ratio(factual_states, hall_states)
        
        # Mahalanobis distance
        delta_sq, delta_corr, risk_prob, shrinkage = compute_mahalanobis_distance(
            factual_states, hall_states
        )
        
        # Effective dimensionality (original)
        eff_dim_f = compute_effective_dimensionality(factual_states)
        eff_dim_h = compute_effective_dimensionality(hall_states)
        
        # NEW: Effective Rank (manifold collapse detection)
        eff_rank_f = compute_effective_rank(factual_states)
        eff_rank_h = compute_effective_rank(hall_states)
        eff_rank_combined = compute_effective_rank(H)
        
        # NEW: Flow magnitude (will be computed in layer-wise analysis)
        # For now, set to 0 (will be updated in experiments that have previous layer)
        flow_mag = 0.0
        
        results[layer_idx] = BasinGeometry(
            layer_idx=layer_idx,
            mu_factual=mu_factual,
            mu_hallucinated=mu_hall,
            steering_vector=steering_vec,
            centroid_separation=centroid_sep,
            mahalanobis_distance_sq=delta_sq,
            corrected_mahalanobis_distance=delta_corr,
            hallucination_risk_probability=risk_prob,
            fisher_ratio=fisher,
            effective_dim_factual=eff_dim_f,
            effective_dim_hallucinated=eff_dim_h,
            ledoit_wolf_shrinkage=shrinkage,
            effective_rank_factual=eff_rank_f,
            effective_rank_hallucinated=eff_rank_h,
            effective_rank_combined=eff_rank_combined,
            flow_magnitude=flow_mag,
        )
        
        if verbose:
            print(f"sep={centroid_sep:.2f}, Fisher={fisher:.4f}, δ²={delta_sq:.2f}, r_eff={eff_rank_combined:.1f}")
    
    return results


def compute_geometric_features(
    hidden_states: Dict[int, np.ndarray],
    basin_geometry: Dict[int, BasinGeometry],
    aggregate: str = 'mean',
) -> np.ndarray:
    # compute geometric feature vectors for detection
    layers = sorted(hidden_states.keys())
    n_samples = hidden_states[layers[0]].shape[0]
    
    layer_features = []
    
    for i, layer_idx in enumerate(layers):
        if layer_idx not in basin_geometry:
            continue
            
        H = hidden_states[layer_idx]
        geom = basin_geometry[layer_idx]
        
        # Distance to hallucination basin center
        d_basin = np.linalg.norm(H - geom.mu_hallucinated, axis=1)
        
        # Fisher ratio (same for all samples)
        rho_fisher = np.full(n_samples, geom.fisher_ratio)
        
        # Residual flow (layer-to-layer change)
        if i > 0:
            prev_layer = layers[i-1]
            if prev_layer in hidden_states:
                H_prev = hidden_states[prev_layer]
                sigma_flow = np.linalg.norm(H - H_prev, axis=1)
            else:
                sigma_flow = np.zeros(n_samples)
        else:
            sigma_flow = np.zeros(n_samples)
        
        layer_features.append(np.stack([d_basin, rho_fisher, sigma_flow], axis=1))
    
    if not layer_features:
        return np.zeros((n_samples, 3))
    
    features = np.stack(layer_features, axis=0)  # (n_layers, n_samples, 3)
    
    if aggregate == 'mean':
        return features.mean(axis=0)
    elif aggregate == 'last':
        return features[-1]
    elif aggregate == 'concat':
        return features.transpose(1, 0, 2).reshape(n_samples, -1)
    else:
        return features.mean(axis=0)


def compute_spectral_radius(
    model,
    layer_idx: int,
    reference_state: np.ndarray,
    epsilon: float = 1e-4,
    n_samples: int = 100,
) -> float:
    # estimate spectral radius of jacobian at reference state
    import torch
    
    d = len(reference_state)
    device = next(model.parameters()).device
    
    # Random initial vector
    v = np.random.randn(d)
    v = v / np.linalg.norm(v)
    
    # Power iteration
    for _ in range(20):
        # Approximate J @ v via finite differences
        h_plus = torch.tensor(reference_state + epsilon * v, dtype=torch.float16, device=device)
        h_minus = torch.tensor(reference_state - epsilon * v, dtype=torch.float16, device=device)
        
        # This would require hooking into the model's forward pass
        # For now, return a placeholder
        # In practice, this requires model-specific implementation
        pass
    
    # Placeholder - actual implementation needs model hooks
    return 0.95  # Typical value < 1


class GeometricRiskScorer:
    # compute geometric risk scores for hallucination detection
    
    def __init__(self, basin_geometry: Dict[int, BasinGeometry]):
        self.basin_geometry = basin_geometry
        self.weights = None
        
    def fit(self, features: np.ndarray, labels: np.ndarray):
        # learn optimal weights via logistic regression
        from sklearn.linear_model import LogisticRegression
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(features, labels)
        
        self.weights = clf.coef_[0]
        self.bias = clf.intercept_[0]
        self.clf = clf
        
        return self
    
    def predict_risk(self, features: np.ndarray) -> np.ndarray:
        # predict hallucination risk probability
        if self.weights is None:
            raise ValueError("Must call fit() first")
        
        return self.clf.predict_proba(features)[:, 1]
    
    def get_decision_boundary(self) -> Dict:
        # return decision boundary parameters
        return {
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
        }
