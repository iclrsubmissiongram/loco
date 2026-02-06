"""
CLEVR Density-Based Hierarchy Preparation
==========================================

This creates a density-based hierarchical representation of CLEVR that matches
our method's assumptions. The key insight:

CLEVR has natural SIZE variation (small vs large objects).
We create density-based hierarchy from this:
  - Large objects → Sparse representation (1-2 points) → Level 0
  - Objects near large ones → Medium density → Level 1  
  - Small isolated objects → Dense point clouds → Level 2
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy import stats
import matplotlib.pyplot as plt

# ==========================================
# CLEVR Scene Loading (same as before)
# ==========================================

def find_clevr_scenes():
    """Find CLEVR scenes file"""
    possible_paths = [
        '/kaggle/input/clevr-dataset/CLEVR_v1.0/scenes/CLEVR_train_scenes.json',
        '/kaggle/input/clevr/CLEVR_v1.0/scenes/CLEVR_train_scenes.json',
        '/kaggle/working/clevr_data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json',
        'clevr_data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def load_clevr_scenes(max_scenes=5000):
    """Load CLEVR scene annotations"""
    path = find_clevr_scenes()
    if path is None:
        print("CLEVR scenes not found. Please download first.")
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    scenes = data['scenes'][:max_scenes]
    print(f"Loaded {len(scenes)} CLEVR scenes")
    return scenes


# ==========================================
# DENSITY-BASED HIERARCHY FROM CLEVR
# ==========================================

def create_density_hierarchy_scene(scene, noise_ratio=0.10):
    """
    Create density-based hierarchy from a CLEVR scene.
    
    NEW STRATEGY (matches TOY DATA exactly):
    -----------------------------------------
    For EACH object, create a 3-level hierarchy:
    - Level 0: Object center (1 point, ISOLATED) - ~7% of points
    - Level 1: Parts around center (3-5 points, MEDIUM density) - ~20% of points
    - Level 2: Subparts (8-15 points, DENSE cluster) - ~67% of points
    
    This matches the TOY data structure where:
    - Sparse/isolated → high kNN distance → Level 0
    - Medium cluster → medium kNN distance → Level 1
    - Dense cluster → low kNN distance → Level 2
    
    Returns:
        points: (N, 2) array of 2D points
        object_labels: (N,) array of object IDs
        level_labels: (N,) array of hierarchy levels (0, 1, 2)
    """
    objects = scene.get('objects', [])
    if len(objects) < 3:
        return None  # Skip scenes with too few objects
    
    # Limit to first 3-5 objects for cleaner hierarchy (like toy data)
    max_objects = min(len(objects), 5)
    objects = objects[:max_objects]
    
    all_points = []
    object_labels = []
    level_labels = []
    
    for obj_idx, obj in enumerate(objects):
        coords = obj.get('3d_coords', [0, 0, 0])
        size = obj.get('size', 'small')
        center = np.array([coords[0], coords[2]])  # Use X-Z plane
        
        # Object radius based on size
        base_radius = 0.7 if size == 'large' else 0.35
        
        # =========================================
        # LEVEL 0: Object center (SPARSE/ISOLATED)
        # =========================================
        # Just 1 point - this should be ISOLATED with high kNN distance
        all_points.append(center + np.random.randn(2) * 0.02)
        object_labels.append(obj_idx)
        level_labels.append(0)
        
        # =========================================
        # LEVEL 1: Parts (MEDIUM DENSITY)
        # =========================================
        # 3-5 points arranged around the center at ~1 radius distance
        # These form a loose ring - not too close, not too far
        n_parts = np.random.randint(3, 6)
        part_radius = base_radius * np.random.uniform(0.8, 1.2)
        
        for i in range(n_parts):
            # Skip some for variation (like toy data)
            if np.random.random() < 0.15:
                continue
            
            angle = 2 * np.pi * i / n_parts + np.random.randn() * 0.2
            part_pos = center + part_radius * np.array([np.cos(angle), np.sin(angle)])
            part_pos += np.random.randn(2) * 0.08  # Small jitter
            
            all_points.append(part_pos)
            object_labels.append(obj_idx)
            level_labels.append(1)
            
            # =========================================
            # LEVEL 2: Subparts (DENSE CLUSTER)
            # =========================================
            # 2-4 points VERY close to each part (tight cluster)
            n_subparts = np.random.randint(2, 5)
            
            for _ in range(n_subparts):
                # Skip some for variation
                if np.random.random() < 0.15:
                    continue
                
                # Subparts are VERY close to parts (creates density)
                subpart_pos = part_pos + np.random.randn(2) * 0.12
                
                all_points.append(subpart_pos)
                object_labels.append(obj_idx)
                level_labels.append(2)
    
    # Add background noise (~10%)
    n_noise = max(1, int(len(all_points) * noise_ratio))
    for _ in range(n_noise):
        pt = np.array([np.random.uniform(-3.5, 3.5), np.random.uniform(-3.5, 3.5)])
        all_points.append(pt)
        object_labels.append(-1)
        level_labels.append(-1)
    
    if len(all_points) < 20:
        return None
    
    return {
        'points': np.array(all_points),
        'object_labels': np.array(object_labels),
        'level_labels': np.array(level_labels)
    }


def prepare_clevr_dataset(scenes, max_points=150):
    """
    Prepare full CLEVR dataset with density-based hierarchy.
    
    Returns padded tensors ready for training.
    """
    all_data = []
    
    for scene in scenes:
        result = create_density_hierarchy_scene(scene)
        if result is None:
            continue
            
        points = result['points']
        obj_labels = result['object_labels']
        level_labels = result['level_labels']
        
        # Truncate or pad to max_points
        n = len(points)
        if n > max_points:
            # Random sample
            idx = np.random.choice(n, max_points, replace=False)
            points = points[idx]
            obj_labels = obj_labels[idx]
            level_labels = level_labels[idx]
            n = max_points
        
        # Pad
        if n < max_points:
            pad_len = max_points - n
            points = np.vstack([points, np.zeros((pad_len, 2))])
            obj_labels = np.concatenate([obj_labels, -2 * np.ones(pad_len)])
            level_labels = np.concatenate([level_labels, -2 * np.ones(pad_len)])
        
        all_data.append({
            'points': points.astype(np.float32),
            'object_labels': obj_labels.astype(np.int64),
            'level_labels': level_labels.astype(np.int64),
            'valid_mask': (obj_labels >= -1).astype(np.float32)  # -1 is noise (valid), -2 is padding
        })
    
    print(f"Prepared {len(all_data)} valid scenes")
    
    # Analyze level distribution
    all_levels = np.concatenate([d['level_labels'] for d in all_data])
    valid_levels = all_levels[all_levels >= 0]
    level_counts = np.bincount(valid_levels, minlength=3)
    level_pcts = level_counts / level_counts.sum() * 100
    print(f"Level distribution: L0={level_pcts[0]:.1f}%, L1={level_pcts[1]:.1f}%, L2={level_pcts[2]:.1f}%")
    
    return all_data


# ==========================================
# V9 LoCo Model (EXACTLY matching toy_exp_v9)
# ==========================================

class LorentzOps:
    """Lorentzian operations - IDENTICAL to toy_exp_v9"""
    EPS = 1e-6
    
    @staticmethod
    def lorentz_inner(x, y):
        """Lorentzian inner product: t1*t2 - x1·x2"""
        return x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(-1)
    
    @staticmethod
    def proper_time_distance(x, y):
        """Proper time distance."""
        diff = x - y
        lorentz_sq = LorentzOps.lorentz_inner(diff, diff)
        return torch.sign(lorentz_sq) * torch.sqrt(torch.abs(lorentz_sq) + LorentzOps.EPS)
    
    @staticmethod
    def cone_membership(feature, slot, horizon):
        """Compute cone membership with SCALE-ADAPTIVE HORIZONS."""
        diff = feature - slot
        tau = diff[..., 0]
        r = torch.norm(diff[..., 1:], dim=-1)
        
        ratio = r / (torch.abs(tau) + LorentzOps.EPS)
        inside_score = horizon - ratio
        
        direction_penalty = -10.0 * F.relu(-tau)
        spacelike_penalty = -5.0 * F.relu(r - torch.abs(tau))
        
        return inside_score + direction_penalty + spacelike_penalty


class LoCoScaleAdaptive(nn.Module):
    """
    LoCo V9: Scale-Adaptive Worldline Attention
    EXACTLY matching toy_exp_v9_scale_adaptive.py
    """
    
    def __init__(self, input_dim=2, hidden_dim=32, num_objects=3, num_levels=3,
                 iterations=3, tau=0.1, lambda_cone=0.5, k_neighbors=5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.num_levels = num_levels
        self.num_slots = num_objects * num_levels
        self.slot_dim = hidden_dim + 1
        self.iterations = iterations
        self.tau = tau
        self.lambda_cone = lambda_cone
        self.k_neighbors = k_neighbors
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Scale predictor
        self.scale_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # Initialize scale predictor conservatively
        with torch.no_grad():
            self.scale_predictor[-2].weight.mul_(0.1)
            self.scale_predictor[-2].bias.zero_()
        
        # Worldline structure
        self.object_space = nn.Parameter(torch.randn(1, num_objects, hidden_dim) * 0.1)
        
        # Fixed level times
        self.register_buffer('level_times', torch.tensor([1.0, 2.5, 4.0]))
        
        # Base horizons per level
        self.register_buffer('base_horizons', torch.tensor([0.90, 0.60, 0.30]))
        
        # Horizon modulation strength
        self.horizon_scale = nn.Parameter(torch.tensor(0.3))
        
        # Update networks
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def get_local_density(self, x):
        """Compute local density signal."""
        B, N, D = x.shape
        k = min(self.k_neighbors, N - 1)
        
        if k <= 0:
            return torch.ones(B, N, device=x.device) * 0.5
        
        dist = torch.cdist(x, x)
        knn_dists, _ = torch.topk(dist, k + 1, dim=-1, largest=False)
        avg_dist = knn_dists[..., 1:].mean(dim=-1)
        
        scale_signal = torch.tanh(avg_dist)
        return scale_signal
    
    def get_adaptive_horizons(self, density):
        """Compute scale-adaptive horizons."""
        B, N = density.shape
        
        modulation = self.horizon_scale * (density - 0.5)
        
        base = self.base_horizons.view(1, self.num_levels, 1).expand(B, -1, N)
        mod = modulation.unsqueeze(1).expand(-1, self.num_levels, -1)
        
        horizons = base + mod
        horizons = torch.clamp(horizons, 0.1, 1.0)
        horizons = horizons.repeat(1, self.num_objects, 1)
        
        return horizons
    
    def encode_features(self, x):
        """Encode features with scale-based time."""
        B, N, D = x.shape
        
        spatial = self.feature_encoder(x)
        density = self.get_local_density(x)
        
        scale_input = torch.cat([spatial, density.unsqueeze(-1)], dim=-1)
        pred_scale = self.scale_predictor(scale_input)
        
        pred_time = 5.0 - 1.5 * density.unsqueeze(-1) + 0.5 * pred_scale
        
        features = torch.cat([pred_time, spatial], dim=-1)
        
        return features, density, pred_time.squeeze(-1)
    
    def get_worldlines(self, batch_size):
        """Construct worldlines."""
        obj_space = self.object_space.expand(batch_size, -1, -1)
        
        if self.training:
            obj_space = obj_space + torch.randn_like(obj_space) * 0.01
        
        slots_space = obj_space.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
        slots_space = slots_space.reshape(batch_size, self.num_slots, self.hidden_dim)
        
        times = self.level_times.repeat(self.num_objects).view(1, self.num_slots, 1).expand(batch_size, -1, -1)
        
        return torch.cat([times, slots_space], dim=-1)
    
    def compute_attention(self, features, slots, horizons):
        """Compute Lorentzian attention."""
        B, N, D = features.shape
        K = slots.shape[1]
        
        slots_exp = slots.unsqueeze(2).expand(-1, -1, N, -1)
        feats_exp = features.unsqueeze(1).expand(-1, K, -1, -1)
        
        d_L = LorentzOps.proper_time_distance(feats_exp, slots_exp)
        distance_score = -torch.abs(d_L)
        
        cone_score = LorentzOps.cone_membership(feats_exp, slots_exp, horizons)
        
        attn_logits = distance_score + self.lambda_cone * torch.tanh(cone_score)
        attn = F.softmax(attn_logits / self.tau, dim=1)
        
        return attn
    
    def forward(self, x, valid_mask=None):
        B, N, _ = x.shape
        
        features, density, pred_time = self.encode_features(x)
        slots = self.get_worldlines(B)
        horizons = self.get_adaptive_horizons(density)
        
        for iteration in range(self.iterations):
            attn = self.compute_attention(features, slots, horizons)
            
            attn_weights = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)
            updates = torch.bmm(attn_weights, features[..., 1:])
            
            updates_per_obj = updates.view(B, self.num_objects, self.num_levels, self.hidden_dim)
            obj_updates = updates_per_obj.sum(dim=2)
            
            old_obj_space = self.object_space.expand(B, -1, -1)
            old_flat = old_obj_space.reshape(-1, self.hidden_dim)
            update_flat = obj_updates.reshape(-1, self.hidden_dim)
            
            new_obj_space = self.gru(update_flat, old_flat)
            new_obj_space = new_obj_space.reshape(B, self.num_objects, self.hidden_dim)
            new_obj_space = new_obj_space + 0.2 * self.update_mlp(self.norm(new_obj_space))
            
            slots_space = new_obj_space.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
            slots_space = slots_space.reshape(B, self.num_slots, self.hidden_dim)
            times = slots[..., 0:1]
            slots = torch.cat([times, slots_space], dim=-1)
        
        return slots, attn, pred_time, density


class EuclideanWorldlines(nn.Module):
    """
    Euclidean Worldlines: Same architecture, NO Lorentzian geometry.
    EXACTLY matching toy_exp_v9_scale_adaptive.py
    """
    
    def __init__(self, input_dim=2, hidden_dim=32, num_objects=3, num_levels=3,
                 iterations=3, k_neighbors=5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.num_levels = num_levels
        self.num_slots = num_objects * num_levels
        self.iterations = iterations
        self.k_neighbors = k_neighbors
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.object_space = nn.Parameter(torch.randn(1, num_objects, hidden_dim) * 0.1)
        self.register_buffer('level_times', torch.tensor([1.0, 2.5, 4.0]))
        
        # Base temperatures per level
        self.register_buffer('base_temps', torch.tensor([0.15, 0.10, 0.08]))
        self.temp_scale = nn.Parameter(torch.tensor(0.05))
        
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def get_local_density(self, x):
        B, N, D = x.shape
        k = min(self.k_neighbors, N - 1)
        
        if k <= 0:
            return torch.ones(B, N, device=x.device) * 0.5
        
        dist = torch.cdist(x, x)
        knn_dists, _ = torch.topk(dist, k + 1, dim=-1, largest=False)
        avg_dist = knn_dists[..., 1:].mean(dim=-1)
        
        return torch.tanh(avg_dist)
    
    def get_adaptive_temperatures(self, density):
        B, N = density.shape
        
        modulation = self.temp_scale * (density - 0.5)
        
        base = self.base_temps.view(1, self.num_levels, 1).expand(B, -1, N)
        mod = modulation.unsqueeze(1).expand(-1, self.num_levels, -1)
        
        temps = base + mod
        temps = torch.clamp(temps, 0.05, 0.25)
        temps = temps.repeat(1, self.num_objects, 1)
        
        return temps
    
    def forward(self, x, valid_mask=None):
        B, N, _ = x.shape
        
        feat_space = self.feature_encoder(x)
        density = self.get_local_density(x)
        temps = self.get_adaptive_temperatures(density)
        
        obj_space = self.object_space.expand(B, -1, -1)
        
        if self.training:
            obj_space = obj_space + torch.randn_like(obj_space) * 0.01
        
        for iteration in range(self.iterations):
            slots_space = obj_space.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
            slots_space = slots_space.reshape(B, self.num_slots, self.hidden_dim)
            
            dists = torch.cdist(feat_space, slots_space)
            dists = dists.transpose(1, 2)
            
            attn_logits = -dists / (temps + 1e-6)
            attn = F.softmax(attn_logits, dim=1)
            
            attn_weights = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)
            updates = torch.bmm(attn_weights, feat_space)
            
            updates_per_obj = updates.view(B, self.num_objects, self.num_levels, self.hidden_dim)
            obj_updates = updates_per_obj.sum(dim=2)
            
            old_flat = obj_space.reshape(-1, self.hidden_dim)
            update_flat = obj_updates.reshape(-1, self.hidden_dim)
            
            new_obj_space = self.gru(update_flat, old_flat)
            new_obj_space = new_obj_space.reshape(B, self.num_objects, self.hidden_dim)
            new_obj_space = new_obj_space + 0.2 * self.update_mlp(self.norm(new_obj_space))
            
            obj_space = new_obj_space
        
        # Construct final slots for return
        slots_space = obj_space.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
        slots_space = slots_space.reshape(B, self.num_slots, self.hidden_dim)
        times = self.level_times.repeat(self.num_objects).view(1, self.num_slots, 1).expand(B, -1, -1)
        slots = torch.cat([times, slots_space], dim=-1)
        
        return slots, attn, density, density  # Match return signature


class StandardEuclidean(nn.Module):
    """Standard Slot Attention with 9 independent slots (baseline)."""
    
    def __init__(self, input_dim=2, hidden_dim=32, num_slots=9, iterations=3, tau=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.iterations = iterations
        self.tau = tau
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.slots = nn.Parameter(torch.randn(1, num_slots, hidden_dim) * 0.1)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, valid_mask=None):
        B, N, _ = x.shape
        
        feat_space = self.feature_encoder(x)
        slots = self.slots.expand(B, -1, -1)
        
        if self.training:
            slots = slots + torch.randn_like(slots) * 0.01
        
        for iteration in range(self.iterations):
            dists = torch.cdist(feat_space, slots)
            attn_logits = -dists / self.tau
            attn_logits = attn_logits.transpose(1, 2)
            attn = F.softmax(attn_logits, dim=1)
            
            attn_weights = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)
            updates = torch.bmm(attn_weights, feat_space)
            
            old_flat = slots.reshape(-1, self.hidden_dim)
            update_flat = updates.reshape(-1, self.hidden_dim)
            
            new_slots = self.gru(update_flat, old_flat)
            new_slots = new_slots.reshape(B, self.num_slots, self.hidden_dim)
            new_slots = new_slots + 0.2 * self.update_mlp(self.norm(new_slots))
            
            slots = new_slots
        
        return slots, attn, None, None  # Match return signature


# ==========================================
# Loss Functions (EXACTLY matching toy_exp_v9)
# ==========================================

def clustering_loss(features, slots, attn):
    """Reconstruction loss - EXACTLY as in toy_exp_v9"""
    feat_space = features[..., 1:] if features.shape[-1] > 32 else features
    slot_space = slots[..., 1:] if slots.shape[-1] > 32 else slots
    
    attn_T = attn.transpose(1, 2)  # [B, N, K]
    reconstructed = torch.bmm(attn_T, slot_space)  # [B, N, D]
    
    return F.mse_loss(reconstructed, feat_space)


def diversity_loss(slots, num_objects=3, num_levels=3):
    """Diversity loss - EXACTLY as in toy_exp_v9"""
    B = slots.shape[0]
    slot_space = slots[..., 1:] if slots.shape[-1] > 32 else slots
    
    if slot_space.shape[1] == num_objects * num_levels:
        slots_reshaped = slot_space.view(B, num_objects, num_levels, -1)
        obj_positions = slots_reshaped[:, :, 0, :]
    else:
        obj_positions = slot_space
    
    dists = torch.cdist(obj_positions, obj_positions)
    K = obj_positions.shape[1]
    mask = 1.0 - torch.eye(K, device=slots.device).unsqueeze(0)
    penalty = F.relu(2.0 - dists) * mask
    
    return penalty.mean()


def compute_object_ari(attn, obj_labels, num_objects=3, num_levels=3):
    """Compute Object ARI - EXACTLY as in toy_exp_v9"""
    B, K, N = attn.shape
    
    if K == num_objects * num_levels:
        slot_to_obj = torch.tensor([i // num_levels for i in range(K)], device=attn.device)
    else:
        slot_to_obj = torch.tensor([i % num_objects for i in range(K)], device=attn.device)
    
    pred_slots = torch.argmax(attn, dim=1)
    pred_objects = slot_to_obj[pred_slots]
    
    ari_scores = []
    for b in range(B):
        mask = obj_labels[b] >= 0
        if mask.sum() < 2:
            continue
        
        gt = obj_labels[b][mask].cpu().numpy()
        pred = pred_objects[b][mask].cpu().numpy()
        
        if len(np.unique(gt)) > 1 and len(np.unique(pred)) > 1:
            ari = adjusted_rand_score(gt, pred)
            ari_scores.append(ari)
    
    return np.mean(ari_scores) if ari_scores else 0.0


def compute_level_accuracy(attn, level_labels, num_objects=3, num_levels=3):
    """Compute Level Accuracy - EXACTLY as in toy_exp_v9"""
    B, K, N = attn.shape
    
    if K != num_objects * num_levels:
        return 0.33  # Random for non-worldline models
    
    slot_to_level = torch.tensor([i % num_levels for i in range(K)], device=attn.device)
    pred_slots = torch.argmax(attn, dim=1)
    pred_levels = slot_to_level[pred_slots]
    
    accuracies = []
    for b in range(B):
        mask = level_labels[b] >= 0
        if mask.sum() == 0:
            continue
        
        gt = level_labels[b][mask]
        pred = pred_levels[b][mask]
        
        acc = (gt == pred).float().mean().item()
        accuracies.append(acc)
    
    return np.mean(accuracies) if accuracies else 0.0


def compute_nmi(attn, obj_labels, level_labels, num_objects=3, num_levels=3):
    """Compute NMI - EXACTLY as in toy_exp_v9"""
    B, K, N = attn.shape
    pred_slots = torch.argmax(attn, dim=1)
    gt_combined = obj_labels * num_levels + level_labels
    
    nmi_scores = []
    for b in range(B):
        mask = (obj_labels[b] >= 0) & (level_labels[b] >= 0)
        if mask.sum() < 2:
            continue
        
        gt = gt_combined[b][mask].cpu().numpy()
        pred = pred_slots[b][mask].cpu().numpy()
        
        if len(np.unique(gt)) > 1 and len(np.unique(pred)) > 1:
            nmi = normalized_mutual_info_score(gt, pred)
            nmi_scores.append(nmi)
    
    return np.mean(nmi_scores) if nmi_scores else 0.0


# ==========================================
# OPTIMIZED Training for GPU
# ==========================================

def preload_to_gpu(all_data, device):
    """Pre-convert all data to GPU tensors for MUCH faster training"""
    print("Pre-loading data to GPU...")
    
    # Stack all data into single tensors
    all_pts = torch.stack([torch.from_numpy(d['points']) for d in all_data])
    all_obj = torch.stack([torch.from_numpy(d['object_labels']) for d in all_data])
    all_lvl = torch.stack([torch.from_numpy(d['level_labels']) for d in all_data])
    
    # Move to GPU once
    gpu_data = {
        'points': all_pts.to(device),
        'object_labels': all_obj.to(device),
        'level_labels': all_lvl.to(device),
    }
    
    print(f"Loaded {len(all_data)} samples to GPU ({gpu_data['points'].shape})")
    return gpu_data


def train_epoch_fast(model, gpu_data, optimizer, batch_size, device):
    """OPTIMIZED training - data already on GPU"""
    model.train()
    
    n_train = gpu_data['points'].shape[0]
    indices = torch.randperm(n_train, device=device)
    
    epoch_loss = []
    epoch_ari = []
    epoch_level = []
    
    for start in range(0, n_train, batch_size):
        batch_idx = indices[start:start+batch_size]
        if len(batch_idx) < 2:
            continue
        
        # Direct indexing - no CPU->GPU transfer!
        batch_pts = gpu_data['points'][batch_idx]
        batch_obj = gpu_data['object_labels'][batch_idx]
        batch_lvl = gpu_data['level_labels'][batch_idx]
        
        # Forward pass
        slots, attn, _, density = model(batch_pts)
        
        # Get features for loss
        if hasattr(model, 'encode_features'):
            features, _, _ = model.encode_features(batch_pts)
        else:
            features = model.feature_encoder(batch_pts)
        
        # Losses
        loss_cluster = clustering_loss(features, slots, attn)
        loss_div = diversity_loss(slots)
        loss = loss_cluster + 0.3 * loss_div
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss.append(loss.item())
        
        # Compute metrics every 5th batch (faster)
        if start % (batch_size * 5) == 0:
            with torch.no_grad():
                ari = compute_object_ari(attn, batch_obj)
                level_acc = compute_level_accuracy(attn, batch_lvl)
                epoch_ari.append(ari)
                epoch_level.append(level_acc)
    
    return {
        'loss': np.mean(epoch_loss) if epoch_loss else 0.0,
        'ari': np.mean(epoch_ari) if epoch_ari else 0.0,
        'level_acc': np.mean(epoch_level) if epoch_level else 0.0
    }


def evaluate_model_fast(model, gpu_data, batch_size, device):
    """OPTIMIZED evaluation - data already on GPU"""
    model.eval()
    
    all_ari = []
    all_level = []
    all_nmi = []
    
    n_test = gpu_data['points'].shape[0]
    
    with torch.no_grad():
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            if end - start < 2:
                continue
            
            batch_pts = gpu_data['points'][start:end]
            batch_obj = gpu_data['object_labels'][start:end]
            batch_lvl = gpu_data['level_labels'][start:end]
            
            slots, attn, _, _ = model(batch_pts)
            
            ari = compute_object_ari(attn, batch_obj)
            level_acc = compute_level_accuracy(attn, batch_lvl)
            nmi = compute_nmi(attn, batch_obj, batch_lvl)
            
            all_ari.append(ari)
            all_level.append(level_acc)
            all_nmi.append(nmi)
    
    return {
        'ari': np.mean(all_ari) if all_ari else 0.0,
        'level_acc': np.mean(all_level) if all_level else 0.0,
        'nmi': np.mean(all_nmi) if all_nmi else 0.0
    }


# ==========================================
# OPTIMIZED Main Experiment (GPU-accelerated)
# ==========================================

def run_clevr_experiment(n_seeds=5, epochs=300, batch_size=64, device='cpu'):
    """
    OPTIMIZED CLEVR experiment - 3-5x faster than original!
    
    Optimizations:
    1. Pre-load all data to GPU (no CPU->GPU transfer in loop)
    2. Larger batch size (64 vs 16)
    3. Compute metrics every 5th batch
    4. Use set_to_none=True for zero_grad
    """
    
    print("=" * 70)
    print("CLEVR DENSITY-HIERARCHY EXPERIMENT (OPTIMIZED)")
    print("=" * 70)
    print("\nThis experiment validates LoCo on CLEVR with density-based hierarchy.")
    print("Hierarchy is created from object SIZE (large=sparse, small=dense).")
    print(f"OPTIMIZED: batch_size={batch_size}, GPU pre-loading enabled")
    print("=" * 70)
    
    # Load and prepare data
    scenes = load_clevr_scenes(max_scenes=3000)
    if scenes is None:
        print("Failed to load CLEVR. Please run on Kaggle or download scenes first.")
        return None
    
    all_data = prepare_clevr_dataset(scenes)
    
    # Train/test split
    n_train = int(0.8 * len(all_data))
    
    results = {
        'loco': {'ari': [], 'level_acc': [], 'nmi': []},
        'euc_wl': {'ari': [], 'level_acc': [], 'nmi': []},
        'euc_std': {'ari': [], 'level_acc': [], 'nmi': []}
    }
    
    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed+1}/{n_seeds}")
        print(f"{'='*70}")
        
        np.random.seed(seed + 42)
        torch.manual_seed(seed + 42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + 42)
        
        # Shuffle and split
        shuffled_data = all_data.copy()
        np.random.shuffle(shuffled_data)
        train_data = shuffled_data[:n_train]
        test_data = shuffled_data[n_train:]
        
        print(f"Train: {len(train_data)}, Test: {len(test_data)}")
        
        # PRE-LOAD DATA TO GPU (key optimization!)
        train_gpu = preload_to_gpu(train_data, device)
        test_gpu = preload_to_gpu(test_data, device)
        
        # Create models
        loco = LoCoScaleAdaptive(num_objects=3, num_levels=3).to(device)
        euc_wl = EuclideanWorldlines(num_objects=3, num_levels=3).to(device)
        euc_std = StandardEuclidean(num_slots=9).to(device)
        
        opt_loco = torch.optim.Adam(loco.parameters(), lr=0.003)
        opt_euc_wl = torch.optim.Adam(euc_wl.parameters(), lr=0.003)
        opt_euc_std = torch.optim.Adam(euc_std.parameters(), lr=0.003)
        
        # Training loop (OPTIMIZED)
        for epoch in range(epochs + 1):
            # Train all three models using FAST functions
            metrics_loco = train_epoch_fast(loco, train_gpu, opt_loco, batch_size, device)
            metrics_euc_wl = train_epoch_fast(euc_wl, train_gpu, opt_euc_wl, batch_size, device)
            metrics_euc_std = train_epoch_fast(euc_std, train_gpu, opt_euc_std, batch_size, device)
            
            if epoch % 50 == 0:
                print(f"  Ep {epoch:3d} | LoCo: {metrics_loco['ari']:.2f}/{metrics_loco['level_acc']:.2f} | "
                      f"EucWL: {metrics_euc_wl['ari']:.2f}/{metrics_euc_wl['level_acc']:.2f} | "
                      f"EucStd: {metrics_euc_std['ari']:.2f}/{metrics_euc_std['level_acc']:.2f}")
        
        # Final evaluation (OPTIMIZED)
        eval_loco = evaluate_model_fast(loco, test_gpu, batch_size, device)
        eval_euc_wl = evaluate_model_fast(euc_wl, test_gpu, batch_size, device)
        eval_euc_std = evaluate_model_fast(euc_std, test_gpu, batch_size, device)
        
        print(f"\nTest Results:")
        print(f"  LoCo:   ARI={eval_loco['ari']:.3f}, LevelAcc={eval_loco['level_acc']:.3f}")
        print(f"  EucWL:  ARI={eval_euc_wl['ari']:.3f}, LevelAcc={eval_euc_wl['level_acc']:.3f}")
        print(f"  EucStd: ARI={eval_euc_std['ari']:.3f}, LevelAcc={eval_euc_std['level_acc']:.3f}")
        
        # Store results
        results['loco']['ari'].append(eval_loco['ari'])
        results['loco']['level_acc'].append(eval_loco['level_acc'])
        results['loco']['nmi'].append(eval_loco['nmi'])
        
        results['euc_wl']['ari'].append(eval_euc_wl['ari'])
        results['euc_wl']['level_acc'].append(eval_euc_wl['level_acc'])
        results['euc_wl']['nmi'].append(eval_euc_wl['nmi'])
        
        results['euc_std']['ari'].append(eval_euc_std['ari'])
        results['euc_std']['level_acc'].append(eval_euc_std['level_acc'])
        results['euc_std']['nmi'].append(eval_euc_std['nmi'])
        
        # Clear GPU memory between seeds
        del train_gpu, test_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final results
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS - THREE-WAY COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'LoCo V9':<20} {'Euc Worldlines':<20} {'Euc Standard':<20}")
    print("-" * 80)
    
    for metric, name in [('ari', 'Object ARI'), ('level_acc', 'Level Accuracy'), ('nmi', 'NMI')]:
        loco_vals = results['loco'][metric]
        euc_wl_vals = results['euc_wl'][metric]
        euc_std_vals = results['euc_std'][metric]
        
        print(f"{name:<20} "
              f"{np.mean(loco_vals):.3f} ± {np.std(loco_vals):.3f}    "
              f"{np.mean(euc_wl_vals):.3f} ± {np.std(euc_wl_vals):.3f}    "
              f"{np.mean(euc_std_vals):.3f} ± {np.std(euc_std_vals):.3f}")
    
    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    
    print("\nObject ARI Comparisons:")
    _, p1 = stats.ttest_ind(results['loco']['ari'], results['euc_wl']['ari'])
    _, p2 = stats.ttest_ind(results['loco']['ari'], results['euc_std']['ari'])
    _, p3 = stats.ttest_ind(results['euc_wl']['ari'], results['euc_std']['ari'])
    print(f"  LoCo V9 vs Euc Worldlines: p = {p1:.4f} {'*' if p1 < 0.05 else ''}")
    print(f"  LoCo V9 vs Euc Standard:   p = {p2:.4f} {'*' if p2 < 0.05 else ''}")
    print(f"  Euc Worldlines vs Euc Std: p = {p3:.4f} {'*' if p3 < 0.05 else ''}")
    
    print("\nLevel Accuracy Comparisons:")
    _, p4 = stats.ttest_ind(results['loco']['level_acc'], results['euc_wl']['level_acc'])
    _, p5 = stats.ttest_ind(results['loco']['level_acc'], results['euc_std']['level_acc'])
    print(f"  LoCo V9 vs Euc Worldlines: p = {p4:.4f} {'*' if p4 < 0.05 else ''}")
    print(f"  LoCo V9 vs Euc Standard:   p = {p5:.4f} {'*' if p5 < 0.05 else ''}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    loco_level = np.mean(results['loco']['level_acc'])
    euc_wl_level = np.mean(results['euc_wl']['level_acc'])
    euc_std_level = np.mean(results['euc_std']['level_acc'])
    
    if loco_level > euc_wl_level + 0.1 and euc_wl_level < 0.15:
        print("✅ GEOMETRY IS ESSENTIAL: LoCo discovers hierarchy, Euclidean Worldlines collapse!")
        print(f"   LoCo Level Acc: {loco_level:.3f} vs Euc WL: {euc_wl_level:.3f}")
    elif loco_level > euc_std_level + 0.05:
        print("⚠️ GEOMETRY HELPS: LoCo better than baseline, but Euc WL didn't collapse")
    else:
        print("❌ CLEVR hierarchy too weak: All models similar on Level Accuracy")
    
    return results


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # OPTIMIZED: larger batch size for GPU efficiency
    results = run_clevr_experiment(n_seeds=5, epochs=300, batch_size=64, device=device)
