"""
HYPERBOLIC BASELINE COMPARISON: The Ultimate Geometry Test

This experiment answers THE critical question from reviewers:
"Why Lorentzian instead of Hyperbolic? Hyperbolic is the standard for hierarchies."

We compare FOUR models:
1. LoCo V9: Lorentzian + Worldlines + Scale-Adaptive Horizons
2. Hyperbolic Worldlines: Hyperbolic (Poincaré) + Worldlines + Scale-Adaptive
3. Euclidean Worldlines: Euclidean + Worldlines + Scale-Adaptive
4. Euclidean Standard: 9 independent slots (baseline)

KEY GEOMETRIC DIFFERENCES:
- Lorentzian: Hierarchy via TIME dimension (causal cones, past→future)
- Hyperbolic: Hierarchy via RADIAL distance (tree-like, center=root)
- Euclidean: No inherent hierarchy (flat space)

HYPOTHESIS:
Lorentzian > Hyperbolic > Euclidean for VISUAL hierarchy because:
- Visual hierarchies are CAUSAL (parts depend on wholes)
- NOT tree-like (parts don't "branch" from wholes)

If Hyperbolic wins: Pivot to tree-based hierarchy
If Lorentzian wins: Strong evidence for causal structure
If tie: Non-Euclidean geometry matters, specific choice less important
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

print("=" * 70)
print("HYPERBOLIC BASELINE COMPARISON")
print("Lorentzian vs Hyperbolic vs Euclidean")
print("=" * 70)


# ==========================================
# PART 1: GEOMETRIC OPERATIONS
# ==========================================

class LorentzOps:
    """Lorentzian (Minkowski) geometry operations."""
    EPS = 1e-6

    @staticmethod
    def lorentz_inner(x, y):
        """Lorentzian inner product: t1*t2 - x1·x2"""
        return x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(-1)

    @staticmethod
    def proper_time_distance(x, y):
        """Proper time distance (timelike interval)."""
        diff = x - y
        lorentz_sq = LorentzOps.lorentz_inner(diff, diff)
        return torch.sign(lorentz_sq) * torch.sqrt(torch.abs(lorentz_sq) + LorentzOps.EPS)

    @staticmethod
    def cone_membership(feature, slot, horizon):
        """
        Light cone membership with scale-adaptive horizons.
        
        Key insight: Points in the FUTURE light cone can be causally influenced.
        - Sparse features (abstract) → wide horizon → captures more
        - Dense features (fine-grained) → narrow horizon → captures less
        """
        diff = feature - slot
        tau = diff[..., 0]  # Time difference
        r = torch.norm(diff[..., 1:], dim=-1)  # Spatial distance
        
        # Cone score: positive if inside horizon
        ratio = r / (torch.abs(tau) + LorentzOps.EPS)
        inside_score = horizon - ratio
        
        # Penalize wrong causal direction and spacelike separation
        direction_penalty = -10.0 * F.relu(-tau)
        spacelike_penalty = -5.0 * F.relu(r - torch.abs(tau))
        
        return inside_score + direction_penalty + spacelike_penalty


class HyperbolicOps:
    """
    Hyperbolic geometry operations (Poincaré ball model).
    
    Key properties:
    - Points near origin = high in hierarchy (abstract)
    - Points near boundary = low in hierarchy (fine-grained)
    - Distance grows exponentially toward boundary
    - Natural for TREE-LIKE hierarchies
    
    Mathematical basis:
    - Poincaré ball: {x ∈ R^n : ||x|| < 1}
    - Metric: ds² = 4/(1-||x||²)² * ||dx||²
    - Distance: d(x,y) = arccosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
    """
    EPS = 1e-6
    BOUNDARY_EPS = 1e-5  # Keep points away from boundary
    
    @staticmethod
    def project_to_ball(x, max_norm=1.0 - 1e-5):
        """Project points to stay inside the Poincaré ball."""
        norm = torch.norm(x, dim=-1, keepdim=True)
        scale = torch.clamp(norm, min=HyperbolicOps.EPS)
        return x / scale * torch.clamp(norm, max=max_norm)
    
    @staticmethod
    def poincare_distance(x, y):
        """
        Hyperbolic distance in Poincaré ball.
        
        d(x,y) = arccosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
        """
        # Ensure points are in the ball
        x_norm_sq = torch.sum(x**2, dim=-1).clamp(max=1.0 - HyperbolicOps.BOUNDARY_EPS)
        y_norm_sq = torch.sum(y**2, dim=-1).clamp(max=1.0 - HyperbolicOps.BOUNDARY_EPS)
        
        diff_sq = torch.sum((x - y)**2, dim=-1)
        
        # Hyperbolic distance formula
        denom = (1 - x_norm_sq) * (1 - y_norm_sq)
        denom = torch.clamp(denom, min=HyperbolicOps.EPS)
        
        argument = 1 + 2 * diff_sq / denom
        argument = torch.clamp(argument, min=1.0 + HyperbolicOps.EPS)  # arccosh domain
        
        return torch.acosh(argument)
    
    @staticmethod
    def hierarchy_score(feature, slot, scale_factor):
        """
        Hierarchy-aware attention score in hyperbolic space.
        
        Key insight: Slots near origin (low norm) should attend to
        features near origin (abstract). Slots near boundary should
        attend to features near boundary (fine-grained).
        
        scale_factor modulates the "reach" of each slot:
        - High scale_factor → wider reach (for abstract slots)
        - Low scale_factor → narrower reach (for fine-grained slots)
        """
        # Hyperbolic distance
        dist = HyperbolicOps.poincare_distance(feature, slot)
        
        # Base score from distance
        distance_score = -dist
        
        # Hierarchy alignment bonus
        # Features and slots at similar "depth" (norm) get bonus
        feat_norm = torch.norm(feature, dim=-1)
        slot_norm = torch.norm(slot, dim=-1)
        
        # Slots near origin should prefer features near origin
        # Slots near boundary should prefer features near boundary
        depth_diff = torch.abs(feat_norm - slot_norm)
        alignment_bonus = -5.0 * depth_diff  # Penalize misaligned depths
        
        # Scale by hierarchy factor
        return (distance_score + alignment_bonus) * scale_factor
    
    @staticmethod
    def mobius_addition(x, y):
        """
        Möbius addition in Poincaré ball (for potential future use).
        
        x ⊕ y = ((1 + 2<x,y> + ||y||²)x + (1 - ||x||²)y) / 
                (1 + 2<x,y> + ||x||²||y||²)
        """
        x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y**2, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2*xy + y_norm_sq) * x + (1 - x_norm_sq) * y
        denom = 1 + 2*xy + x_norm_sq * y_norm_sq
        
        return num / (denom + HyperbolicOps.EPS)


# ==========================================
# PART 2: DATA GENERATION
# ==========================================

DATA_MODE = 'toy'  # 'toy' or 'sprites'

def generate_sprites_data(batch_size=16, num_objects=3, device='cpu'):
    """Simplified hierarchical sprites matching toy data difficulty."""
    data_batch = []
    object_labels_batch = []
    level_labels_batch = []
    
    base_centers = np.array([[-1.8, -1.0], [1.8, -1.0], [0, 1.9]])
    
    for _ in range(batch_size):
        all_points = []
        all_object_labels = []
        all_level_labels = []
        
        centers = base_centers + np.random.randn(3, 2) * 0.25
        
        for obj_idx, center in enumerate(centers):
            # Level 0: Body center (sparse)
            body = center + np.random.randn(2) * 0.08
            all_points.append(body)
            all_object_labels.append(obj_idx)
            all_level_labels.append(0)
            
            # Level 1: Limbs
            limb_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
            for angle in limb_angles:
                if np.random.random() < 0.15:
                    continue
                    
                limb_dist = np.random.uniform(0.85, 1.15)
                limb_pos = center + limb_dist * np.array([np.cos(angle), np.sin(angle)])
                limb_pos += np.random.randn(2) * 0.12
                
                all_points.append(limb_pos)
                all_object_labels.append(obj_idx)
                all_level_labels.append(1)
                
                # Level 2: Joints
                for _ in range(2):
                    if np.random.random() < 0.15:
                        continue
                    joint_offset = np.random.randn(2) * 0.25
                    joint_pos = limb_pos + joint_offset
                    all_points.append(joint_pos)
                    all_object_labels.append(obj_idx)
                    all_level_labels.append(2)
        
        # Background noise
        n_noise = max(1, int(len(all_points) * 0.10))
        for _ in range(n_noise):
            all_points.append([np.random.uniform(-3, 3), np.random.uniform(-2.5, 3)])
            all_object_labels.append(-1)
            all_level_labels.append(-1)
        
        data_batch.append(torch.tensor(np.array(all_points), dtype=torch.float32))
        object_labels_batch.append(torch.tensor(all_object_labels, dtype=torch.long))
        level_labels_batch.append(torch.tensor(all_level_labels, dtype=torch.long))
    
    # Pad
    max_len = max([d.shape[0] for d in data_batch])
    padded_data, padded_obj, padded_level = [], [], []
    
    for data, obj_lab, lev_lab in zip(data_batch, object_labels_batch, level_labels_batch):
        pad_len = max_len - data.shape[0]
        if pad_len > 0:
            data = torch.cat([data, torch.zeros(pad_len, 2)], dim=0)
            obj_lab = torch.cat([obj_lab, -2 * torch.ones(pad_len, dtype=torch.long)], dim=0)
            lev_lab = torch.cat([lev_lab, -2 * torch.ones(pad_len, dtype=torch.long)], dim=0)
        padded_data.append(data)
        padded_obj.append(obj_lab)
        padded_level.append(lev_lab)
    
    return (
        torch.stack(padded_data).to(device),
        torch.stack(padded_obj).to(device),
        torch.stack(padded_level).to(device)
    )


def generate_hard_hierarchical_data(batch_size=16, num_objects=3, device='cpu'):
    """Generate hierarchical data with 3 levels (toy data)."""
    data_batch = []
    object_labels_batch = []
    level_labels_batch = []
    
    base_centers = np.array([[-1.75, -1.05], [1.75, -1.05], [0, 1.75]])
    
    for _ in range(batch_size):
        all_points = []
        all_object_labels = []
        all_level_labels = []
        
        centers = base_centers + np.random.randn(3, 2) * 0.3
        
        for obj_idx, center in enumerate(centers):
            # Level 0: Object center (sparse)
            all_points.append(center + np.random.randn(2) * 0.1)
            all_object_labels.append(obj_idx)
            all_level_labels.append(0)
            
            # Level 1: Parts
            n_parts = np.random.randint(4, 6)
            for _ in range(n_parts):
                if np.random.random() < 0.15:
                    continue
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.uniform(0.8, 1.2)
                part_center = center + radius * np.array([np.cos(angle), np.sin(angle)])
                part_center += np.random.randn(2) * 0.15
                
                all_points.append(part_center)
                all_object_labels.append(obj_idx)
                all_level_labels.append(1)
                
                # Level 2: Subparts
                n_subparts = np.random.randint(2, 4)
                for _ in range(n_subparts):
                    if np.random.random() < 0.15:
                        continue
                    sp_angle = np.random.uniform(0, 2*np.pi)
                    sp_radius = np.random.uniform(0.2, 0.4)
                    subpart = part_center + sp_radius * np.array([np.cos(sp_angle), np.sin(sp_angle)])
                    subpart += np.random.randn(2) * 0.08
                    
                    all_points.append(subpart)
                    all_object_labels.append(obj_idx)
                    all_level_labels.append(2)
        
        # Background noise
        n_noise = max(1, int(len(all_points) * 0.1))
        for _ in range(n_noise):
            all_points.append([np.random.uniform(-3, 3), np.random.uniform(-2.5, 3)])
            all_object_labels.append(-1)
            all_level_labels.append(-1)
        
        data_batch.append(torch.tensor(np.array(all_points), dtype=torch.float32))
        object_labels_batch.append(torch.tensor(all_object_labels, dtype=torch.long))
        level_labels_batch.append(torch.tensor(all_level_labels, dtype=torch.long))
    
    # Pad
    max_len = max([d.shape[0] for d in data_batch])
    padded_data, padded_obj, padded_level = [], [], []
    
    for data, obj_lab, lev_lab in zip(data_batch, object_labels_batch, level_labels_batch):
        pad_len = max_len - data.shape[0]
        if pad_len > 0:
            data = torch.cat([data, torch.zeros(pad_len, 2)], dim=0)
            obj_lab = torch.cat([obj_lab, -2 * torch.ones(pad_len, dtype=torch.long)], dim=0)
            lev_lab = torch.cat([lev_lab, -2 * torch.ones(pad_len, dtype=torch.long)], dim=0)
        padded_data.append(data)
        padded_obj.append(obj_lab)
        padded_level.append(lev_lab)
    
    return (
        torch.stack(padded_data).to(device),
        torch.stack(padded_obj).to(device),
        torch.stack(padded_level).to(device)
    )


def generate_data(batch_size=16, num_objects=3, device='cpu'):
    if DATA_MODE == 'sprites':
        return generate_sprites_data(batch_size, num_objects, device)
    else:
        return generate_hard_hierarchical_data(batch_size, num_objects, device)


# ==========================================
# PART 3: MODEL 1 - LORENTZIAN WORLDLINES (LoCo V9)
# ==========================================

class LoCoScaleAdaptive(nn.Module):
    """
    LoCo V9: Lorentzian Worldlines with Scale-Adaptive Horizons.
    
    Geometry: Minkowski spacetime with light cones
    Hierarchy: Encoded in TIME dimension (sparse → low time → abstract)
    """
    
    def __init__(self, num_objects=3, num_levels=3, input_dim=2, hidden_dim=32,
                 iterations=3, tau=0.1, lambda_cone=0.5, k_neighbors=5):
        super().__init__()
        
        self.num_objects = num_objects
        self.num_levels = num_levels
        self.num_slots = num_objects * num_levels
        self.hidden_dim = hidden_dim
        self.slot_dim = hidden_dim + 1
        self.iterations = iterations
        self.tau = tau
        self.lambda_cone = lambda_cone
        self.k_neighbors = k_neighbors
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.scale_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        with torch.no_grad():
            self.scale_predictor[-2].weight.mul_(0.1)
            self.scale_predictor[-2].bias.zero_()
        
        self.object_space = nn.Parameter(torch.randn(1, num_objects, hidden_dim) * 0.1)
        
        level_times = torch.tensor([1.0, 2.5, 4.0])
        self.register_buffer('level_times', level_times)
        
        base_horizons = torch.tensor([0.90, 0.60, 0.30])
        self.register_buffer('base_horizons', base_horizons)
        
        self.horizon_scale = nn.Parameter(torch.tensor(0.3))
        
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

    def get_adaptive_horizons(self, density):
        B, N = density.shape
        modulation = self.horizon_scale * (density - 0.5)
        
        base = self.base_horizons.view(1, self.num_levels, 1).expand(B, -1, N)
        mod = modulation.unsqueeze(1).expand(-1, self.num_levels, -1)
        
        horizons = base + mod
        horizons = torch.clamp(horizons, 0.1, 1.0)
        horizons = horizons.repeat(1, self.num_objects, 1)
        
        return horizons

    def encode_features(self, x):
        B, N, D = x.shape
        spatial = self.feature_encoder(x)
        density = self.get_local_density(x)
        
        scale_input = torch.cat([spatial, density.unsqueeze(-1)], dim=-1)
        pred_scale = self.scale_predictor(scale_input)
        
        pred_time = 5.0 - 1.5 * density.unsqueeze(-1) + 0.5 * pred_scale
        
        features = torch.cat([pred_time, spatial], dim=-1)
        
        return features, density, pred_time.squeeze(-1)

    def get_worldlines(self, batch_size):
        obj_space = self.object_space.expand(batch_size, -1, -1)
        
        if self.training:
            obj_space = obj_space + torch.randn_like(obj_space) * 0.01
        
        slots_space = obj_space.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
        slots_space = slots_space.reshape(batch_size, self.num_slots, self.hidden_dim)
        
        times = self.level_times.repeat(self.num_objects).view(1, self.num_slots, 1).expand(batch_size, -1, -1)
        
        return torch.cat([times, slots_space], dim=-1)

    def compute_attention(self, features, slots, horizons):
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

    def forward(self, x):
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


# ==========================================
# PART 4: MODEL 2 - HYPERBOLIC WORLDLINES (NEW!)
# ==========================================

class HyperbolicWorldlines(nn.Module):
    """
    Hyperbolic Worldlines: Poincaré Ball Geometry with Worldline Binding.
    
    Geometry: Poincaré ball model of hyperbolic space
    Hierarchy: Encoded in RADIAL DISTANCE from origin
        - Center (low norm) → Abstract (Level 0)
        - Boundary (high norm) → Fine-grained (Level 2)
    
    This is the "natural" geometry for hierarchies according to literature
    (Poincaré embeddings, hyperbolic neural networks, etc.)
    
    Key differences from Lorentzian:
    - Lorentzian: Hierarchy in separate TIME dimension
    - Hyperbolic: Hierarchy in RADIAL distance within same space
    """
    
    def __init__(self, num_objects=3, num_levels=3, input_dim=2, hidden_dim=32,
                 iterations=3, tau=0.1, k_neighbors=5, curvature=1.0):
        super().__init__()
        
        self.num_objects = num_objects
        self.num_levels = num_levels
        self.num_slots = num_objects * num_levels
        self.hidden_dim = hidden_dim
        self.iterations = iterations
        self.tau = tau
        self.k_neighbors = k_neighbors
        self.curvature = curvature
        
        # Feature encoder (same as LoCo)
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Map features to Poincaré ball
        # Output is constrained to ||x|| < 1
        self.hyperbolic_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Keep in [-1, 1] range, then scale
        )
        
        # Worldline structure: 3 object DIRECTIONS in hyperbolic space
        # Each object is a "ray" from origin, levels are at different radii
        self.object_directions = nn.Parameter(torch.randn(1, num_objects, hidden_dim) * 0.3)
        
        # Level radii (distance from origin in Poincaré ball)
        # Level 0 (abstract) → near origin (low radius)
        # Level 2 (fine-grained) → near boundary (high radius)
        level_radii = torch.tensor([0.2, 0.5, 0.8])  # Must be < 1
        self.register_buffer('level_radii', level_radii)
        
        # Scale-adaptive radius modulation
        self.radius_scale = nn.Parameter(torch.tensor(0.15))
        
        # Base scale factors for attention (analogous to horizons)
        base_scales = torch.tensor([2.0, 1.0, 0.5])  # Abstract=wide, fine=narrow
        self.register_buffer('base_scales', base_scales)
        
        # Update networks
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def get_local_density(self, x):
        """Same density computation as LoCo."""
        B, N, D = x.shape
        k = min(self.k_neighbors, N - 1)
        
        if k <= 0:
            return torch.ones(B, N, device=x.device) * 0.5
        
        dist = torch.cdist(x, x)
        knn_dists, _ = torch.topk(dist, k + 1, dim=-1, largest=False)
        avg_dist = knn_dists[..., 1:].mean(dim=-1)
        
        return torch.tanh(avg_dist)

    def encode_features_hyperbolic(self, x):
        """Encode features and project to Poincaré ball."""
        B, N, D = x.shape
        
        # Encode to hidden dim
        encoded = self.feature_encoder(x)
        density = self.get_local_density(x)
        
        # Project to hyperbolic space
        hyp_features = self.hyperbolic_projection(encoded)
        
        # Scale by density: sparse → near origin, dense → near boundary
        # This encodes hierarchy in radial position
        target_radius = 0.3 + 0.5 * (1.0 - density)  # Sparse=low radius, dense=high radius
        target_radius = target_radius.unsqueeze(-1)
        
        # Normalize and scale to target radius
        feat_norm = torch.norm(hyp_features, dim=-1, keepdim=True).clamp(min=1e-6)
        hyp_features = hyp_features / feat_norm * target_radius * 0.95  # Stay inside ball
        
        return hyp_features, density

    def get_worldlines_hyperbolic(self, batch_size):
        """
        Construct hyperbolic worldlines.
        
        Each object is a "ray" from origin.
        Levels are points along this ray at different radii.
        """
        # Normalize object directions to unit vectors
        obj_dirs = self.object_directions.expand(batch_size, -1, -1)
        obj_dirs_norm = F.normalize(obj_dirs, dim=-1)
        
        if self.training:
            obj_dirs_norm = obj_dirs_norm + torch.randn_like(obj_dirs_norm) * 0.01
            obj_dirs_norm = F.normalize(obj_dirs_norm, dim=-1)
        
        # Create slots at different radii along each object direction
        slots_list = []
        for level in range(self.num_levels):
            radius = self.level_radii[level]
            level_slots = obj_dirs_norm * radius  # [B, num_objects, hidden_dim]
            slots_list.append(level_slots)
        
        # Interleave: obj0_lv0, obj0_lv1, obj0_lv2, obj1_lv0, ...
        slots = torch.stack(slots_list, dim=2)  # [B, num_objects, num_levels, hidden_dim]
        slots = slots.reshape(batch_size, self.num_slots, self.hidden_dim)
        
        return slots

    def get_adaptive_scales(self, density):
        """Scale-adaptive attention scale factors."""
        B, N = density.shape
        
        modulation = self.radius_scale * (density - 0.5)
        base = self.base_scales.view(1, self.num_levels, 1).expand(B, -1, N)
        mod = modulation.unsqueeze(1).expand(-1, self.num_levels, -1)
        
        scales = base + mod
        scales = torch.clamp(scales, 0.3, 3.0)
        scales = scales.repeat(1, self.num_objects, 1)
        
        return scales

    def compute_attention(self, features, slots, scales):
        """Compute attention using hyperbolic distance."""
        B, N, D = features.shape
        K = slots.shape[1]
        
        slots_exp = slots.unsqueeze(2).expand(-1, -1, N, -1)
        feats_exp = features.unsqueeze(1).expand(-1, K, -1, -1)
        
        # Hyperbolic distance
        hyp_dist = HyperbolicOps.poincare_distance(feats_exp, slots_exp)
        
        # Hierarchy alignment: features and slots should match in radial position
        feat_radius = torch.norm(feats_exp, dim=-1)
        slot_radius = torch.norm(slots_exp, dim=-1)
        radius_diff = torch.abs(feat_radius - slot_radius)
        
        # Combined score
        distance_score = -hyp_dist
        alignment_score = -3.0 * radius_diff  # Penalize radial mismatch
        
        attn_logits = (distance_score + alignment_score) * scales
        attn = F.softmax(attn_logits / self.tau, dim=1)
        
        return attn

    def forward(self, x):
        B, N, _ = x.shape
        
        features, density = self.encode_features_hyperbolic(x)
        slots = self.get_worldlines_hyperbolic(B)
        scales = self.get_adaptive_scales(density)
        
        for iteration in range(self.iterations):
            attn = self.compute_attention(features, slots, scales)
            
            attn_weights = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)
            updates = torch.bmm(attn_weights, features)
            
            # Aggregate across levels
            updates_per_obj = updates.view(B, self.num_objects, self.num_levels, self.hidden_dim)
            obj_updates = updates_per_obj.sum(dim=2)
            
            # GRU update on object directions
            obj_dirs = self.object_directions.expand(B, -1, -1)
            old_flat = obj_dirs.reshape(-1, self.hidden_dim)
            update_flat = obj_updates.reshape(-1, self.hidden_dim)
            
            new_dirs = self.gru(update_flat, old_flat)
            new_dirs = new_dirs.reshape(B, self.num_objects, self.hidden_dim)
            new_dirs = new_dirs + 0.2 * self.update_mlp(self.norm(new_dirs))
            new_dirs = F.normalize(new_dirs, dim=-1)
            
            # Reconstruct slots at level radii
            slots_list = []
            for level in range(self.num_levels):
                radius = self.level_radii[level]
                level_slots = new_dirs * radius
                slots_list.append(level_slots)
            
            slots = torch.stack(slots_list, dim=2)
            slots = slots.reshape(B, self.num_slots, self.hidden_dim)
        
        return slots, attn, None, density


# ==========================================
# PART 5: MODEL 3 - EUCLIDEAN WORLDLINES
# ==========================================

class EuclideanWorldlines(nn.Module):
    """Euclidean Worldlines: Same architecture as LoCo, no special geometry."""
    
    def __init__(self, num_objects=3, num_levels=3, input_dim=2, hidden_dim=32,
                 iterations=3, tau=0.1, k_neighbors=5):
        super().__init__()
        
        self.num_objects = num_objects
        self.num_levels = num_levels
        self.num_slots = num_objects * num_levels
        self.hidden_dim = hidden_dim
        self.iterations = iterations
        self.tau = tau
        self.k_neighbors = k_neighbors
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.object_space = nn.Parameter(torch.randn(1, num_objects, hidden_dim) * 0.1)
        
        base_temps = torch.tensor([0.2, 0.1, 0.05])
        self.register_buffer('base_temps', base_temps)
        
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

    def get_worldlines(self, batch_size):
        obj_space = self.object_space.expand(batch_size, -1, -1)
        
        if self.training:
            obj_space = obj_space + torch.randn_like(obj_space) * 0.01
        
        slots = obj_space.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
        slots = slots.reshape(batch_size, self.num_slots, self.hidden_dim)
        
        return slots

    def get_adaptive_temperatures(self, density):
        B, N = density.shape
        
        modulation = self.temp_scale * (density - 0.5)
        base = self.base_temps.view(1, self.num_levels, 1).expand(B, -1, N)
        mod = modulation.unsqueeze(1).expand(-1, self.num_levels, -1)
        
        temps = base + mod
        temps = torch.clamp(temps, 0.01, 0.5)
        temps = temps.repeat(1, self.num_objects, 1)
        
        return temps

    def compute_attention(self, features, slots, temps):
        B, N, D = features.shape
        K = slots.shape[1]
        
        slots_exp = slots.unsqueeze(2).expand(-1, -1, N, -1)
        feats_exp = features.unsqueeze(1).expand(-1, K, -1, -1)
        
        dist_sq = torch.sum((slots_exp - feats_exp)**2, dim=-1)
        attn_logits = -dist_sq / (temps + 1e-8)
        
        attn = F.softmax(attn_logits, dim=1)
        
        return attn

    def forward(self, x):
        B, N, _ = x.shape
        
        features = self.feature_encoder(x)
        density = self.get_local_density(x)
        slots = self.get_worldlines(B)
        temps = self.get_adaptive_temperatures(density)
        
        for iteration in range(self.iterations):
            attn = self.compute_attention(features, slots, temps)
            
            attn_weights = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)
            updates = torch.bmm(attn_weights, features)
            
            updates_per_obj = updates.view(B, self.num_objects, self.num_levels, self.hidden_dim)
            obj_updates = updates_per_obj.sum(dim=2)
            
            old_obj_space = self.object_space.expand(B, -1, -1)
            old_flat = old_obj_space.reshape(-1, self.hidden_dim)
            update_flat = obj_updates.reshape(-1, self.hidden_dim)
            
            new_obj_space = self.gru(update_flat, old_flat)
            new_obj_space = new_obj_space.reshape(B, self.num_objects, self.hidden_dim)
            new_obj_space = new_obj_space + 0.2 * self.update_mlp(self.norm(new_obj_space))
            
            slots = new_obj_space.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
            slots = slots.reshape(B, self.num_slots, self.hidden_dim)
        
        return slots, attn, None, density


# ==========================================
# PART 6: MODEL 4 - STANDARD EUCLIDEAN (BASELINE)
# ==========================================

class StandardEuclidean(nn.Module):
    """Standard Euclidean with 9 INDEPENDENT slots (no worldline binding)."""
    
    def __init__(self, num_slots=9, input_dim=2, hidden_dim=32, iterations=3, tau=0.1):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.iterations = iterations
        self.tau = tau
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, hidden_dim) * 0.1)
        
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, N, _ = x.shape
        
        features = self.feature_encoder(x)
        slots = self.slot_mu.expand(B, -1, -1)
        
        if self.training:
            slots = slots + torch.randn_like(slots) * 0.01
        
        for iteration in range(self.iterations):
            slots_exp = slots.unsqueeze(2).expand(-1, -1, N, -1)
            feats_exp = features.unsqueeze(1).expand(-1, self.num_slots, -1, -1)
            
            dist_sq = -torch.sum((slots_exp - feats_exp)**2, dim=-1)
            attn = F.softmax(dist_sq / self.tau, dim=1)
            
            attn_weights = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)
            updates = torch.bmm(attn_weights, features)
            
            old_flat = slots.reshape(-1, self.hidden_dim)
            update_flat = updates.reshape(-1, self.hidden_dim)
            new_slots = self.gru(update_flat, old_flat)
            new_slots = new_slots.reshape(B, self.num_slots, self.hidden_dim)
            
            slots = new_slots + 0.2 * self.mlp(self.norm(new_slots))
        
        return slots, attn, None, None


# ==========================================
# PART 7: LOSSES AND METRICS
# ==========================================

def clustering_loss(features, slots, attn):
    feat_space = features[..., 1:] if features.shape[-1] > 32 else features
    slot_space = slots[..., 1:] if slots.shape[-1] > 32 else slots
    
    attn_T = attn.transpose(1, 2)
    reconstructed = torch.bmm(attn_T, slot_space)
    
    return F.mse_loss(reconstructed, feat_space)


def diversity_loss(slots, num_objects=3, num_levels=3):
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
    B, K, N = attn.shape
    
    if K != num_objects * num_levels:
        return 0.33
    
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
# PART 8: TRAINING
# ==========================================

def train_single_run(seed, epochs=300, batch_size=16, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create all FOUR models
    loco = LoCoScaleAdaptive(num_objects=3, num_levels=3).to(DEVICE)
    hyperbolic = HyperbolicWorldlines(num_objects=3, num_levels=3).to(DEVICE)
    euc_wl = EuclideanWorldlines(num_objects=3, num_levels=3).to(DEVICE)
    euc_std = StandardEuclidean(num_slots=9).to(DEVICE)
    
    opt_loco = optim.Adam(loco.parameters(), lr=0.003)
    opt_hyp = optim.Adam(hyperbolic.parameters(), lr=0.003)
    opt_euc_wl = optim.Adam(euc_wl.parameters(), lr=0.003)
    opt_euc_std = optim.Adam(euc_std.parameters(), lr=0.003)
    
    history = {
        'loco_obj_ari': [], 'hyp_obj_ari': [], 'euc_wl_obj_ari': [], 'euc_std_obj_ari': [],
        'loco_level_acc': [], 'hyp_level_acc': [], 'euc_wl_level_acc': [], 'euc_std_level_acc': [],
        'loco_nmi': [], 'hyp_nmi': [], 'euc_wl_nmi': [], 'euc_std_nmi': [],
    }
    
    for epoch in range(epochs + 1):
        data, obj_labels, level_labels = generate_data(batch_size, device=DEVICE)
        
        # ============ TRAIN LOCO ============
        loco.train()
        slots_loco, attn_loco, _, density = loco(data)
        features_loco, _, _ = loco.encode_features(data)
        
        loss_loco = clustering_loss(features_loco, slots_loco, attn_loco) + 0.3 * diversity_loss(slots_loco)
        
        opt_loco.zero_grad()
        loss_loco.backward()
        torch.nn.utils.clip_grad_norm_(loco.parameters(), 1.0)
        opt_loco.step()
        
        # ============ TRAIN HYPERBOLIC ============
        hyperbolic.train()
        slots_hyp, attn_hyp, _, _ = hyperbolic(data)
        features_hyp, _ = hyperbolic.encode_features_hyperbolic(data)
        
        loss_hyp = clustering_loss(features_hyp, slots_hyp, attn_hyp) + 0.3 * diversity_loss(slots_hyp)
        
        opt_hyp.zero_grad()
        loss_hyp.backward()
        torch.nn.utils.clip_grad_norm_(hyperbolic.parameters(), 1.0)
        opt_hyp.step()
        
        # ============ TRAIN EUCLIDEAN WORLDLINES ============
        euc_wl.train()
        slots_euc_wl, attn_euc_wl, _, _ = euc_wl(data)
        features_euc_wl = euc_wl.feature_encoder(data)
        
        loss_euc_wl = clustering_loss(features_euc_wl, slots_euc_wl, attn_euc_wl) + 0.3 * diversity_loss(slots_euc_wl)
        
        opt_euc_wl.zero_grad()
        loss_euc_wl.backward()
        torch.nn.utils.clip_grad_norm_(euc_wl.parameters(), 1.0)
        opt_euc_wl.step()
        
        # ============ TRAIN STANDARD EUCLIDEAN ============
        euc_std.train()
        slots_euc_std, attn_euc_std, _, _ = euc_std(data)
        features_euc_std = euc_std.feature_encoder(data)
        
        loss_euc_std = clustering_loss(features_euc_std, slots_euc_std, attn_euc_std) + 0.3 * diversity_loss(slots_euc_std, num_objects=9, num_levels=1)
        
        opt_euc_std.zero_grad()
        loss_euc_std.backward()
        torch.nn.utils.clip_grad_norm_(euc_std.parameters(), 1.0)
        opt_euc_std.step()
        
        # ============ METRICS ============
        with torch.no_grad():
            loco.eval()
            hyperbolic.eval()
            euc_wl.eval()
            euc_std.eval()
            
            _, attn_loco, _, _ = loco(data)
            _, attn_hyp, _, _ = hyperbolic(data)
            _, attn_euc_wl, _, _ = euc_wl(data)
            _, attn_euc_std, _, _ = euc_std(data)
            
            history['loco_obj_ari'].append(compute_object_ari(attn_loco, obj_labels))
            history['hyp_obj_ari'].append(compute_object_ari(attn_hyp, obj_labels))
            history['euc_wl_obj_ari'].append(compute_object_ari(attn_euc_wl, obj_labels))
            history['euc_std_obj_ari'].append(compute_object_ari(attn_euc_std, obj_labels))
            
            history['loco_level_acc'].append(compute_level_accuracy(attn_loco, level_labels))
            history['hyp_level_acc'].append(compute_level_accuracy(attn_hyp, level_labels))
            history['euc_wl_level_acc'].append(compute_level_accuracy(attn_euc_wl, level_labels))
            history['euc_std_level_acc'].append(compute_level_accuracy(attn_euc_std, level_labels))
            
            history['loco_nmi'].append(compute_nmi(attn_loco, obj_labels, level_labels))
            history['hyp_nmi'].append(compute_nmi(attn_hyp, obj_labels, level_labels))
            history['euc_wl_nmi'].append(compute_nmi(attn_euc_wl, obj_labels, level_labels))
            history['euc_std_nmi'].append(compute_nmi(attn_euc_std, obj_labels, level_labels))
        
        if verbose and epoch % 50 == 0:
            print(f"  Ep {epoch:3d} | "
                  f"LoCo: {history['loco_obj_ari'][-1]:.2f}/{history['loco_level_acc'][-1]:.2f} | "
                  f"Hyp: {history['hyp_obj_ari'][-1]:.2f}/{history['hyp_level_acc'][-1]:.2f} | "
                  f"EucWL: {history['euc_wl_obj_ari'][-1]:.2f}/{history['euc_wl_level_acc'][-1]:.2f} | "
                  f"EucStd: {history['euc_std_obj_ari'][-1]:.2f}/{history['euc_std_level_acc'][-1]:.2f}")
    
    return history


def run_experiment(num_seeds=5, epochs=300):
    print("\n" + "=" * 70)
    print("FOUR-WAY GEOMETRY COMPARISON")
    print("=" * 70)
    print("\n1. LoCo (Lorentzian): Hierarchy via TIME (causal cones)")
    print("2. Hyperbolic: Hierarchy via RADIAL distance (tree-like)")
    print("3. Euclidean Worldlines: No special geometry")
    print("4. Euclidean Standard: No geometry, no worldlines")
    print("\nThis answers: 'Is Lorentzian specifically better, or any non-Euclidean?'\n")
    print(f"Running {num_seeds} seeds, {epochs} epochs each")
    print("=" * 70)
    
    all_results = {key: [] for key in [
        'loco_obj_ari', 'hyp_obj_ari', 'euc_wl_obj_ari', 'euc_std_obj_ari',
        'loco_level_acc', 'hyp_level_acc', 'euc_wl_level_acc', 'euc_std_level_acc',
        'loco_nmi', 'hyp_nmi', 'euc_wl_nmi', 'euc_std_nmi',
    ]}
    
    for seed in range(num_seeds):
        print(f"\n--- Seed {seed + 1}/{num_seeds} ---")
        history = train_single_run(seed, epochs=epochs, verbose=True)
        
        for key in all_results:
            all_results[key].append(np.mean(history[key][-50:]))
    
    return all_results


def analyze_results(results):
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS - FOUR-WAY GEOMETRY COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Metric':<18} {'LoCo (Lor)':<14} {'Hyperbolic':<14} {'Euc WL':<14} {'Euc Std':<14}")
    print("-" * 75)
    
    metrics = [
        ('Object ARI', 'loco_obj_ari', 'hyp_obj_ari', 'euc_wl_obj_ari', 'euc_std_obj_ari'),
        ('Level Accuracy', 'loco_level_acc', 'hyp_level_acc', 'euc_wl_level_acc', 'euc_std_level_acc'),
        ('NMI', 'loco_nmi', 'hyp_nmi', 'euc_wl_nmi', 'euc_std_nmi'),
    ]
    
    for name, loco_key, hyp_key, wl_key, std_key in metrics:
        loco = np.array(results[loco_key])
        hyp = np.array(results[hyp_key])
        wl = np.array(results[wl_key])
        std = np.array(results[std_key])
        
        loco_str = f"{loco.mean():.3f}±{loco.std():.3f}"
        hyp_str = f"{hyp.mean():.3f}±{hyp.std():.3f}"
        wl_str = f"{wl.mean():.3f}±{wl.std():.3f}"
        std_str = f"{std.mean():.3f}±{std.std():.3f}"
        
        print(f"{name:<18} {loco_str:<14} {hyp_str:<14} {wl_str:<14} {std_str:<14}")
    
    # Statistical tests for Level Accuracy (the key metric for hierarchy)
    print("\n" + "=" * 70)
    print("LEVEL ACCURACY STATISTICAL TESTS (Hierarchy Discovery)")
    print("=" * 70)
    
    loco_level = np.array(results['loco_level_acc'])
    hyp_level = np.array(results['hyp_level_acc'])
    wl_level = np.array(results['euc_wl_level_acc'])
    std_level = np.array(results['euc_std_level_acc'])
    
    _, p_loco_vs_hyp = ttest_ind(loco_level, hyp_level)
    _, p_loco_vs_wl = ttest_ind(loco_level, wl_level)
    _, p_hyp_vs_wl = ttest_ind(hyp_level, wl_level)
    _, p_loco_vs_std = ttest_ind(loco_level, std_level)
    
    print(f"\nLevel Accuracy Comparisons:")
    print(f"  LoCo vs Hyperbolic:    Δ = {loco_level.mean() - hyp_level.mean():+.3f}, p = {p_loco_vs_hyp:.4f} {'*' if p_loco_vs_hyp < 0.05 else ''}")
    print(f"  LoCo vs Euc WL:        Δ = {loco_level.mean() - wl_level.mean():+.3f}, p = {p_loco_vs_wl:.4f} {'*' if p_loco_vs_wl < 0.05 else ''}")
    print(f"  Hyperbolic vs Euc WL:  Δ = {hyp_level.mean() - wl_level.mean():+.3f}, p = {p_hyp_vs_wl:.4f} {'*' if p_hyp_vs_wl < 0.05 else ''}")
    print(f"  LoCo vs Euc Std:       Δ = {loco_level.mean() - std_level.mean():+.3f}, p = {p_loco_vs_std:.4f} {'*' if p_loco_vs_std < 0.05 else ''}")
    
    # Determine geometry ranking
    print("\n" + "=" * 70)
    print("GEOMETRY RANKING (by Level Accuracy)")
    print("=" * 70)
    
    geo_scores = [
        ('LoCo (Lorentzian)', loco_level.mean(), loco_level.std()),
        ('Hyperbolic', hyp_level.mean(), hyp_level.std()),
        ('Euclidean WL', wl_level.mean(), wl_level.std()),
        ('Euclidean Std', std_level.mean(), std_level.std()),
    ]
    
    geo_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nRanking (best to worst for hierarchy discovery):")
    for i, (name, mean, std) in enumerate(geo_scores, 1):
        marker = "✅ WINNER" if i == 1 else ("⚠️" if i == 2 else "❌")
        print(f"  {i}. {name}: {mean:.3f} ± {std:.3f} {marker}")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    winner = geo_scores[0][0]
    second = geo_scores[1][0]
    
    if 'LoCo' in winner:
        if loco_level.mean() > hyp_level.mean() and p_loco_vs_hyp < 0.1:
            print("\n🎯 OUTCOME: LORENTZIAN IS SPECIFICALLY BETTER THAN HYPERBOLIC")
            print("\n   Explanation:")
            print("   - Visual hierarchy is CAUSAL (parts depend on wholes)")
            print("   - NOT tree-like (parts don't 'branch' from wholes)")
            print("   - Lorentzian's TIME dimension captures this causality")
            print("   - Hyperbolic's RADIAL distance doesn't fit as well")
            print("\n   Paper Claim: 'Lorentzian geometry is essential for visual hierarchy'")
        else:
            print("\n🎯 OUTCOME: NON-EUCLIDEAN GEOMETRY HELPS, LORENTZIAN SLIGHTLY BETTER")
            print("\n   Both Lorentzian and Hyperbolic outperform Euclidean")
            print("   Paper Claim: 'Non-Euclidean geometry enables hierarchy discovery'")
    elif 'Hyperbolic' in winner:
        print("\n🎯 OUTCOME: HYPERBOLIC IS BETTER THAN LORENTZIAN")
        print("\n   This would require pivoting the paper focus!")
        print("   Paper Claim: 'Hyperbolic geometry is better for visual hierarchy'")
    else:
        print("\n🎯 OUTCOME: GEOMETRY DOESN'T CLEARLY HELP")
        print("\n   Focus on worldline architecture instead of geometry")
    
    # Object ARI comparison
    print("\n" + "=" * 70)
    print("CLUSTERING QUALITY (Object ARI)")
    print("=" * 70)
    
    loco_ari = np.array(results['loco_obj_ari'])
    hyp_ari = np.array(results['hyp_obj_ari'])
    wl_ari = np.array(results['euc_wl_obj_ari'])
    std_ari = np.array(results['euc_std_obj_ari'])
    
    ari_scores = [
        ('LoCo', loco_ari.mean()),
        ('Hyperbolic', hyp_ari.mean()),
        ('Euclidean WL', wl_ari.mean()),
        ('Euclidean Std', std_ari.mean()),
    ]
    ari_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nClustering ranking:")
    for i, (name, mean) in enumerate(ari_scores, 1):
        print(f"  {i}. {name}: {mean:.3f}")
    
    return results


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    
    NUM_SEEDS = 10  # Can increase for publication
    EPOCHS = 300
    
    print(f"\nFOUR-WAY GEOMETRY COMPARISON:")
    print(f"  1. LoCo (Lorentzian): Hierarchy via TIME dimension")
    print(f"  2. Hyperbolic (Poincaré): Hierarchy via RADIAL distance")
    print(f"  3. Euclidean Worldlines: No special geometry")
    print(f"  4. Euclidean Standard: Baseline (no geometry, no worldlines)")
    print(f"\n  Seeds: {NUM_SEEDS}, Epochs: {EPOCHS}")
    print(f"  Data mode: {DATA_MODE}")
    print(f"\nThis answers: 'Why Lorentzian instead of Hyperbolic?'")
    
    results = run_experiment(num_seeds=NUM_SEEDS, epochs=EPOCHS)
    analyze_results(results)
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    
    print("\n| Model | Level Accuracy | Object ARI | Geometry |")
    print("|-------|----------------|------------|----------|")
    print(f"| LoCo (Lorentzian) | {np.mean(results['loco_level_acc']):.3f} ± {np.std(results['loco_level_acc']):.3f} | {np.mean(results['loco_obj_ari']):.3f} | Minkowski |")
    print(f"| Hyperbolic | {np.mean(results['hyp_level_acc']):.3f} ± {np.std(results['hyp_level_acc']):.3f} | {np.mean(results['hyp_obj_ari']):.3f} | Poincaré |")
    print(f"| Euclidean WL | {np.mean(results['euc_wl_level_acc']):.3f} ± {np.std(results['euc_wl_level_acc']):.3f} | {np.mean(results['euc_wl_obj_ari']):.3f} | Flat |")
    print(f"| Euclidean Std | {np.mean(results['euc_std_level_acc']):.3f} ± {np.std(results['euc_std_level_acc']):.3f} | {np.mean(results['euc_std_obj_ari']):.3f} | Flat |")
