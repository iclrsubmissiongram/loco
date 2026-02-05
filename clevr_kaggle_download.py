"""
CLEVR Dataset Download & Preparation for Kaggle
================================================

HOW TO USE ON KAGGLE:
1. Create a new Kaggle notebook
2. Add the CLEVR dataset: 
   - Click "Add Data" → Search "CLEVR" → Add "CLEVR Dataset" by timoboz
   - OR download directly from Stanford (see Option B below)
3. Copy this code into a cell and run

This script:
1. Loads CLEVR scene annotations
2. Converts objects to hierarchical point clouds
3. Provides analysis of the dataset structure
4. Prepares data for LoCo vs Euclidean comparison
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# OPTION A: Using Kaggle's Pre-loaded CLEVR
# ==========================================

def find_clevr_scenes_kaggle():
    """Find CLEVR scenes file in Kaggle's input directory"""
    
    # Common Kaggle paths for CLEVR
    possible_paths = [
        '/kaggle/input/clevr-dataset/CLEVR_v1.0/scenes/CLEVR_train_scenes.json',
        '/kaggle/input/clevr/CLEVR_v1.0/scenes/CLEVR_train_scenes.json',
        '/kaggle/input/clevr-dataset/scenes/CLEVR_train_scenes.json',
        '/kaggle/input/clevr/scenes/CLEVR_train_scenes.json',
        # For val set (smaller, good for quick tests)
        '/kaggle/input/clevr-dataset/CLEVR_v1.0/scenes/CLEVR_val_scenes.json',
        '/kaggle/input/clevr/CLEVR_v1.0/scenes/CLEVR_val_scenes.json',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found CLEVR scenes at: {path}")
            return path
    
    # List what's available
    print("❌ CLEVR scenes not found in expected locations.")
    print("\nSearching /kaggle/input/...")
    
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            for f in files:
                if 'scene' in f.lower() and f.endswith('.json'):
                    full_path = os.path.join(root, f)
                    print(f"  Found: {full_path}")
                    return full_path
    
    return None


# ==========================================
# OPTION B: Direct Download from Stanford
# ==========================================

def download_clevr_scenes_direct():
    """
    Download CLEVR scenes directly from Stanford.
    This downloads only the scene annotations (~100MB), not full images (~18GB).
    """
    import urllib.request
    import zipfile
    
    # CLEVR scene annotations URL
    # Note: Full dataset is ~18GB, but we only need scenes (~100MB unzipped)
    scenes_url = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip"
    
    output_dir = '/kaggle/working/clevr_data'
    os.makedirs(output_dir, exist_ok=True)
    
    zip_path = os.path.join(output_dir, 'clevr_scenes.zip')
    
    print("Downloading CLEVR scene annotations (~100MB)...")
    print(f"URL: {scenes_url}")
    
    try:
        urllib.request.urlretrieve(scenes_url, zip_path)
        print("✅ Download complete!")
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(output_dir)
        print("✅ Extraction complete!")
        
        # Find the scenes file
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if 'train_scenes' in f and f.endswith('.json'):
                    return os.path.join(root, f)
        
        # Try val scenes
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if 'val_scenes' in f and f.endswith('.json'):
                    return os.path.join(root, f)
                    
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nAlternative: Add CLEVR dataset manually in Kaggle:")
        print("  1. Click 'Add Data' button")
        print("  2. Search for 'CLEVR'")
        print("  3. Add 'CLEVR Dataset' by timoboz")
        
    return None


# ==========================================
# Load and Parse CLEVR Scenes
# ==========================================

def load_clevr_scenes(scenes_path, max_scenes=None):
    """Load CLEVR scene annotations from JSON"""
    
    print(f"\nLoading scenes from: {scenes_path}")
    
    with open(scenes_path, 'r') as f:
        data = json.load(f)
    
    scenes = data.get('scenes', data)  # Handle both formats
    
    if max_scenes:
        scenes = scenes[:max_scenes]
    
    print(f"✅ Loaded {len(scenes)} scenes")
    
    return scenes


def analyze_clevr_scenes(scenes):
    """Analyze CLEVR dataset statistics"""
    
    print("\n" + "="*60)
    print("CLEVR DATASET ANALYSIS")
    print("="*60)
    
    # Objects per scene
    objects_per_scene = [len(s['objects']) for s in scenes]
    print(f"\nObjects per scene:")
    print(f"  Min: {min(objects_per_scene)}")
    print(f"  Max: {max(objects_per_scene)}")
    print(f"  Mean: {np.mean(objects_per_scene):.2f}")
    print(f"  Distribution: {Counter(objects_per_scene)}")
    
    # Object properties
    all_sizes = []
    all_shapes = []
    all_materials = []
    all_colors = []
    
    for scene in scenes:
        for obj in scene['objects']:
            all_sizes.append(obj.get('size', 'unknown'))
            all_shapes.append(obj.get('shape', 'unknown'))
            all_materials.append(obj.get('material', 'unknown'))
            all_colors.append(obj.get('color', 'unknown'))
    
    print(f"\nObject sizes: {Counter(all_sizes)}")
    print(f"Object shapes: {Counter(all_shapes)}")
    print(f"Object materials: {Counter(all_materials)}")
    print(f"Object colors: {Counter(all_colors)}")
    
    # 3D coordinate ranges
    all_x = []
    all_y = []
    all_z = []
    
    for scene in scenes:
        for obj in scene['objects']:
            coords = obj.get('3d_coords', [0, 0, 0])
            all_x.append(coords[0])
            all_y.append(coords[1])
            all_z.append(coords[2])
    
    print(f"\n3D Coordinate ranges:")
    print(f"  X: [{min(all_x):.2f}, {max(all_x):.2f}]")
    print(f"  Y: [{min(all_y):.2f}, {max(all_y):.2f}]")
    print(f"  Z: [{min(all_z):.2f}, {max(all_z):.2f}]")
    
    return {
        'num_scenes': len(scenes),
        'objects_per_scene': objects_per_scene,
        'sizes': Counter(all_sizes),
        'shapes': Counter(all_shapes),
        'coord_ranges': {
            'x': (min(all_x), max(all_x)),
            'y': (min(all_y), max(all_y)),
            'z': (min(all_z), max(all_z))
        }
    }


# ==========================================
# Convert to Hierarchical Point Clouds
# ==========================================

def clevr_scene_to_hierarchical_pointcloud(scene, points_per_object=60):
    """
    Convert a CLEVR scene to hierarchical point cloud.
    
    Hierarchy based on distance from object center:
    - Level 0 (Core): Object center (1 point) - SPARSE
    - Level 1 (Surface): Points on surface (~20%) - MEDIUM
    - Level 2 (Interior): Points inside object (~80%) - DENSE
    
    This matches our toy experiment hierarchy structure.
    """
    
    all_points = []
    object_labels = []
    level_labels = []
    
    objects = scene.get('objects', [])
    
    for obj_idx, obj in enumerate(objects):
        # Get object properties
        coords = obj.get('3d_coords', [0, 0, 0])
        size = obj.get('size', 'small')
        shape = obj.get('shape', 'sphere')
        
        # Use X and Z coordinates (top-down view, Y is height)
        center_x = coords[0]
        center_z = coords[2]
        center = np.array([center_x, center_z])
        
        # Size to radius mapping (CLEVR uses specific sizes)
        radius_map = {
            'small': 0.35,
            'large': 0.7
        }
        base_radius = radius_map.get(size, 0.5)
        
        # Shape affects point distribution slightly
        # (spheres are rounder, cubes have corners, cylinders are elongated)
        
        # Level 0: Center point (1 point)
        all_points.append(center)
        object_labels.append(obj_idx)
        level_labels.append(0)
        
        # Level 1: Surface points (~20%)
        num_surface = int(0.2 * points_per_object)
        
        if shape == 'cube':
            # Cube: points on edges
            for i in range(num_surface):
                side = np.random.randint(4)
                t = np.random.uniform(-1, 1)
                if side == 0:
                    point = center + base_radius * np.array([1, t])
                elif side == 1:
                    point = center + base_radius * np.array([-1, t])
                elif side == 2:
                    point = center + base_radius * np.array([t, 1])
                else:
                    point = center + base_radius * np.array([t, -1])
                all_points.append(point)
                object_labels.append(obj_idx)
                level_labels.append(1)
        else:
            # Sphere/cylinder: points on circle
            angles = np.linspace(0, 2*np.pi, num_surface, endpoint=False)
            angles += np.random.uniform(0, 0.1)  # Small random offset
            for angle in angles:
                r = base_radius * (1 + np.random.randn() * 0.02)
                point = center + r * np.array([np.cos(angle), np.sin(angle)])
                all_points.append(point)
                object_labels.append(obj_idx)
                level_labels.append(1)
        
        # Level 2: Interior points (~80%)
        num_interior = points_per_object - num_surface - 1
        
        for _ in range(num_interior):
            if shape == 'cube':
                # Uniform in square
                dx = np.random.uniform(-0.8, 0.8) * base_radius
                dz = np.random.uniform(-0.8, 0.8) * base_radius
                point = center + np.array([dx, dz])
            else:
                # Uniform in circle
                r = base_radius * np.sqrt(np.random.uniform(0, 0.64))  # sqrt for uniform area
                angle = np.random.uniform(0, 2*np.pi)
                point = center + r * np.array([np.cos(angle), np.sin(angle)])
            
            all_points.append(point)
            object_labels.append(obj_idx)
            level_labels.append(2)
    
    return np.array(all_points), np.array(object_labels), np.array(level_labels)


def create_clevr_dataset(scenes, points_per_object=60, max_scenes=None):
    """Convert all CLEVR scenes to point cloud dataset"""
    
    if max_scenes:
        scenes = scenes[:max_scenes]
    
    data_list = []
    obj_labels_list = []
    level_labels_list = []
    
    print(f"\nConverting {len(scenes)} scenes to point clouds...")
    
    for i, scene in enumerate(scenes):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(scenes)} scenes")
        
        try:
            points, obj_labels, level_labels = clevr_scene_to_hierarchical_pointcloud(
                scene, points_per_object
            )
            
            # Only keep scenes with at least 3 objects
            num_objects = len(np.unique(obj_labels))
            if num_objects >= 3:
                data_list.append(points)
                obj_labels_list.append(obj_labels)
                level_labels_list.append(level_labels)
                
        except Exception as e:
            print(f"  Warning: Scene {i} failed: {e}")
            continue
    
    print(f"\n✅ Created {len(data_list)} valid point cloud scenes")
    
    # Statistics
    if data_list:
        avg_points = np.mean([len(d) for d in data_list])
        print(f"   Average points per scene: {avg_points:.1f}")
        
        # Level distribution
        all_levels = np.concatenate(level_labels_list)
        print(f"   Level distribution: L0={np.mean(all_levels==0):.1%}, "
              f"L1={np.mean(all_levels==1):.1%}, L2={np.mean(all_levels==2):.1%}")
    
    return data_list, obj_labels_list, level_labels_list


# ==========================================
# Visualization
# ==========================================

def visualize_clevr_scene(points, obj_labels, level_labels, title="CLEVR Scene"):
    """Visualize a single CLEVR scene as point cloud"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Color by object
    ax1 = axes[0]
    unique_objs = np.unique(obj_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_objs)))
    
    for i, obj_id in enumerate(unique_objs):
        mask = obj_labels == obj_id
        ax1.scatter(points[mask, 0], points[mask, 1], 
                   c=[colors[i]], s=20, alpha=0.7, label=f'Object {obj_id}')
    
    ax1.set_title(f'{title} - By Object')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Right: Color by level
    ax2 = axes[1]
    level_colors = ['red', 'orange', 'blue']
    level_names = ['L0: Core', 'L1: Surface', 'L2: Interior']
    level_sizes = [100, 40, 15]
    
    for level in [2, 1, 0]:  # Draw interior first, then surface, then core on top
        mask = level_labels == level
        ax2.scatter(points[mask, 0], points[mask, 1],
                   c=level_colors[level], s=level_sizes[level], 
                   alpha=0.7, label=level_names[level])
    
    ax2.set_title(f'{title} - By Hierarchy Level')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.legend(loc='upper right')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/clevr_scene_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Saved visualization to /kaggle/working/clevr_scene_example.png")


def visualize_multiple_scenes(data_list, obj_labels_list, level_labels_list, num_scenes=4):
    """Visualize multiple CLEVR scenes"""
    
    num_scenes = min(num_scenes, len(data_list))
    
    fig, axes = plt.subplots(2, num_scenes, figsize=(4*num_scenes, 8))
    
    for i in range(num_scenes):
        points = data_list[i]
        obj_labels = obj_labels_list[i]
        level_labels = level_labels_list[i]
        
        # Top row: by object
        ax1 = axes[0, i]
        unique_objs = np.unique(obj_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_objs)))
        
        for j, obj_id in enumerate(unique_objs):
            mask = obj_labels == obj_id
            ax1.scatter(points[mask, 0], points[mask, 1], 
                       c=[colors[j]], s=10, alpha=0.7)
        
        ax1.set_title(f'Scene {i+1}: {len(unique_objs)} objects')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Bottom row: by level
        ax2 = axes[1, i]
        level_colors = ['red', 'orange', 'blue']
        
        for level in [2, 1, 0]:
            mask = level_labels == level
            ax2.scatter(points[mask, 0], points[mask, 1],
                       c=level_colors[level], s=[100, 30, 8][level], alpha=0.7)
        
        ax2.set_title(f'Hierarchy Levels')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
    
    axes[0, 0].set_ylabel('By Object')
    axes[1, 0].set_ylabel('By Level')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/clevr_multiple_scenes.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Saved visualization to /kaggle/working/clevr_multiple_scenes.png")


# ==========================================
# Save Prepared Dataset
# ==========================================

def save_prepared_dataset(data_list, obj_labels_list, level_labels_list, 
                         output_path='/kaggle/working/clevr_pointclouds.npz'):
    """Save prepared dataset for training"""
    
    # Convert to arrays with padding
    max_points = max(len(d) for d in data_list)
    
    n_scenes = len(data_list)
    padded_points = np.zeros((n_scenes, max_points, 2))
    padded_obj_labels = -np.ones((n_scenes, max_points), dtype=np.int32)
    padded_level_labels = -np.ones((n_scenes, max_points), dtype=np.int32)
    valid_mask = np.zeros((n_scenes, max_points), dtype=bool)
    
    for i, (pts, obj, lvl) in enumerate(zip(data_list, obj_labels_list, level_labels_list)):
        n = len(pts)
        padded_points[i, :n] = pts
        padded_obj_labels[i, :n] = obj
        padded_level_labels[i, :n] = lvl
        valid_mask[i, :n] = True
    
    np.savez_compressed(
        output_path,
        points=padded_points,
        object_labels=padded_obj_labels,
        level_labels=padded_level_labels,
        valid_mask=valid_mask
    )
    
    print(f"\n✅ Saved prepared dataset to {output_path}")
    print(f"   Shape: {padded_points.shape}")
    print(f"   File size: {os.path.getsize(output_path) / 1e6:.1f} MB")
    
    return output_path


# ==========================================
# MAIN: Run on Kaggle
# ==========================================

def main():
    print("="*60)
    print("CLEVR DATASET PREPARATION FOR LOCO EXPERIMENTS")
    print("="*60)
    
    # Step 1: Find or download CLEVR scenes
    print("\n📥 STEP 1: Loading CLEVR dataset...")
    
    scenes_path = find_clevr_scenes_kaggle()
    
    if scenes_path is None:
        print("\nCLEVR not found in Kaggle input. Trying direct download...")
        scenes_path = download_clevr_scenes_direct()
    
    if scenes_path is None:
        print("\n❌ Could not load CLEVR data!")
        print("\nPLEASE ADD CLEVR DATASET MANUALLY:")
        print("1. Click 'Add Data' in the right panel")
        print("2. Search for 'CLEVR'")
        print("3. Add 'CLEVR Dataset' or similar")
        print("4. Re-run this notebook")
        return
    
    # Step 2: Load scenes
    print("\n📊 STEP 2: Loading scene annotations...")
    scenes = load_clevr_scenes(scenes_path, max_scenes=10000)  # Use 10k for quick test
    
    # Step 3: Analyze dataset
    print("\n🔍 STEP 3: Analyzing dataset...")
    stats = analyze_clevr_scenes(scenes[:1000])  # Analyze first 1000
    
    # Step 4: Convert to point clouds
    print("\n🔄 STEP 4: Converting to hierarchical point clouds...")
    data_list, obj_labels_list, level_labels_list = create_clevr_dataset(
        scenes, 
        points_per_object=60,
        max_scenes=5000  # Start with 5000 scenes
    )
    
    # Step 5: Visualize examples
    print("\n📈 STEP 5: Visualizing examples...")
    if len(data_list) > 0:
        visualize_clevr_scene(
            data_list[0], obj_labels_list[0], level_labels_list[0],
            title="Example CLEVR Scene"
        )
        
        if len(data_list) >= 4:
            visualize_multiple_scenes(data_list, obj_labels_list, level_labels_list, num_scenes=4)
    
    # Step 6: Save prepared dataset
    print("\n💾 STEP 6: Saving prepared dataset...")
    output_path = save_prepared_dataset(data_list, obj_labels_list, level_labels_list)
    
    # Summary
    print("\n" + "="*60)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nPrepared dataset saved to: {output_path}")
    print(f"Total scenes: {len(data_list)}")
    print(f"\nNext steps:")
    print("1. Share this notebook output with the insights")
    print("2. We'll create the LoCo vs Euclidean training script")
    print("3. Run the comparison experiment")
    print("="*60)
    
    return data_list, obj_labels_list, level_labels_list


if __name__ == "__main__":
    main()
