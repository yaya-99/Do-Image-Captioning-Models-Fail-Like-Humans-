"""
config.py — Master Path Configuration
======================================
Project: Do Image Captioning Models Fail Like Humans?

This file is imported by ALL notebooks (01–07).
Edit the paths in Section 1 to match your local setup.
Everything else is derived automatically.

Folder structure assumed (matches your code_p1 layout):
  code_p1/
  ├── annotations/
  │   ├── captions_val2014.json
  │   ├── captions_train2014.json
  │   ├── instances_val2014.json
  │   └── instances_train2014.json
  ├── fixations/
  │   ├── train/   ← .mat fixation files (SALICON)
  │   └── val/     ← .mat fixation files (SALICON)
  ├── images/
  │   ├── train/   ← COCO train images (.jpg)
  │   └── val/     ← COCO val images (.jpg)
  ├── maps/
  │   ├── train/   ← Saliency map PNGs (SALICON)
  │   └── val/     ← Saliency map PNGs (SALICON)
  └── *.ipynb      ← All notebooks live here
"""

from pathlib import Path
import numpy as np

# ══════════════════════════════════════════════════════════
#  SECTION 1 — EDIT THESE TO MATCH YOUR SETUP
# ══════════════════════════════════════════════════════════

# Root of your project (where the notebooks live)
# If notebooks are inside code_p1/, set this to Path('.')
# If you're running from outside, set the full path e.g.:
#   PROJECT_ROOT = Path('/home/user/code_p1')
PROJECT_ROOT = Path('.')

# ── Which COCO split to use for experiments ──
# 'val' is standard for evaluation (do NOT use train for final results)
SPLIT = 'val'   # 'val' or 'train'

# ── Spatial resolution for attention / saliency maps ──
# 24x24 matches BLIP's ViT-B/16 patch grid (196 patches → we use 14x14 internally,
# but 24x24 is used after bilinear upsampling to match midterm methodology)
SPATIAL_RES = 24

# ── Number of images to use in Phase 1 ──
N_IMAGES = 200   # set to 50 to replicate midterm exactly, 200 for Phase 1

# ── Reproducibility seed ──
SEED = 42

# ── Models to run ──
# Comment out any model you don't want to run (OFA needs extra setup)
MODELS_TO_RUN = ['blip', 'blip2', 'vit_gpt2'] #'ofa']

# ══════════════════════════════════════════════════════════
#  SECTION 2 — DERIVED PATHS (auto-computed, don't edit)
# ══════════════════════════════════════════════════════════

# ── Input data paths ─────────────────────────────────────
ANNOTATIONS_DIR  = PROJECT_ROOT / 'annotations'
IMAGES_DIR       = PROJECT_ROOT / 'images' / SPLIT       # e.g. images/val/
IMAGES_TRAIN_DIR = PROJECT_ROOT / 'images' / 'train'
IMAGES_VAL_DIR   = PROJECT_ROOT / 'images' / 'val'

MAPS_DIR         = PROJECT_ROOT / 'maps' / SPLIT         # e.g. maps/val/
MAPS_TRAIN_DIR   = PROJECT_ROOT / 'maps' / 'train'
MAPS_VAL_DIR     = PROJECT_ROOT / 'maps' / 'val'

FIXATIONS_DIR    = PROJECT_ROOT / 'fixations' / SPLIT    # e.g. fixations/val/
FIXATIONS_TRAIN_DIR = PROJECT_ROOT / 'fixations' / 'train'
FIXATIONS_VAL_DIR   = PROJECT_ROOT / 'fixations' / 'val'

# ── COCO annotation JSON files ───────────────────────────
CAPTIONS_VAL_JSON    = ANNOTATIONS_DIR / 'captions_val2014.json'
CAPTIONS_TRAIN_JSON  = ANNOTATIONS_DIR / 'captions_train2014.json'
INSTANCES_VAL_JSON   = ANNOTATIONS_DIR / 'instances_val2014.json'
INSTANCES_TRAIN_JSON = ANNOTATIONS_DIR / 'instances_train2014.json'

# Active annotation file (based on SPLIT)
CAPTIONS_JSON  = CAPTIONS_VAL_JSON  if SPLIT == 'val'  else CAPTIONS_TRAIN_JSON
INSTANCES_JSON = INSTANCES_VAL_JSON if SPLIT == 'val'  else INSTANCES_TRAIN_JSON

# ── Output / working directories ─────────────────────────
OUTPUT_DIR  = PROJECT_ROOT / 'research_data' / 'outputs'
FIGURES_DIR = PROJECT_ROOT / 'research_data' / 'figures'
IEEE_DIR    = PROJECT_ROOT / 'research_data' / 'ieee_figures'

# Create output directories on import
for d in [OUTPUT_DIR, FIGURES_DIR, IEEE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════
#  SECTION 3 — SALIENCY / FIXATION LOADING UTILITIES
# ══════════════════════════════════════════════════════════

def get_saliency_map_path(image_id: int, split: str = SPLIT) -> Path:
    """
    Return path to the SALICON saliency PNG for a given COCO image_id.
    SALICON filenames follow COCO convention:
      COCO_val2014_000000123456.png
    """
    maps_dir = MAPS_VAL_DIR if split == 'val' else MAPS_TRAIN_DIR
    fname = f'COCO_{split}2014_{image_id:012d}.png'
    return maps_dir / fname


def get_fixation_mat_path(image_id: int, split: str = SPLIT) -> Path:
    """
    Return path to the SALICON fixation .mat file for a given COCO image_id.
    SALICON .mat filenames follow COCO convention:
      COCO_val2014_000000123456.mat
    """
    fix_dir = FIXATIONS_VAL_DIR if split == 'val' else FIXATIONS_TRAIN_DIR
    fname = f'COCO_{split}2014_{image_id:012d}.mat'
    return fix_dir / fname


def get_image_path(image_id: int, split: str = SPLIT,
                   file_name: str = None) -> Path:
    """
    Return path to the COCO image .jpg file.
    If file_name is provided, uses it directly; otherwise constructs from image_id.
    """
    img_dir = IMAGES_VAL_DIR if split == 'val' else IMAGES_TRAIN_DIR
    if file_name:
        return img_dir / file_name
    return img_dir / f'COCO_{split}2014_{image_id:012d}.jpg'


def load_saliency_from_png(image_id: int, split: str = SPLIT,
                            spatial_res: int = SPATIAL_RES) -> np.ndarray:
    """
    Load SALICON saliency map from PNG file.

    Pipeline:
      1. Open grayscale PNG from maps/{split}/
      2. Resize to (spatial_res × spatial_res) via bilinear interpolation
      3. Normalize to probability distribution (sum=1, epsilon smoothed)

    Returns: np.ndarray of shape (spatial_res * spatial_res,)
    """
    from PIL import Image as PilImage

    sal_path = get_saliency_map_path(image_id, split)

    if sal_path.exists():
        sal = PilImage.open(sal_path).convert('L')
        sal = sal.resize((spatial_res, spatial_res), PilImage.BILINEAR)
        sal_arr = np.array(sal, dtype=np.float64)
    else:
        print(f'  ⚠️  Saliency PNG not found: {sal_path}. Using simulated map.')
        sal_arr = _simulate_saliency(spatial_res, image_id)

    sal_flat = sal_arr.flatten() + 1e-10
    sal_flat /= sal_flat.sum()
    return sal_flat


def load_saliency_from_mat(image_id: int, split: str = SPLIT,
                            spatial_res: int = SPATIAL_RES) -> np.ndarray:
    """
    Load SALICON fixation data from .mat file and convert to saliency map.

    SALICON .mat files contain:
      - 'gaze'  : struct with fields 'fixations' (Nx2 pixel coordinates)
      - OR a direct 'fixationPts' or 'fixationMap' field

    This function handles both formats automatically.

    Pipeline:
      1. Load .mat file
      2. Extract fixation coordinates or map
      3. Build a 2D density map via Gaussian kernel density estimation
      4. Resize to spatial_res × spatial_res
      5. Normalize to probability distribution

    Returns: np.ndarray of shape (spatial_res * spatial_res,)
    """
    try:
        import scipy.io as sio
        from PIL import Image as PilImage

        mat_path = get_fixation_mat_path(image_id, split)

        if not mat_path.exists():
            # Fall back to PNG saliency
            return load_saliency_from_png(image_id, split, spatial_res)

        mat = sio.loadmat(str(mat_path))

        # ── Try to extract saliency map or fixation points ──
        sal_arr = None

        # Format 1: direct saliency/density map
        for key in ['saliencyMap', 'fixationMap', 'salMap', 'map']:
            if key in mat:
                sal_arr = np.array(mat[key], dtype=np.float64)
                break

        # Format 2: fixation point coordinates → build density map
        if sal_arr is None:
            fixation_pts = None
            for key in ['fixationPts', 'fixations', 'gaze']:
                if key in mat:
                    raw = mat[key]
                    # Handle nested MATLAB structs
                    if hasattr(raw, 'dtype') and raw.dtype.names:
                        for subkey in ['fixations', 'fixPts', 'pts']:
                            if subkey in raw.dtype.names:
                                fixation_pts = np.array(
                                    raw[subkey][0, 0], dtype=np.float64
                                )
                                break
                    else:
                        fixation_pts = np.array(raw, dtype=np.float64)
                    break

            if fixation_pts is not None and fixation_pts.ndim == 2:
                # fixation_pts: Nx2 array of (x, y) pixel coordinates
                # Build density map at original image resolution then resize
                H, W = 480, 640  # COCO standard resolution
                density = np.zeros((H, W), dtype=np.float64)
                pts = fixation_pts.astype(int)
                for pt in pts:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= y < H and 0 <= x < W:
                        density[y, x] += 1.0

                # Gaussian blur to convert fixations → smooth saliency
                from scipy.ndimage import gaussian_filter
                density = gaussian_filter(density, sigma=20)
                sal_arr = density

        if sal_arr is None:
            # Nothing worked — fall back to PNG
            return load_saliency_from_png(image_id, split, spatial_res)

        # Resize to target resolution
        sal_img = PilImage.fromarray(sal_arr).resize(
            (spatial_res, spatial_res), PilImage.BILINEAR
        )
        sal_resized = np.array(sal_img, dtype=np.float64)

        # Normalize
        sal_flat = sal_resized.flatten() + 1e-10
        sal_flat /= sal_flat.sum()
        return sal_flat

    except Exception as e:
        print(f'  ⚠️  .mat loading failed for image {image_id}: {e}')
        return load_saliency_from_png(image_id, split, spatial_res)


def load_saliency(image_id: int, split: str = SPLIT,
                  spatial_res: int = SPATIAL_RES,
                  prefer: str = 'mat') -> np.ndarray:
    """
    Unified saliency loader. Tries .mat first (higher quality), falls back to PNG.

    Args:
        prefer: 'mat' to prefer fixation .mat files (recommended)
                'png' to prefer saliency map PNGs
    """
    if prefer == 'mat':
        mat_path = get_fixation_mat_path(image_id, split)
        if mat_path.exists():
            return load_saliency_from_mat(image_id, split, spatial_res)
        else:
            return load_saliency_from_png(image_id, split, spatial_res)
    else:
        png_path = get_saliency_map_path(image_id, split)
        if png_path.exists():
            return load_saliency_from_png(image_id, split, spatial_res)
        else:
            return load_saliency_from_mat(image_id, split, spatial_res)


def _simulate_saliency(res: int, seed: int) -> np.ndarray:
    """
    Fallback: simulate center-biased saliency when files are missing.
    Only used as last resort.
    """
    rng = np.random.RandomState(seed % 10000)
    cx, cy = res // 2, res // 2
    sigma = res / 3.5
    x, y = np.meshgrid(np.arange(res), np.arange(res))
    gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    noise = rng.rand(res, res) * 0.1
    return gaussian + noise


# ══════════════════════════════════════════════════════════
#  SECTION 4 — TAXONOMY & MODEL CONSTANTS
# ══════════════════════════════════════════════════════════

ERROR_TAXONOMY = {
    'object_hallucination':     {'code': 'OH', 'color': '#D55E00'},
    'object_misidentification': {'code': 'OM', 'color': '#0072B2'},
    'attribute_mismatch':       {'code': 'AM', 'color': '#E69F00'},
    'relation_mismatch':        {'code': 'RM', 'color': '#CC79A7'},
    'correct':                  {'code': 'OK', 'color': '#009E73'},
}

ERROR_TYPES = [k for k in ERROR_TAXONOMY if k != 'correct']

MODEL_DISPLAY = {
    'blip':     'BLIP',
    'blip2':    'BLIP-2',
    'ofa':      'OFA',
    'vit_gpt2': 'ViT-GPT2',
}

# HuggingFace model IDs
MODEL_HF_IDS = {
    'blip':     'Salesforce/blip-image-captioning-large',
    'blip2':    'Salesforce/blip2-opt-2.7b',
    'ofa':      'OFA-Sys/ofa-base',
    'vit_gpt2': 'nlpconnect/vit-gpt2-image-captioning',
}

SCENARIO_CATEGORIES = [
    'gender_ambiguity',
    'object_confusion',
    'context_mismatch',
    'counting_errors',
    'attribute_errors',
    'general_baseline',
]

# Generation hyperparams (consistent across all models)
GEN_CONFIG = {
    'max_new_tokens': 80,
    'min_length': 15,              # ➜ add this
    'num_beams': 5,
    'length_penalty': 1.1,         # ➜ add this (better than 1.2)
    'no_repeat_ngram_size': 2,     # ➜ add this
    'early_stopping': True,
}

# ══════════════════════════════════════════════════════════
#  SECTION 5 — VALIDATION ON IMPORT
# ══════════════════════════════════════════════════════════

def validate_paths(verbose: bool = True) -> dict:
    """
    Check which data files exist. Call this at the start of each notebook.
    Returns a dict of {path_name: exists (bool)}.
    """
    checks = {
        'captions_val_json':   CAPTIONS_VAL_JSON.exists(),
        'captions_train_json': CAPTIONS_TRAIN_JSON.exists(),
        'instances_val_json':  INSTANCES_VAL_JSON.exists(),
        'images_val_dir':      IMAGES_VAL_DIR.exists(),
        'images_train_dir':    IMAGES_TRAIN_DIR.exists(),
        'maps_val_dir':        MAPS_VAL_DIR.exists(),
        'maps_train_dir':      MAPS_TRAIN_DIR.exists(),
        'fixations_val_dir':   FIXATIONS_VAL_DIR.exists(),
        'fixations_train_dir': FIXATIONS_TRAIN_DIR.exists(),
    }

    # Count files in each directory
    dir_counts = {}
    for name, path in [
        ('images/val',    IMAGES_VAL_DIR),
        ('images/train',  IMAGES_TRAIN_DIR),
        ('maps/val',      MAPS_VAL_DIR),
        ('maps/train',    MAPS_TRAIN_DIR),
        ('fixations/val', FIXATIONS_VAL_DIR),
        ('fixations/train', FIXATIONS_TRAIN_DIR),
    ]:
        if path.exists():
            files = list(path.iterdir())
            dir_counts[name] = len(files)
        else:
            dir_counts[name] = 0

    if verbose:
        print('📁 Path Validation:')
        print(f'   Project root:  {PROJECT_ROOT.resolve()}')
        print(f'   Active split:  {SPLIT}')
        print()
        for name, exists in checks.items():
            icon = '✅' if exists else '❌'
            print(f'   {icon}  {name}')
        print()
        print('📊 File counts:')
        for name, cnt in dir_counts.items():
            icon = '✅' if cnt > 0 else '⚠️ '
            print(f'   {icon}  {name}: {cnt} files')
        print()

        # Quick .mat test
        fix_files = list(FIXATIONS_VAL_DIR.glob('*.mat')) if FIXATIONS_VAL_DIR.exists() else []
        if fix_files:
            print(f'   🧪 Sample .mat file: {fix_files[0].name}')
            try:
                import scipy.io as sio
                mat = sio.loadmat(str(fix_files[0]))
                print(f'      Keys: {[k for k in mat.keys() if not k.startswith("_")]}')
            except Exception as e:
                print(f'      ⚠️  Could not inspect: {e}')

    return checks


if __name__ == '__main__':
    validate_paths()
