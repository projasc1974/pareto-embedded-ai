"""
Fashion-MNIST Dataset Partition Script

Creates a reproducible dataset partition for Fashion-MNIST:
- Train: 48,000 images (80%)
- Validation: 12,000 images (20% of train)
- Test PC: 10,000 images (original)
- Test Embedded: 1,000 images (100 per class)
"""

import numpy as np
import pickle
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIGURATION
# =============================================================================

SEMILLA = 42
N_EMBEDDED_POR_CLASE = 100  # 100 images per class for embedded testing

print("=" * 80)
print("CREATING REPRODUCIBLE PARTITION FOR FASHION-MNIST")
print("=" * 80)

# =============================================================================
# LOAD FASHION-MNIST DATASET
# =============================================================================

print("\nLoading Fashion-MNIST...")
(x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()

# Fashion-MNIST class names (for reference)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Original train set: {x_train_full.shape}, {len(np.unique(y_train_full))} classes")
print(f"Original test set:  {x_test_full.shape}, {len(np.unique(y_test_full))} classes")

# =============================================================================
# VERIFY ORIGINAL DISTRIBUTION
# =============================================================================

print("\nOriginal distribution:")
train_classes, train_counts = np.unique(y_train_full, return_counts=True)
test_classes, test_counts = np.unique(y_test_full, return_counts=True)

print("Train (expected: 6,000 per class):")
for cls, cnt in zip(train_classes, train_counts):
    print(f"   Class {cls} ({class_names[cls]}): {cnt} images")

print("Test (expected: 1,000 per class):")
for cls, cnt in zip(test_classes, test_counts):
    print(f"   Class {cls} ({class_names[cls]}): {cnt} images")

# =============================================================================
# SET RANDOM SEED FOR REPRODUCIBILITY
# =============================================================================

np.random.seed(SEMILLA)

# =============================================================================
# 1. SPLIT TRAIN INTO TRAIN (80%) AND VALIDATION (20%)
# =============================================================================

# Use the full training set (60,000) to create:
# 48,000 train and 12,000 validation

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=0.2,
    stratify=y_train_full,
    random_state=SEMILLA
)

print(f"\nTrain/validation split:")
print(f"   Train: {len(x_train)} images (expected: 48,000)")
print(f"   Validation: {len(x_val)} images (expected: 12,000)")

# Verify stratified distribution
train_classes, train_counts = np.unique(y_train, return_counts=True)
val_classes, val_counts = np.unique(y_val, return_counts=True)

print("\n   Train distribution (~4,800 per class expected):")
for cls, cnt in zip(train_classes, train_counts):
    print(f"      Class {cls} ({class_names[cls]}): {cnt} images")

print("   Validation distribution (~1,200 per class expected):")
for cls, cnt in zip(val_classes, val_counts):
    print(f"      Class {cls} ({class_names[cls]}): {cnt} images")

# =============================================================================
# 2. SELECT EMBEDDED TEST SET (100 PER CLASS FROM ORIGINAL TEST)
# =============================================================================

print(f"\nSelecting {N_EMBEDDED_POR_CLASE} images per class for embedded testing...")

test_embedded_indices = []
test_embedded_labels = []

for clase in range(10):
    # Indices of this class in the original test set
    idx_clase = np.where(y_test_full == clase)[0]
    print(f"   Class {clase} ({class_names[clase]}): {len(idx_clase)} available", end="")
    
    # Randomly select 100 samples
    seleccionados = np.random.choice(idx_clase, N_EMBEDDED_POR_CLASE, replace=False)
    test_embedded_indices.extend(seleccionados)
    test_embedded_labels.extend([clase] * N_EMBEDDED_POR_CLASE)
    print(f" → selected {len(seleccionados)}")

test_embedded_indices = np.array(test_embedded_indices)
test_embedded_labels = np.array(test_embedded_labels)

print(f"\nTotal embedded test images: {len(test_embedded_indices)}")
print(f"Images per class: {N_EMBEDDED_POR_CLASE}")

# =============================================================================
# 3. VERIFY THAT EMBEDDED TEST ⊆ ORIGINAL TEST
# =============================================================================

# Boolean mask for embedded test samples
test_embedded_mask = np.zeros(len(x_test_full), dtype=bool)
test_embedded_mask[test_embedded_indices] = True

# Verify index validity
assert np.all(test_embedded_indices < len(x_test_full)), "Indices out of range"

print("\nVerification: embedded test set is a subset of the original test set")

# =============================================================================
# 4. SAVE PARTITION
# =============================================================================

particion_fashion = {
    'train': {
        'indices': np.arange(len(x_train)),
        'x': x_train,
        'y': y_train
    },
    'validation': {
        'indices': np.arange(len(x_val)),
        'x': x_val,
        'y': y_val
    },
    'test_pc': {
        'indices': np.arange(len(x_test_full)),
        'x': x_test_full,
        'y': y_test_full
    },
    'test_embedded': {
        'indices': test_embedded_indices,
        'mask': test_embedded_mask,
        'y': test_embedded_labels,
        'per_class': N_EMBEDDED_POR_CLASE
    },
    'metadata': {
        'total_train': len(x_train),
        'total_val': len(x_val),
        'total_test_pc': len(x_test_full),
        'total_test_embedded': len(test_embedded_indices),
        'seed': SEMILLA,
        'classes': 10,
        'class_names': class_names,
        'description': 'Fashion-MNIST partitioned: train(48k), val(12k), test_pc(10k), test_embedded(1k)'
    }
}

with open('fashion_mnist_particion.pkl', 'wb') as f:
    pickle.dump(particion_fashion, f)

print(f"\nPartition saved in 'fashion_mnist_particion.pkl'")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("FASHION-MNIST PARTITION SUMMARY")
print("=" * 80)
print(f"""
STATISTICS:
   • Train: {len(x_train)} images
   • Validation: {len(x_val)} images
   • Test PC: {len(x_test_full)} images
   • Test Embedded: {len(test_embedded_indices)} images
   • Images per class (embedded): {N_EMBEDDED_POR_CLASE}

This partition must be used in ALL experiments:
   - Training on PC (train + validation)
   - Evaluation on PC (test_pc)
   - Evaluation on embedded devices (ESP32, Pico, Zero W)
""")

# =============================================================================
# VERIFY EMBEDDED DISTRIBUTION
# =============================================================================

print("\nEmbedded test distribution:")
embedded_classes, embedded_counts = np.unique(test_embedded_labels, return_counts=True)
for cls, cnt in zip(embedded_classes, embedded_counts):
    print(f"   Class {cls} ({class_names[cls]}): {cnt} images")
