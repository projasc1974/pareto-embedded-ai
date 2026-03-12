"""
STEP 1: Scan and organize files – CORRECTED VERSION (2100 images)

This script scans a directory containing synthetic images of geometric figures
(circle, triangle, square), extracts the class from the filename, verifies the
dataset distribution, and splits the dataset into training, validation, and test
sets. Additionally, it selects a balanced subset of test images intended for
embedded device evaluation.

Expected filename format:
circle_perfect_0351.png
triangle_distorted_0093.png
square_perfect_0120.png

Final dataset split:
- Train: 60%
- Validation: 10%
- Test (PC): 30%

From the test set, the script randomly selects:
- 100 images per class (300 total) for embedded-device testing.

The resulting structure is saved in a pickle file for later use in experiments.
"""

import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# 1. Define the path to the image dataset
RUTA_IMAGENES = "/home/patricio/experimento/Synthetic_Figures"  # ADJUST THIS PATH!

# 2. Scan all .png files in the directory
todos_archivos = [f for f in os.listdir(RUTA_IMAGENES) if f.endswith('.png')]
print(f"Total de archivos encontrados: {len(todos_archivos)}")

# 3. Extract the class (shape: circle, triangle, square) from each filename
nombres_validos = []
clases = []

for archivo in todos_archivos:
    try:
        # Expected format: shape_subtype_number.png
        # Example: circle_perfect_0351.png
        partes = archivo.split('_')
        
        if len(partes) >= 3:
            forma = partes[0]  # circle, triangle, square
            
            # Normalize class names to Spanish for consistency
            if forma == 'circle':
                clase = 'circulo'
            elif forma == 'triangle':
                clase = 'triangulo'
            elif forma == 'square':
                clase = 'cuadrado'
            else:
                print(f"  Forma desconocida '{forma}' en: {archivo}")
                continue
                
            nombres_validos.append(archivo)
            clases.append(clase)
        else:
            print(f"  Formato inesperado (menos de 3 partes): {archivo}")
            
    except Exception as e:
        print(f"  Error procesando {archivo}: {e}")

# Convert lists to numpy arrays
nombres_validos = np.array(nombres_validos)
clases = np.array(clases)

print(f"\nArchivos válidos clasificados: {len(nombres_validos)}")
if len(nombres_validos) == 0:
    print(" ERROR: No se clasificó ningún archivo válido")
    exit()

# Verify that the dataset contains the expected number of images
print(f"\n VERIFICACIÓN DE CANTIDADES:")
clases_unicas, counts = np.unique(clases, return_counts=True)
total_esperado = 2100
total_actual = len(nombres_validos)

if total_actual != total_esperado:
    print(f"  ADVERTENCIA: Se esperaban {total_esperado} imágenes, se encontraron {total_actual}")
else:
    print(f" Total correcto: {total_actual} imágenes")

print("\nDistribución por clase:")
for clase, count in zip(clases_unicas, counts):
    print(f"  {clase}: {count} imágenes (esperado: 700)")

# Verify that each class contains exactly 700 images
for clase, count in zip(clases_unicas, counts):
    if count != 700:
        print(f"  ⚠  Clase {clase}: tiene {count}, debería tener 700")

# 4. Fix the random seed to ensure reproducibility
SEMILLA = 42
np.random.seed(SEMILLA)

# 5. Split the dataset filenames
# First split: 60% training, 40% temporary
X_train_nombres, X_temp_nombres, y_train, y_temp = train_test_split(
    nombres_validos, clases, 
    test_size=0.4, 
    stratify=clases, 
    random_state=SEMILLA
)

# Second split: from the 40% temporary set
# 25% validation (10% total), 75% test (30% total)
X_val_nombres, X_test_nombres, y_val, y_test = train_test_split(
    X_temp_nombres, y_temp,
    test_size=0.75,  # 75% of 40% = 30% of total
    stratify=y_temp,
    random_state=SEMILLA
)

print(f"\n PARTICIÓN 60/10/30:")
print(f"   - Train (60%): {len(X_train_nombres)} imágenes")
print(f"     Esperado: 1260 imágenes")
print(f"   - Validation (10%): {len(X_val_nombres)} imágenes")
print(f"     Esperado: 210 imágenes")
print(f"   - Test PC (30%): {len(X_test_nombres)} imágenes")
print(f"     Esperado: 630 imágenes")

# Verify split integrity
total_partido = len(X_train_nombres) + len(X_val_nombres) + len(X_test_nombres)
print(f"\n Total particionado: {total_partido} (debe ser {total_actual})")

# 6. Select 100 files per class for embedded-device testing
test_embedded_nombres = []
clases_unicas_test = np.unique(y_test)

print(f"\n SELECCIONANDO 100 IMÁGENES POR CLASE PARA TEST EMBEBIDO:")
total_seleccionadas = 0

for clase in clases_unicas_test:
    # Get all test filenames for this class
    mascara = (y_test == clase)
    nombres_clase = X_test_nombres[mascara]
    n_disponibles = len(nombres_clase)
    
    # Verify that at least 100 images are available
    if n_disponibles < 100:
        print(f"  ⚠  Clase '{clase}': solo {n_disponibles} disponibles en test (se usarán todas)")
        n_seleccionar = n_disponibles
    else:
        print(f"   Clase '{clase}': {n_disponibles} disponibles, seleccionando 100")
        n_seleccionar = 100
    
    # Random selection
    seleccionados = np.random.choice(nombres_clase, n_seleccionar, replace=False)
    test_embedded_nombres.extend(seleccionados)
    total_seleccionadas += n_seleccionar

test_embedded_nombres = np.array(test_embedded_nombres)

print(f"\n TOTAL TEST EMBEBIDO: {total_seleccionadas} imágenes")
print(f"   Esperado: 300 imágenes (3 clases × 100)")

# 7. Verify that test_embedded is a subset of test_pc
embedded_set = set(test_embedded_nombres)
test_pc_set = set(X_test_nombres)

if embedded_set.issubset(test_pc_set):
    print(f"\n Verificación: test_embedded ⊆ test_pc")
    print(f"   {len(embedded_set)} imágenes de test_embedded están en test_pc")
else:
    print(f"\n ERROR: test_embedded NO es subconjunto de test_pc")
    fuera = embedded_set - test_pc_set
    print(f"   {len(fuera)} imágenes fuera: {list(fuera)[:5]}")

# 8. Save the file lists and metadata
indices_archivos = {
    'train': X_train_nombres.tolist(),
    'validation': X_val_nombres.tolist(),
    'test_pc': X_test_nombres.tolist(),
    'test_embedded': test_embedded_nombres.tolist(),
    'clases': {
        'train': y_train.tolist(),
        'validation': y_val.tolist(),
        'test_pc': y_test.tolist()
    },
    'metadata': {
        'total_imagenes': total_actual,
        'train_count': len(X_train_nombres),
        'val_count': len(X_val_nombres),
        'test_pc_count': len(X_test_nombres),
        'test_embedded_count': len(test_embedded_nombres),
        'seed': SEMILLA
    }
}

# Save the structure to a pickle file
with open('split_figuras_2100.pkl', 'wb') as f:
    pickle.dump(indices_archivos, f)

print(f"\n✅ Archivo 'split_figuras_2100.pkl' creado")

# 9. Final summary
print(f"\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)
print(f"Total imágenes: {total_actual}")
print(f"Train: {len(X_train_nombres)} ({len(X_train_nombres)/total_actual*100:.1f}%)")
print(f"Validation: {len(X_val_nombres)} ({len(X_val_nombres)/total_actual*100:.1f}%)")
print(f"Test PC: {len(X_test_nombres)} ({len(X_test_nombres)/total_actual*100:.1f}%)")
print(f"Test Embedded: {len(test_embedded_nombres)} ({len(test_embedded_nombres)/total_actual*100:.1f}%)")
print("="*60)

# 10. Show some examples from the embedded test set
print(f"\n Ejemplos de archivos en test_embedded (primeros 10):")
for i, nombre in enumerate(test_embedded_nombres[:10]):
    # Find the class associated with this file
    idx = list(X_test_nombres).index(nombre) if nombre in X_test_nombres else -1
    clase = y_test[idx] if idx >= 0 else "desconocida"
    print(f"   {i+1:2d}. {nombre} ({clase})")
