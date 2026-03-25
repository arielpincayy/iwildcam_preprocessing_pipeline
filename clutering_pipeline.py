from processing.auto_cluster_dl import auto_cluster_dl_hdbscan
from processing.clahe import chalhe_images
from processing.make_crops import make_crops
from processing.megadetector_step import megadetector_classify
from processing.divide import divide_images
from processing.remove_footer import remove_footer
import os
import shutil

# --- CONFIGURACIÓN ---
PATH = os.getcwd()
SRC_IMAGES = os.path.join(PATH, 'dataset_ecuador')
IMAGES = os.path.join(PATH, 'images')
# Carpetas intermedias
SORTED_DIR = os.path.join(PATH, 'images_sorted')
CROPS_RAW_DIR = os.path.join(PATH, 'crops')
CROPS_CLAHE_DIR = os.path.join(PATH, 'crops_clahe_processed')
CLUSTERS_OUTPUT = os.path.join(PATH, 'Clusters')       # Salida final
# Definimos rutas claras

ANIMALS_FOLDER = 'Animales'
EMPTY_FOLDER = 'Vacias'

path_animales_img = os.path.join(SORTED_DIR, ANIMALS_FOLDER)

# --- PIPELINE ---

print("=== INICIANDO PIPELINE (MegaDetector -> Crop -> CLAHE -> ResNet50) ===")

# 0. RECORTAR FOOTER
remove_footer(
    input_folder=SRC_IMAGES,
    output_folder=IMAGES,
    pixels_to_cut=400,
    quality=95
)
# 1. MEGADETECTOR
print("\n--- Paso 1: MegaDetector ---")
if os.path.exists('resultados_megadetector.json'):
    print("JSON detectado. Saltando.")
else:
    megadetector_classify(
        input_folder=IMAGES,
        output_file='resultados_megadetector.json',
        model_version='MDV5A',
        conf_threshold=0.2,
        recursive=True
    )

# 2. DIVIDE
print("\n--- Paso 2: Dividir ---")
divide_images(
    json_file='resultados_megadetector.json',
    source_folder=IMAGES,
    dest_root=SORTED_DIR,
    animals_folder_name=ANIMALS_FOLDER,
    empty_folder_name=EMPTY_FOLDER,
    conf_threshold=0.4,
    accepted_categories=['1']
)

shutil.rmtree(IMAGES)

# 3. MAKE CROPS
print("\n--- Paso 3: Recortes (Crops) ---")

make_crops(
    json_file='resultados_megadetector.json',
    input_folder=path_animales_img,
    output_folder=CROPS_RAW_DIR,
    conf_threshold=0.4,
    accepted_categories=['1']
)

shutil.rmtree(SORTED_DIR)

# 4. CLAHE
print("\n--- Paso 4: CLAHE ---")

chalhe_images(
    input_dir=CROPS_RAW_DIR,
    output_dir=CROPS_CLAHE_DIR
)

shutil.rmtree(CROPS_RAW_DIR)

# 5. AUTO CLUSTER (RESNET)
print("\n--- Paso 5: Clustering con ResNet50 + UMAP ---")
# CAMBIO: Usamos la función que integra ResNet
auto_cluster_dl_hdbscan(
    input_dir=CROPS_CLAHE_DIR,    # Usamos los crops mejorados
    output_dir=CLUSTERS_OUTPUT,
    min_cluster_size=15
)

print(f"\n¡LISTO! Revisa la carpeta: {CLUSTERS_OUTPUT}")
