import numpy as np
import itertools
import random
from time import time
from functions import *

### --- Configuración de la Simulación --- ###

print("--- Iniciando validación del Algoritmo Final ---")

# --- Parámetros de Cámara ---
K = np.array([[800, 0, 640],
              [0, 800, 360],
              [0, 0, 1]])

# Altura de las cámaras sobre el suelo (Y=0)
CAM_HEIGHT = 1.0

# Poses de las cámaras (R: rotación, t: traslación)
# R_world_to_cam, t_world_to_cam
# Para que C = -R.T @ t, necesitamos R,t de la transformación Xc = R @ Xw + t
# C_world = [x, y, z]
# t_cam1 = np.array([[0, CAM_HEIGHT, 0]]).T
# R_cam1 = rot_y(0) # Mirando recto
# R1 = R_cam1.T
# t1 = -R_cam1.T @ t_cam1

# Simplificación: Usemos la convención del código original donde
# R y t son directamente los parámetros extrínsecos de W->C (Xc = R@Xw + t)
# El origen (0,0,0) del mundo está en el suelo, bajo la cámara 1
# t1: Cam1 está en (0, 1, 0) y mira a +Z
R1 = rot_y(0)  # Rotación de mundo a cam1 (identidad)
t1 = np.array([[0, -CAM_HEIGHT, 0]]).T  # Traslación: mover mundo 1m ABAJO

# t2: Cam2 está en (-0.5, 1, 0) y mira 30 grados a la derecha
C2_world = np.array([[-0.5, CAM_HEIGHT, 0]]).T
R_cam2 = rot_y(30)
R2 = R_cam2.T
t2 = -R_cam2.T @ C2_world

# t3: Cam3 está en (0.5, 1, 0) y mira 30 grados a la izquierda
C3_world = np.array([[0.5, CAM_HEIGHT, 0]]).T
R_cam3 = rot_y(-30)
R3 = R_cam3.T
t3 = -R_cam3.T @ C3_world

Ks = [K, K, K]
Rs = [R1, R2, R3]
ts = [t1, t2, t3]
NUM_CAMERAS = len(Ks)

# --- Parámetros de Conos y Validación ---
Y_GROUND = 0.0
CONE_REAL_HEIGHT = 0.30
CONE_REAL_WIDTH = 0.20
CONE_CENTER_Y = Y_GROUND + CONE_REAL_HEIGHT / 2.0  # 0.15m

# 1. Validación de Altura
HEIGHT_TOLERANCE_M = 0.05  # +/- 5cm
Y_MIN_VALID = CONE_CENTER_Y - HEIGHT_TOLERANCE_M  # 0.10m
Y_MAX_VALID = CONE_CENTER_Y + HEIGHT_TOLERANCE_M  # 0.20m

# 2. Validación de Coincidencia de Rayos (del código original)
MAX_RAY_DISTANCE_M = 0.05  # 10 cm de error de triangulación
MIN_RAY_MATCH = 3  # Al menos 2 de 3 rayos deben coincidir

# 3. Validación de Tamaño de BBox (NUEVO)
BBOX_SIZE_TOLERANCE = 0.50  # 50% de error permitido
MIN_BBOX_MATCHES = 3  # Al menos 2 de 3 cámaras deben validar el tamaño

# 4. Parámetros de Métrica
CLUSTER_DISTANCE_M = 0.5  # 50cm para agrupar puntos validados
MATCH_DISTANCE_M = 0.5  # 50cm para asociar predicción a GT

print(f"Sistema de coordenadas: Y=Arriba. Suelo en Y={Y_GROUND}m.")
print(f"Altura Y del centro del cono: {CONE_CENTER_Y:.3f}m")
print(f"Rango de 'Confirmación por altura' (Y): [{Y_MIN_VALID:.3f}m, {Y_MAX_VALID:.3f}m]\n")

### --- Generación de Escena y Detecciones --- ###

# Generar escena aleatoria
N_REAL_CONES = 8
N_FALSE_OBJECTS = 2
ground_truth_cones, false_cones = generate_scene(
    num_real=N_REAL_CONES,
    num_false=N_FALSE_OBJECTS,
    y_cone=CONE_CENTER_Y,
    y_range_false=(Y_MAX_VALID + 0.1, Y_MAX_VALID + 0.2),  # Objetos falsos por encima
    x_range=(-5, 5),  # 10m de ancho
    z_range=(3, 15)  # 3m a 15m de distancia
)

all_objects_3d = ground_truth_cones + false_cones

print(f"--- 🌎 Escena Aleatoria Generada ({len(all_objects_3d)} objetos) ---")
print(f"  Conos Reales (GT): {len(ground_truth_cones)}")
for i, cone in enumerate(ground_truth_cones):
    print(f"    GT {i + 1}: {np.round(cone.ravel(), 2)}")
print(f"  Objetos Falsos: {len(all_objects_3d) - len(ground_truth_cones)}")
for i, obj in enumerate(false_cones):
    print(f"    False {i + 1}: {np.round(obj.ravel(), 2)}")

# Simular Detecciones YOLO (BBoxes) para CADA cámara
all_detections_by_cam = []
for i in range(NUM_CAMERAS):
    K_cam, R_cam, t_cam = Ks[i], Rs[i], ts[i]
    detections_this_cam = []

    for obj in all_objects_3d:
        # Calcular el BBox ideal
        bbox = get_expected_bbox(K_cam, R_cam, t_cam, obj, CONE_REAL_WIDTH, CONE_REAL_HEIGHT)

        if bbox is not None:
            # Simular una detección real de YOLO
            u, v, w, h = bbox

            # Añadir ruido al centro
            u_noisy = u + random.normalvariate(0, 0.5)  # Ruido de +/- 0.5px
            v_noisy = v + random.normalvariate(0, 0.5)

            # Añadir ruido al tamaño (simulando errores de YOLO)
            w_noisy = w * random.uniform(0.8, 1.2)  # +/- 20% de error de tamaño
            h_noisy = h * random.uniform(0.8, 1.2)

            # Formato de "Detección": (u_center, v_center, width, height)
            detections_this_cam.append((u_noisy, v_noisy, w_noisy, h_noisy))

    all_detections_by_cam.append(detections_this_cam)

# `all_detections_by_cam` es una lista de listas:
# [ [bbox_c1_o1, bbox_c1_o2, ...],  <- Detecciones Cam 1
#   [bbox_c2_o1, bbox_c2_o2, ...],  <- Detecciones Cam 2
#   ... ]

### --- Pipeline de Fusión y Validación --- ###

start = time()

# Probar todas las combinaciones de detecciones (1 por cámara)
# (N_det_c1 * N_det_c2 * N_det_c3) combinaciones
try:
    posibles_combinaciones = list(itertools.product(*all_detections_by_cam))
except MemoryError:
    print("Error: Demasiadas detecciones, el producto cartesiano es muy grande.")
    exit()

print(f"\n--- Procesando {len(posibles_combinaciones)} Combinaciones de Rayos ---")

conos_validados_final = []
conos_descartados_altura = 0
conos_descartados_rayos = 0
conos_descartados_bbox = 0

for combo in posibles_combinaciones:
    # combo = (bbox_cam1, bbox_cam2, bbox_cam3)

    # Extraer centros (u,v) para la triangulación
    points_2d = [(bbox[0], bbox[1]) for bbox in combo]

    # 1. Estimar punto 3D
    X_est = triangulate_from_rays(Ks, Rs, ts, points_2d)

    # --- FILTRO 1: Coincidencia de Rayos (Calidad de Triangulación) ---
    matches = 0
    X_vec = X_est.ravel()
    for i in range(NUM_CAMERAS):
        K_cam, R_cam, t_cam = Ks[i], Rs[i], ts[i]
        u, v, _, _ = combo[i]

        C = -R_cam.T @ t_cam
        pixel_h = np.array([[u, v, 1.0]]).T
        d = R_cam.T @ np.linalg.inv(K_cam) @ pixel_h
        d = d.ravel() / np.linalg.norm(d)

        v_to_X = X_vec - C.ravel()
        dist = np.linalg.norm(v_to_X - np.dot(v_to_X, d) * d)

        if dist <= MAX_RAY_DISTANCE_M:
            matches += 1

    if matches < MIN_RAY_MATCH:
        conos_descartados_rayos += 1
        continue

    # --- FILTRO 2: Confirmación por Altura ---
    y_estimado = X_est[1]  # Extraemos la coordenada de altura (Y)

    if not (Y_MIN_VALID <= y_estimado <= Y_MAX_VALID):
        conos_descartados_altura += 1
        continue

    # --- FILTRO 3: Validación de Tamaño de BBox ---
    bbox_matches = 0
    for i in range(NUM_CAMERAS):
        K_cam, R_cam, t_cam = Ks[i], Rs[i], ts[i]
        detected_bbox = combo[i]

        if is_bbox_size_valid(K_cam, R_cam, t_cam, X_est, detected_bbox,
                              (CONE_REAL_WIDTH, CONE_REAL_HEIGHT),
                              BBOX_SIZE_TOLERANCE):
            bbox_matches += 1

    if bbox_matches < MIN_BBOX_MATCHES:
        conos_descartados_bbox += 1
        continue

    # Si pasa todos los filtros, es un punto validado
    conos_validados_final.append(X_est)

### --- Resultados y Métricas --- ###

print("\n--- ✅ Resultados de la Validación ---")
print(f"Total de Combinaciones Estimadas: {len(posibles_combinaciones)}")
print(f"  Descartados (Coincidencia Rayos): {conos_descartados_rayos}")
print(f"  Descartados (Altura):           {conos_descartados_altura}")
print(f"  Descartados (Tamaño BBox):      {conos_descartados_bbox}")
print(f"Puntos 3D Validados (Pre-Clúster): {len(conos_validados_final)}")

# Agrupar puntos validados para obtener predicciones finales
predicciones_finales = cluster_points(conos_validados_final, CLUSTER_DISTANCE_M)
print(f"Predicciones de Conos (Post-Clúster): {len(predicciones_finales)}")

for i, cono in enumerate(predicciones_finales):
    print(f"    Pred {i + 1}: {np.round(cono, 2)}")

# Calcular métricas
precision, recall, f1, TP, FP, FN = calculate_metrics(
    predicciones_finales,
    [gt.ravel() for gt in ground_truth_cones],
    MATCH_DISTANCE_M
)

print("\n--- Métricas de Rendimiento ---")
print(f"Ground Truth (GT): {N_REAL_CONES}")
print(f"Predicciones (P):  {len(predicciones_finales)}")
print("---------------------------------")
print(f"True Positives (TP):  {TP}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print("---------------------------------")
print(f"Precisión: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-Score:  {f1:.2%}")

end = time()
print(f"\nTiempo de Ejecución: {end - start:.2f} segundos")