import numpy as np
import itertools
import random
from time import time


### --- 1. Definiciones y Funciones Base --- ###

# Funciones del código de referencia
def project_point(K, R, t, Xw):
    """Proyecta un punto 3D (Xw) en la imagen de una cámara con K, R, t."""
    if Xw.shape != (3, 1):
        Xw = Xw.reshape(3, 1)

    # Transformar de Coordenadas de Mundo (W) a Coordenadas de Cámara (C)
    # t_cam = -R @ C_world. En nuestro caso, R/t son de W->C
    # Xc = R @ Xw + t

    # El código de referencia asume que t es t_W_C (World-to-Cam)
    # Pero la definición estándar de R,t es [R|t] que transforma Xw -> Xc
    # C = -R.T @ t (Si t es t_W_C)

    # Asumamos que R,t son los parámetros extrínsecos que llevan Xw a Xc
    # R_cam, t_cam (posición de la cámara en el mundo)
    # Xc = R_cam.T @ (Xw - t_cam)

    # El código de triangulación usa C = -R.T @ t, lo que implica
    # que R,t son [R|t] para Xc = R @ Xw + t
    # Esta es la convención de OpenCV (extrinsics = rvec, tvec)

    Xc = R @ Xw + t

    if Xc[2] <= 0:  # Punto detrás de la cámara
        return np.array([np.nan, np.nan])

    x_projected = K @ (Xc / Xc[2])
    return x_projected[:2].ravel()  # (u, v)


def triangulate_from_rays(Ks, Rs, ts, points_2d):
    """Reconstruye punto 3D minimizando distancia a los rayos (para 2 o más rayos)."""
    I = np.eye(3)
    A = np.zeros((3, 3))
    b = np.zeros((3, 1))

    if len(points_2d) < 2:
        raise ValueError("Se necesitan al menos 2 rayos para triangular")

    for K, R, t, (u, v) in zip(Ks, Rs, ts, points_2d):
        # Centro de cámara en mundo (C = -R.T @ t)
        C = -R.T @ t

        # Dirección del rayo en mundo (d = R.T @ inv(K) @ [u, v, 1])
        pixel_h = np.array([[u, v, 1.0]]).T
        d = R.T @ np.linalg.inv(K) @ pixel_h
        d = d / np.linalg.norm(d)

        # Matriz de proyección perpendicular (I - d*d.T)
        P = I - d @ d.T
        A += P
        b += P @ C

    try:
        X = np.linalg.inv(A) @ b
    except np.linalg.LinAlgError:
        X = np.linalg.pinv(A) @ b

    return X.ravel()


def rot_y(theta_deg):
    t = np.radians(theta_deg)
    # Rotación en Y (yaw)
    return np.array([
        [np.cos(t), 0, np.sin(t)],
        [0, 1, 0],
        [-np.sin(t), 0, np.cos(t)]
    ])


### --- 2. Nuevas Funciones para Simulación y Validación --- ###

def get_expected_bbox(K, R, t, X_3d, real_w, real_h):
    """
    Calcula el bounding box 2D (u,v,w,h) esperado para un objeto 3D.
    Sistema: Y es ARRIBA.
    """
    X_center = X_3d.ravel()

    # Puntos 3D del Bounding Box del cono
    X_top = X_center + np.array([0, real_h / 2, 0])
    X_bottom = X_center - np.array([0, real_h / 2, 0])
    X_left = X_center - np.array([real_w / 2, 0, 0])
    X_right = X_center + np.array([real_w / 2, 0, 0])

    # Proyectar todos los puntos
    uv_c = project_point(K, R, t, X_center)
    uv_t = project_point(K, R, t, X_top)
    uv_b = project_point(K, R, t, X_bottom)
    uv_l = project_point(K, R, t, X_left)
    uv_r = project_point(K, R, t, X_right)

    if np.isnan(uv_c).any() or np.isnan(uv_t).any() or np.isnan(uv_b).any() or \
            np.isnan(uv_l).any() or np.isnan(uv_r).any():
        return None  # Objeto no visible o detrás de la cámara

    w_expected = abs(uv_r[0] - uv_l[0])
    h_expected = abs(uv_b[1] - uv_t[1])  # v_b (abajo) > v_t (arriba) en píxeles

    return (uv_c[0], uv_c[1], w_expected, h_expected)


def is_bbox_size_valid(K, R, t, X_est_3d, detected_bbox, real_dims, tolerance):
    """
    Comprueba si un BBox detectado coincide con el tamaño esperado en X_est_3d.
    """
    _, _, w_detected, h_detected = detected_bbox
    real_w, real_h = real_dims

    # Obtener el bbox esperado si el objeto estuviera en X_est_3d
    expected_bbox = get_expected_bbox(K, R, t, X_est_3d, real_w, real_h)

    if expected_bbox is None:
        return False  # Punto estimado proyecta fuera de la imagen

    _, _, w_expected, h_expected = expected_bbox

    if w_expected < 1 or h_expected < 1:  # Evitar división por cero
        return False

    # Calcular error relativo
    w_error = abs(w_detected - w_expected) / w_expected
    h_error = abs(h_detected - h_expected) / h_expected

    return (w_error < tolerance) and (h_error < tolerance)


def generate_scene(num_real, num_false, y_cone, y_range_false, x_range, z_range):
    """Genera una escena aleatoria con conos reales y objetos falsos."""
    ground_truth_cones = []
    false_cones = []
    all_objects = []

    # Generar conos reales
    for _ in range(num_real):
        x = random.uniform(*x_range)
        z = random.uniform(*z_range)
        cone = np.array([[x, y_cone, z]]).T
        ground_truth_cones.append(cone)
        all_objects.append(cone)

    # Generar objetos falsos
    for _ in range(num_false):
        x = random.uniform(*x_range)
        z = random.uniform(*z_range)
        # Altura aleatoria garantizada fuera del rango de validación del cono
        y = random.uniform(*y_range_false)
        false_obj = np.array([[x, y, z]]).T
        false_cones.append(false_obj)
        all_objects.append(false_obj)

    return ground_truth_cones, false_cones


def cluster_points(points, cluster_dist):
    """Agrupa puntos 3D cercanos en clústeres y devuelve sus centroides."""
    if not points:
        return []

    points_arr = np.array(points)
    clusters = []
    remaining_indices = list(range(len(points_arr)))

    while remaining_indices:
        current_idx = remaining_indices.pop(0)
        current_point = points_arr[current_idx]

        # Encontrar todos los puntos cercanos (incluyéndose a sí mismo)
        distances = np.linalg.norm(points_arr[remaining_indices] - current_point, axis=1)
        nearby_indices_rel = np.where(distances < cluster_dist)[0]

        # Convertir a índices absolutos y añadirlos al clúster
        cluster_indices = [current_idx] + [remaining_indices[i] for i in nearby_indices_rel]

        # Calcular centroide del clúster
        cluster_points = points_arr[cluster_indices]
        centroid = np.mean(cluster_points, axis=0)
        clusters.append(centroid)

        # Eliminar puntos agrupados de la lista de restantes
        remaining_indices = [idx for idx in remaining_indices if idx not in cluster_indices]

    return clusters


def calculate_metrics(predictions, ground_truth, match_dist):
    """Calcula Precisión, Recall y F1-Score."""
    if not ground_truth:
        print("Advertencia: No hay conos reales (Ground Truth) para calcular métricas.")
        return 0, 0, 0

    if not predictions:
        print("No se encontraron predicciones. Recall = 0.")
        return 0, 0, 0  # Precisión es indefinida (0/0), Recall es 0, F1 es 0

    TP = 0
    FP = 0

    gt_arr = np.array(ground_truth).reshape(-1, 3)
    pred_arr = np.array(predictions).reshape(-1, 3)

    gt_matched = [False] * len(gt_arr)

    # Por cada predicción, ver si coincide con un GT
    for pred in pred_arr:
        distances = np.linalg.norm(gt_arr - pred, axis=1)
        best_match_idx = np.argmin(distances)

        if distances[best_match_idx] < match_dist:
            # Es un True Positive si el GT no ha sido 'gastado'
            if not gt_matched[best_match_idx]:
                TP += 1
                gt_matched[best_match_idx] = True
            else:
                # La predicción coincide con un GT ya 'gastado' por otra predicción
                # Esto es un FP (detección duplicada)
                FP += 1
        else:
            # La predicción no coincide con ningún GT
            FP += 1

    # Los GT no 'gastados' son False Negatives
    FN = gt_matched.count(False)

    # Calcular métricas
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, TP, FP, FN


### --- 3. Configuración de la Simulación --- ###

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

### --- 4. Generación de Escena y Detecciones --- ###

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

### --- 5. Pipeline de Fusión y Validación --- ###

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

### --- 6. Resultados y Métricas --- ###

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