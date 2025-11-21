import numpy as np
import random

### --- Definiciones y Funciones Base --- ###

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


### --- Funciones para Simulación y Validación --- ###

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