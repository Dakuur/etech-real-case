import numpy as np

def project_point(K, R, t, Xw):
    """Proyecta un punto 3D (Xw) en la imagen de una cámara con K, R, t."""
    Xc = R @ Xw + t
    x = K @ (Xc / Xc[2])
    return x[:2].ravel()  # (u, v)

def triangulate_from_rays(Ks, Rs, ts, points_2d):
    """Reconstruye punto 3D minimizando distancia a los rayos."""
    I = np.eye(3)
    A = np.zeros((3, 3))
    b = np.zeros((3, 1))

    for K, R, t, (u, v) in zip(Ks, Rs, ts, points_2d):
        # Centro de cámara en mundo
        C = -R.T @ t
        # Dirección del rayo en mundo
        pixel_h = np.array([u, v, 1.0])
        d = R.T @ np.linalg.inv(K) @ pixel_h
        d = d / np.linalg.norm(d)

        P = I - np.outer(d, d)
        A += P
        b += P @ C

    X = np.linalg.inv(A) @ b
    return X.ravel()

# --- Definición de cámaras ---

# Matriz intrínseca (todas iguales)
K = np.array([[800, 0, 640],
              [0, 800, 360],
              [0, 0, 1]])

# Rotaciones (cámaras mirando hacia delante con ligeras rotaciones Y)
def rot_y(theta_deg):
    t = np.radians(theta_deg)
    return np.array([
        [np.cos(t), 0, np.sin(t)],
        [0, 1, 0],
        [-np.sin(t), 0, np.cos(t)]
    ])

R1 = rot_y(0)
R2 = rot_y(30)
R3 = rot_y(-30)

# Traslaciones (separadas en eje X)
t1 = np.array([[0, 0, 0]]).T
t2 = np.array([[-0.5, 0, 0]]).T
t3 = np.array([[0.5, 0, 0]]).T

Ks = [K, K, K]
Rs = [R1, R2, R3]
ts = [t1, t2, t3]

# --- Punto real ---
X_true = np.array([[0.0, 0.0, 3.0]]).T

# --- Proyecciones simuladas ---
points_2d = [project_point(K, R, t, X_true) for K, R, t in zip(Ks, Rs, ts)]

# --- Reconstrucción 3D ---
X_est = triangulate_from_rays(Ks, Rs, ts, points_2d)

print("Proyecciones (u, v):")
for i, (u, v) in enumerate(points_2d, 1):
    print(f"  Cámara {i}: ({u:.2f}, {v:.2f})")

print("\nPunto real:", X_true.ravel())
print(f"Punto estimado:", X_est.ravel())
print("Error (m):", np.linalg.norm(X_true.ravel() - X_est.ravel()))
