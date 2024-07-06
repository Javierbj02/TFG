import numpy as np

class filtroKalman:
    # En nuestro caso de estudio, el filtro de Kalman se encarga de estimar la posición del robot en el mundo.
    # El robot se mueve en un plano 2D, por lo que la posición se representa con dos coordenadas (x, y).
    # El modelo de movimiento del robot es un modelo de movimiento rectilíneo uniforme (MRU). Va a una velocidad constante en una dirección.
    # A lo largo de la ruta, el robot recibe mediciones de las señales inalámbricas de los puntos de acceso (APs) que están en el mundo.
    # Estas mediciones sirven de entrada a un modelo de red neuronal ya entrenado sobre un fingerprint, estimando así la posición que tiene el robot en ese momento en el mundo.
    # Por lo tanto, tenemos las lecturas de las mediciones de los APs y la estimación de la posición del robot en el mundo.
    # La idea es implementar un filtro de Kalman que ayude a mejorar la estimación de la posición del robot en el mundo.

    # Inicialización de las variables del filtro de Kalman
    def __init__(self, x0, P0, A, B, H, Q, R):
        self.x = x0 # Estado inicial del sistema. En nuestro caso, la posición del robot en el mundo
        self.P = P0 # Covarianza inicial del sistema. 
        self.A = A # Matriz de transición de estado. 
        self.B = B # Matriz de control. 
        self.H = H # Matriz de observación.
        self.Q = Q # Covarianza del ruido del proceso.
        self.R = R # Covarianza del ruido de la medición.

    # Predicción del estado del sistema
    def predict(self, u):
        # Predicción del estado del sistema. El estado es la posición del robot en el mundo.
        self.x = (self.A @ self.x) + (self.B @ u) # Explicación: x(k) = A*x(k-1) + B*u(k)
        
        # Predicción de la covarianza del sistema. La covarianza es la incertidumbre de la posición del robot en el mundo.
        self.P = ((self.A @ self.P) @ self.A.T) + self.Q # Explicación: P(k) = A*P(k-1)*A^T + Q

    # Actualización del estado del sistema
    def update(self, z):
        # Cálculo de la ganancia de Kalman
        K = (self.P @ self.H.T) @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        # Actualización del estado del sistema.
        self.x = self.x + K @ (z - self.H @ self.x)


        # Actualización de la covarianza del estado
        I = np.eye(self.A.shape[0]) # Matriz identidad
        self.P = (I - (K @ self.H)) @ self.P

        

