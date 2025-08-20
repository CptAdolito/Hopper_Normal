import math  # For cosine calculation

def compute_dense_reward(self, obs, action) -> float:
    # === Desempaquetar observación ===
    (_,                  # Ignorado (puede ser torso z-pos)
     top_ang,           # Ángulo del torso
     thigh_ang,         # Ángulo del muslo
     leg_ang,           # Ángulo de la pierna
     foot_ang,          # Ángulo del pie
     vel_x,             # Velocidad horizontal (forward)
     _,                 # Ignorado (posiblemente vel_y)
     ang_vel_top,       # Velocidad angular del torso
     ang_vel_thigh,
     ang_vel_leg,
     ang_vel_foot) = obs

    # === Recompensa por velocidad hacia adelante ===
    forward_vel = vel_x

    # === Recompensa por mantener el torso erguido ===
    upright_reward = math.cos(top_ang)  # +1 erguido, -1 invertido

    # === Penalización por altura si está debajo de 1.25m ===
    # Suponiendo que obs[0] es la altura del torso
    current_height = obs[0]
    height_penalty = max(0, 1.25 - current_height) ** 2

    # === Penalización por torque (esfuerzo de control) ===
    torque_penalty = 0.001 * (action[0]**2 + action[1]**2 + action[2]**2)

    # === Recompensa total ponderada ===
    reward = (
        1.0 * forward_vel +      # Prioridad alta al avance
        0.5 * upright_reward -   # Bonificación por estabilidad
        0.1 * height_penalty -   # Penalización si cae
        torque_penalty           # Penalización por esfuerzo
    )
    return reward
