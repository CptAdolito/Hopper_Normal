import numpy as np

def compute_dense_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
    """
    Compute dense reward from Gymnasium observation and action.

    Args:
      obs    (ndarray, shape=(11,)): [z, top_ang, thigh_ang, leg_ang, foot_ang,
                                      vel_x, vel_z, ang_vel_top, ang_vel_thigh,
                                      ang_vel_leg, ang_vel_foot]
      action (ndarray, shape=(3,)):  [torque_thigh, torque_leg, torque_foot]

    Returns:
      reward (float)
    """
    # Unpack observation
    (_, top_ang, thigh_ang, leg_ang, foot_ang,
     vel_x, _, ang_vel_top, ang_vel_thigh,
     ang_vel_leg, ang_vel_foot) = obs

    # 1. Forward velocity reward (primary goal)
    reward_forward = 1.0 * vel_x

    # 2. Alive bonus (small constant for not falling)
    reward_alive = 1.0

    # 3. Control cost (penalize large torques lightly)
    ctrl_cost = 0.001 * (action[0]**2 + action[1]**2 + action[2]**2)

    # 4. Torso uprightness cost (keep top hinge near 0)
    angle_cost = 0.1 * (top_ang**2)

    # Total reward
    reward = reward_forward + reward_alive - ctrl_cost - angle_cost
    return float(reward)
