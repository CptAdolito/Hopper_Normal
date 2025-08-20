from stable_baselines3 import PPO, PPO, TD3, A2C, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
import numpy as np
import gymnasium as gym
import seals


SEED = 42

#p = "T2R_HOPPER_COMPARE\GTP_SAC\Hopper-v4\checkpoints\ppo_terminate_on_4000000_steps.zip"
#p = "T2R_HOPPER_COMPARE\GTP_PPO\Hopper-v4\checkpoints\ppo_terminate_on_4000000_steps.zip"
#p = "T2R_HOPPER_COMPARE\DEEPSEEK\Hopper-v4\checkpoints\ppo_terminate_on_4000000_steps.zip"
p = "T2R_HOPPER_COMPARE\DEEPSEEK_SAC\Hopper-v4\checkpoints\ppo_terminate_on_4000000_steps.zip"
learner = SAC.load(p)

# Create environment with custom wrapper to remove unhealthy termination
def make_no_unhealthy_env():
    env = gym.make("Hopper-v4", terminate_when_unhealthy=False)
    return env

venv = make_no_unhealthy_env()
venv = RolloutInfoWrapper(venv)

mean_reward, std_reward = evaluate_policy(
    learner,
    venv,
    n_eval_episodes=50,
    deterministic=True,
)

print(f"Learner Evaluation: Mean Reward = {mean_reward}, Std Reward = {std_reward}")


# record_seals_PPO_video.py

import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

# --- Configuración ---
VIDEO_FOLDER = "T2R_HOPPER_COMPARE/Videos"
MODEL_PATH   = p
VIDEO_LENGTH = 1000
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# 1) Factory para crear el entorno sin terminación unhealthy
def make_seals_env_no_unhealthy():
    env = gym.make("Hopper-v4", render_mode="rgb_array", terminate_when_unhealthy=False)
    env = RolloutInfoWrapper(env)
    return env

# 2) Crea un DummyVecEnv de un solo entorno
vec_env = DummyVecEnv([make_seals_env_no_unhealthy])

# 3) Envuélvelo con VecVideoRecorder
vec_env = VecVideoRecorder(
    vec_env,
    video_folder=VIDEO_FOLDER,
    record_video_trigger=lambda step: step == 0,
    video_length=1000,
    name_prefix="PPO_seals_hopper_video",
)

# 4) Carga tu agente PPO
model = SAC.load(MODEL_PATH, env=vec_env)

# 5) Ejecuta y graba
for i in range(10):
    obs = vec_env.reset()
    for _ in range(VIDEO_LENGTH):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        if dones.any():
            obs = vec_env.reset()

# 6) Finaliza y guarda el vídeo
vec_env.close()
print(f"Vídeo grabado en: {VIDEO_FOLDER}/")