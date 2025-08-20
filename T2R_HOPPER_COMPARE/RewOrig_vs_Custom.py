import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC, PPO
import math

SEED = 42

# ---------------------------
# 1) Cargar el agente
# ---------------------------
# Usa cadena raw para evitar escapes de '\H' etc.
#p = r"T2R_HOPPER_COMPARE\GTP_SAC\Hopper-v4\checkpoints\ppo_terminate_on_4000000_steps.zip"
#p = r"T2R_HOPPER_COMPARE\GTP_PPO\Hopper-v4\checkpoints\ppo_terminate_on_4000000_steps.zip"
#p = r"T2R_HOPPER_COMPARE\DEEPSEEK\Hopper-v4\checkpoints\ppo_terminate_on_4000000_steps.zip"
#p = r"T2R_HOPPER_COMPARE\DEEPSEEK_SAC\Hopper-v4\checkpoints\ppo_terminate_on_4000000_steps.zip"

for p in [
    r"T2R_HOPPER_COMPARE\GTP_SAC\Hopper-v4\checkpoints\ppo_terminate_on_4000000_steps.zip",
    r"T2R_HOPPER_COMPARE\GTP_PPO\Hopper-v4\checkpoints\ppo_terminate_on_4000000_steps.zip",
    r"T2R_HOPPER_COMPARE\DEEPSEEK_PPO\Hopper-v4\checkpoints\ppo_terminate_on_4000000_steps.zip",
    r"T2R_HOPPER_COMPARE\DEEPSEEK_SAC\Hopper-v4\checkpoints\ppo_terminate_on_4000000_steps.zip"
]:
    print(f"Loading model from: {p}")
    if "SAC" in p:
        learner = SAC.load(p)
    else:
        learner = PPO.load(p)

    # ---------------------------
    # 2) Entorno
    # ---------------------------
    # env = gym.make("Hopper-v4")  # entorno "original"
    env = gym.make("Hopper-v4", terminate_when_unhealthy=False)

    # Semillas
    try:
        env.reset(seed=SEED)
        env.action_space.seed(SEED)
        np.random.seed(SEED)
    except TypeError:
        pass

    # ---------------------------
    # 3) Rewards densos
    # ---------------------------
    def compute_dense_reward_gpt(obs: np.ndarray, action: np.ndarray) -> float:
        (_, top_ang, thigh_ang, leg_ang, foot_ang,
        vel_x, _, ang_vel_top, ang_vel_thigh,
        ang_vel_leg, ang_vel_foot) = obs
        reward_forward = 1.0 * vel_x
        reward_alive = 1.0
        ctrl_cost = 0.001 * (action[0]**2 + action[1]**2 + action[2]**2)
        angle_cost = 0.1 * (top_ang**2)
        reward = reward_forward + reward_alive - ctrl_cost - angle_cost
        return float(reward)

    def compute_dense_reward_deepseek(obs: np.ndarray, action: np.ndarray) -> float:
        (_, top_ang, thigh_ang, leg_ang, foot_ang,
        vel_x, _, ang_vel_top, ang_vel_thigh,
        ang_vel_leg, ang_vel_foot) = obs
        upright_reward = math.cos(top_ang)
        current_height = obs[0]
        height_penalty = max(0.0, 1.25 - current_height) ** 2
        torque_penalty = 0.001 * (action[0]**2 + action[1]**2 + action[2]**2)
        reward = (1.0 * vel_x +
                0.5 * upright_reward -
                0.1 * height_penalty -
                torque_penalty)
        return float(reward)

    # ---------------------------
    # 4) Rollout y comparación
    # ---------------------------
    def rollout_and_compare(model, env, n_episodes=50, deterministic=True, render=False):
        """
        Devuelve:
        - summaries: lista de dicts con retornos por episodio (env, gpt, deepseek)
        - trajectories: lista de listas con dicts por paso con los tres rewards
        """
        summaries, trajectories = [], []

        for ep in range(n_episodes):
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                obs, info = reset_out
            else:
                obs, info = reset_out, {}

            ret_env = 0.0
            ret_gpt = 0.0
            ret_deepseek = 0.0
            steps = 0
            ep_traj = []

            while True:
                if render:
                    env.render()

                action, _ = model.predict(obs, deterministic=deterministic)
                step_out = env.step(action)

                # Gymnasium API
                if len(step_out) == 5:
                    next_obs, rew_env, terminated, truncated, info = step_out
                    done = terminated or truncated
                else:
                    next_obs, rew_env, done, info = step_out

                # Recompensas densas sobre la MISMA transición (obs, action)
                a = np.asarray(action, dtype=np.float32)
                o = np.asarray(obs, dtype=np.float32)
                rew_gpt = compute_dense_reward_gpt(o, a)
                rew_deep = compute_dense_reward_deepseek(o, a)

                # Acumular
                ret_env += float(rew_env)
                ret_gpt += float(rew_gpt)
                ret_deepseek += float(rew_deep)
                steps += 1

                ep_traj.append({
                    "obs": obs,
                    "act": action,
                    "rew_env": float(rew_env),
                    "rew_gpt": float(rew_gpt),
                    "rew_deepseek": float(rew_deep),
                    "done": bool(done),
                    "info": info,
                })

                obs = next_obs
                if done:
                    break

            summaries.append({
                "episode": ep,
                "steps": steps,
                "return_env": ret_env,
                "return_gpt": ret_gpt,
                "return_deepseek": ret_deepseek,
            })
            trajectories.append(ep_traj)

        return summaries, trajectories

    # Ejecutar
    summaries, trajectories = rollout_and_compare(
        learner, env, n_episodes=100, deterministic=True, render=False
    )

    # ---------------------------
    # 5) Reporte
    # ---------------------------
    #print("Resumen por episodio:")
    #for s in summaries:
    #    print(f"Ep {s['episode']:02d} | steps={s['steps']:4d} | "
    #        f"Return(env)={s['return_env']:.2f} | "
    #        f"Return(gpt)={s['return_gpt']:.2f} | "
    #        f"Return(deepseek)={s['return_deepseek']:.2f}")

    # Agregados
    env_returns = np.array([s["return_env"] for s in summaries], dtype=np.float64)
    gpt_returns = np.array([s["return_gpt"] for s in summaries], dtype=np.float64)
    deep_returns = np.array([s["return_deepseek"] for s in summaries], dtype=np.float64)

    def safe_std(x): 
        return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    print("\nAggregate:")
    print(f"Env return:       mean={env_returns.mean():.2f}, std={safe_std(env_returns):.2f}")
    print(f"GPT dense return: mean={gpt_returns.mean():.2f}, std={safe_std(gpt_returns):.2f}")
    print(f"Deep dense return:mean={deep_returns.mean():.2f}, std={safe_std(deep_returns):.2f}")




    # ---------------------------
    # 6) Métrica de similitud entre rewards
    # ---------------------------
    def _rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
        """
        Asigna rangos (1..n) con promedio en empates, similar a scipy.stats.rankdata(method='average').
        """
        x = np.asarray(x)
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1, dtype=float)

        # Promediar empates
        # Recorremos bloques de valores iguales en el ordenado estable (mergesort)
        sorted_x = x[order]
        i = 0
        while i < len(x):
            j = i + 1
            while j < len(x) and sorted_x[j] == sorted_x[i]:
                j += 1
            if j - i > 1:
                avg = (ranks[order[i]] + ranks[order[j-1]]) / 2.0
                ranks[order[i:j]] = avg
            i = j
        return ranks

    def compare_rewards(
        a: np.ndarray,
        b: np.ndarray,
        name_a: str = "A",
        name_b: str = "B",
        verbose: bool = True,
    ) -> dict:
        """
        Compara cuán parecidas son dos secuencias (p.ej., retornos por episodio).
        Devuelve y/o imprime: Pearson, Spearman, Cosine y NRMSE.

        a, b: arrays 1D de la MISMA longitud.
        """
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        assert a.shape == b.shape and a.ndim == 1, "a y b deben ser vectores 1D de misma longitud"

        # Pearson
        std_a = np.std(a)
        std_b = np.std(b)
        pearson = float(np.corrcoef(a, b)[0, 1]) if std_a > 0 and std_b > 0 else float("nan")

        # Spearman (rangos con manejo de empates)
        ra = _rankdata_avg_ties(a)
        rb = _rankdata_avg_ties(b)
        std_ra = np.std(ra)
        std_rb = np.std(rb)
        spearman = float(np.corrcoef(ra, rb)[0, 1]) if std_ra > 0 and std_rb > 0 else float("nan")

        # Cosine similarity
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        #cosine = float(np.dot(a, b) / denom) if denom > 0 else float("nan")

        # NRMSE (normalizado por rango de b; cambia a 'a' si prefieres)
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        range_b = float(np.max(b) - np.min(b)) if len(b) > 0 else 0.0
        nrmse = float(rmse / range_b) if range_b > 0 else float("nan")

        out = {
            "pearson": pearson,
            "spearman": spearman,
            #"cosine": cosine,
            #"rmse": rmse,
            "nrmse": nrmse,
        }

        if verbose:
            print(f"\nSimilaridad {name_a} vs {name_b}:")
            print(f"  Pearson : {pearson:.3f}" if not np.isnan(pearson) else "  Pearson : nan")
            print(f"  Spearman: {spearman:.3f}" if not np.isnan(spearman) else "  Spearman: nan")
            #print(f"  Cosine  : {cosine:.3f}" if not np.isnan(cosine) else "  Cosine  : nan")
            #print(f"  RMSE    : {rmse:.3f}")
            print(f"  NRMSE   : {nrmse:.3f}" if not np.isnan(nrmse) else "  NRMSE   : nan")

        return out

    # --- Ejemplos de uso (puedes dejarlos o quitarlos) ---
    #compare_rewards(gpt_returns, deep_returns, "GPT", "Deepseek")
    compare_rewards(gpt_returns, env_returns, "GPT", "Env")
    compare_rewards(deep_returns, env_returns, "Deepseek", "Env")
