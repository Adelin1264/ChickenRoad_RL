import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from chicken_env import CrossyRoadEnv # Importăm clasa ta
import os


# Creăm directoare pentru a salva modelele
models_dir = "models"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# 1. Inițializăm mediul
env = CrossyRoadEnv()

# Verificăm dacă mediul respectă standardele (foarte important pt debugging)
# Dacă ai erori aici, înseamnă că observation_space nu se potrivește cu ce returnează _get_observation
print("Verific mediul...")
check_env(env) 
print("Mediul este OK!")


TIMESTEPS = 1000000  # Câți pași să învețe fiecare

print("--- Începe antrenarea cu PPO ---")
model_ppo = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model_ppo.learn(total_timesteps=TIMESTEPS)
model_ppo.save(f"{models_dir}/ppo_chicken")
print("PPO Salvat!")