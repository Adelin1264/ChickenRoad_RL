import sys
import time
import pygame
from stable_baselines3 import A2C
from chicken_env import CrossyRoadEnv
import gymnasium as gym
import os
# Inițializăm mediul
env = CrossyRoadEnv()

# Încărcăm modelul (asigură-te că fișierul există)
model = A2C.load("models/a2c_chicken", env=env)

obs, _ = env.reset()

print("Apasa X pe fereastra jocului pentru a iesi.")

while True:
    # --- PARTEA CARE PREVINE CRASH-UL ---
    # Trebuie să golim coada de evenimente la fiecare frame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit() # Oprește scriptul elegant
    # ------------------------------------

    # Întrebăm modelul ce acțiune să facă
    action, _ = model.predict(obs, deterministic=True)
    
    # Executăm pasul
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Desenăm
    env.render()
    
    # Încetinim puțin (altfel se mișcă prea repede pentru ochiul uman)
    time.sleep(0.05)
    
    # Resetăm dacă a murit
    if terminated:
        obs, _ = env.reset()