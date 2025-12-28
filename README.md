# Crossy Road Reinforcement Learning

Acest proiect implementează un mediu personalizat de tip "Crossy Road" folosind `pygame` și `gymnasium`, și antrenează agenți inteligenți să joace acest joc folosind algoritmi de Reinforcement Learning (RL) din biblioteca `stable-baselines3`.

## Algoritmi Utilizați

Proiectul explorează performanța a trei algoritmi populari de RL:
- **PPO** (Proximal Policy Optimization)
- **DQN** (Deep Q-Network)
- **A2C** (Advantage Actor Critic)

## Structura Proiectului

- `chicken_env.py`: Codul sursă al mediului de joc (Environment). Definește regulile jocului, stările, acțiunile și recompensele.
- `train_ppo.py`, `train_dqn.py`, `train_a2c.py`: Scripturi pentru antrenarea agenților.
- `test_ppo.py`, `test_dqn.py`, `test_a2c.py`: Scripturi pentru testarea și vizualizarea agenților antrenați.
- `models/`: Folder unde sunt salvate modelele antrenate.
- `logs/`: Folder pentru log-urile TensorBoard.
- `Grafice/`: Conține date și grafice despre performanța antrenamentului.

## Cerințe

Pentru a rula acest proiect, aveți nevoie de Python instalat și de următoarele biblioteci:

```bash
pip install gymnasium stable-baselines3 shimmy pygame tensorboard
```

*(Notă: `shimmy` poate fi necesar pentru compatibilitatea între anumite versiuni de gym/gymnasium).*

## Joc Manual

Puteți încerca jocul chiar dumneavoastră! Pentru a juca manual, rulați scriptul mediului:

```bash
python chicken_env.py
```

**Controale:** Folosiți săgețile de la tastatură pentru a vă deplasa.

## Cum se utilizează

### 1. Antrenarea unui agent

Pentru a antrena un agent, rulați unul dintre scripturile de antrenare. De exemplu, pentru PPO:

```bash
python train_ppo.py
```

Acesta va începe procesul de învățare și va salva modelul în folderul `models/` și log-urile în `logs/`.

### 2. Testarea unui agent

**Notă:** Proiectul include deja modele pre-antrenate pentru fiecare algoritm în folderul `models/`. Puteți testa direct performanța acestora fără a fi nevoie să rulați antrenamentul de la zero.

Pentru a vedea un agent în acțiune (de exemplu, PPO), rulați scriptul de test corespunzător:

```bash
python test_ppo.py
```

O fereastră Pygame se va deschide și veți vedea agentul jucând.

### 3. Monitorizarea antrenamentului

Puteți vizualiza graficele de performanță (recompense, lungimea episoadelor etc.) folosind TensorBoard:

```bash
tensorboard --logdir logs
```

Apoi deschideți browserul la adresa indicată (de obicei `http://localhost:6006`).

## Detalii despre Mediu (`chicken_env.py`)

- **Obiectiv**: Puiul trebuie să traverseze străzi cu mașini și benzi de iarbă pentru a obține un scor cât mai mare.
- **Acțiuni**:
  - `0`: Mergi înainte (Față)
  - `1`: Mergi stânga
  - `2`: Mergi dreapta
  - `3`: Stai pe loc (Wait)
- **Observații**: O matrice locală (grid) în jurul puiului, aplatizată, care îi spune ce se află în vecinătate (drum, iarbă, mașină etc.).
- **Recompense**:
  - Pozitivă (+1.0) pentru avansare (trecerea la o bandă nouă).
  - Negativă mică (-0.01) pentru așteptare (Wait).
  - Negativă (-0.1) pentru mișcări blocate (ex: lovirea unui copac).
  - Penalizare mare și terminarea episodului pentru coliziune cu o mașină sau dacă agentul nu avansează timp de 150 de frame-uri.
