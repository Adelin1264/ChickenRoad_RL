import pygame
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

# --- Configurații ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
GRID_SIZE = 12       
BLOCK_SIZE = SCREEN_HEIGHT // GRID_SIZE
FPS = 30

# Culori
COLORS = {
    'grass': (118, 203, 70),  
    'road': (80, 80, 80),
    'chicken': (255, 215, 0),
    'beak': (255, 100, 0),
    'car': (230, 50, 50),
    'car_window': (150, 200, 255),
    'tree': (34, 100, 34)     
}

class Lane:
    def __init__(self, row_index, type_):
        self.row_index = row_index 
        self.type = type_ 
        self.obstacles = [] 
        self.speed = 0 
        
        if self.type == 'road':
            # Viteză: 6-9 pixeli per frame
            base_speed = random.randint(6, 9)
            direction = random.choice([-1, 1])
            self.speed = base_speed * direction
            
            num_cars = random.randint(1, 2)
            positions = []
            for _ in range(num_cars):
                pos = random.randint(0, SCREEN_WIDTH)
                valid = True
                for p in positions:
                    if abs(p - pos) < BLOCK_SIZE * 2.5: valid = False
                if valid:
                    self.obstacles.append(pos)
                    positions.append(pos)
        
        elif self.type == 'grass':
            num_trees = random.randint(0, 3)
            for _ in range(num_trees):
                col = random.randint(0, (SCREEN_WIDTH // BLOCK_SIZE) - 1)
                self.obstacles.append(col * BLOCK_SIZE)

    def update(self):
        if self.type == 'road':
            new_obstacles = []
            for x in self.obstacles:
                x += self.speed
                # Resetare infinită trafic
                if x > SCREEN_WIDTH + BLOCK_SIZE: x = -BLOCK_SIZE * 2
                if x < -BLOCK_SIZE * 2: x = SCREEN_WIDTH + BLOCK_SIZE
                new_obstacles.append(x)
            self.obstacles = new_obstacles

class CrossyRoadEnv(gym.Env):
    def __init__(self):
        super(CrossyRoadEnv, self).__init__()
        # MODIFICARE: 4 acțiuni acum. 3 = WAIT (Stai pe loc)
        self.action_space = spaces.Discrete(4) 
        self.observation_space = spaces.Box(low=0, high=1, shape=(49,), dtype=np.float32)

        pygame.init()
        self.screen = None
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24, bold=True)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.score = 0
        self.chicken_x = 5 
        self.player_visual_row = 8 
        
        # --- MODIFICARE: Contor pentru "stat degeaba" ---
        self.steps_since_last_advance = 0
        
        self.lanes = []
        for i in range(GRID_SIZE):
            if i > 5: 
                self.lanes.append(Lane(i, 'grass'))
                self.lanes[-1].obstacles = [] 
            else:
                self.lanes.append(self._generate_random_lane(i))
        
        return self._get_observation(), {}

    def _generate_random_lane(self, row_index):
        if random.random() < 0.65:
            return Lane(row_index, 'road')
        else:
            return Lane(row_index, 'grass')

    def step(self, action):
        reward = 0
        terminated = False
        
        # --- MODIFICARE: Creștem contorul de plictiseală ---
        self.steps_since_last_advance += 1
        
        current_lane = self.lanes[self.player_visual_row]
        
        # --- LOGICA DE MIȘCARE ---
        if action == 0: # FAȚĂ
            next_lane = self.lanes[self.player_visual_row - 1]
            blocked = False
            if next_lane.type == 'grass':
                for tree_x in next_lane.obstacles:
                    if abs(tree_x - self.chicken_x * BLOCK_SIZE) < 20: blocked = True
            
            if not blocked:
                self.score += 1
                reward = 1.0
                
                # --- MODIFICARE: Resetăm contorul dacă am avansat! ---
                self.steps_since_last_advance = 0
                
                self.lanes.pop() 
                for lane in self.lanes: lane.row_index += 1
                new_lane = self._generate_random_lane(0)
                self.lanes.insert(0, new_lane)
            else:
                reward = -0.1

        elif action == 1: # STÂNGA
            if self.chicken_x > 0:
                blocked = False
                if current_lane.type == 'grass':
                     for tree_x in current_lane.obstacles:
                        if abs(tree_x - (self.chicken_x - 1) * BLOCK_SIZE) < 20: blocked = True
                if not blocked:
                    self.chicken_x -= 1
            
        elif action == 2: # DREAPTA
            if self.chicken_x < (SCREEN_WIDTH // BLOCK_SIZE) - 1:
                blocked = False
                if current_lane.type == 'grass':
                     for tree_x in current_lane.obstacles:
                        if abs(tree_x - (self.chicken_x + 1) * BLOCK_SIZE) < 20: blocked = True
                if not blocked:
                    self.chicken_x += 1
        
        elif action == 3: # WAIT
            reward = -0.01

        # --- UPDATE MEDIU ---
        for lane in self.lanes:
            lane.update()

        # --- COLIZIUNI CU MAȘINI ---
        offset = 12 
        player_rect = pygame.Rect(
            self.chicken_x * BLOCK_SIZE + offset, 
            self.player_visual_row * BLOCK_SIZE + offset, 
            BLOCK_SIZE - (offset * 2), 
            BLOCK_SIZE - (offset * 2)
        )
        
        lane_of_death = self.lanes[self.player_visual_row]
        if lane_of_death.type == 'road':
            for car_x in lane_of_death.obstacles:
                car_rect = pygame.Rect(car_x + 2, self.player_visual_row * BLOCK_SIZE + 5, BLOCK_SIZE - 4, BLOCK_SIZE - 10)
                if player_rect.colliderect(car_rect):
                    terminated = True
                    reward = -10
                    print(f"SPLAT! Lovit în timp ce stăteai! Score: {self.score}")

        # --- MODIFICARE: Verificăm dacă a "murit de foame" (TIMEOUT) ---
        # Dacă au trecut 150 de frame-uri (5 secunde) fără să avanseze
        if self.steps_since_last_advance > 150:
            terminated = True
            reward = -5 # Penalizare că a fost prea lent sau s-a blocat
            print("TIMEOUT! Prea lent sau blocat între copaci.")

        return self._get_observation(), reward, terminated, False, {}

    def _get_observation(self):
        # Vedem o zonă de 7x7 în jurul puiului
        # Puiul este mereu în centrul acestei matrici (la indexul 3,3)
        view_radius = 3
        grid_size = (view_radius * 2) + 1 # 7
        observation = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Iterăm prin rândurile din jurul puiului
        for i in range(grid_size):
            # Calculăm indexul real al benzii (relativ la poziția vizuală a puiului)
            # player_visual_row este 8. Deci ne uităm de la rândul 5 la 11.
            lane_idx_relative = self.player_visual_row - view_radius + i
            
            # Verificăm dacă banda există (nu suntem în afara hărții)
            if 0 <= lane_idx_relative < GRID_SIZE:
                lane = self.lanes[lane_idx_relative]
                
                # Iterăm prin coloanele din jurul puiului
                for j in range(grid_size):
                    col_idx = self.chicken_x - view_radius + j
                    
                    # Verificăm limitele ecranului stânga-dreapta
                    if 0 <= col_idx < (SCREEN_WIDTH // BLOCK_SIZE):
                        # Verificăm obstacolele de pe această bandă
                        is_obstacle = False
                        
                        # Verificăm mașinile sau copacii
                        for obs_x in lane.obstacles:
                            # Convertim pixelii obstacolului în coordonate de grilă
                            obs_col = int(obs_x // BLOCK_SIZE)
                            if obs_col == col_idx:
                                is_obstacle = True
                                break
                        
                        if is_obstacle:
                            if lane.type == 'road':
                                observation[i][j] = 1.0 # MAȘINĂ
                            else:
                                observation[i][j] = 0.5 # COPAC
        
        # Flatten: transformăm matricea 7x7 într-un vector lung de 49 de numere
        # Rețelele neuronale simple preferă vectori.
        return observation.flatten()

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Chicken Road AI")

        self.screen.fill(COLORS['grass'])

        for lane in self.lanes:
            y = lane.row_index * BLOCK_SIZE
            if lane.type == 'road':
                pygame.draw.rect(self.screen, COLORS['road'], (0, y, SCREEN_WIDTH, BLOCK_SIZE))
                pygame.draw.line(self.screen, (255, 255, 255), (0, y + BLOCK_SIZE//2), (SCREEN_WIDTH, y+BLOCK_SIZE//2), 2)
                
                for car_x in lane.obstacles:
                     pygame.draw.rect(self.screen, COLORS['car'], (car_x, y + 5, BLOCK_SIZE, BLOCK_SIZE-10))
                     window_offset = 5 if lane.speed < 0 else BLOCK_SIZE - 15
                     pygame.draw.rect(self.screen, COLORS['car_window'], (car_x + window_offset, y + 8, 10, BLOCK_SIZE-16))
            
            elif lane.type == 'grass':
                pygame.draw.rect(self.screen, COLORS['grass'], (0, y, SCREEN_WIDTH, BLOCK_SIZE))
                for tree_x in lane.obstacles:
                    pygame.draw.circle(self.screen, (30, 80, 30), (int(tree_x + BLOCK_SIZE/2), int(y + BLOCK_SIZE/2)), BLOCK_SIZE//2.5)
                    pygame.draw.circle(self.screen, COLORS['tree'], (int(tree_x + BLOCK_SIZE/2), int(y + BLOCK_SIZE/2)), BLOCK_SIZE//3.5)

        px = self.chicken_x * BLOCK_SIZE
        py = self.player_visual_row * BLOCK_SIZE
        offset = 12
        size = BLOCK_SIZE - (offset * 2)
        
        pygame.draw.rect(self.screen, COLORS['chicken'], (px + offset, py + offset, size, size))
        pygame.draw.rect(self.screen, COLORS['beak'], (px + offset + size//2 - 2, py + offset - 4, 4, 4))
        pygame.draw.circle(self.screen, (0,0,0), (int(px + offset + size//3), int(py + offset + size//3)), 2)
        pygame.draw.circle(self.screen, (0,0,0), (int(px + offset + 2*size//3), int(py + offset + size//3)), 2)
        
        score_text = f"Score: {self.score}"
        text_shadow = self.font.render(score_text, True, (0, 0, 0))
        text_fg = self.font.render(score_text, True, (255, 255, 255))
        self.screen.blit(text_shadow, (12, 12))
        self.screen.blit(text_fg, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)
        
    def close(self):
        if self.screen:
            pygame.quit()

if __name__ == "__main__":
    env = CrossyRoadEnv()
    env.reset()
    running = True
    
    print("CONTROL: Săgeți pentru mișcare. Dacă nu apeși nimic, puiul AȘTEAPTĂ (dar mașinile vin!)")

    while running:
        # Default action este 3 (WAIT)
        action = 3 
        
        # Procesăm inputul uman
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action = 0
                if event.key == pygame.K_LEFT: action = 1
                if event.key == pygame.K_RIGHT: action = 2
        
        # Executăm pasul INDIFERENT dacă am apăsat ceva sau nu
        # Dacă nu am apăsat, action e 3, deci mașinile se actualizează și verifică coliziunea
        obs, reward, done, _, _ = env.step(action)
        
        env.render()
        
        if done:
            print("--- MORT! RESET ---")
            env.reset()