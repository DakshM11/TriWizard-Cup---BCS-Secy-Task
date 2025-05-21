import numpy as np
import pygame
import random
import matplotlib.pyplot as plt
import requests
import time
from collections import deque, defaultdict
from tqdm import trange


import time
import os, sys
file_id = "1gRZjq7IMywWxnB34DpygJg4f52oR69yK"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
response = requests.get(url)
data = response.text
maze = [[list(line) for line in data.strip().split('\n')]]

file_id = "1wuNf4Nce5URt9gMJThGd7qVM9X38xmgd"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
response = requests.get(url)
data = response.text
maze.append([list(line) for line in data.strip().split('\n')])

file_id = "1xs8ccjway-rCVNsg2NPEE_oje0RKGxVc"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
response = requests.get(url)
data = response.text
maze.append([list(line) for line in data.strip().split('\n')])


def get_neighbour(pos,grid): #returns a list of the neighbours of a given position
  direction=[(1,0),(0,1),(-1,0),(0,-1)]
  neighbours=[]
  x,y=pos
  for dx,dy in direction:
    nx,ny=x+dx,y+dy
    if 0<=nx<len(grid) and 0<=ny<len(grid[0]) and grid[nx][ny]!='X':
      neighbours.append((nx,ny))
  return neighbours

def check(start,goal,grid): #to check if its possible to reach the goal from our starting position(that is walls are not blocking it)
  x,y=start
  if(grid[x][y]=='X'):
    return False
  x,y=goal
  if(grid[x][y]=='X'):
    return False
  queue=deque([start])
  visited=set()
  visited.add(start)

  while queue:
    current=queue.popleft()
    if current==goal:
      return True
    neighbours=get_neighbour(current,grid)
    for neighbour in neighbours:
      if neighbour not in visited:
        queue.append(neighbour)
        visited.add(neighbour)
  return False
def dist(x,y): #to find euclidean distance
  a,b=x
  c,d=y
  return np.linalg.norm(np.array([a,b])-np.array([c,d]))
def sign(x): return 1 if x>0 else (-1 if x<0 else 0)




class GridWorld:
    def __init__(self, grid, gamma=0.95):
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
        self.n_states = self.rows * self.cols
        self.grid = grid
        self.prev_harry = None         # Harry’s position one step ago
        self.prev_prev_harry = None    # Harry’s position two steps ago
        self.prev_prev_prev_harry=None #Harry's position three steps ago
        self.gamma = gamma
        self.clock = None
        self.step_n=0
        self.window_size = 600
        self.screen = None
        self.reset()

    def reset(self):
        flag = False
        self.step_n=0
        while not flag:
            x_harry = np.random.randint(0, self.rows)
            y_harry = np.random.randint(0, self.cols)
            x_death = np.random.randint(0, self.rows)
            y_death = np.random.randint(0, self.cols)
            x_cup = np.random.randint(0, self.rows)
            y_cup = np.random.randint(0, self.cols)

            if (check((x_harry, y_harry), (x_cup, y_cup), self.grid) and
                check((x_death, y_death), (x_harry, y_harry), self.grid) and
                (x_harry,y_harry)!=(x_death,y_death) and (x_cup,y_cup)!=(x_harry,y_harry)):
                flag = True

        self.harry_pos = (x_harry, y_harry)
        self.death_pos = (x_death, y_death)
        self.cup_pos = (x_cup, y_cup)
        self.prev_prev_harry = self.harry_pos
        self.prev_harry      = self.harry_pos
        self.prev_prev_prev_harry=self.harry_pos
        return (*self.harry_pos, *self.death_pos, *self.cup_pos)

    def is_valid(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x][y] != 'X'

    def bfs(self, start, goal):
        queue = deque([(start, [])])
        visited = set([start])
        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal:
                return path[0] if path else start
            for dx, dy in self.actions:
                nx, ny = x + dx, y + dy
                if self.is_valid(nx, ny) and (nx, ny) not in visited:
                    queue.append(((nx, ny), path + [(nx, ny)]))
                    visited.add((nx, ny))

    def step(self, action):
        done = False
        win = False
        x, y = self.harry_pos
        dx, dy = self.actions[action]
        nx, ny = x + dx, y + dy
        if self.is_valid(nx, ny):
            self.harry_pos = (nx, ny)
        else:
            return self.encode_state(), 0, done, {"win": win}
        self.step_n+=1
        prev_harry = (x, y)
        prev_death = self.death_pos
        self.death_pos = self.bfs(self.death_pos, self.harry_pos)
        self.prev_prev_prev_harry=self.prev_prev_harry
        self.prev_prev_harry = self.prev_harry
        self.prev_harry      = prev_harry
        reward = self.reward_func(*prev_harry, *prev_death)

        if self.harry_pos == self.cup_pos:
            done = True
            win = True
        if self.harry_pos == self.death_pos:
            done = True
            win = False

        return self.encode_state(), reward, done, {"win": win}
    def encode_state(self):
        hx, hy = self.harry_pos
        cx, cy = self.cup_pos
        dx, dy = self.death_pos

        # 1) compute raw offsets
        dcx = cx - hx
        dcy = cy - hy
        ddx = dx - hx
        ddy = dy - hy

        # 2) turn each offset into a quadrant: (−1, 0, +1)
        dir_cx = sign(dcx)   
        dir_cy = sign(dcy)  
        dir_dx = sign(ddx)   
        dir_dy = sign(ddy)   

        # 3) bucket the magnitude into “close/medium/far”
        rng_c = min(abs(dcx), abs(dcy)) // 3
        rng_d = min(abs(ddx), abs(ddy)) // 3

        return (dir_cx, dir_cy, rng_c, dir_dx, dir_dy, rng_d)


    def reward_func(self, px, py, pa, pb):
        x, y = self.harry_pos
        backtrack_penalty=0
        if(self.prev_prev_harry==self.harry_pos):
          backtrack_penalty=-5
        if(self.prev_harry==self.prev_prev_prev_harry):
          backtrack_penalty+=-10
        a, b = self.death_pos
        c, d = self.cup_pos
        d1_HC = dist((px, py), (c, d))
        d2_HC = dist((x, y), (c, d))
        d1_DH = dist((pa, pb), (px, py))
        d2_DH = dist((x, y), (a, b))

        if d2_HC == 0:
            return 100
        if d2_DH == 0:
            return -150

        α, β, γ_wall = 1.3, 1.2, 0.75
        step_coeff = 0.02
        R_old = -α * d1_HC + β * d1_DH - γ_wall * self._wall_penalty((px, py))
        R_new = -α * d2_HC + β * d2_DH - γ_wall * self._wall_penalty(self.harry_pos)
        return -1 + self.gamma * R_new - R_old + backtrack_penalty +step_coeff*self.step_n

    def _wall_penalty(self, pos):
        free = len(get_neighbour(pos, self.grid))
        return (4 - free) / 4.0

    def _render_frame(self):
        pygame.init()
        pygame.display.init()

        if not hasattr(self, 'window') or self.window is None:
            self.window_size = 600  # Adjust as needed
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if not hasattr(self, 'clock') or self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        cell_width = self.window_size / self.cols
        cell_height = self.window_size / self.rows

        # Draw walls
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 'X':
                    pygame.draw.rect(canvas, (0, 0, 0),
                                    pygame.Rect(j * cell_width, i * cell_height, cell_width, cell_height))

        # Draw Cup
        cx, cy = self.cup_pos
        pygame.draw.rect(canvas, (0, 255, 0),
                        pygame.Rect(cy * cell_width, cx * cell_height, cell_width, cell_height))

        # Draw Harry
        hx, hy = self.harry_pos
        center_x = int((hy + 0.5) * cell_width)
        center_y = int((hx + 0.5) * cell_height)
        radius = int(min(cell_width, cell_height) / 2.5)
        pygame.draw.circle(canvas, (255, 255, 0), (center_x, center_y), radius)

        # Draw Death Eater
        dx, dy = self.death_pos
        center_x = int((dy + 0.5) * cell_width)
        center_y = int((dx + 0.5) * cell_height)
        pygame.draw.circle(canvas, (128, 0, 128), (center_x, center_y), radius)

        # Draw grid lines
        for x in range(self.rows + 1):
            pygame.draw.line(canvas, (200, 200, 200),
                            (0, cell_height * x), (self.window_size, cell_height * x), width=2)
        for y in range(self.cols + 1):
            pygame.draw.line(canvas, (200, 200, 200),
                            (cell_width * y, 0), (cell_width * y, self.window_size), width=2)

        self.window.blit(canvas, (0, 0))
        pygame.display.update()
        pygame.event.pump()
        time.sleep(0.2)

    def render(self):
        self._render_frame()
        pygame.display.update()
        pygame.event.pump()



import random
import numpy as np
from collections import defaultdict
from tqdm import trange 

def epsilon_greedy(Q, state, n_actions, ε):
    """Pick a random action w.p. ε, otherwise greedy."""
    if random.random() < ε:
        return random.randrange(n_actions)
    qs = Q[state]
    return int(np.argmax(qs))

def train(env,
          episodes=5000,
          α=0.175,
          #γ=0.95,
          γ=0.995,
          ε_start=1,
          ε_end=0.175,
          ε_decay=0.999):
    Q = defaultdict(lambda: np.zeros(len(env.actions)))
    ε = ε_start

    rewards_per_ep = []
    success_per_ep = []

    for ep in trange(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 1) choose action
            action = epsilon_greedy(Q, state, len(env.actions), ε)

            # 2) step
            next_state, reward, done, info = env.step(action)

            # 3) Q-update
            best_next = np.max(Q[next_state])
            td = reward + γ * best_next - Q[state][action]
            Q[state][action] += α * td

            state = next_state
            total_reward += reward

        # record
        rewards_per_ep.append(total_reward)
        success_per_ep.append(int(info.get('win', False)))

        # decay ε
        ε = max(ε_end, ε * ε_decay)

    return Q, rewards_per_ep, success_per_ep


env = GridWorld(maze[0])
Q, rewards, successes = train(env)

window = 100  # for rolling average

# Compute rolling average success rate
rolling_success = np.convolve(successes, np.ones(window)/window, mode='valid')

plt.figure(figsize=(15, 5))

# Reward plot
plt.subplot(1, 3, 1)
plt.plot(rewards)
plt.title("Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")

# Success rate
plt.subplot(1, 3, 2)
plt.plot(rolling_success)
plt.title(f"Success Rate (rolling {window})")
plt.xlabel("Episode")
plt.ylabel("Success Rate")

plt.tight_layout()
plt.show()


state = env.reset()
total_reward = 0
done = False
while not done:
    action = epsilon_greedy(Q, state, len(env.actions), 0.1)
    a=env.harry_pos
    new_state, reward, done, info = env.step(action)
    if env.harry_pos != a or done:
        total_reward += reward
        env.render() 
        state = new_state
print(total_reward)
