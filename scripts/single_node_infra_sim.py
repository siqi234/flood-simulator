import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FloodSimEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(1)  # meaningless action space for disaster simulation

        self.observation_space = spaces.Box(
            low = np.array([0, 0, 0, 0, 0], dtype=np.float32), #time, water level, infrastructure1, infrastructure2, infrastructure3
            high = np.array([np.inf, 100, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0
        self.water_level = 0.0
        self.infra_states = [1, 1, 1]  # all infrastructures are initially functional

        state = np.array([self.time, self.water_level] + self.infra_states, dtype=np.float32)

        return state, {}
    
    def update_disaster(self):
        prop = self.np_random.uniform(0, 1) # every second, water level increases by 0-1
        self.water_level += prop

    def simulate_infrastructure_failure(self):
        old_states = list(self.infra_states)

        for i in range(len(self.infra_states)):
            if self.infra_states[i] == 1:
                p_failure = 0.0
                if i == 0:
                    p_failure = self.water_level*0.02  # Infra 1 failure probability increases by 2% for every unit increase in water level
                else:
                    if old_states[i-1] == 0:
                        p_failure = 0.2  # if parent failed, 20% chance of failure

                if self.np_random.uniform(0, 1) < p_failure:
                    self.infra_states[i] = 0  # infrastructure fails

    def step(self, action):
        self.time += 1

        self.update_disaster()
        self.simulate_infrastructure_failure()

        terminated = bool(
            self.water_level >= 50 
            or sum(self.infra_states) == 0)
        
        state = np.array([self.time, self.water_level] + self.infra_states, dtype=np.float32)
        reward = 0

        return state, reward, terminated, False, {}
    
    def render(self):

        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 4))
            self.fig.canvas.manager.set_window_title('Flood Simulation')

        self.ax.clear()

        # Bar chart for water level
        self.ax.bar(['Flood Level'], [self.water_level], color='blue', width=0.3)
        self.ax.set_ylim(0, 50)

        node_x = [1, 2, 3]
        node_y = [25, 25, 25]
        node_names = ['Power Grid', 'Water Station', 'Hospital']

        colors = ['green' if state == 1 else 'red' for state in self.infra_states]

        self.ax.scatter(node_x, node_y, s=2000, c=colors, zorder=5)

        # Bayesian Network (I1 -> I2 -> I3)
        # 1: Power Grid -> Water Station
        arrow1_color = 'red' if self.infra_states[0] == 0 else 'black'
        self.ax.annotate('', xy=(1.8, 25), xytext=(1.2, 25), 
                         arrowprops=dict(arrowstyle="->", lw=4, color=arrow1_color))
        # 2: Water Station -> Hospital
        arrow2_color = 'red' if self.infra_states[1] == 0 else 'black'
        self.ax.annotate('', xy=(2.8, 25), xytext=(2.2, 25), 
                         arrowprops=dict(arrowstyle="->", lw=4, color=arrow2_color))

        for i, (x, y) in enumerate(zip(node_x, node_y)):
            self.ax.text(x, y + 10, node_names[i], ha='center', va='bottom', 
                         fontsize=12, fontweight='bold')
            
            self.ax.text(x, y, f'I_{i+1}', ha='center', va='center', 
                         color='white', fontweight='bold', fontsize=14)
            
            status_text = "OK" if self.infra_states[i] == 1 else "FAILED"
            status_color = "green" if self.infra_states[i] == 1 else "red"
            self.ax.text(x, y - 10, status_text, ha='center', va='top', 
                         fontsize=10, fontweight='bold', color=status_color)

        self.ax.set_title(f"Time Step: {self.time} | Flood Level: {self.water_level:.2f}")
        # 1. Hide the top, right, and bottom borders so the nodes still look clean
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        
        # 2. Hide the X-axis tick marks (since we don't need numbers under the nodes)
        self.ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        # 3. Add a bold, clear label to the Y-axis for the flood level
        self.ax.set_ylabel('Flood Level', fontweight='bold', fontsize=12)
        
        # 4. (Optional) Make sure the Y-axis has clear tick marks from 0 to 50
        self.ax.set_yticks([0, 10, 20, 30, 40, 50])

        plt.draw()
        plt.pause(0.5)

if __name__ == "__main__":
    env = FloodSimEnv()
    state, info = env.reset()

    env.render()

    terminated = False

    print ('Begin simulation...')

    while not terminated:
        action = 0

        next_state, reward, terminated, truncated, info = env.step(action)

        env.render()

        time_step = next_state[0]
        water_lvl = next_state[1]
        infra = next_state[2:]
        print(f"Time: {time_step:2.0f} | Flood: {water_lvl:5.2f} | Infra States: {infra}")

        print('Simulation terminated (water level >= 50 or all infrastructures failed)')
    
    plt.ioff()
    plt.show()