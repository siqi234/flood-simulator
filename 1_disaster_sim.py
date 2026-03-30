import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FloodSimEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(1)  # meaningless action space for disaster simulation

        # 21 variables: time + (flood level + 3 infra states) for each of the 5 nodes
        self.observation_space = spaces.Box(
            low=0.0, 
            high=100.0, 
            shape=(21,), 
            dtype=np.float32
        )


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0
        
        # Define the map
        # Node A gets hit at t=0. Neighbors B & C get hit at t=20. D gets hit at t=40.
        self.map_nodes = {
            'A': {'arrival': 0,  'water_level': 0.0, 'infra': [1, 1, 1]},
            'B': {'arrival': 20, 'water_level': 0.0, 'infra': [1, 1, 1]},
            'C': {'arrival': 40, 'water_level': 0.0, 'infra': [1, 1, 1]},
            'D': {'arrival': 40, 'water_level': 0.0, 'infra': [1, 1, 1]},
            'E': {'arrival': 60, 'water_level': 0.0, 'infra': [1, 1, 1]},
        }

        state = self._get_obs()

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
        reward = 0.0

        for node_id, data in self.map_nodes.items():
            if self.time >= data['arrival']:
                # Increase flood level by a random amount each step after arrival
                data['water_level'] += self.np_random.uniform(0, 1)

                # Simulate infrastructure failure based on water level
                for i in range(len(data['infra'])):
                    if data['infra'][i] == 1:  # only consider failure if currently functional
                        p_failure = 0.0
                        if i == 0:
                            p_failure = data['water_level'] * 0.02
                        else:
                            if data['infra'][i-1] == 0:
                                p_failure = 0.2

                        if self.np_random.uniform(0, 1) < p_failure:
                            data['infra'][i] = 0


        # 2. Check Termination: Sum up the health of ALL infrastructure on the map
        total_health = 0
        for data in self.map_nodes.values():
            total_health += sum(data['infra'])
            
        # Terminate ONLY when every single node on the map is destroyed (total_health == 0)
        terminated = bool(total_health == 0)

        # 3. Return the standard Gymnasium tuple
        state = self._get_obs() # Our helper function that flattens the dictionary

        return state, reward, terminated, False, {}
    
    def render(self):
        import matplotlib.pyplot as plt

        # Initialize the figure only once
        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.fig.canvas.manager.set_window_title('Spatial Disaster Simulation')

        self.ax.clear()

        # 1. Define the physical coordinates to match your new layout
        positions = {
            'A': (1, 3),
            'B': (2, 3),
            'C': (3, 4),  # B's upper neighbor
            'D': (3, 2),  # B's lower neighbor
            'E': (4, 2)   # E connects to D, so we place it to the right of D
        }

        # 2. Draw lines connecting the cities (The exact path you described)
        edges = [('A', 'B'), ('B', 'C'), ('B', 'D'), ('D', 'E')]
        
        for start, end in edges:
            x_values = [positions[start][0], positions[end][0]]
            y_values = [positions[start][1], positions[end][1]]
            self.ax.plot(x_values, y_values, 'black', lw=2, zorder=1)

        # 3. Draw the cities and their specific stats from self.map_nodes!
        for node, pos in positions.items():
            data = self.map_nodes[node] 
            water_lvl = data['water_level']
            infra = data['infra']
            
            # Determine color: 3 = Green (Perfect), 0 = Red (Destroyed)
            health = sum(infra)
            if health == 3:
                color = 'green'
            elif health > 0:
                color = 'orange'
            else:
                color = 'red'

            # Draw the city circle on layer 5
            self.ax.scatter(*pos, s=1500, c=color, zorder=5)
            
            # Put the letter on layer 10 so it stays on top of the circle!
            self.ax.text(*pos, node, ha='center', va='center', 
                         color='white', fontweight='bold', fontsize=18, zorder=10)

            # Display the Bayes Net status and Water Level BELOW the circle
            status_text = f"Infra: {infra}\nFlood: {water_lvl:.1f}"
            self.ax.text(pos[0], pos[1] - 0.3, status_text, ha='center', va='top', fontsize=10, 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Add a big clock at the top
        self.ax.set_title(f"Simulation Time: t = {self.time}", fontsize=16, fontweight='bold')
        self.ax.axis('off')

        plt.draw()
        plt.pause(0.2)

    def _get_obs(self):
        """
        Flattens the map_nodes dictionary into a single 1D numpy array for Gymnasium.
        Format: [time, A_flood, A_i1, A_i2, A_i3, B_flood, B_i1, B_i2, B_i3, ...]
        """
        # Start the list with the current time
        state_list = [self.time]
        
        # Loop through nodes A, B, C, D, E in order
        for node in ['A', 'B', 'C', 'D', 'E']:
            # Add this city's flood level
            state_list.append(self.map_nodes[node]['water_level'])
            # Add this city's 3 infrastructure states
            state_list.extend(self.map_nodes[node]['infra'])
            
        return np.array(state_list, dtype=np.float32)
    
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
        infra = next_state[2:]  # Adjusted index to match the correct range
        print(f"Time: {time_step:2.0f} | Flood: {water_lvl:5.2f} | Infra States: {infra}")

        print('Simulation terminated (water level >= 50 or all infrastructures failed)')
    
    plt.ioff()
    plt.show()