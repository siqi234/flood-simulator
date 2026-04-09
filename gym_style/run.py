"""
Run the flood disaster simulation.

Usage (from the simulator root):
    python gym_style/run.py            # combined (all 3 algos, one window)
    python gym_style/algo2_powerline.py  # algo 2 only
    python gym_style/algo3_road.py       # algo 3 only
    python gym_style/algo4_telecom.py    # algo 4 only (drives algo 2 internally)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from gym_style.flood_env import FloodDisasterEnv

if __name__ == "__main__":
    env = FloodDisasterEnv(render_mode="human")
    obs, _ = env.reset(seed=42)

    print(f"Observation size : {len(obs)}")
    print(f"  Power lines    : {env.n_lines}")
    print(f"  Roads          : {env.n_roads}")
    print(f"  Telecom towers : {env.n_towers}")
    print(f"  Time horizon   : {env.T} hours\n")

    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(0)
        env.render()
        print(
            f"Hour {env.t:>3d} | "
            f"lines failed: {info['lines_failed']:>3d}/{env.n_lines} | "
            f"roads blocked: {info['roads_blocked']:>4d}/{env.n_roads} | "
            f"towers failed: {info['towers_failed']:>3d}/{env.n_towers}"
        )

    print("\nEpisode complete.")
    plt.ioff()
    plt.show()
