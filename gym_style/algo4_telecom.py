"""
Algorithm 4 (Flood) — Failure Propagation to Telecommunication Network
Gymnasium-style: step() advances one timestep at a time.

Depends on L_status from Algorithm 2 at each step.
Accepts a live PowerlineFailureEnv instance so it can read
the current L_status after each step.

Run standalone (runs Algo 2 internally to drive Algo 4):
    python gym_style/algo4_telecom.py
"""

import json
from collections import defaultdict

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TelecomFailureEnv(gym.Env):
    """
    Wraps Algorithm 4 (Flood): Failure Propagation to Telecom Network.

    Takes a live PowerlineFailureEnv (algo2_env) and reads its L_status
    each step to determine which substations have lost grid power.

    Observation: [t, flood_radius, flood_max_depth,
                  L_status (N_lines,), T_status (N_towers,)]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        algo2_env,                                         # PowerlineFailureEnv instance
        flood_path="synthetic_data/flood_data.geojson",
        powerline_path="synthetic_data/powerline.geojson",
        substation_path="synthetic_data/substation.geojson",
        tower_path="synthetic_data/tower.geojson",
        boundary_path="jsons/Precinct_Boundaries.geojson",
        epsg=32618,
        render_mode=None,
    ):
        super().__init__()
        self.algo2 = algo2_env
        self.render_mode = render_mode

        # Load data
        with open(flood_path) as f:
            raw = json.load(f)
        self.flood_data = [
            {
                "t":            feat["properties"]["t"],
                "origin":       tuple(feat["geometry"]["coordinates"]),
                "flood_radius": feat["properties"]["flood_radius"],
                "max_depth":    feat["properties"]["max_depth"],
            }
            for feat in raw["features"]
        ]

        self.lines_proj   = gpd.read_file(powerline_path).to_crs(epsg=epsg).reset_index(drop=True)
        self.subs_proj    = gpd.read_file(substation_path).to_crs(epsg=epsg)
        self.towers_proj  = gpd.read_file(tower_path).to_crs(epsg=epsg)
        self.boundary_gdf = gpd.read_file(boundary_path).to_crs(epsg=epsg)

        self.T        = len(self.flood_data)
        self.n_lines  = len(self.lines_proj)
        self.n_towers = len(self.towers_proj)

        # Substation → line index map
        self._sub_to_lines = defaultdict(set)
        for idx, row in enumerate(self.lines_proj.itertuples()):
            self._sub_to_lines[row.from_node].add(idx)
            self._sub_to_lines[row.to_node].add(idx)

        self._tower_sub    = self.towers_proj["substation_id"].values
        self._tower_backup = self.towers_proj["backup_hours"].values.astype(int)

        obs_size = 3 + self.n_lines + self.n_towers
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self._T_status      = np.zeros(self.n_towers, dtype=int)
        self._backup_timer  = np.zeros(self.n_towers, dtype=int)
        self._fig = None
        return self._get_obs(), {}

    def step(self, action):
        # Read current L_status from algo 2 env
        L_status = self.algo2.L_status

        # Determine which substations have grid power
        sub_powered = {
            sub_id: any(L_status[i] == 1 for i in line_idxs)
            for sub_id, line_idxs in self._sub_to_lines.items()
        }

        for j in range(self.n_towers):
            if self._T_status[j] == 1:
                continue
            if sub_powered.get(self._tower_sub[j], False):
                self._backup_timer[j] = 0
            else:
                self._backup_timer[j] += 1
            if self._backup_timer[j] > self._tower_backup[j]:
                self._T_status[j] = 1

        self.t += 1
        terminated = self.t >= self.T

        info = {
            "towers_failed": int(np.sum(self._T_status == 1)),
            "lines_failed":  int(np.sum(L_status == 0)),
        }
        return self._get_obs(), 0.0, terminated, False, info

    def render(self):
        if self.render_mode != "human":
            return

        t_idx  = min(self.t, self.T - 1)
        flood  = self.flood_data[t_idx]
        origin = flood["origin"]
        radius = flood["flood_radius"]

        tower_x = self.towers_proj.geometry.x.values
        tower_y = self.towers_proj.geometry.y.values

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(11, 9))
            self._fig.canvas.manager.set_window_title("Algo 4 — Telecom Failure Propagation")

            self.boundary_gdf.boundary.plot(ax=self._ax, linewidth=1, color="black", zorder=1)

            self.subs_proj.plot(ax=self._ax, color="goldenrod", markersize=15, marker="s", zorder=2)

            self._sc = self._ax.scatter(
                tower_x, tower_y,
                c=["limegreen"] * self.n_towers, s=40, marker="^",
                edgecolors="black", linewidths=0.4, zorder=3,
            )

            self._flood_fill = Circle(origin, 0, color="steelblue", alpha=0.15, zorder=4)
            self._flood_ring = Circle(origin, 0, fill=False, edgecolor="steelblue", linewidth=1.5, zorder=4)
            self._ax.add_patch(self._flood_fill)
            self._ax.add_patch(self._flood_ring)
            self._ax.plot(*origin, "r^", markersize=10, zorder=5)

            bounds = self.towers_proj.total_bounds
            m = 8000
            self._ax.set_xlim(bounds[0] - m, bounds[2] + m)
            self._ax.set_ylim(bounds[1] - m, bounds[3] + m)
            self._ax.set_aspect("equal")
            self._ax.set_xlabel("X  (m, EPSG:32618)")
            self._ax.set_ylabel("Y  (m, EPSG:32618)")

            legend_handles = [
                Line2D([0], [0], marker="^", color="w",
                       markerfacecolor="limegreen", markersize=9,
                       markeredgecolor="black",     label="Tower operational"),
                Line2D([0], [0], marker="^", color="w",
                       markerfacecolor="red",       markersize=9,
                       markeredgecolor="black",     label="Tower failed"),
                Patch(facecolor="goldenrod",  edgecolor="none",  label="Substation"),
                Patch(facecolor="steelblue",  alpha=0.2,
                      edgecolor="steelblue",                     label="Flood extent"),
            ]
            self._ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

            self._info_text = self._ax.text(
                0.02, 0.98, "", transform=self._ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=6,
            )

        # Update towers
        t_colors = np.where(self._T_status == 1, "red", "limegreen")
        self._sc.set_facecolors(t_colors)

        # Update flood circle
        self._flood_fill.set_radius(radius)
        self._flood_ring.set_radius(radius)

        n_tower_failed = int(np.sum(self._T_status == 1))
        self._ax.set_title(
            f"Algo 4 (Flood) — Telecom Failure Propagation   |   Hour {t_idx}", fontsize=12
        )
        self._info_text.set_text(
            f"Hour         : {t_idx:>3d}\n"
            f"Flood radius : {radius/1000:>6.2f} km\n"
            f"Max depth    : {flood['max_depth']:>5.2f} m\n"
            f"Towers failed: {n_tower_failed:>3d} / {self.n_towers}"
        )
        plt.draw()
        plt.pause(0.3)

    # ------------------------------------------------------------------

    def _get_obs(self):
        t_idx = min(self.t, self.T - 1)
        flood = self.flood_data[t_idx]
        return np.concatenate([
            [float(self.t), flood["flood_radius"], flood["max_depth"]],
            self.algo2.L_status.astype(np.float32),
            self._T_status.astype(np.float32),
        ]).astype(np.float32)

    @property
    def T_status(self):
        return self._T_status.copy()


# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from gym_style.algo2_powerline import PowerlineFailureEnv

    algo2 = PowerlineFailureEnv(render_mode="human")
    algo4 = TelecomFailureEnv(algo2_env=algo2, render_mode="human")

    obs2, _ = algo2.reset(seed=42)
    obs4, _ = algo4.reset(seed=42)

    print(f"Power lines: {algo2.n_lines}  |  Towers: {algo4.n_towers}  |  Hours: {algo2.T}\n")

    terminated = False
    while not terminated:
        obs2, _, terminated, _, info2 = algo2.step(0)
        obs4, _, _,          _, info4 = algo4.step(0)

        # algo2.render()
        algo4.render()

        print(
            f"Hour {algo2.t:>3d} | "
            f"lines failed: {info2['lines_failed']:>3d}/{algo2.n_lines} | "
            f"towers failed: {info4['towers_failed']:>3d}/{algo4.n_towers}"
        )

    plt.ioff()
    plt.show()
