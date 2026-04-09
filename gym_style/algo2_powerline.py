"""
Algorithm 2 (Flood) — Power Line Failure Assessment
Gymnasium-style: step() advances one timestep at a time.

Run standalone:
    python gym_style/algo2_powerline.py
"""

import json
from math import erfc, log, sqrt

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PowerlineFailureEnv(gym.Env):
    """
    Wraps Algorithm 2 (Flood): Power Line Failure Assessment.

    Observation: [t, flood_radius, flood_max_depth, L_status (N_lines,)]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        flood_path="synthetic_data/flood_data.geojson",
        powerline_path="synthetic_data/powerline.geojson",
        substation_path="synthetic_data/substation.geojson",
        boundary_path="jsons/Precinct_Boundaries.geojson",
        epsg=32618,
        mu=-0.22,
        sigma=0.30,
        render_mode=None,
    ):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
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
        self.lines_proj = (
            gpd.read_file(powerline_path).to_crs(epsg=epsg).reset_index(drop=True)
        )
        self.subs_proj    = gpd.read_file(substation_path).to_crs(epsg=epsg)
        self.boundary_gdf = gpd.read_file(boundary_path).to_crs(epsg=epsg)

        self.T       = len(self.flood_data)
        self.n_lines = len(self.lines_proj)

        self._mid_x = (
            self.lines_proj.geometry.interpolate(0.5, normalized=True).x.values.astype(float)
        )
        self._mid_y = (
            self.lines_proj.geometry.interpolate(0.5, normalized=True).y.values.astype(float)
        )

        obs_size = 3 + self.n_lines
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self._L_status  = np.ones(self.n_lines, dtype=int)
        self._L_flooded = np.zeros(self.n_lines, dtype=bool)
        self._fig = None  # reset figure handle
        return self._get_obs(), {}

    def step(self, action):
        flood  = self.flood_data[self.t]
        ox, oy = flood["origin"]
        radius = flood["flood_radius"]
        depth  = flood["max_depth"]

        dist = np.sqrt((self._mid_x - ox) ** 2 + (self._mid_y - oy) ** 2)
        self._L_flooded |= dist < radius

        for i in np.where(self._L_flooded)[0]:
            if self._L_status[i] == 1:
                local_depth = depth * max(0.0, 1.0 - dist[i] / radius) if radius > 0 else 0.0
                if self.np_random.random() < self._lognormal_cdf(local_depth):
                    self._L_status[i] = 0

        self.t += 1
        terminated = self.t >= self.T

        info = {"lines_failed": int(np.sum(self._L_status == 0))}
        return self._get_obs(), 0.0, terminated, False, info

    def render(self):
        if self.render_mode != "human":
            return

        t_idx  = min(self.t, self.T - 1)
        flood  = self.flood_data[t_idx]
        origin = flood["origin"]
        radius = flood["flood_radius"]

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(11, 9))
            self._fig.canvas.manager.set_window_title("Algo 2 — Power Line Failure")

            self.boundary_gdf.boundary.plot(ax=self._ax, linewidth=1, color="black", zorder=1)

            line_segs = [list(geom.coords) for geom in self.lines_proj.geometry]
            self._lc = mc.LineCollection(line_segs, colors="limegreen", linewidths=1.5, zorder=2)
            self._ax.add_collection(self._lc)

            self.subs_proj.plot(ax=self._ax, color="blue", markersize=20, marker="s", zorder=4)

            self._flood_fill = Circle(origin, 0, color="steelblue", alpha=0.20, zorder=3)
            self._flood_ring = Circle(origin, 0, fill=False, edgecolor="steelblue", linewidth=1.5, zorder=3)
            self._ax.add_patch(self._flood_fill)
            self._ax.add_patch(self._flood_ring)
            self._ax.plot(*origin, "r^", markersize=10, zorder=5)

            bounds = self.lines_proj.total_bounds
            m = 8000
            self._ax.set_xlim(bounds[0] - m, bounds[2] + m)
            self._ax.set_ylim(bounds[1] - m, bounds[3] + m)
            self._ax.set_aspect("equal")
            self._ax.set_xlabel("X  (m, EPSG:32618)")
            self._ax.set_ylabel("Y  (m, EPSG:32618)")

            legend_handles = [
                Line2D([0], [0], color="limegreen", lw=2, label="Intact line"),
                Line2D([0], [0], color="red",       lw=2, label="Failed line"),
                Patch(facecolor="steelblue", alpha=0.2, edgecolor="steelblue", label="Flood extent"),
                Line2D([0], [0], marker="s", color="w", markerfacecolor="blue",
                       markersize=8, label="Substation"),
            ]
            self._ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

            self._info_text = self._ax.text(
                0.02, 0.98, "", transform=self._ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=6,
            )

        # Update colours and flood circle
        colors = np.where(self._L_status == 1, "limegreen", "red")
        self._lc.set_colors(colors)
        self._flood_fill.set_radius(radius)
        self._flood_ring.set_radius(radius)

        n_failed = int(np.sum(self._L_status == 0))
        self._ax.set_title(
            f"Algo 2 (Flood) — Power Line Failure   |   Hour {t_idx}", fontsize=12
        )
        self._info_text.set_text(
            f"Hour        : {t_idx:>3d}\n"
            f"Flood radius: {radius/1000:>6.2f} km\n"
            f"Max depth   : {flood['max_depth']:>5.2f} m\n"
            f"Failed      : {n_failed:>3d} / {self.n_lines}"
        )
        plt.draw()
        plt.pause(0.3)

    # ------------------------------------------------------------------

    def _get_obs(self):
        t_idx = min(self.t, self.T - 1)
        flood = self.flood_data[t_idx]
        return np.concatenate([
            [float(self.t), flood["flood_radius"], flood["max_depth"]],
            self._L_status.astype(np.float32),
        ]).astype(np.float32)

    def _lognormal_cdf(self, x):
        if x <= 0:
            return 0.0
        z = (log(x) - self.mu) / (self.sigma * sqrt(2))
        return 0.5 * erfc(-z)

    # Expose status for downstream envs (Algo 4)
    @property
    def L_status(self):
        return self._L_status.copy()


# ----------------------------------------------------------------------

if __name__ == "__main__":
    env = PowerlineFailureEnv(render_mode="human")
    obs, _ = env.reset(seed=42)
    print(f"Power lines: {env.n_lines}  |  Time horizon: {env.T} hours\n")

    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(0)
        env.render()
        print(f"Hour {env.t:>3d} | failed: {info['lines_failed']:>3d} / {env.n_lines}")

    plt.ioff()
    plt.show()
