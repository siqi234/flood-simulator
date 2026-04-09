"""
Algorithm 3 (Flood) — Road Blockage Assessment
Gymnasium-style: step() advances one timestep at a time.

Run standalone:
    python gym_style/algo3_road.py
"""

import json
from math import erfc, log, sqrt

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.patches import Circle, Patch, Polygon as MplPolygon
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RoadBlockageEnv(gym.Env):
    """
    Wraps Algorithm 3 (Flood): Road Blockage Assessment.

    Observation: [t, flood_radius, flood_max_depth, R_status (N_roads,)]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        flood_path="synthetic_data/flood_data.geojson",
        roads_path="synthetic_data/roads_with_trees.geojson",
        boundary_path="jsons/Precinct_Boundaries.geojson",
        epsg=32618,
        mu=-0.92,
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

        roads = gpd.read_file(roads_path).to_crs(epsg=epsg).copy()
        roads["cent_x"] = roads.geometry.centroid.x
        roads["cent_y"] = roads.geometry.centroid.y
        self.roads_proj   = roads
        self.boundary_gdf = gpd.read_file(boundary_path).to_crs(epsg=epsg)

        self.T       = len(self.flood_data)
        self.n_roads = len(self.roads_proj)

        self._cent_x = roads["cent_x"].values.astype(float)
        self._cent_y = roads["cent_y"].values.astype(float)

        obs_size = 3 + self.n_roads
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)

        # Build road patches once (used in render)
        self._road_patches = []
        for geom in self.roads_proj.geometry:
            poly = max(geom.geoms, key=lambda p: p.area) if geom.geom_type == "MultiPolygon" else geom
            self._road_patches.append(MplPolygon(list(poly.exterior.coords)))

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self._R_status  = np.zeros(self.n_roads, dtype=int)
        self._R_flooded = np.zeros(self.n_roads, dtype=bool)
        self._fig = None
        return self._get_obs(), {}

    def step(self, action):
        flood  = self.flood_data[self.t]
        ox, oy = flood["origin"]
        radius = flood["flood_radius"]
        depth  = flood["max_depth"]

        if radius > 0:
            dist = np.sqrt((self._cent_x - ox) ** 2 + (self._cent_y - oy) ** 2)
            self._R_flooded |= dist < radius

            for i in np.where(self._R_flooded)[0]:
                if self._R_status[i] == 0:
                    local_depth = depth * max(0.0, 1.0 - dist[i] / radius)
                    if self.np_random.random() < self._lognormal_cdf(local_depth):
                        self._R_status[i] = 1

        self.t += 1
        terminated = self.t >= self.T

        info = {"roads_blocked": int(np.sum(self._R_status == 1))}
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
            self._fig.canvas.manager.set_window_title("Algo 3 — Road Blockage")

            self.boundary_gdf.boundary.plot(ax=self._ax, linewidth=1, color="black", zorder=1)

            self._pc = mc.PatchCollection(
                self._road_patches, facecolors="limegreen", edgecolors="none", zorder=2
            )
            self._ax.add_collection(self._pc)

            self._flood_fill = Circle(origin, 0, color="steelblue", alpha=0.20, zorder=3)
            self._flood_ring = Circle(origin, 0, fill=False, edgecolor="steelblue", linewidth=1.5, zorder=3)
            self._ax.add_patch(self._flood_fill)
            self._ax.add_patch(self._flood_ring)
            self._ax.plot(*origin, "r^", markersize=10, zorder=4)

            bounds = self.roads_proj.total_bounds
            m = 5000
            self._ax.set_xlim(bounds[0] - m, bounds[2] + m)
            self._ax.set_ylim(bounds[1] - m, bounds[3] + m)
            self._ax.set_aspect("equal")
            self._ax.set_xlabel("X  (m, EPSG:32618)")
            self._ax.set_ylabel("Y  (m, EPSG:32618)")

            legend_handles = [
                Patch(facecolor="limegreen", edgecolor="none", label="Clear road"),
                Patch(facecolor="red",       edgecolor="none", label="Blocked road"),
                Patch(facecolor="steelblue", alpha=0.2, edgecolor="steelblue", label="Flood extent"),
            ]
            self._ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

            self._info_text = self._ax.text(
                0.02, 0.98, "", transform=self._ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=5,
            )

        # Update colours and flood circle
        colors = np.where(self._R_status == 1, "red", "limegreen")
        self._pc.set_facecolors(colors)
        self._flood_fill.set_radius(radius)
        self._flood_ring.set_radius(radius)

        n_blocked = int(np.sum(self._R_status == 1))
        self._ax.set_title(
            f"Algo 3 (Flood) — Road Blockage   |   Hour {t_idx}", fontsize=12
        )
        self._info_text.set_text(
            f"Hour        : {t_idx:>3d}\n"
            f"Flood radius: {radius/1000:>6.2f} km\n"
            f"Max depth   : {flood['max_depth']:>5.2f} m\n"
            f"Blocked     : {n_blocked:>4d} / {self.n_roads}"
        )
        plt.draw()
        plt.pause(0.3)

    # ------------------------------------------------------------------

    def _get_obs(self):
        t_idx = min(self.t, self.T - 1)
        flood = self.flood_data[t_idx]
        return np.concatenate([
            [float(self.t), flood["flood_radius"], flood["max_depth"]],
            self._R_status.astype(np.float32),
        ]).astype(np.float32)

    def _lognormal_cdf(self, x):
        if x <= 0:
            return 0.0
        z = (log(x) - self.mu) / (self.sigma * sqrt(2))
        return 0.5 * erfc(-z)

    @property
    def R_status(self):
        return self._R_status.copy()


# ----------------------------------------------------------------------

if __name__ == "__main__":
    env = RoadBlockageEnv(render_mode="human")
    obs, _ = env.reset(seed=42)
    print(f"Roads: {env.n_roads}  |  Time horizon: {env.T} hours\n")

    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(0)
        env.render()
        print(f"Hour {env.t:>3d} | blocked: {info['roads_blocked']:>4d} / {env.n_roads}")

    plt.ioff()
    plt.show()
