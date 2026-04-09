"""
FloodDisasterEnv — Combined Gymnasium environment for Algorithms 2, 3, and 4.

Composes PowerlineFailureEnv, RoadBlockageEnv, and TelecomFailureEnv into a
single env. Each call to step() advances all three algorithms by one timestep.

Observation vector (float32):
    [t, flood_radius, flood_max_depth,
     L_status (N_lines,),   # 1=intact,      0=failed
     R_status (N_roads,),   # 0=clear,        1=blocked
     T_status (N_towers,)]  # 0=operational,  1=failed

Action space: Discrete(1) — passive observer; no intervention modelled yet.

Run:
    python gym_style/flood_env.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon as MplPolygon
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_style.algo2_powerline import PowerlineFailureEnv
from gym_style.algo3_road import RoadBlockageEnv
from gym_style.algo4_telecom import TelecomFailureEnv


class FloodDisasterEnv(gym.Env):
    """
    Combined environment: Algorithms 2, 3, and 4 run together each step.
    Visualisation shows all three infrastructure layers in one figure.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        flood_path="synthetic_data/flood_data.geojson",
        powerline_path="synthetic_data/powerline.geojson",
        substation_path="synthetic_data/substation.geojson",
        tower_path="synthetic_data/tower.geojson",
        roads_path="synthetic_data/roads_with_trees.geojson",
        boundary_path="jsons/Precinct_Boundaries.geojson",
        epsg=32618,
        mu_line=-0.22,  sigma_line=0.30,
        mu_road=-0.92,  sigma_road=0.30,
        render_mode=None,
    ):
        super().__init__()
        self.render_mode = render_mode

        # Build sub-envs
        self._algo2 = PowerlineFailureEnv(
            flood_path=flood_path, powerline_path=powerline_path,
            substation_path=substation_path, boundary_path=boundary_path,
            epsg=epsg, mu=mu_line, sigma=sigma_line,
        )
        self._algo3 = RoadBlockageEnv(
            flood_path=flood_path, roads_path=roads_path,
            boundary_path=boundary_path, epsg=epsg,
            mu=mu_road, sigma=sigma_road,
        )
        self._algo4 = TelecomFailureEnv(
            algo2_env=self._algo2,
            flood_path=flood_path, powerline_path=powerline_path,
            substation_path=substation_path, tower_path=tower_path,
            boundary_path=boundary_path, epsg=epsg,
        )

        self.T        = self._algo2.T
        self.n_lines  = self._algo2.n_lines
        self.n_roads  = self._algo3.n_roads
        self.n_towers = self._algo4.n_towers

        obs_size = 3 + self.n_lines + self.n_roads + self.n_towers
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)

        # Road patches for render (built once)
        self._road_patches = []
        for geom in self._algo3.roads_proj.geometry:
            poly = max(geom.geoms, key=lambda p: p.area) if geom.geom_type == "MultiPolygon" else geom
            self._road_patches.append(MplPolygon(list(poly.exterior.coords)))

        self._fig = None

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._algo2.reset(seed=seed)
        self._algo3.reset(seed=seed)
        self._algo4.reset(seed=seed)
        self.t = 0
        self._fig = None

        return self._get_obs(), {}

    def step(self, action):
        self._algo2.step(action)
        self._algo3.step(action)
        self._algo4.step(action)
        self.t += 1

        terminated = self.t >= self.T

        info = {
            "lines_failed":  int(np.sum(self._algo2.L_status == 0)),
            "roads_blocked": int(np.sum(self._algo3.R_status == 1)),
            "towers_failed": int(np.sum(self._algo4.T_status == 1)),
        }
        return self._get_obs(), 0.0, terminated, False, info

    def render(self):
        if self.render_mode != "human":
            return

        t_idx   = min(self.t, self.T - 1)
        flood   = self._algo2.flood_data[t_idx]
        origin  = flood["origin"]
        radius  = flood["flood_radius"]

        L_status = self._algo2.L_status
        R_status = self._algo3.R_status
        T_status = self._algo4.T_status

        tower_x = self._algo4.towers_proj.geometry.x.values
        tower_y = self._algo4.towers_proj.geometry.y.values

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(13, 10))
            self._fig.canvas.manager.set_window_title("Flood Disaster — Combined (Algo 2 + 3 + 4)")

            # Boundary
            self._algo2.boundary_gdf.boundary.plot(ax=self._ax, linewidth=1, color="black", zorder=1)

            # Roads (layer 2)
            self._pc_roads = mc.PatchCollection(
                self._road_patches, facecolors="limegreen", edgecolors="none", alpha=0.6, zorder=2
            )
            self._ax.add_collection(self._pc_roads)

            # Power lines (layer 3)
            line_segs = [list(geom.coords) for geom in self._algo2.lines_proj.geometry]
            self._lc = mc.LineCollection(line_segs, colors="limegreen", linewidths=1.2, zorder=3)
            self._ax.add_collection(self._lc)

            # Substations (layer 4)
            self._algo4.subs_proj.plot(ax=self._ax, color="goldenrod", markersize=15, marker="s", zorder=4)

            # Towers (layer 5)
            self._sc = self._ax.scatter(
                tower_x, tower_y,
                c=["limegreen"] * self.n_towers, s=50, marker="^",
                edgecolors="black", linewidths=0.4, zorder=5,
            )

            # Flood circle (layer 6)
            self._flood_fill = Circle(origin, 0, color="steelblue", alpha=0.15, zorder=6)
            self._flood_ring = Circle(origin, 0, fill=False, edgecolor="steelblue", linewidth=1.5, zorder=6)
            self._ax.add_patch(self._flood_fill)
            self._ax.add_patch(self._flood_ring)
            self._ax.plot(*origin, "r^", markersize=10, zorder=7, label="Flood origin")

            # Axis limits
            bounds = self._algo2.lines_proj.total_bounds
            m = 8000
            self._ax.set_xlim(bounds[0] - m, bounds[2] + m)
            self._ax.set_ylim(bounds[1] - m, bounds[3] + m)
            self._ax.set_aspect("equal")
            self._ax.set_xlabel("X  (m, EPSG:32618)")
            self._ax.set_ylabel("Y  (m, EPSG:32618)")

            legend_handles = [
                Patch(facecolor="limegreen", alpha=0.6, edgecolor="none", label="Clear road"),
                Patch(facecolor="red",       alpha=0.6, edgecolor="none", label="Blocked road"),
                Line2D([0], [0], color="limegreen", lw=2,         label="Intact power line"),
                Line2D([0], [0], color="dimgrey",   lw=2,         label="Failed power line"),
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
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=8,
            )

        # Update roads
        road_colors = np.where(R_status == 1, "red", "limegreen")
        self._pc_roads.set_facecolors(road_colors)

        # Update power lines
        line_colors = np.where(L_status == 1, "limegreen", "dimgrey")
        self._lc.set_colors(line_colors)

        # Update towers
        t_colors = np.where(T_status == 1, "red", "limegreen")
        self._sc.set_facecolors(t_colors)

        # Update flood circle
        self._flood_fill.set_radius(radius)
        self._flood_ring.set_radius(radius)

        self._ax.set_title(
            f"Flood Disaster Simulation (Algo 2 + 3 + 4)   |   Hour {t_idx}", fontsize=13
        )
        self._info_text.set_text(
            f"Hour         : {t_idx:>3d}\n"
            f"Flood radius : {radius/1000:>6.2f} km\n"
            f"Max depth    : {flood['max_depth']:>5.2f} m\n"
            f"Lines failed : {int(np.sum(L_status==0)):>3d} / {self.n_lines}\n"
            f"Roads blocked: {int(np.sum(R_status==1)):>4d} / {self.n_roads}\n"
            f"Towers failed: {int(np.sum(T_status==1)):>3d} / {self.n_towers}"
        )
        plt.draw()
        plt.pause(0.3)

    # ------------------------------------------------------------------

    def _get_obs(self):
        t_idx = min(self.t, self.T - 1)
        flood = self._algo2.flood_data[t_idx]
        return np.concatenate([
            [float(self.t), flood["flood_radius"], flood["max_depth"]],
            self._algo2.L_status.astype(np.float32),
            self._algo3.R_status.astype(np.float32),
            self._algo4.T_status.astype(np.float32),
        ]).astype(np.float32)


# ----------------------------------------------------------------------

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
