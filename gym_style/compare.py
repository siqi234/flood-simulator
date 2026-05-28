"""
Side-by-side comparison of IID vs SI cascading failure models.

Uses the real algo2 / algo3 / algo4 environments directly.

Left panel  — IID: power lines fail only from direct flood stress.
Right panel — SI:  power lines also cascade through network topology;
                   this propagates to more telecom tower outages via algo4.

Roads (algo3) are independent of the power model so a single instance
drives both panels with the same seed.

Run from project root:
    python gym_style/compare.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon as MplPolygon

from gym_style.algo2_powerline import PowerlineFailureEnv
from gym_style.algo3_road      import RoadBlockageEnv
from gym_style.algo4_telecom   import TelecomFailureEnv


# ---------------------------------------------------------------------------
# Build figure — 2 panels (IID | SI), each showing all three infra layers
# ---------------------------------------------------------------------------

def build_figure(algo2_iid, algo3, algo4_iid, algo2_si, algo4_si):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)
    fig.suptitle(
        "IID (direct only)  vs  SI (direct + cascade)\n"
        "Power lines · Roads · Telecom towers",
        fontsize=13, fontweight="bold",
    )

    panels = []

    line_segs  = [list(geom.coords) for geom in algo2_iid.lines_proj.geometry]
    road_polys = []
    for geom in algo3.roads_proj.geometry:
        poly = max(geom.geoms, key=lambda p: p.area) if geom.geom_type == "MultiPolygon" else geom
        road_polys.append(MplPolygon(list(poly.exterior.coords)))
    tower_x = algo4_iid.towers_proj.geometry.x.values
    tower_y = algo4_iid.towers_proj.geometry.y.values

    bounds = algo2_iid.lines_proj.total_bounds
    m = 8000

    for ax, title, algo2, algo4 in [
        (axes[0], "IID — direct failures only",       algo2_iid, algo4_iid),
        (axes[1], "SI  — direct + cascade failures",  algo2_si,  algo4_si),
    ]:
        ax.set_title(title, fontsize=11)
        algo2.boundary_gdf.boundary.plot(ax=ax, linewidth=1, color="black", zorder=1)

        # Power lines
        lc = mc.LineCollection(line_segs, colors="limegreen", linewidths=1.5, zorder=2)
        ax.add_collection(lc)

        # Roads
        pc = mc.PatchCollection(
            [MplPolygon(list(
                (max(g.geoms, key=lambda p: p.area) if g.geom_type == "MultiPolygon" else g)
                .exterior.coords
            )) for g in algo3.roads_proj.geometry],
            facecolors="limegreen", edgecolors="none", alpha=0.6, zorder=3,
        )
        ax.add_collection(pc)

        # Substations
        algo2.subs_proj.plot(ax=ax, color="goldenrod", markersize=14, marker="s", zorder=4)

        # Telecom towers
        sc = ax.scatter(tower_x, tower_y, c=["limegreen"] * algo4.n_towers,
                        s=35, marker="^", edgecolors="black", linewidths=0.4, zorder=5)

        # Flood circle
        origin = algo2.flood_data[0]["origin"]
        flood_fill = Circle(origin, 0, color="steelblue", alpha=0.12, zorder=6)
        flood_ring = Circle(origin, 0, fill=False, edgecolor="steelblue", linewidth=1.5, zorder=6)
        ax.add_patch(flood_fill)
        ax.add_patch(flood_ring)
        ax.plot(*origin, "r^", markersize=9, zorder=7)

        ax.set_xlim(bounds[0] - m, bounds[2] + m)
        ax.set_ylim(bounds[1] - m, bounds[3] + m)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m, EPSG:32618)")
        ax.set_ylabel("Y (m, EPSG:32618)")

        legend_handles = [
            Line2D([0], [0], color="limegreen", lw=2,         label="Line intact"),
            Line2D([0], [0], color="red",       lw=2,         label="Line failed"),
            Patch(facecolor="limegreen", alpha=0.6,           label="Road clear"),
            Patch(facecolor="red",       alpha=0.6,           label="Road blocked"),
            Line2D([0], [0], marker="^", color="w", markersize=8,
                   markerfacecolor="limegreen", markeredgecolor="black", label="Tower ok"),
            Line2D([0], [0], marker="^", color="w", markersize=8,
                   markerfacecolor="red",       markeredgecolor="black", label="Tower failed"),
            Patch(facecolor="goldenrod",                       label="Substation"),
            Patch(facecolor="steelblue", alpha=0.2,            label="Flood extent"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=7)

        txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=8,
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), zorder=8)

        panels.append(dict(lc=lc, pc=pc, sc=sc, flood_fill=flood_fill,
                           flood_ring=flood_ring, txt=txt, algo2=algo2, algo4=algo4))

    return fig, panels, algo3


# ---------------------------------------------------------------------------
# Update one panel for the current timestep
# ---------------------------------------------------------------------------

def update_panel(panel, flood, t):
    algo2 = panel["algo2"]
    algo4 = panel["algo4"]

    # Power lines
    line_colors = np.where(algo2.L_status == 1, "limegreen", "red")
    panel["lc"].set_colors(line_colors)

    # Roads (shared R_status passed in via algo3 reference held by caller)
    # updated separately — see main loop

    # Telecom towers
    tower_colors = np.where(algo4.T_status == 1, "red", "limegreen")
    panel["sc"].set_facecolors(tower_colors)

    # Flood circle
    origin = flood["origin"]
    radius = flood["flood_radius"]
    panel["flood_fill"].center = origin
    panel["flood_ring"].center = origin
    panel["flood_fill"].set_radius(radius)
    panel["flood_ring"].set_radius(radius)

    n_lines_failed  = int(np.sum(algo2.L_status == 0))
    n_towers_failed = int(np.sum(algo4.T_status == 1))

    panel["txt"].set_text(
        f"Hour         : {t:>3d}\n"
        f"Flood radius : {radius/1000:>6.2f} km\n"
        f"Max depth    : {flood['max_depth']:>5.2f} m\n"
        f"Lines failed : {n_lines_failed:>3d} / {algo2.n_lines}\n"
        f"Towers failed: {n_towers_failed:>3d} / {algo4.n_towers}"
    )


def update_roads(panels, R_status):
    n_roads = len(R_status)
    road_colors = np.where(R_status == 1, "red", "limegreen")
    for p in panels:
        p["pc"].set_facecolors(road_colors)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    edge_factor = 0.8, 
    mu          = -0.22,  # paper default; median failure at exp(-0.22)≈0.80m
    sigma       = 0.30,
    seed        = 42,
):
    print("Loading environments...")

    algo2_iid = PowerlineFailureEnv(mu=mu, sigma=sigma, use_si_model=False)
    algo2_si  = PowerlineFailureEnv(mu=mu, sigma=sigma, use_si_model=True,
                                    edge_factor=edge_factor)
    algo3     = RoadBlockageEnv()
    algo4_iid = TelecomFailureEnv(algo2_env=algo2_iid)
    algo4_si  = TelecomFailureEnv(algo2_env=algo2_si)

    algo2_iid.reset(seed=seed)
    algo2_si.reset(seed=seed)
    algo3.reset(seed=seed)
    algo4_iid.reset(seed=seed)
    algo4_si.reset(seed=seed)

    T = algo2_iid.T

    _, panels, _ = build_figure(algo2_iid, algo3, algo4_iid, algo2_si, algo4_si)
    plt.ion()
    plt.show()

    for t in range(T):
        flood = algo2_iid.flood_data[t]

        algo2_iid.step(0)
        algo2_si.step(0)
        algo3.step(0)
        algo4_iid.step(0)
        algo4_si.step(0)

        update_panel(panels[0], flood, t + 1)
        update_panel(panels[1], flood, t + 1)
        update_roads(panels, algo3.R_status)

        plt.draw()
        plt.pause(0.05)
        input(f"  Hour {t + 1}/{T} — press Enter for next step...")

    plt.ioff()
    n_iid = int(np.sum(algo2_iid.L_status == 0))
    n_si  = int(np.sum(algo2_si.L_status  == 0))
    print(f"\nFinal summary:")
    print(f"  IID — lines failed : {n_iid} / {algo2_iid.n_lines}")
    print(f"  SI  — lines failed : {n_si}  / {algo2_si.n_lines}  (+{n_si - n_iid} from cascade)")
    print(f"  IID — towers failed: {int(np.sum(algo4_iid.T_status == 1))} / {algo4_iid.n_towers}")
    print(f"  SI  — towers failed: {int(np.sum(algo4_si.T_status  == 1))} / {algo4_si.n_towers}")
    print(f"  Roads blocked      : {int(np.sum(algo3.R_status == 1))} / {algo3.n_roads}")
    plt.show()


if __name__ == "__main__":
    main()
