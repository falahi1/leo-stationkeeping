"""
Phase 1b: Ground Track Visualisation
--------------------------------------
Loads the saved NUSAT-26 TLE from data/, uses skyfield to convert SGP4
positions to geodetic latitude/longitude, and plots the 24-hour ground
track on a world map with each orbit coloured individually.

Coastlines are fetched from Natural Earth (110m resolution, ~200 KB).
If the fetch fails the track is plotted on a plain background — the
orbital mechanics output is identical either way.
"""

import json
import math
import os
import urllib.request
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec
from skyfield.api import load, EarthSatellite, wgs84


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TLE_FILE   = "data/nusat26_tle.txt"
HOURS      = 24      # window to plot
STEP_MIN   = 0.5     # time step in minutes (30 s — smoother polar turnarounds)

COASTLINE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_coastline.geojson"
)


# ---------------------------------------------------------------------------
# TLE loader — reads the saved file, skips # comment lines
# ---------------------------------------------------------------------------
def load_tle(path: str):
    """Return (name, line1, line2) from the saved TLE file."""
    lines = []
    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    for i in range(len(lines) - 2):
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            return lines[i], lines[i + 1], lines[i + 2]
    raise ValueError(f"No valid TLE found in {path}")


# ---------------------------------------------------------------------------
# Coastline fetch
# ---------------------------------------------------------------------------
def fetch_coastlines():
    """
    Download Natural Earth 110m coastline GeoJSON and return a list of
    (lons, lats) arrays — one per coastline segment.
    Returns None if the fetch fails.
    """
    try:
        req = urllib.request.Request(
            COASTLINE_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        segments = []
        for feat in data["features"]:
            geom = feat["geometry"]
            raw_coords = (
                [geom["coordinates"]]
                if geom["type"] == "LineString"
                else geom["coordinates"]
            )
            for seg in raw_coords:
                segments.append(([c[0] for c in seg], [c[1] for c in seg]))

        print(f"  Loaded {len(segments)} coastline segments.")
        return segments

    except Exception as exc:
        print(f"  Coastline fetch failed ({exc}) — plotting without map.")
        return None


# ---------------------------------------------------------------------------
# Track computation — separated so app.py can reuse without plotting
# ---------------------------------------------------------------------------
def compute_track(name: str, line1: str, line2: str,
                  hours: int = 24, step_min: int = 1):
    """
    Propagate and convert to geodetic.
    Returns (lats_array, lons_array, epoch_dt, period_min).
    """
    sat_sgp4   = Satrec.twoline2rv(line1, line2)
    period_min = 2 * math.pi / sat_sgp4.no_kozai

    ts  = load.timescale(builtin=True)
    sat = EarthSatellite(line1, line2, name, ts)
    epoch_dt   = sat.epoch.utc_datetime()

    n         = int(hours * 60 // step_min)
    datetimes = [epoch_dt + timedelta(minutes=i * step_min) for i in range(n)]
    times     = ts.from_datetimes(datetimes)

    geocentric = sat.at(times)
    subpoint   = wgs84.subpoint_of(geocentric)
    return (subpoint.latitude.degrees,
            subpoint.longitude.degrees,
            epoch_dt,
            period_min)


# ---------------------------------------------------------------------------
# Figure path — descriptive name when run_id provided, versioned fallback
# ---------------------------------------------------------------------------
def figure_path(run_id: str, plot_type: str, folder: str = "figures") -> str:
    if run_id:
        return f"{folder}/{run_id}_{plot_type}.png"
    v = 1
    while os.path.exists(f"{folder}/phase1_{plot_type}_v{v}.png"):
        v += 1
    return f"{folder}/phase1_{plot_type}_v{v}.png"


# ---------------------------------------------------------------------------
# Core analysis — called by main.py or the standalone main() below
# ---------------------------------------------------------------------------
def run(name: str, line1: str, line2: str, run_id: str = None, show: bool = True):
    """Compute 24-hour ground track and produce the world-map figure."""
    sat_sgp4   = Satrec.twoline2rv(line1, line2)
    period_min = (2 * math.pi / sat_sgp4.no_kozai)

    ts  = load.timescale(builtin=True)
    sat = EarthSatellite(line1, line2, name, ts)

    epoch_dt        = sat.epoch.utc_datetime()
    n               = int(HOURS * 60 // STEP_MIN)
    datetimes       = [epoch_dt + timedelta(minutes=i * STEP_MIN) for i in range(n)]
    times           = ts.from_datetimes(datetimes)
    n_orbits        = int(round(HOURS * 60 / period_min))
    steps_per_orbit = int(round(period_min / STEP_MIN))

    print(f"Satellite : {name}")
    print(f"Epoch     : {epoch_dt.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Period    : {period_min:.1f} min")
    print(f"Computing {n} ground-track points ({HOURS} h, ~{n_orbits} orbits)...")

    geocentric = sat.at(times)
    subpoint   = wgs84.subpoint_of(geocentric)
    lats = subpoint.latitude.degrees
    lons = subpoint.longitude.degrees

    print("Fetching coastlines...")
    coastlines = fetch_coastlines()

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_facecolor("#cce5f0")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90,   90)

    if coastlines:
        for seg_lons, seg_lats in coastlines:
            ax.plot(seg_lons, seg_lats, color="#7a6a55", linewidth=0.5, zorder=2)

    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90,   91, 30))
    ax.grid(True, linestyle="--", alpha=0.25, color="white", zorder=1)
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.6, zorder=1)

    cmap         = plt.cm.plasma
    orbit_starts = list(range(0, n, steps_per_orbit)) + [n]

    for k, (i0, i1) in enumerate(zip(orbit_starts[:-1], orbit_starts[1:])):
        color = cmap(k / max(n_orbits - 1, 1))
        # Extend segment by one point into the next orbit to close the gap
        # at the ascending node (avoids missing-step breaks at the equator)
        i1_ext   = min(i1 + 1, n)
        seg_lons = lons[i0:i1_ext]
        seg_lats = lats[i0:i1_ext]
        jumps    = np.where(np.abs(np.diff(seg_lons)) > 180)[0] + 1
        starts   = np.concatenate([[0], jumps])
        ends     = np.concatenate([jumps, [len(seg_lons)]])
        for s, e in zip(starts, ends):
            ax.plot(seg_lons[s:e], seg_lats[s:e],
                    color=color, linewidth=1.1, alpha=0.9, zorder=3)

    ax.plot(lons[0],  lats[0],  "o", color="lime", markersize=9, zorder=5,
            label=f"Start ({epoch_dt.strftime('%Y-%m-%d %H:%M UTC')})")
    ax.plot(lons[-1], lats[-1], "s", color="red",  markersize=8, zorder=5,
            label="End (+24 h)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_orbits))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.018, pad=0.01)
    cbar.set_label("Orbit number", fontsize=10)
    cbar.set_ticks([1, n_orbits // 2, n_orbits])

    ax.set_xlabel("Longitude (°)", fontsize=11)
    ax.set_ylabel("Latitude (°)",  fontsize=11)
    ax.set_title(
        f"Ground Track — {name}\n"
        f"24-hour window · ~{n_orbits} orbits · Max latitude: ±{abs(lats).max():.1f}°",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.7)
    plt.tight_layout()
    if show:
        out = figure_path(run_id, "groundtrack")
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved → {out}")
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Standalone entry point (loads TLE from saved data file)
# ---------------------------------------------------------------------------
def main():
    name, line1, line2 = load_tle(TLE_FILE)
    run(name, line1, line2)


if __name__ == "__main__":
    main()
