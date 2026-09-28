"""
Phase 1: Altitude vs Time from TLE propagation
-----------------------------------------------
Fetches a Satellogic NewSat TLE from CelesTrak, propagates it with SGP4
over 7 days, and produces a two-panel figure:
  Left  — first 3 orbits, raw altitude (shows perigee/apogee oscillation)
  Right — 7-day orbit-averaged decay with linear trend
"""

import math
import os
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday


# ---------------------------------------------------------------------------
# CelesTrak GP element data API (correct endpoint)
# ---------------------------------------------------------------------------
CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?NAME=NUSAT-&FORMAT=TLE"
)


def _checksum(line68: str) -> str:
    """Append the correct TLE checksum digit to a 68-character line."""
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line68)
    return line68 + str(total % 10)


# Fallback: a physically realistic sun-synchronous orbit at ~500 km.
# Used automatically if the CelesTrak fetch fails (no internet, rate limit, etc.)
_L1 = _checksum("1 43768U 18099F   26269.50000000  .00002179  00000-0  13426-3 0  999")
_L2 = _checksum("2 43768  97.4023 123.4567 0010523  87.3456 272.8765 15.19453012 4567")
FALLBACK_TLE = ("NEWSAT DEMO (~500 km SSO)", _L1, _L2)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_first_tle(url: str):
    """
    Download TLE text from `url` and return (name, line1, line2) for the
    first satellite found.  Returns None on any error.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i in range(len(lines) - 2):
            if (lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 ")
                    and lines[i].upper().startswith("NUSAT-")):
                return lines[i], lines[i + 1], lines[i + 2]
    except Exception as exc:
        print(f"  Fetch failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------
def propagate(line1: str, line2: str, days: float = 7.0, step_s: int = 60):
    """
    Run SGP4 from TLE epoch forward by `days`, sampling every `step_s` seconds.
    Returns (sat, time_days_array, altitude_km_array).
    """
    sat = Satrec.twoline2rv(line1, line2)

    jd_epoch = sat.jdsatepoch + sat.jdsatepochF
    J2000 = datetime(2000, 1, 1, 12, 0, 0)          # JD 2451545.0
    epoch = J2000 + timedelta(days=jd_epoch - 2451545.0)

    n = int(days * 86400 / step_s)
    times_days = np.empty(n)
    altitudes  = np.empty(n)

    for i in range(n):
        dt = epoch + timedelta(seconds=i * step_s)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        err, r, _ = sat.sgp4(jd, fr)
        times_days[i] = i * step_s / 86400.0
        altitudes[i] = (
            np.nan if err != 0
            else math.sqrt(r[0]**2 + r[1]**2 + r[2]**2) - 6371.0
        )

    return sat, times_days, altitudes


# ---------------------------------------------------------------------------
# Orbit averaging
# ---------------------------------------------------------------------------
def orbit_average(times_days: np.ndarray, altitudes: np.ndarray,
                  period_s: float, step_s: int = 60):
    """
    Average altitude over each complete orbit.
    Returns (orbit_times_days, orbit_mean_altitudes).
    """
    chunk = max(1, int(round(period_s / step_s)))
    n_orbits = len(altitudes) // chunk
    orbit_times = np.empty(n_orbits)
    orbit_alts  = np.empty(n_orbits)
    for k in range(n_orbits):
        orbit_times[k] = np.nanmean(times_days[k * chunk : (k + 1) * chunk])
        orbit_alts[k]  = np.nanmean(altitudes[k * chunk : (k + 1) * chunk])
    return orbit_times, orbit_alts


# ---------------------------------------------------------------------------
# Figure path — descriptive name when run_id provided, versioned fallback
# ---------------------------------------------------------------------------
def figure_path(run_id: str, plot_type: str, folder: str = "figures") -> str:
    """
    With run_id  → 'figures/nusat26_20260926_altitude.png'
    Without      → 'figures/phase1_altitude_v1.png' (auto-incremented)
    """
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
    """Propagate TLE and produce the two-panel altitude decay figure."""
    print(f"\nTLE in use:\n  {name}\n  {line1}\n  {line2}")
    print("\nPropagating 7 days with SGP4 (60-second steps)...")

    sat, t, alt = propagate(line1, line2, days=7.0, step_s=60)
    period_s    = (2 * math.pi / sat.no_kozai) * 60
    ot, oa      = orbit_average(t, alt, period_s)
    coeffs      = np.polyfit(ot, oa, 1)
    trend       = np.poly1d(coeffs)
    decay       = oa[0] - oa[-1]

    print(f"  Orbital period         : {period_s/60:.1f} min")
    print(f"  Mean altitude at start : {oa[0]:.2f} km")
    print(f"  Mean altitude at end   : {oa[-1]:.2f} km")
    print(f"  Total decay (7 days)   : {decay:.3f} km  →  {decay/7:.4f} km/day")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"SGP4 Altitude Analysis — {name}",
                 fontsize=12, fontweight="bold")

    # Left panel: first 3 orbits raw altitude
    zoom_days = 3 * period_s / 86400
    mask  = t <= zoom_days
    ax1.plot(t[mask] * 24 * 60, alt[mask], color="steelblue", linewidth=1.2)
    ax1.set_xlabel("Time from epoch (minutes)", fontsize=11)
    ax1.set_ylabel("Altitude — |r| − Rₑ  (km)", fontsize=11)
    ax1.set_title(f"(a)  Osculating altitude — first 3 orbits\n"
                  f"Period ≈ {period_s/60:.1f} min",
                  fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.35)

    first   = t <= period_s / 86400
    t_min   = t[first] * 24 * 60
    a_first = alt[first]
    idx_lo  = int(np.nanargmin(a_first))
    idx_hi  = int(np.nanargmax(a_first))

    # Orbit-mean line to visually connect the two panels
    orb_mean = np.nanmean(a_first)
    ax1.axhline(orb_mean, color="#888888", linewidth=0.9,
                linestyle=":", alpha=0.7, label=f"Orbit mean  {orb_mean:.1f} km")
    ax1.legend(fontsize=8.5, framealpha=0.75, loc="lower left")

    ax1.annotate(
        f"perigee  {a_first[idx_lo]:.1f} km",
        xy=(t_min[idx_lo], a_first[idx_lo]),
        xytext=(t_min[idx_lo] + 8, a_first[idx_lo] + 1.8),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
        fontsize=8.5, color="#333333",
    )
    ax1.annotate(
        f"apogee  {a_first[idx_hi]:.1f} km",
        xy=(t_min[idx_hi], a_first[idx_hi]),
        xytext=(t_min[idx_hi] + 8, a_first[idx_hi] - 2.5),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
        fontsize=8.5, color="#333333",
    )

    # Right panel: 7-day orbit-averaged decay
    # Shade the 3-orbit window shown on the left panel
    ax2.axvspan(0, 3 * period_s / 86400, color="#e8f4fb", alpha=0.55,
                label="← left panel")

    ax2.plot(ot, oa, color="steelblue", linewidth=1.4, label="Orbit-mean altitude")
    ax2.plot(ot, trend(ot), "--", color="tomato", linewidth=1.8,
             label=f"SGP4 drag trend: {coeffs[0]:.3f} km/day")

    # ΔV equivalent annotation
    a_m_mid    = (oa[len(oa)//2] + 6371.0) * 1e3
    v_ms_mid   = math.sqrt(398600.4418e9 / a_m_mid)
    dv_per_day = abs(coeffs[0]) * v_ms_mid * 1e3 / (2.0 * a_m_mid)  # m/s/day
    ax2.text(0.97, 0.05,
             f"≈ {dv_per_day:.3f} m/s/day drag cost",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=8.5, color="tomato",
             bbox=dict(facecolor="white", edgecolor="#dddddd", alpha=0.85, pad=3))

    ax2.set_xlabel("Time from epoch (days)", fontsize=11)
    ax2.set_ylabel("Orbit-mean altitude — |r| − Rₑ  (km)", fontsize=11)
    ax2.set_title(
        f"(b)  7-day orbit-averaged decay\n"
        f"Decay: {decay:.2f} km  ·  Rate: {abs(coeffs[0]):.3f} km/day",
        fontsize=10,
    )
    ax2.legend(fontsize=9.5, framealpha=0.85, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.35)

    # Provenance box (bottom-left of right panel)
    jd_epoch   = sat.jdsatepoch + sat.jdsatepochF
    epoch_prov = (datetime(2000, 1, 1, 12, 0, 0)
                  + timedelta(days=jd_epoch - 2451545.0))
    ax2.text(
        0.01, 0.02,
        f"NORAD {sat.satnum}  ·  Epoch {epoch_prov.strftime('%Y-%m-%d %H:%M UTC')}"
        f"  ·  B* {sat.bstar:.2e}  ·  SGP4 / python-sgp4  ·  60 s steps",
        transform=ax2.transAxes,
        fontsize=7, color="#777777", va="bottom",
        family="monospace",
    )

    plt.tight_layout()
    if show:
        out = figure_path(run_id, "altitude")
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved → {out}")
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Standalone entry point (fetches its own TLE)
# ---------------------------------------------------------------------------
def main():
    print("Fetching TLE from CelesTrak...")
    result = fetch_first_tle(CELESTRAK_URL)
    if result is None:
        print("  Using fallback demo TLE.")
        name, line1, line2 = FALLBACK_TLE
    else:
        name, line1, line2 = result
        print(f"  Found: {name}")
    run(name, line1, line2)


if __name__ == "__main__":
    main()
