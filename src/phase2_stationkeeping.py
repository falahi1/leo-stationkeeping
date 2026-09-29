"""
Phase 2: Stationkeeping Simulation
------------------------------------
Simulates deadband stationkeeping for any LEO satellite from a live TLE.

Physics:
  - Atmospheric drag is modelled as a constant decay rate (km/day) measured
    by fitting a linear trend to 7-day SGP4 orbit-averaged altitudes.
  - When altitude drops to the deadband floor, an impulsive Hohmann raise
    fires immediately.  Delta-V is computed from the exact two-impulse formula.
  - Propellant consumed per burn uses the Tsiolkovsky rocket equation:
      dm = m * (1 - exp(-dv / (Isp * g0)))

Outputs:
  - Staircase altitude plot  (decay segments + instantaneous burn jumps)
  - Maneuver log             (time, dv, propellant consumed, cumulative totals)
  - Summary stats            (total dv, n burns, propellant consumed, decay rate)

Usage:
  run(name, line1, line2, target_alt, half_width, duration, isp, wet_mass,
      raise_to="top", run_id=None, show=True)
  Returns (fig, results_dict).
"""

import math
import os
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sgp4.api import Satrec, jday


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU = 398600.4418   # km³/s²  Earth gravitational parameter
RE = 6371.0        # km      Earth mean radius
G0 = 9.80665       # m/s²   standard gravity


# ---------------------------------------------------------------------------
# Hohmann delta-V  (exact two-impulse, circular-to-circular)
# ---------------------------------------------------------------------------
def hohmann_dv(alt1_km: float, alt2_km: float) -> float:
    """
    Total delta-V (m/s) for a Hohmann transfer between two circular orbits.
    alt1_km, alt2_km: altitudes above Earth's surface in km.
    """
    r1 = (alt1_km + RE) * 1e3   # m
    r2 = (alt2_km + RE) * 1e3   # m
    mu = MU * 1e9                # m³/s²

    v1   = math.sqrt(mu / r1)
    v2   = math.sqrt(mu / r2)
    vt1  = math.sqrt(mu * (2.0 / r1 - 2.0 / (r1 + r2)))
    vt2  = math.sqrt(mu * (2.0 / r2 - 2.0 / (r1 + r2)))

    return (vt1 - v1) + (v2 - vt2)   # m/s


# ---------------------------------------------------------------------------
# Decay rate measurement via SGP4 linear fit
# ---------------------------------------------------------------------------
def measure_decay_rate(line1: str, line2: str,
                       days: float = 7.0, step_s: int = 60) -> float:
    """
    Propagate TLE for `days`, orbit-average the altitude series, fit a line.
    Returns positive decay rate in km/day.
    Minimum floor of 0.005 km/day so the simulation always terminates.
    """
    sat      = Satrec.twoline2rv(line1, line2)
    jd_epoch = sat.jdsatepoch + sat.jdsatepochF
    epoch    = (datetime(2000, 1, 1, 12, 0, 0)
                + timedelta(days=jd_epoch - 2451545.0))

    period_s = (2 * math.pi / sat.no_kozai) * 60.0
    chunk    = max(1, int(round(period_s / step_s)))
    n        = int(days * 86400 / step_s)

    t_arr  = np.empty(n)
    h_arr  = np.empty(n)

    for i in range(n):
        dt      = epoch + timedelta(seconds=i * step_s)
        jd, fr  = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        err, r, _ = sat.sgp4(jd, fr)
        t_arr[i] = i * step_s / 86400.0
        h_arr[i] = (np.nan if err != 0
                    else math.sqrt(r[0]**2 + r[1]**2 + r[2]**2) - RE)

    n_orb = len(h_arr) // chunk
    ot = np.array([np.nanmean(t_arr[k * chunk:(k + 1) * chunk]) for k in range(n_orb)])
    oa = np.array([np.nanmean(h_arr[k * chunk:(k + 1) * chunk]) for k in range(n_orb)])

    slope = np.polyfit(ot, oa, 1)[0]     # km/day  (negative = decaying)
    return max(-slope, 0.005)            # return as positive value


# ---------------------------------------------------------------------------
# Core simulation engine
# ---------------------------------------------------------------------------
def simulate(
    name:         str,
    line1:        str,
    line2:        str,
    target_alt:   float,         # km  — centre of deadband
    half_width:   float,         # km  — deadband = target ± half_width
    duration:     float,         # days
    isp:          float,         # s   — specific impulse
    wet_mass:     float,         # kg  — spacecraft mass at start of simulation
    raise_to:             str   = "top", # "top" → raise to upper limit; "target" → raise to centre
    thrust_n:             float = 1.0,   # N   — thruster force (used for burn duration only)
    solar_factor:         float = 1.0,   # ×   — scales measured decay rate (0.3 = min, 1 = mean, 3 = max)
    decay_rate_override:  float = None,  # km/day — if set, skips internal measurement (use history fit)
) -> dict:
    """
    Simulate stationkeeping for `duration` days.
    Returns a results dict with time series, maneuver log, and summary stats.
    """
    if decay_rate_override is not None:
        decay_rate = decay_rate_override * solar_factor
    else:
        decay_rate = measure_decay_rate(line1, line2) * solar_factor

    lower = target_alt - half_width
    upper = target_alt + half_width
    alt   = upper         # start at top of band (freshly raised)
    t     = 0.0
    mass  = wet_mass
    total_dv = 0.0

    ts        = [0.0]
    alts      = [alt]
    maneuvers = []

    while t < duration:
        # Time until altitude hits the lower limit
        days_to_trigger = (alt - lower) / decay_rate

        if t + days_to_trigger >= duration:
            # Simulation ends before the next trigger — just decay to end
            end_alt = alt - decay_rate * (duration - t)
            ts.append(duration)
            alts.append(max(end_alt, lower - 2))   # allow slight overshoot visually
            break

        t_trig = t + days_to_trigger

        # Decay segment: current altitude → lower limit
        ts.append(t_trig)
        alts.append(lower)

        # Choose raise target
        alt_raise = upper if raise_to == "top" else target_alt

        # Hohmann delta-V and propellant
        dv       = hohmann_dv(lower, alt_raise)               # m/s
        v_e      = isp * G0                                   # exhaust velocity (m/s)
        dm       = mass * (1.0 - math.exp(-dv / v_e))         # kg consumed
        t_burn_s = dm * v_e / thrust_n                        # s — finite burn duration
        mass = max(mass - dm, 0.0)
        total_dv += dv

        maneuvers.append({
            "Day":             round(t_trig, 1),
            "From (km)":       round(lower, 2),
            "To (km)":         round(alt_raise, 2),
            "ΔV (m/s)":        round(dv, 3),
            "Burn dur. (s)":   int(round(t_burn_s, 0)),
            "Propellant (kg)": round(dm, 4),
            "Mass after (kg)": round(mass, 3),
            "Cumul. ΔV (m/s)": round(total_dv, 3),
        })

        # Instantaneous burn — vertical jump in the plot
        ts.append(t_trig)
        alts.append(alt_raise)

        alt = alt_raise
        t   = t_trig

        if mass <= 0:
            break

    # Steady-state annual delta-V from decay rate (independent of simulation window).
    # Each km of altitude raised costs dv_per_km = v/(2a) m/s  (vis-viva linearised).
    # dv/yr = decay_rate [km/day] × 365.25 [day/yr] × dv_per_km [m/s per km]
    a_m        = (target_alt + RE) * 1e3          # semi-major axis (m)
    v_ms       = math.sqrt(MU * 1e9 / a_m)        # circular orbital speed (m/s)
    dv_per_km  = v_ms * 1e3 / (2.0 * a_m)         # m/s per km of altitude raised
    dv_per_year = decay_rate * 365.25 * dv_per_km  # m/s/yr

    # Propellant consumed
    prop_consumed = wet_mass - mass

    # Estimated mission life — how many more days at current rate until 10 % of
    # wet mass (a rough propellant fraction for small LEO sats) is used up.
    prop_budget = wet_mass * 0.10
    days_budget = (prop_budget / prop_consumed * duration) if prop_consumed > 0 else float("inf")

    avg_burn_dur_s = (
        sum(m["Burn dur. (s)"] for m in maneuvers) / len(maneuvers)
        if maneuvers else 0.0
    )

    return {
        "ts":             np.array(ts),
        "alts":           np.array(alts),
        "maneuvers":      maneuvers,
        "decay_rate":     decay_rate,
        "total_dv_ms":    total_dv,
        "dv_per_year_ms": dv_per_year,
        "n_maneuvers":    len(maneuvers),
        "prop_consumed":  prop_consumed,
        "final_mass":     mass,
        "days_budget":    days_budget,
        "target_alt":     target_alt,
        "lower":          lower,
        "upper":          upper,
        "duration":       duration,
        "isp":            isp,
        "wet_mass":       wet_mass,
        "thrust_n":       thrust_n,
        "solar_factor":   solar_factor,
        "avg_burn_dur_s": avg_burn_dur_s,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_results(name: str, res: dict) -> plt.Figure:
    """Build and return the stationkeeping staircase figure."""
    ts   = res["ts"]
    alts = res["alts"]
    mvrs = res["maneuvers"]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # ── Deadband shading ──────────────────────────────────────────────────
    ax.axhspan(res["lower"], res["upper"],
               color="#cce5f0", alpha=0.30, zorder=1)
    ax.axhline(res["upper"],  color="#023e8a", linewidth=1.0,
               linestyle="--", alpha=0.8, zorder=2)
    ax.axhline(res["target_alt"], color="#444444", linewidth=1.2,
               linestyle="-",  alpha=0.75, zorder=2)
    ax.axhline(res["lower"],  color="#023e8a", linewidth=1.0,
               linestyle="--", alpha=0.8, zorder=2)

    # Direct-label the limit lines at the right edge
    x_lbl = res["duration"] * 0.985
    ax.text(x_lbl, res["upper"],      f"{res['upper']:.1f} km",
            va="bottom", ha="right", fontsize=8, color="#023e8a", alpha=0.85)
    ax.text(x_lbl, res["target_alt"], f"target {res['target_alt']:.1f} km",
            va="center", ha="right", fontsize=8, color="#444444",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))
    ax.text(x_lbl, res["lower"],      f"{res['lower']:.1f} km",
            va="top",    ha="right", fontsize=8, color="#023e8a", alpha=0.85)

    # ── Staircase line ────────────────────────────────────────────────────
    ax.plot(ts, alts, color="#023e8a", linewidth=1.6, zorder=3)

    # ── Burn markers ─────────────────────────────────────────────────────
    if mvrs:
        burn_days = [m["Day"] for m in mvrs]
        burn_from = [m["From (km)"] for m in mvrs]
        ax.scatter(burn_days, burn_from, color="#ef233c", s=45,
                   zorder=5, label="Maneuver")

        # Annotate first burn with ΔV + burn duration — label all if ≤10, else first only
        label_count = len(mvrs) if len(mvrs) <= 10 else 1
        for k, m in enumerate(mvrs[:label_count]):
            suffix = " (each)" if k == 0 and len(mvrs) > 1 else ""
            ax.annotate(
                f"Δv = {m['ΔV (m/s)']:.2f} m/s{suffix}\n{m['Burn dur. (s)']} s",
                xy=(m["Day"], m["From (km)"]),
                xytext=(0, -24),
                textcoords="offset points",
                fontsize=7.5, color="#ef233c", ha="center",
                arrowprops=dict(arrowstyle="-", color="#ef233c",
                               lw=0.6, alpha=0.6),
            )

    # ── Axes formatting ───────────────────────────────────────────────────
    margin = max(res["upper"] - res["lower"], 2) * 1.6
    ax.set_ylim(res["lower"] - margin * 0.35,
                res["upper"] + margin * 0.5)
    ax.set_xlim(0, res["duration"])
    ax.set_xlabel("Time from epoch (days)", fontsize=11)
    ax.set_ylabel("Mean altitude — |r| − Rₑ  (km)", fontsize=11)
    ax.set_title(
        f"Stationkeeping Simulation — {name}",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.30, color="#888888")
    # Minimal legend: only burn marker
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    # ── Summary stats text box ────────────────────────────────────────────
    prop_str = (f"{res['prop_consumed']:.3f} kg"
                if res['prop_consumed'] < 1
                else f"{res['prop_consumed']:.2f} kg")
    a_m       = (res["target_alt"] + 6371.0) * 1e3
    v_ms      = math.sqrt(398600.4418e9 / a_m)
    dv_per_km = v_ms * 1e3 / (2.0 * a_m)
    sf = res["solar_factor"]
    solar_tag = ("solar max" if sf > 1.5 else "solar min" if sf < 0.7 else "mean solar")
    stats = (
        f"Decay rate   {res['decay_rate']:.3f} km/day  [{sf:.1f}× {solar_tag}]\n"
        f"             ({dv_per_km * res['decay_rate']:.3f} m/s/day)\n"
        f"Maneuvers    {res['n_maneuvers']}\n"
        f"Total ΔV     {res['total_dv_ms']:.2f} m/s\n"
        f"ΔV/year      {res['dv_per_year_ms']:.1f} m/s/yr  (steady-state)\n"
        f"Propellant   {prop_str}\n"
        f"Isp          {res['isp']:.0f} s  ·  Wet mass {res['wet_mass']:.0f} kg\n"
        f"Thrust       {res['thrust_n']:.2f} N  ·  Burn dur {res['avg_burn_dur_s']:.0f} s/burn\n"
        f"Model: constant decay rate (30-day TLE history or 7-day SGP4 fit)"
    )
    ax.text(
        0.01, 0.98, stats,
        transform=ax.transAxes,
        fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#cccccc", alpha=0.92),
        family="monospace",
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure path helper
# ---------------------------------------------------------------------------
def figure_path(run_id: str, folder: str = "figures") -> str:
    if run_id:
        return f"{folder}/{run_id}_stationkeeping.png"
    v = 1
    while os.path.exists(f"{folder}/phase2_stationkeeping_v{v}.png"):
        v += 1
    return f"{folder}/phase2_stationkeeping_v{v}.png"


# ---------------------------------------------------------------------------
# Public run() — called by app.py and standalone main()
# ---------------------------------------------------------------------------
def run(
    name:         str,
    line1:        str,
    line2:        str,
    target_alt:   float,
    half_width:   float,
    duration:     float,
    isp:          float,
    wet_mass:     float,
    raise_to:            str   = "top",
    thrust_n:            float = 1.0,
    solar_factor:        float = 1.0,
    decay_rate_override: float = None,
    run_id:              str   = None,
    show:                bool  = True,
):
    """Simulate, plot, optionally save and show. Returns (fig, results)."""
    print(f"\nRunning stationkeeping simulation for {name}...")
    print(f"  Target alt   : {target_alt:.1f} km  ±{half_width:.1f} km")
    print(f"  Duration     : {duration:.0f} days")
    print(f"  Isp          : {isp:.0f} s    Wet mass: {wet_mass:.1f} kg")
    print(f"  Thrust       : {thrust_n:.2f} N    Solar factor: {solar_factor:.1f}×")

    res = simulate(name, line1, line2, target_alt, half_width,
                   duration, isp, wet_mass, raise_to, thrust_n, solar_factor,
                   decay_rate_override)

    print(f"  Decay rate   : {res['decay_rate']:.4f} km/day")
    print(f"  Maneuvers    : {res['n_maneuvers']}")
    print(f"  Total ΔV     : {res['total_dv_ms']:.3f} m/s")
    print(f"  ΔV/year      : {res['dv_per_year_ms']:.2f} m/s/yr")
    print(f"  Prop used    : {res['prop_consumed']:.4f} kg")

    fig = plot_results(name, res)

    if show:
        out = figure_path(run_id)
        os.makedirs("figures", exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"\n  Plot saved → {out}")
        plt.show()

    return fig, res


# ---------------------------------------------------------------------------
# Standalone entry point — uses hardcoded demo values
# ---------------------------------------------------------------------------
def main():
    from phase1_altitude_plot import fetch_first_tle, CELESTRAK_URL, FALLBACK_TLE
    print("Fetching TLE from CelesTrak...")
    result = fetch_first_tle(CELESTRAK_URL)
    if result is None:
        print("  Using fallback TLE.")
        name, line1, line2 = FALLBACK_TLE
    else:
        name, line1, line2 = result
        print(f"  Found: {name}")

    # Defaults match Satellogic ÑuSat (cold gas N₂, ~40 kg, ~437 km SSO)
    run(
        name, line1, line2,
        target_alt=437.0,
        half_width=5.0,
        duration=90.0,
        isp=65.0,
        wet_mass=40.0,
        thrust_n=0.5,
    )


if __name__ == "__main__":
    main()
