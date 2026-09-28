"""
Custom Numerical Propagator
---------------------------
Integrates the equations of motion in ECI using scipy RK45.

Forces modelled:
  1. Two-body gravity          — dominant force
  2. J2 oblateness perturbation — Earth's equatorial bulge; drives RAAN precession
  3. Atmospheric drag           — piecewise exponential atmosphere (USSA76, mean
                                  solar activity); main LEO decay driver

Usage
-----
  from propagator import propagate, decay_rate, comparison_figure

  t, h = propagate(line1, line2, days=7, Cd=2.2, Am=0.010)
  rate = decay_rate(line1, line2, Cd=2.2, Am=0.010)  # km/day
  fig  = comparison_figure(name, line1, line2, Cd=2.2, Am=0.010)

Notes on Cd and A/m
--------------------
Cd (drag coefficient) ≈ 2.2 for most satellites in free-molecular flow.
Am (area-to-mass ratio, m²/kg) depends on satellite geometry:
  - Large microsat  (40 kg, ~0.4 m² mean area)  → Am ≈ 0.010
  - Small CubeSat   ( 5 kg, ~0.03 m² mean area) → Am ≈ 0.006
  - ISS             (420 t, ~2500 m² area)        → Am ≈ 0.006

Unlike SGP4, this propagator does NOT use the TLE B* term.  Instead Cd and Am
are set explicitly.  Tuning Am until the decay rate matches SGP4 gives the
satellite's effective ballistic coefficient at current solar conditions.
"""

import math
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sgp4.api import Satrec, jday


# ---------------------------------------------------------------------------
# Physical constants  (SI throughout)
# ---------------------------------------------------------------------------
MU      = 3.986004418e14    # m³/s²  gravitational parameter
RE_EQ   = 6.378137e6        # m      Earth equatorial radius  (for J2)
RE_MEAN = 6.371e6           # m      Earth mean radius        (for altitude)
J2      = 1.08262668e-3     # J2 oblateness coefficient
OMEGA_E = 7.2921150e-5      # rad/s  Earth sidereal rotation rate


# ---------------------------------------------------------------------------
# Atmosphere model  — piecewise exponential, USSA76, mean solar activity
# Columns: base altitude (km), density at base (kg/m³), scale height (km)
# ---------------------------------------------------------------------------
_ATM = np.array([
    [200, 2.789e-10, 26.3],
    [250, 7.248e-11, 25.5],
    [300, 2.418e-11, 26.3],
    [350, 9.158e-12, 27.6],
    [400, 3.725e-12, 29.1],
    [450, 1.585e-12, 30.8],
    [500, 6.967e-13, 33.2],
    [550, 1.454e-13, 38.8],
    [600, 3.614e-14, 46.2],
    [650, 6.967e-15, 53.3],
    [700, 1.454e-15, 53.3],
], dtype=float)


def atm_density(alt_m: float) -> float:
    """
    Atmospheric density (kg/m³) at altitude alt_m (metres).
    Piecewise exponential: within each layer rho = rho_ref * exp(-(h-h0)/H).
    Returns ~0 for altitudes below the table minimum.
    """
    alt_km = alt_m / 1e3
    idx = int(np.searchsorted(_ATM[:, 0], alt_km, side="right")) - 1
    idx = max(0, min(idx, len(_ATM) - 1))
    h0, rho0, H = _ATM[idx]
    return rho0 * math.exp(-(alt_km - h0) / H)


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------
def _eom(t: float, state: list, Cd_Am: float) -> list:
    """
    Equations of motion in ECI frame.  SI units (m, m/s, s).

    State vector: [x, y, z, vx, vy, vz]
    Cd_Am:  Cd * A/m  (m²/kg) — combined drag parameter

    Returns: [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state

    r2 = x*x + y*y + z*z
    r  = math.sqrt(r2)
    r3 = r2 * r
    r5 = r2 * r3

    # ── Two-body ──────────────────────────────────────────────────────────
    c   = -MU / r3
    ax  = c * x
    ay  = c * y
    az  = c * z

    # ── J2 perturbation ───────────────────────────────────────────────────
    # a_J2 from the gradient of the J2 gravitational potential (Vallado eq 8-20)
    fac  = 1.5 * J2 * MU * RE_EQ**2 / r5
    zr2  = (z / r) ** 2
    ax  += fac * x * (5.0 * zr2 - 1.0)
    ay  += fac * y * (5.0 * zr2 - 1.0)
    az  += fac * z * (5.0 * zr2 - 3.0)

    # ── Atmospheric drag ──────────────────────────────────────────────────
    # Velocity relative to the rotating atmosphere  (omega x r subtracted)
    vrx = vx + OMEGA_E * y
    vry = vy - OMEGA_E * x
    vrz = vz
    vr  = math.sqrt(vrx*vrx + vry*vry + vrz*vrz)

    rho      = atm_density(r - RE_MEAN)
    drag_fac = -0.5 * rho * Cd_Am * vr    # multiply by v_rel vector below
    ax      += drag_fac * vrx
    ay      += drag_fac * vry
    az      += drag_fac * vrz

    return [vx, vy, vz, ax, ay, az]


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------
def propagate(
    line1:  str,
    line2:  str,
    days:   float = 7.0,
    step_s: int   = 60,
    Cd:     float = 2.2,
    Am:     float = 0.010,
) -> tuple:
    """
    Integrate equations of motion from TLE epoch.

    SGP4 is used only to get the initial position and velocity at epoch.
    All subsequent integration uses the custom RK45 propagator.

    Parameters
    ----------
    Am : float
        Drag area-to-mass ratio (m²/kg).  Tune this to match the satellite's
        measured decay rate: larger Am → faster decay.

    Returns
    -------
    times_days : np.ndarray   time from epoch (days)
    alts_km    : np.ndarray   altitude above mean Earth radius (km)
    """
    sat      = Satrec.twoline2rv(line1, line2)
    jd_epoch = sat.jdsatepoch + sat.jdsatepochF
    epoch    = (datetime(2000, 1, 1, 12, 0, 0)
                + timedelta(days=jd_epoch - 2451545.0))

    # Initial state from SGP4 at epoch  (km → m,  km/s → m/s)
    jd, fr = jday(epoch.year, epoch.month, epoch.day,
                  epoch.hour, epoch.minute, epoch.second)
    err, r_km, v_kms = sat.sgp4(jd, fr)
    if err != 0:
        raise ValueError(f"SGP4 error at epoch: {err}")

    state0 = [
        r_km[0] * 1e3, r_km[1] * 1e3, r_km[2] * 1e3,
        v_kms[0] * 1e3, v_kms[1] * 1e3, v_kms[2] * 1e3,
    ]

    t_end  = days * 86400.0
    t_eval = np.arange(0.0, t_end + step_s, step_s)

    sol = solve_ivp(
        _eom,
        (0.0, t_end),
        state0,
        method       = "RK45",
        t_eval       = t_eval,
        args         = (Cd * Am,),
        rtol         = 1e-6,
        atol         = 1e-7,
        max_step     = float(step_s),
        dense_output = False,
    )

    radii_m    = np.sqrt(sol.y[0]**2 + sol.y[1]**2 + sol.y[2]**2)
    alts_km    = (radii_m - RE_MEAN) / 1e3
    times_days = sol.t / 86400.0

    return times_days, alts_km


# ---------------------------------------------------------------------------
# Decay rate  (same interface as phase2_stationkeeping.measure_decay_rate)
# ---------------------------------------------------------------------------
def _orbit_average(times: np.ndarray, alts: np.ndarray,
                   period_min: float, step_s: int = 60) -> tuple:
    chunk = max(1, int(round(period_min * 60.0 / step_s)))
    n_orb = len(alts) // chunk
    ot = np.array([np.mean(times[k*chunk:(k+1)*chunk]) for k in range(n_orb)])
    oa = np.array([np.mean(alts [k*chunk:(k+1)*chunk]) for k in range(n_orb)])
    return ot, oa


def decay_rate(
    line1:  str,
    line2:  str,
    days:   float = 7.0,
    step_s: int   = 60,
    Cd:     float = 2.2,
    Am:     float = 0.010,
) -> float:
    """
    Measure decay rate (km/day, positive) from the numerical propagator.
    Linear fit to orbit-averaged altitudes — same method as the SGP4 approach
    in phase2_stationkeeping, but using the custom integrator.
    """
    sat        = Satrec.twoline2rv(line1, line2)
    period_min = 2 * math.pi / sat.no_kozai  # no_kozai in rad/min

    t, h   = propagate(line1, line2, days=days, step_s=step_s, Cd=Cd, Am=Am)
    ot, oa = _orbit_average(t, h, period_min, step_s)
    slope  = np.polyfit(ot, oa, 1)[0]         # km/day  (negative = decaying)
    return max(-slope, 0.005)


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------
def comparison_figure(
    name:   str,
    line1:  str,
    line2:  str,
    days:   float = 7.0,
    Cd:     float = 2.2,
    Am:     float = 0.010,
) -> plt.Figure:
    """
    Two-panel comparison:
      Left  — first 6 orbits raw altitude  (SGP4 vs RK45 overlay)
      Right — 7-day orbit-averaged decay   (both rates annotated)
    """
    step_s = 60

    # ── SGP4 propagation ─────────────────────────────────────────────────
    sat      = Satrec.twoline2rv(line1, line2)
    jd_epoch = sat.jdsatepoch + sat.jdsatepochF
    epoch    = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=jd_epoch - 2451545.0)
    period_min = 2 * math.pi / sat.no_kozai

    n_sgp4 = int(days * 86400 / step_s)
    t_sgp4 = np.empty(n_sgp4)
    h_sgp4 = np.empty(n_sgp4)
    for i in range(n_sgp4):
        dt = epoch + timedelta(seconds=i * step_s)
        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute, dt.second)
        err, r, _ = sat.sgp4(jd, fr)
        t_sgp4[i] = i * step_s / 86400.0
        h_sgp4[i] = (np.nan if err != 0
                     else math.sqrt(r[0]**2 + r[1]**2 + r[2]**2) - 6371.0)

    # ── Custom propagator ─────────────────────────────────────────────────
    t_num, h_num = propagate(line1, line2, days=days,
                             step_s=step_s, Cd=Cd, Am=Am)

    # ── Orbit-averaged decay ──────────────────────────────────────────────
    ot_s, oa_s = _orbit_average(t_sgp4, h_sgp4, period_min, step_s)
    ot_n, oa_n = _orbit_average(t_num,  h_num,  period_min, step_s)

    rate_sgp4 = max(-float(np.polyfit(ot_s, oa_s, 1)[0]), 0.005)
    rate_num  = max(-float(np.polyfit(ot_n, oa_n, 1)[0]), 0.005)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Propagator Comparison — {name}\n"
        f"SGP4 (B*-calibrated)  vs  Custom RK45  (two-body + J2 + drag,  "
        f"Cd = {Cd},  A/m = {Am} m²/kg)",
        fontsize=10, fontweight="bold",
    )

    # Left: raw altitude — first 6 orbits
    zoom_h = 6 * period_min / 60.0
    m_s = t_sgp4 * 24 <= zoom_h
    m_n = t_num  * 24 <= zoom_h
    ax1.plot(t_sgp4[m_s] * 24, h_sgp4[m_s],
             color="#2176ae", lw=1.2, alpha=0.9, label="SGP4")
    ax1.plot(t_num[m_n]  * 24, h_num[m_n],
             color="#ef233c", lw=1.2, alpha=0.85,
             linestyle="--", label="Custom RK45")
    ax1.set_xlabel("Time from epoch (hours)", fontsize=10)
    ax1.set_ylabel("Altitude (km)", fontsize=10)
    ax1.set_title("First 6 orbits — raw altitude", fontsize=10)
    ax1.legend(fontsize=9, framealpha=0.85, loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Right: orbit-averaged decay
    ax2.plot(ot_s, oa_s, color="#2176ae", lw=1.5,
             label=f"SGP4          {rate_sgp4:.3f} km/day")
    ax2.plot(ot_n, oa_n, color="#ef233c", lw=1.5, linestyle="--",
             label=f"Custom RK45   {rate_num:.3f} km/day")

    # Trend lines
    c_s = np.polyfit(ot_s, oa_s, 1)
    c_n = np.polyfit(ot_n, oa_n, 1)
    ax2.plot(ot_s, np.poly1d(c_s)(ot_s), color="#2176ae",
             lw=1.6, linestyle=":", alpha=0.6)
    ax2.plot(ot_n, np.poly1d(c_n)(ot_n), color="#ef233c",
             lw=1.6, linestyle=":", alpha=0.6)

    ax2.set_xlabel("Time from epoch (days)", fontsize=10)
    ax2.set_ylabel("Mean altitude per orbit (km)", fontsize=10)
    ax2.set_title(
        f"7-day orbit-averaged decay\n"
        f"Difference: {abs(rate_sgp4 - rate_num):.4f} km/day  "
        f"({abs(rate_sgp4 - rate_num) / rate_sgp4 * 100:.1f} %)",
        fontsize=10,
    )
    ax2.legend(fontsize=9, framealpha=0.85, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.35)

    # Derive Cd·A/m implied by the TLE's B* term for comparison
    # SGP4 definition: B* = (1/2)(Cd·A/m) × ρ₀  where ρ₀ = 2.461×10⁻⁵ kg/m² per Earth radius
    # Cd·A/m = B* × 2 / ρ₀   (B* in 1/Re, 1 Re = 6378.137 km)
    sat_check   = Satrec.twoline2rv(line1, line2)
    rho0_per_Re = 2.461e-5     # kg/m²/Earth-radius  (SGP4 reference)
    Re_km       = 6378.137     # km
    bstar       = sat_check.bstar                   # 1/Re units
    bstar_si    = bstar / Re_km                     # 1/km → but B* is 1/Earth-radius
    CdAm_bstar  = abs(bstar) * 2.0 / rho0_per_Re   # m²/kg  (approximate)

    match_pct = max(0.0, 100.0 - abs(rate_sgp4 - rate_num) / rate_sgp4 * 100)
    ax2.text(
        0.01, 0.04,
        f"SGP4 B* = {sat_check.bstar:.3e} (1/Re)\n"
        f"  → implied Cd·A/m ≈ {CdAm_bstar:.4f} m²/kg\n"
        f"RK45 assumed Cd·A/m = {Cd * Am:.4f} m²/kg\n"
        f"Decay-rate agreement: {match_pct:.0f}%\n"
        f"Initial state: SGP4 at TLE epoch  ·  Frame: TEME\n"
        f"Atmosphere: USSA76 piecewise exponential (mean solar)\n"
        f"Integrator: RK45  rtol=1e-6  atol=1e-7  max_step=60 s",
        transform=ax2.transAxes,
        fontsize=7.5, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#cccccc", alpha=0.93),
        family="monospace",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    return fig
