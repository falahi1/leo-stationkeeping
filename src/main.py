"""
LEO Stationkeeping — Satellite Analysis Platform
--------------------------------------------------
Interactive launcher: pick a satellite from the registry, fetch its TLE
from CelesTrak, then run altitude decay and/or ground track analysis.

All output figures are named:  {satellite}_{epoch_date}_{plot_type}.png
Example:                        nusat26_20260926_altitude.png
                                nusat26_20260926_groundtrack.png

Usage:
    python src/main.py
"""

import math
import sys
import urllib.request
from datetime import datetime, timedelta

from sgp4.api import Satrec

# Import the run() functions from each phase script
sys.path.insert(0, "src")
from phase1_altitude_plot import run as run_altitude
from phase1_groundtrack   import run as run_groundtrack


# ---------------------------------------------------------------------------
# Satellite registry
# Each entry: display label, CelesTrak URL, name filter function
# ---------------------------------------------------------------------------
SATELLITES = {
    "1": {
        "label":  "Satellogic ÑuSat",
        "url":    "https://celestrak.org/NORAD/elements/gp.php?NAME=NUSAT-&FORMAT=TLE",
        "filter": lambda n: n.upper().startswith("NUSAT-"),
    },
    "2": {
        "label":  "Planet Labs Dove",
        "url":    "https://celestrak.org/NORAD/elements/gp.php?NAME=FLOCK&FORMAT=TLE",
        "filter": lambda n: "FLOCK" in n.upper(),
    },
    "3": {
        "label":  "Spire LEMUR",
        "url":    "https://celestrak.org/NORAD/elements/gp.php?NAME=LEMUR&FORMAT=TLE",
        "filter": lambda n: n.upper().startswith("LEMUR"),
    },
    "4": {
        "label":  "International Space Station",
        "url":    "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=TLE",
        "filter": lambda n: "ISS" in n.upper() or "ZARYA" in n.upper(),
    },
}


# ---------------------------------------------------------------------------
# TLE fetch — returns ALL matching satellites
# ---------------------------------------------------------------------------
def fetch_all_tles(url: str, name_filter) -> list:
    """
    Download TLE data from CelesTrak and return a list of
    (name, line1, line2) tuples for every satellite that passes name_filter.
    Returns an empty list on failure.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8")
        lines   = [ln.strip() for ln in text.splitlines() if ln.strip()]
        matches = []
        for i in range(len(lines) - 2):
            if (lines[i + 1].startswith("1 ")
                    and lines[i + 2].startswith("2 ")
                    and name_filter(lines[i])):
                matches.append((lines[i], lines[i + 1], lines[i + 2]))
        if not matches:
            print("  No matching satellites found in response.")
        return matches
    except Exception as exc:
        print(f"  Fetch failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Run ID  — satellite slug + TLE epoch date
# ---------------------------------------------------------------------------
def make_run_id(name: str, line1: str, line2: str) -> str:
    """
    Build a unique identifier for this TLE fetch.
    'NUSAT-26 (M SOMERVILLE)' + epoch 2026-09-26  →  'nusat26_20260926'
    """
    sat      = Satrec.twoline2rv(line1, line2)
    jd       = sat.jdsatepoch + sat.jdsatepochF
    epoch_dt = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=jd - 2451545.0)
    slug     = name.split()[0].lower().replace("-", "")
    return f"{slug}_{epoch_dt.strftime('%Y%m%d')}"


# ---------------------------------------------------------------------------
# Satellite info summary
# ---------------------------------------------------------------------------
def print_satellite_info(name: str, line1: str, line2: str, run_id: str):
    sat      = Satrec.twoline2rv(line1, line2)
    jd       = sat.jdsatepoch + sat.jdsatepochF
    epoch_dt = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=jd - 2451545.0)

    # Altitude from mean motion
    mu       = 398600.4418                         # km³/s²
    n_rad_s  = sat.no_kozai / 60                   # rad/min → rad/s
    a        = (mu / n_rad_s ** 2) ** (1 / 3)
    alt_km   = a - 6371.0
    period   = 2 * math.pi / sat.no_kozai          # minutes

    print(f"\n  {'Satellite':<14}: {name}")
    print(f"  {'NORAD ID':<14}: {sat.satnum}")
    print(f"  {'Epoch':<14}: {epoch_dt.strftime('%Y-%m-%d  %H:%M UTC')}")
    print(f"  {'Altitude':<14}: ~{alt_km:.0f} km")
    print(f"  {'Period':<14}: {period:.1f} min")
    print(f"  {'Inclination':<14}: {math.degrees(sat.inclo):.2f}°")
    print(f"  {'B* (drag)':<14}: {sat.bstar:.4e}")
    print(f"  {'Run ID':<14}: {run_id}")


# ---------------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------------
def satellite_menu() -> str:
    """Show satellite list and return the chosen key, or 'q' to quit."""
    print("\n" + "=" * 56)
    print("  LEO Stationkeeping — Satellite Analysis Platform")
    print("=" * 56)
    print("\n  Select a satellite:\n")
    for key, entry in SATELLITES.items():
        print(f"    {key}  {entry['label']}")
    print("\n    q  Quit")
    print()
    return input("  Enter choice: ").strip().lower()


def instance_menu(matches: list):
    """
    Show all fetched satellites with altitude and inclination, let the user
    pick one.  Returns (name, line1, line2) or None if user goes back.
    """
    print(f"\n  Found {len(matches)} satellite(s). Select one:\n")
    print(f"    {'#':<4}  {'Name':<36}  {'Alt':>6}  {'Incl':>6}")
    print(f"    {'-'*4}  {'-'*36}  {'-'*6}  {'-'*6}")

    for idx, (name, line1, line2) in enumerate(matches, start=1):
        sat     = Satrec.twoline2rv(line1, line2)
        mu      = 398600.4418
        n_rad_s = sat.no_kozai / 60
        alt_km  = (mu / n_rad_s ** 2) ** (1 / 3) - 6371.0
        incl    = math.degrees(sat.inclo)
        print(f"    {idx:<4}  {name:<36}  {alt_km:>5.0f}km  {incl:>5.1f}°")

    print(f"\n    b  Back")
    print()
    raw = input("  Enter number: ").strip().lower()
    if raw == "b":
        return None
    try:
        idx = int(raw)
        if 1 <= idx <= len(matches):
            return matches[idx - 1]
    except ValueError:
        pass
    print("  Invalid choice.")
    return None


def analysis_menu() -> str:
    """Show analysis options and return the chosen key, or 'b' to go back."""
    print("\n  Select analysis:\n")
    print("    1  Altitude decay  (7-day SGP4 propagation)")
    print("    2  Ground track    (24-hour world map)")
    print("    3  Both")
    print("\n    b  Back to satellite selection")
    print()
    return input("  Enter choice: ").strip().lower()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    while True:
        choice = satellite_menu()

        if choice == "q":
            print("\n  Goodbye.\n")
            break

        if choice not in SATELLITES:
            print("  Invalid choice — try again.")
            continue

        entry = SATELLITES[choice]
        print(f"\n  Fetching TLEs for {entry['label']}...")
        matches = fetch_all_tles(entry["url"], entry["filter"])

        if not matches:
            print("  Could not fetch TLEs. Check your internet connection.")
            continue

        print(f"  Retrieved {len(matches)} satellite(s).")

        # If only one match, use it directly; otherwise let the user pick
        if len(matches) == 1:
            name, line1, line2 = matches[0]
        else:
            result = instance_menu(matches)
            if result is None:
                continue
            name, line1, line2 = result

        run_id = make_run_id(name, line1, line2)
        print_satellite_info(name, line1, line2, run_id)

        while True:
            analysis = analysis_menu()

            if analysis == "b":
                break
            elif analysis == "1":
                run_altitude(name, line1, line2, run_id)
            elif analysis == "2":
                run_groundtrack(name, line1, line2, run_id)
            elif analysis == "3":
                run_altitude(name, line1, line2, run_id)
                run_groundtrack(name, line1, line2, run_id)
                print(f"\n  Both figures saved with prefix: {run_id}")
            else:
                print("  Invalid choice — try again.")
                continue

            again = input("\n  Run another analysis for this satellite? (y/n): ").strip().lower()
            if again != "y":
                break


if __name__ == "__main__":
    main()
