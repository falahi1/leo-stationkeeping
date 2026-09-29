"""
LEO Stationkeeping — Web Interface
------------------------------------
Streamlit application.

Run from the repo root with:
    streamlit run src/app.py
"""

import csv
import io
import json
import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — must precede pyplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from sgp4.api import Satrec

# set_page_config MUST be the very first Streamlit call
st.set_page_config(
    page_title="LEO Stationkeeping",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Password gate (active only when APP_PASSWORD secret is set)
# ---------------------------------------------------------------------------
_secret_pw = st.secrets.get("APP_PASSWORD", "")
if _secret_pw:
    if not st.session_state.get("_authenticated"):
        st.title("LEO Stationkeeping")
        pw = st.text_input("Password", type="password")
        if st.button("Enter"):
            if pw == _secret_pw:
                st.session_state["_authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

# Make sibling scripts importable
sys.path.insert(0, os.path.dirname(__file__))
from phase1_altitude_plot    import run as _run_altitude
from phase1_groundtrack      import run as _run_groundtrack, compute_track as _compute_track
from phase2_stationkeeping   import run as _run_sk
from propagator              import comparison_figure as _compare_propagators


# ---------------------------------------------------------------------------
# Satellite registry
# ---------------------------------------------------------------------------
_CT = "https://celestrak.org/NORAD/elements/gp.php"
# Space-Track GP class — wildcard uses ~~ tilde syntax (~~value~~ = LIKE '%value%')
_ST = "https://www.space-track.org/basicspacedata/query/class/gp"

SATELLITES = {
    "1": {
        "label":    "Satellogic ÑuSat",
        "url":      f"{_CT}?NAME=NUSAT-&FORMAT=TLE",
        "st_query": f"{_ST}/OBJECT_NAME/~~NUSAT-~~/EPOCH/>now-30/FORMAT/json",
        "filter":   lambda n: n.upper().startswith("NUSAT-"),
    },
    "2": {
        "label":    "Planet Labs Dove",
        "url":      f"{_CT}?NAME=FLOCK&FORMAT=TLE",
        "st_query": f"{_ST}/OBJECT_NAME/~~FLOCK~~/EPOCH/>now-30/limit/150/FORMAT/json",
        "filter":   lambda n: "FLOCK" in n.upper(),
    },
    "3": {
        "label":    "Spire LEMUR",
        "url":      f"{_CT}?NAME=LEMUR&FORMAT=TLE",
        "st_query": f"{_ST}/OBJECT_NAME/~~LEMUR~~/EPOCH/>now-30/limit/150/FORMAT/json",
        "filter":   lambda n: n.upper().startswith("LEMUR"),
    },
    "4": {
        "label":    "International Space Station",
        "url":      f"{_CT}?GROUP=stations&FORMAT=TLE",
        "st_query": f"{_ST}/NORAD_CAT_ID/25544/EPOCH/>now-30/FORMAT/json",
        "filter":   lambda n: "ISS" in n.upper() or "ZARYA" in n.upper(),
    },
}


# ---------------------------------------------------------------------------
# Satellite propulsion specs — used as stationkeeping simulation defaults
# Sources: public mission documentation for each constellation
# ---------------------------------------------------------------------------
SATELLITE_SPECS = {
    "1": {  # Satellogic ÑuSat
        "wet_mass_kg": 40.0,
        "isp_s":       65,
        "thrust_n":    0.5,    # cold-gas thruster, estimated ~0.5 N
        "prop_type":   "Cold gas (N₂)",
        "source":      "Satellogic public mission documentation (~40 kg, cold-gas propulsion)",
        # Dimensions ~0.45×0.45×0.70 m  →  surface area ≈ 1.66 m²
        # Mean projected area (random tumble) = SA/4 ≈ 0.41 m²  →  A/m ≈ 0.010 m²/kg
        "est_Am":      0.010,
        "dims":        "0.45 × 0.45 × 0.70 m",
        "mean_area_m2": 0.41,
    },
    "2": {  # Planet Labs Dove
        "wet_mass_kg": 5.8,
        "isp_s":       60,
        "thrust_n":    0.1,    # cold-gas thruster on 3U CubeSat, ~0.1 N
        "prop_type":   "Cold gas (Dove+ / Pelican; early Doves had no propulsion)",
        "source":      "Planet Labs public specifications (~5.8 kg 3U CubeSat)",
        # 3U CubeSat: 0.10×0.10×0.30 m  →  SA ≈ 0.14 m²,  mean area ≈ 0.035 m²
        "est_Am":      0.006,
        "dims":        "0.10 × 0.10 × 0.30 m  (3U)",
        "mean_area_m2": 0.035,
    },
    "3": {  # Spire LEMUR-2
        "wet_mass_kg": 4.5,
        "isp_s":       55,
        "thrust_n":    0.1,    # cold-gas thruster on 3U CubeSat, ~0.1 N
        "prop_type":   "Cold gas",
        "source":      "Spire Global public specifications (~4.5 kg 3U CubeSat)",
        # 3U CubeSat: 0.10×0.10×0.30 m  →  mean area ≈ 0.035 m²
        "est_Am":      0.008,
        "dims":        "0.10 × 0.10 × 0.30 m  (3U)",
        "mean_area_m2": 0.035,
    },
    "4": {  # ISS
        "wet_mass_kg": 420000.0,
        "isp_s":       310,
        "thrust_n":    400.0,  # Progress/Zvezda reboost engines, ~400 N combined
        "prop_type":   "UDMH/N₂O₄  (Zvezda main engines · Progress / Cygnus reboost)",
        "source":      "NASA/Roscosmos ISS operations data (~420 t, bipropellant reboost)",
        # Solar arrays dominate drag area: ~2500 m² effective mean projected area
        "est_Am":      0.006,
        "dims":        "~73 m wingspan (solar arrays)",
        "mean_area_m2": 2500.0,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

_ST_LOGIN    = "https://www.space-track.org/ajaxauth/login"
_ST_BASE     = "https://www.space-track.org/basicspacedata/query/class"
_F107_NOAA   = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"


def _fetch_f107() -> float:
    """
    Fetch current F10.7 solar flux (SFU) from NOAA SWPC.
    Returns 150.0 (mean solar) on any failure.
    """
    try:
        resp = requests.get(_F107_NOAA, timeout=8, headers=_BROWSER_HEADERS)
        resp.raise_for_status()
        data = resp.json()
        if data:
            latest = data[-1]
            for key in ("f10.7", "F10.7", "f107", "radio_flux"):
                if latest.get(key) is not None:
                    return float(latest[key])
    except Exception:
        pass
    return 150.0


def _spacetrack_decay_rate(norad_id: int) -> float | None:
    """
    Measure decay rate (km/day, positive) from 30-day Space-Track TLE history.
    Extracts semi-major axis from each TLE's mean motion, fits linear trend.
    Returns None if Space-Track is unavailable or fewer than 3 TLEs exist.
    """
    user = st.secrets.get("SPACETRACK_USER", "")
    pwd  = st.secrets.get("SPACETRACK_PASS", "")
    if not user or not pwd:
        return None
    try:
        sess = requests.Session()
        sess.headers.update(_BROWSER_HEADERS)
        login = sess.post(_ST_LOGIN, data={"identity": user, "password": pwd}, timeout=20)
        if "Failed" in login.text:
            return None
        url = (
            f"{_ST_BASE}/gp_history/NORAD_CAT_ID/{norad_id}"
            f"/EPOCH/%3Enow-30/orderby/EPOCH%20asc/FORMAT/json"
        )
        resp = sess.get(url, timeout=30)
        resp.raise_for_status()
        records = resp.json()
        if not records or len(records) < 3:
            return None
        mu_km3 = 398600.4418
        altitudes, days = [], []
        t0 = None
        for rec in records:
            try:
                n_rev_day = float(rec["MEAN_MOTION"])
                n_rad_s   = n_rev_day * 2 * math.pi / 86400.0
                a_km      = (mu_km3 / n_rad_s ** 2) ** (1 / 3)
                alt_km    = a_km - 6371.0
                ep_str    = rec["EPOCH"][:19].replace("T", " ")
                ep        = datetime.strptime(ep_str, "%Y-%m-%d %H:%M:%S")
                if t0 is None:
                    t0 = ep
                days.append((ep - t0).total_seconds() / 86400.0)
                altitudes.append(alt_km)
            except Exception:
                continue
        if len(altitudes) < 3:
            return None
        slope = np.polyfit(days, altitudes, 1)[0]   # km/day (negative = decaying)
        return max(-slope, 0.005)
    except Exception:
        return None


def _parse_tles(text: str, name_filter=None) -> list:
    """Parse 3-line (name + L1 + L2) or 2-line (L1 + L2) TLE blocks."""
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.startswith("#")]
    matches = []
    i = 0
    while i < len(lines):
        # 3-line block: name line followed by line-1 and line-2
        if (i + 2 < len(lines)
                and not lines[i].startswith("1 ")
                and not lines[i].startswith("2 ")
                and lines[i + 1].startswith("1 ")
                and lines[i + 2].startswith("2 ")):
            name = lines[i]
            if name_filter is None or name_filter(name):
                matches.append((name, lines[i + 1], lines[i + 2]))
            i += 3
        # 2-line block: line-1 and line-2 with no name (Space-Track FORMAT/tle)
        elif (i + 1 < len(lines)
                and lines[i].startswith("1 ")
                and lines[i + 1].startswith("2 ")):
            norad = lines[i][2:7].strip()
            name  = f"NORAD {norad}"
            matches.append((name, lines[i], lines[i + 1]))
            i += 2
        else:
            i += 1
    return matches


def _parse_tles_json(data: list, name_filter=None) -> list:
    """Parse Space-Track GP JSON records into (name, line1, line2) tuples."""
    matches = []
    for rec in data:
        name  = rec.get("OBJECT_NAME", "").strip()
        line1 = rec.get("TLE_LINE1", "").strip()
        line2 = rec.get("TLE_LINE2", "").strip()
        if not (line1 and line2):
            continue
        if name_filter is None or name_filter(name):
            matches.append((name, line1, line2))
    return matches


def _spacetrack_fetch(st_query: str, name_filter) -> list | None:
    """
    Try Space-Track.org (primary source).
    Returns a non-empty TLE list on success, None on any failure or empty result.
    """
    user = st.secrets.get("SPACETRACK_USER", "")
    pwd  = st.secrets.get("SPACETRACK_PASS", "")
    if not user or not pwd:
        return None
    try:
        sess = requests.Session()
        sess.headers.update(_BROWSER_HEADERS)
        login = sess.post(_ST_LOGIN,
                          data={"identity": user, "password": pwd},
                          timeout=20)
        if "Failed" in login.text:
            st.warning("Space-Track login failed — check SPACETRACK_USER/SPACETRACK_PASS in secrets.")
            return None
        resp = sess.get(st_query, timeout=30)
        resp.raise_for_status()
        raw = resp.text.strip()
        if not raw:
            return None
        # Space-Track FORMAT/json returns a JSON array
        if raw.startswith("["):
            try:
                data = json.loads(raw)
            except Exception:
                return None
            if not data:
                return None
            result = _parse_tles_json(data, name_filter)
            return result if result else None
        # { ... } is an error response from Space-Track
        if raw.startswith("{"):
            return None
        # Legacy plain TLE text (FORMAT/tle)
        result = _parse_tles(raw, name_filter)
        return result if result else None
    except Exception as exc:
        st.warning(f"Space-Track exception: {exc}")
        return None


def fetch_all_tles(url: str, name_filter, st_query: str = "") -> list:
    # 1 — Try Space-Track (authoritative source, works from cloud IPs)
    if st_query:
        result = _spacetrack_fetch(st_query, name_filter)
        if result:
            return result

    # 2 — Try CelesTrak (works locally)
    try:
        sess = requests.Session()
        sess.headers.update(_BROWSER_HEADERS)
        resp = sess.get(url, timeout=30)
        resp.raise_for_status()
        result = _parse_tles(resp.text, name_filter)
        if result:
            return result
        st.error("No satellites matched. The TLE catalogue may be empty for this family.")
        return []
    except Exception as exc:
        has_creds = bool(st.secrets.get("SPACETRACK_USER", ""))
        hint = (
            "Space-Track returned no data — check your credentials in Streamlit secrets."
            if has_creds else
            "Add `SPACETRACK_USER` and `SPACETRACK_PASS` to Streamlit secrets "
            "(free account at space-track.org) to enable the primary source."
        )
        st.error(
            f"Could not fetch TLEs.\n\n"
            f"**Space-Track:** {hint}\n\n"
            f"**CelesTrak:** {exc}\n\n"
            "Or paste a TLE directly using **Enter TLE manually** below."
        )
        return []


def make_run_id(name: str, line1: str, line2: str) -> str:
    sat      = Satrec.twoline2rv(line1, line2)
    jd       = sat.jdsatepoch + sat.jdsatepochF
    epoch_dt = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=jd - 2451545.0)
    slug     = name.split()[0].lower().replace("-", "")
    return f"{slug}_{epoch_dt.strftime('%Y%m%d')}"


def satellite_info(name: str, line1: str, line2: str) -> dict:
    sat      = Satrec.twoline2rv(line1, line2)
    jd       = sat.jdsatepoch + sat.jdsatepochF
    epoch_dt = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=jd - 2451545.0)
    mu       = 398600.4418
    n_rad_s  = sat.no_kozai / 60
    alt_km   = (mu / n_rad_s ** 2) ** (1 / 3) - 6371.0
    a_m       = (alt_km + 6371.0) * 1e3          # semi-major axis in metres
    speed_kms = math.sqrt(398600.4418e9 / a_m) / 1e3  # km/s
    return {
        "norad_id":   sat.satnum,
        "epoch":      epoch_dt.strftime("%Y-%m-%d  %H:%M UTC"),
        "alt_km":     alt_km,
        "period_min": 2 * math.pi / sat.no_kozai,
        "incl_deg":   math.degrees(sat.inclo),
        "bstar":      sat.bstar,
        "eccentricity": sat.ecco,
        "speed_kms":  speed_kms,
    }


def fig_to_bytes(fig) -> bytes:
    """Render matplotlib figure to PNG bytes and close it to free memory."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# About-tab: hardcoded family data
# ---------------------------------------------------------------------------
FAMILY_INFO = {
    "1": {
        "operator":     "Satellogic",
        "country":      "Argentina",
        "mission":      "High-resolution Earth observation",
        "payload":      "Multispectral & hyperspectral pushbroom imager",
        "resolution":   "~1 m optical / ~30 m hyperspectral",
        "orbit_desc":   "Sun-synchronous LEO  ·  ~500 km  ·  i ≈ 97°",
        "constel_size": "~30+ operational microsats (~40 kg each)",
        "description": (
            "Satellogic's ÑuSat constellation delivers sub-metre optical and "
            "hyperspectral imagery from LEO. The name derives from the Spanish / "
            "Quechua word ñu (wildebeest). Each satellite carries a pushbroom "
            "imager and weighs roughly 40–45 kg. The sun-synchronous orbit "
            "provides consistent solar illumination for imaging and keeps the "
            "local crossing time fixed across seasons via J2 RAAN precession."
        ),
    },
    "2": {
        "operator":     "Planet Labs PBC",
        "country":      "United States",
        "mission":      "Daily global optical imaging",
        "payload":      "RGB + NIR pushbroom imager",
        "resolution":   "~3 m",
        "orbit_desc":   "Sun-synchronous LEO  ·  ~475–520 km  ·  i ≈ 97–98°",
        "constel_size": "~180+ active Doves (3U–6U CubeSats, ~5 kg each)",
        "description": (
            "Planet's Dove (Flock) constellation is one of the largest "
            "commercial Earth-observation fleets ever deployed. The sheer number "
            "of satellites enables near-daily global revisit at 3 m resolution. "
            "Doves are regularly replenished via rideshare launches. Because of "
            "their small ballistic coefficient, atmospheric drag is the dominant "
            "perturbation — decay rates are high and stationkeeping propellant is "
            "a critical mission-lifetime driver."
        ),
    },
    "3": {
        "operator":     "Spire Global",
        "country":      "United States",
        "mission":      "Weather forecasting · maritime & aviation tracking",
        "payload":      "GNSS-RO receiver + AIS / ADS-B receiver",
        "resolution":   "N/A  (radio occultation & signals intelligence)",
        "orbit_desc":   "Multi-plane LEO  ·  400–600 km  ·  various inclinations",
        "constel_size": "~100+ LEMUR-2 CubeSats (~5 kg each)",
        "description": (
            "Spire's LEMUR-2 satellites use GPS radio occultation (GNSS-RO) to "
            "profile atmospheric temperature, humidity, and pressure — feeding "
            "global numerical weather prediction models. Additional payloads "
            "track ships (AIS) and aircraft (ADS-B). The constellation is spread "
            "across multiple orbital planes to achieve rapid global coverage."
        ),
    },
    "4": {
        "operator":     "NASA · Roscosmos · ESA · JAXA · CSA",
        "country":      "Multinational (15 nations)",
        "mission":      "Crewed microgravity research platform",
        "payload":      "Hundreds of science experiments across 8 pressurised modules",
        "resolution":   "N/A",
        "orbit_desc":   "ISS orbit  ·  ~408 km  ·  i = 51.6°",
        "constel_size": "Single station  (~420 tonnes, 109 m truss span)",
        "description": (
            "The International Space Station is the largest artificial structure "
            "in orbit, continuously crewed since November 2000. At i = 51.6° the "
            "orbit is accessible from both Baikonur and Kennedy Space Center. "
            "Atmospheric drag at ~400 km requires regular reboosts — performed by "
            "onboard thrusters and visiting vehicles (Progress, Cygnus) — to "
            "maintain altitude."
        ),
    },
}

OPS_STATUS_LABELS = {
    "+": ("Operational",            "#00ff88"),
    "-": ("Non-operational",        "#ff4444"),
    "P": ("Partially operational",  "#ffaa00"),
    "B": ("Standby",                "#aaaaff"),
    "S": ("Spare",                  "#88aaff"),
    "X": ("Extended mission",       "#00ccff"),
    "D": ("Debris",                 "#888888"),
    "U": ("Unknown",                "#666666"),
}

OWNER_LABELS = {
    "ARG": "Argentina",  "AUS": "Australia",  "BEL": "Belgium",
    "BRAZ":"Brazil",     "CA":  "Canada",     "CIS": "Russia",
    "COL": "Colombia",   "DEN": "Denmark",    "ECU": "Ecuador",
    "ESRO":"ESA",        "FR":  "France",     "GER": "Germany",
    "IND": "India",      "INDO":"Indonesia",  "IRID":"Iridium",
    "ISRA":"Israel",     "IT":  "Italy",      "ITSO":"INTELSAT",
    "JPN": "Japan",      "KAZ": "Kazakhstan", "MULT":"Multinational",
    "NATO":"NATO",       "NETH":"Netherlands","NOR": "Norway",
    "NZ":  "New Zealand","PRC": "China",      "ROC": "Taiwan",
    "RUS": "Russia",     "SAFR":"South Africa","SAUD":"Saudi Arabia",
    "SPN": "Spain",      "SWED":"Sweden",     "SWTZ":"Switzerland",
    "THAI":"Thailand",   "TUR": "Turkey",     "UAE": "UAE",
    "UK":  "United Kingdom","UKR":"Ukraine",  "US":  "United States",
}

LAUNCH_SITE_LABELS = {
    "AFETR":"Cape Canaveral AFS, USA",
    "AFWTR":"Vandenberg SFB, USA",
    "FRGUI":"Kourou, French Guiana",
    "JSC":  "Jiuquan SC, China",
    "KESC": "Kennedy SC, USA",
    "MGINI":"Mahia Peninsula, NZ (Rocket Lab)",
    "PLSET":"Plesetsk, Russia",
    "RLLC": "Rocket Lab LC-1, NZ",
    "SHAR": "Sriharikota, India",
    "TANSC":"Tanegashima, Japan",
    "TASC": "Taiyuan SC, China",
    "TTMTR":"Baikonur, Kazakhstan",
    "WLPIS":"Wallops Island, USA",
    "WSC":  "Wenchang SC, China",
    "XISC": "Xichang SC, China",
}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_satcat() -> dict:
    """Download CelesTrak SATCAT CSV. Returns dict keyed by NORAD ID (int)."""
    url = "https://celestrak.org/pub/satcat.csv"
    try:
        sess = requests.Session()
        sess.headers.update(_BROWSER_HEADERS)
        resp = sess.get(url, timeout=45)
        resp.raise_for_status()
        reader = csv.DictReader(resp.text.splitlines())
        result = {}
        for row in reader:
            try:
                norad_id = int(row.get("NORAD_CAT_ID", "").strip())
                result[norad_id] = row
            except ValueError:
                continue
        return result
    except Exception:
        return {}



def _spacetrack_satcat(norad_id: int) -> dict:
    """
    Fetch one SATCAT record from Space-Track for a given NORAD ID.
    Returns a dict with CelesTrak-style keys, or {} on any failure.
    """
    user = st.secrets.get("SPACETRACK_USER", "")
    pwd  = st.secrets.get("SPACETRACK_PASS", "")
    if not user or not pwd:
        return {}
    try:
        sess = requests.Session()
        sess.headers.update(_BROWSER_HEADERS)
        login = sess.post(_ST_LOGIN, data={"identity": user, "password": pwd}, timeout=20)
        if "Failed" in login.text:
            return {}
        url = (
            f"https://www.space-track.org/basicspacedata/query/class/satcat"
            f"/NORAD_CAT_ID/{norad_id}/FORMAT/json"
        )
        resp = sess.get(url, timeout=20)
        resp.raise_for_status()
        data = json.loads(resp.text)
        if not data:
            return {}
        rec = data[0]
        # Normalise to the CelesTrak-style keys used by the About tab
        return {
            "OBJECT_ID":       rec.get("INTLDES", ""),
            "LAUNCH_DATE":     rec.get("LAUNCH", ""),
            "DECAY_DATE":      rec.get("DECAY", "") or "",
            "OWNER":           rec.get("COUNTRY", ""),
            "LAUNCH_SITE":     rec.get("SITE", ""),
            "OPS_STATUS_CODE": (
                "+" if rec.get("CURRENT") == "Y" and not rec.get("DECAY")
                else "-" if rec.get("DECAY")
                else "U"
            ),
            "OBJECT_TYPE":     rec.get("OBJECT_TYPE", ""),
            "RCS_SIZE":        rec.get("RCS_SIZE", ""),
        }
    except Exception:
        return {}


def classify_orbit(alt_km: float, incl_deg: float) -> str:
    if 95 <= incl_deg <= 100:
        return f"Sun-Synchronous LEO  ({alt_km:.0f} km, i = {incl_deg:.1f}°)"
    elif incl_deg >= 80:
        return f"Polar LEO  ({alt_km:.0f} km, i = {incl_deg:.1f}°)"
    elif 50 <= incl_deg <= 56:
        return f"ISS-inclined LEO  ({alt_km:.0f} km, i = {incl_deg:.1f}°)"
    elif incl_deg <= 10:
        return f"Equatorial LEO  ({alt_km:.0f} km, i = {incl_deg:.1f}°)"
    else:
        return f"Inclined LEO  ({alt_km:.0f} km, i = {incl_deg:.1f}°)"


def _kv_row(label: str, value: str) -> str:
    """Return one HTML table row for the info card."""
    return (
        f"<tr>"
        f"<td style='padding:5px 14px 5px 0;color:#5c6280;font-size:0.78rem;"
        f"white-space:nowrap;font-weight:500;text-transform:uppercase;"
        f"letter-spacing:0.7px;'>{label}</td>"
        f"<td style='padding:5px 0;color:#c9cdd8;font-size:0.85rem;'>{value}</td>"
        f"</tr>"
    )


# ---------------------------------------------------------------------------
# Custom theme — dark background, Space Grotesk font, electric-blue accents
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Main background ── */
.stApp {
    background-color: #0b0c10;
    color: #e8eaf0;
}

/* ── Headings ── */
h1 { font-size: 1.9rem !important; font-weight: 700 !important; color: #ffffff !important; letter-spacing: -0.5px; }
h2 { font-size: 1.3rem !important; font-weight: 600 !important; color: #ffffff !important; }
h3 { font-size: 1.05rem !important; font-weight: 500 !important; color: #c9cdd8 !important; }

/* ── Body text ── */
p, li, label, .stMarkdown { color: #a8adc0 !important; font-size: 0.88rem !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background-color: #13151f;
    border: 1px solid #1e2130;
    border-radius: 8px;
    padding: 14px 18px !important;
}
[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #5c6280 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #00b4d8 !important;
}

/* ── Primary buttons ── */
.stButton > button[kind="primary"] {
    background-color: #00b4d8 !important;
    color: #0b0c10 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.3px;
    padding: 0.45rem 1.1rem !important;
    transition: background-color 0.15s ease;
}
.stButton > button[kind="primary"]:hover {
    background-color: #0096c7 !important;
}

/* ── Secondary / plain buttons ── */
.stButton > button {
    background-color: #13151f !important;
    color: #c9cdd8 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 6px !important;
    font-size: 0.83rem !important;
}
.stButton > button:hover {
    border-color: #00b4d8 !important;
    color: #00b4d8 !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background-color: #13151f !important;
    color: #00b4d8 !important;
    border: 1px solid #00b4d8 !important;
    border-radius: 6px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid #1e2130 !important;
    gap: 4px;
}
[data-baseweb="tab"] {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #5c6280 !important;
    padding: 8px 18px !important;
    border-radius: 6px 6px 0 0 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #00b4d8 !important;
    border-bottom: 2px solid #00b4d8 !important;
    background-color: transparent !important;
}

/* ── Selectbox / inputs ── */
[data-baseweb="select"] {
    background-color: #13151f !important;
    border-color: #1e2130 !important;
    border-radius: 6px !important;
}
[data-baseweb="select"] * { color: #c9cdd8 !important; font-size: 0.85rem !important; }

/* ── Divider ── */
hr { border-color: #1e2130 !important; }

/* ── Caption / small text ── */
.stCaption, small { color: #3d4260 !important; font-size: 0.75rem !important; }

/* ── Success / error alerts ── */
[data-testid="stAlert"] { border-radius: 6px !important; font-size: 0.83rem !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00b4d8 !important; }

/* ── Plot placeholder ── */
.plot-placeholder {
    height: 260px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px dashed #1e2130;
    border-radius: 8px;
    color: #2e3350;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Hide sidebar entirely ── */
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
section[data-testid="stSidebar"]          { display: none !important; }

/* ── Sliders ── */
[data-testid="stSlider"] {
    padding-top: 0.2rem;
    padding-bottom: 0.6rem;
}
[data-testid="stSlider"] label {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #5c6280 !important;
}

/* ── Tighten page padding for more usable vertical space ── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in {
    "matches":           [],
    "current_run_id":    None,
    "altitude_bytes":    None,
    "groundtrack_bytes": None,
    "globe_data":        None,
    "sk_bytes":          None,
    "sk_results":        None,
    "sk_last_params":    None,
    "compare_bytes":     None,
    "f107_value":        None,   # fetched from NOAA on first use
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Page header — always visible, no sidebar
# ---------------------------------------------------------------------------
st.markdown(
    "<p style='font-size:0.65rem;letter-spacing:2.5px;text-transform:uppercase;"
    "color:#3d4260;margin:0 0 2px 0;'>LEO Stationkeeping</p>"
    "<p style='font-size:1.6rem;font-weight:700;color:#ffffff;margin:0;'>"
    "Satellite Analysis Platform</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Satellite selection — inline, full width
# ---------------------------------------------------------------------------
fam_col, fetch_col = st.columns([4, 1])
with fam_col:
    family_key = st.selectbox(
        "Satellite Family",
        options=list(SATELLITES.keys()),
        format_func=lambda k: SATELLITES[k]["label"],
    )
with fetch_col:
    st.markdown("<div style='margin-top:27px'></div>", unsafe_allow_html=True)
    if st.button("Fetch Satellites", type="primary", use_container_width=True):
        entry = SATELLITES[family_key]
        with st.spinner("Fetching TLEs..."):
            matches = fetch_all_tles(entry["url"], entry["filter"], entry["st_query"])
        if matches:
            st.session_state.matches           = matches
            st.session_state.current_run_id    = None
            st.session_state.altitude_bytes    = None
            st.session_state.groundtrack_bytes = None
            st.session_state.globe_data        = None
            st.session_state.sk_bytes          = None
            st.session_state.sk_results        = None
            st.success(f"{len(matches)} satellite(s) found.")
        else:
            st.info("No satellites returned — try the manual entry below.")

# ---------------------------------------------------------------------------
# Manual TLE entry
# ---------------------------------------------------------------------------
with st.expander("Or enter a TLE manually"):
    st.caption(
        "Paste any 3-line TLE from [CelesTrak](https://celestrak.org) or "
        "[Space-Track](https://www.space-track.org). "
        "The name line is optional — leave it blank to use 'Custom Satellite'."
    )
    tle_name = st.text_input("Satellite name (optional)", key="manual_name",
                             placeholder="e.g. NUSAT-26 (M SOMERVILLE)")
    tle_l1   = st.text_input("TLE Line 1", key="manual_l1",
                             placeholder="1 52184U 22033AD  26269.60424588 ...")
    tle_l2   = st.text_input("TLE Line 2", key="manual_l2",
                             placeholder="2 52184  97.2277 348.1020 ...")
    if st.button("Load TLE", type="primary"):
        l1 = tle_l1.strip()
        l2 = tle_l2.strip()
        nm = tle_name.strip() or "Custom Satellite"
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            st.error("Line 1 must start with '1 ' and Line 2 with '2 '. Check your paste.")
        else:
            try:
                Satrec.twoline2rv(l1, l2)   # validates the TLE
                st.session_state.matches           = [(nm, l1, l2)]
                st.session_state.current_run_id    = None
                st.session_state.altitude_bytes    = None
                st.session_state.groundtrack_bytes = None
                st.session_state.globe_data        = None
                st.session_state.sk_bytes          = None
                st.session_state.sk_results        = None
                st.session_state.sk_last_params    = None
                st.session_state.compare_bytes     = None
                st.success(f"Loaded: {nm}")
                st.rerun()
            except Exception as exc:
                st.error(f"TLE parsing failed: {exc}")

if st.session_state.matches:
    # Filter out satellites with clearly bad TLEs (below 300 km = already deorbiting)
    valid_matches = []
    for m in st.session_state.matches:
        try:
            _s = Satrec.twoline2rv(m[1], m[2])
            _n = _s.no_kozai / 60          # rad/s
            _a = (398600.4418 / _n**2)**(1/3)
            if _a - 6371.0 >= 300:
                valid_matches.append(m)
        except Exception:
            pass
    if not valid_matches:
        valid_matches = st.session_state.matches  # fallback: show all if filter removes everything
    names         = [m[0] for m in valid_matches]
    selected_name = st.selectbox(
        f"Select satellite  ({len(names)} found)",
        options=names,
    )
    match          = next(m for m in valid_matches
                          if m[0] == selected_name)
    name, line1, line2 = match
    run_id         = make_run_id(name, line1, line2)

    if run_id != st.session_state.current_run_id:
        st.session_state.current_run_id    = run_id
        st.session_state.altitude_bytes    = None
        st.session_state.groundtrack_bytes = None
        st.session_state.globe_data        = None
        st.session_state.sk_bytes          = None
        st.session_state.sk_results        = None
        st.session_state.sk_last_params    = None
        st.session_state.compare_bytes     = None

st.divider()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if not st.session_state.matches:
    # ---- Welcome screen ----
    st.markdown("#### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1 — Choose a family**")
        st.markdown(
            "Pick a satellite operator above: Satellogic ÑuSat, "
            "Planet Labs Dove, Spire LEMUR, or ISS."
        )
    with c2:
        st.markdown("**2 — Pick a satellite**")
        st.markdown(
            "After fetching, a dropdown lists every individual satellite "
            "with its current altitude and inclination."
        )
    with c3:
        st.markdown("**3 — Generate plots**")
        st.markdown(
            "Switch between tabs: altitude decay, flat ground track, "
            "interactive 3D globe, or stationkeeping simulation. "
            "Download or save on demand."
        )

else:
    # ---- Satellite info header ----
    info = satellite_info(name, line1, line2)

    st.title(name)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Altitude",      f"{info['alt_km']:.0f} km")
    c2.metric("Speed",         f"{info['speed_kms']:.3f} km/s")
    c3.metric("Period",        f"{info['period_min']:.1f} min")
    c4.metric("Inclination",   f"{info['incl_deg']:.2f}°")
    c5.metric("NORAD ID",      str(info["norad_id"]))

    st.caption(
        f"Epoch: {info['epoch']}  ·  "
        f"e = {info['eccentricity']:.6f}  ·  "
        f"B* = {info['bstar']:.2e}  ·  "
        f"Run ID: {run_id}"
    )

    st.divider()

    # Specs available to all tabs (propagator uses est_Am; stationkeeping uses isp_s etc.)
    specs = SATELLITE_SPECS.get(family_key, {
        "wet_mass_kg": 50.0, "isp_s": 220, "thrust_n": 1.0,
        "prop_type": "—", "source": "generic defaults",
        "est_Am": 0.010, "dims": "unknown", "mean_area_m2": 0.1,
    })

    # ---- Analysis tabs ----
    tab_about, tab_alt, tab_gt, tab_globe, tab_sk, tab_docs = st.tabs(
        ["About", "Altitude Decay", "Ground Track", "3D Globe", "Stationkeeping", "How it works"]
    )

    # --- Tab 0: About ---
    with tab_about:
        fam  = FAMILY_INFO.get(family_key, {})

        # ── Constellation banner ──────────────────────────────────────────
        st.markdown(
            f"<div style='background:#13151f;border:1px solid #1e2130;"
            f"border-radius:10px;padding:20px 24px 16px;margin-bottom:18px;'>"
            f"<p style='font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;"
            f"color:#5c6280;margin:0 0 4px 0;'>Constellation</p>"
            f"<p style='font-size:1.35rem;font-weight:700;color:#ffffff;margin:0 0 8px 0;'>"
            f"{fam.get('operator', '—')}</p>"
            f"<p style='font-size:0.88rem;color:#a8adc0;line-height:1.55;margin:0;'>"
            f"{fam.get('description', '')}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Constellation spec table ──────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                "<p style='font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;"
                "color:#5c6280;margin:0 0 8px 0;'>Constellation</p>"
                "<table style='border-collapse:collapse;width:100%;'>"
                + _kv_row("Operator",          fam.get("operator",     "—"))
                + _kv_row("Country",           fam.get("country",      "—"))
                + _kv_row("Mission",           fam.get("mission",      "—"))
                + _kv_row("Fleet size",        fam.get("constel_size", "—"))
                + _kv_row("Orbit",             fam.get("orbit_desc",   "—"))
                + "</table>",
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                "<p style='font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;"
                "color:#5c6280;margin:0 0 8px 0;'>Payload</p>"
                "<table style='border-collapse:collapse;width:100%;'>"
                + _kv_row("Instrument",  fam.get("payload",     "—"))
                + _kv_row("Resolution",  fam.get("resolution",  "—"))
                + _kv_row("Orbit class", classify_orbit(info["alt_km"], info["incl_deg"]))
                + "</table>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Satellite-specific section ────────────────────────────────────
        st.markdown(
            "<p style='font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;"
            "color:#5c6280;margin:0 0 12px 0;'>Satellite Catalog Data</p>",
            unsafe_allow_html=True,
        )

        with st.spinner("Looking up SATCAT..."):
            sat_row = _spacetrack_satcat(info["norad_id"])
            if not sat_row:
                satcat = fetch_satcat()
                sat_row = satcat.get(info["norad_id"], {})


        if sat_row:
            # Parse fields
            intl_des    = sat_row.get("OBJECT_ID",       "").strip()
            launch_date = sat_row.get("LAUNCH_DATE",     "").strip()
            decay_date  = sat_row.get("DECAY_DATE",      "").strip() or "—"
            owner_code  = sat_row.get("OWNER",           "").strip()
            site_code   = sat_row.get("LAUNCH_SITE",     "").strip()
            status_code = sat_row.get("OPS_STATUS_CODE", "").strip()
            obj_type    = sat_row.get("OBJECT_TYPE",     "").strip()
            rcs_size    = sat_row.get("RCS_SIZE",        "").strip()

            owner_name  = OWNER_LABELS.get(owner_code,  owner_code)
            site_name   = LAUNCH_SITE_LABELS.get(site_code, site_code)
            status_lbl, status_col = OPS_STATUS_LABELS.get(
                status_code, ("Unknown", "#666666")
            )
            status_html = (
                f"<span style='color:{status_col};font-weight:600;'>"
                f"{status_lbl}</span>"
            )

            # TLE epoch age
            epoch_age = (datetime.now() -
                         datetime.strptime(info["epoch"].split()[0], "%Y-%m-%d")).days

            col_c, col_d = st.columns(2)
            with col_c:
                st.markdown(
                    "<table style='border-collapse:collapse;width:100%;'>"
                    + _kv_row("NORAD ID",            str(info["norad_id"]))
                    + _kv_row("Intl. designator",    intl_des)
                    + _kv_row("Object type",         obj_type or "—")
                    + _kv_row("Owner",               f"{owner_name} ({owner_code})")
                    + _kv_row("Operational status",  status_html)
                    + "</table>",
                    unsafe_allow_html=True,
                )
            with col_d:
                st.markdown(
                    "<table style='border-collapse:collapse;width:100%;'>"
                    + _kv_row("Launch date",  launch_date or "—")
                    + _kv_row("Launch site",  f"{site_name} ({site_code})")
                    + _kv_row("Decay date",   decay_date)
                    + _kv_row("Radar cross-section", rcs_size or "—")
                    + _kv_row("TLE age",      f"{epoch_age} days")
                    + "</table>",
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                f"SATCAT entry not found for NORAD ID {info['norad_id']}. "
                "The catalog may be loading or the satellite may be unlisted."
            )

        # ── TLE text ─────────────────────────────────────────────────────
        with st.expander("Raw TLE"):
            st.code(f"{name}\n{line1}\n{line2}", language="text")

    # --- Tab 1: Altitude Decay ---
    with tab_alt:
        st.caption(
            "SGP4 propagation · 7 days · 60 s steps  |  "
            "Left: first 3 orbits (perigee / apogee)  |  "
            "Right: orbit-averaged decay + drag trend"
        )
        if st.button("Generate Altitude Plot", key="btn_alt", type="primary"):
            with st.spinner("Propagating 7 days..."):
                try:
                    fig = _run_altitude(name, line1, line2, run_id, show=False)
                    st.session_state.altitude_bytes = fig_to_bytes(fig)
                except Exception as _e:
                    st.error(f"Altitude plot failed: {_e}")

        if st.session_state.altitude_bytes:
            st.image(st.session_state.altitude_bytes, use_container_width=True)
            dl_col, sv_col, _ = st.columns([1, 1, 5])
            with dl_col:
                st.download_button(
                    "Download PNG",
                    data=st.session_state.altitude_bytes,
                    file_name=f"{run_id}_altitude.png",
                    mime="image/png",
                )
            with sv_col:
                if st.button("Save to disk", key="save_alt"):
                    out = f"figures/{run_id}_altitude.png"
                    os.makedirs("figures", exist_ok=True)
                    with open(out, "wb") as f:
                        f.write(st.session_state.altitude_bytes)
                    st.success(f"Saved → `{out}`")
        else:
            st.markdown(
                "<div class='plot-placeholder'>Click Generate Altitude Plot to run the analysis</div>",
                unsafe_allow_html=True,
            )

        # ── Propagator comparison ─────────────────────────────────────────
        st.divider()
        st.markdown("#### Propagator Comparison")
        st.caption(
            "Custom RK45 integrator  ·  Two-body + J2 + exponential atmosphere  ·  "
            "Compare against SGP4 (B*-calibrated)"
        )

        # ── B*-implied Cd·A/m reference ──────────────────────────────────
        sat_bstar = Satrec.twoline2rv(line1, line2).bstar
        CdAm_bstar = abs(sat_bstar) * 2.0 / 2.461e-5   # m²/kg
        est_Am = specs.get("est_Am", 0.010)
        st.info(
            f"**B\\*-implied Cd·A/m = {CdAm_bstar:.4f} m²/kg** — derived from TLE drag term "
            f"(B\\* = {sat_bstar:.3e} /Rₑ). "
            f"**Geometry estimate: {est_Am:.3f} m²/kg** "
            f"({specs.get('dims','—')}, mean projected area "
            f"{specs.get('mean_area_m2', 0):.3g} m², mass {specs['wet_mass_kg']} kg). "
            f"Set A/m below to match SGP4 decay rate."
        )

        # ── F10.7 fetch (once per session) ───────────────────────────────
        if st.session_state.f107_value is None:
            with st.spinner("Fetching F10.7 solar flux from NOAA SWPC…"):
                st.session_state.f107_value = _fetch_f107()

        cmp_c1, cmp_c2, cmp_c3 = st.columns(3)
        with cmp_c1:
            cmp_cd = st.slider(
                "Drag coefficient  Cd",
                min_value=1.5, max_value=3.0,
                value=2.2, step=0.1,
                key="cmp_cd",
                help="Cd ≈ 2.2 for most satellites in free-molecular flow LEO.",
            )
        with cmp_c2:
            cmp_am = st.slider(
                "Area-to-mass ratio  A/m  (m²/kg)",
                min_value=0.001, max_value=0.050,
                value=est_Am, step=0.001,
                key="cmp_am",
                help=(
                    f"Geometry estimate for this satellite: {est_Am:.3f} m²/kg "
                    f"({specs.get('dims','—')}). "
                    f"B*-implied: {CdAm_bstar:.4f} m²/kg. "
                    "Tune until RK45 decay rate matches SGP4."
                ),
            )
        with cmp_c3:
            f107_default = float(st.session_state.f107_value or 150.0)
            f107_label   = ("solar min" if f107_default < 100 else
                            "solar max" if f107_default > 200 else "mean solar")
            cmp_f107 = st.slider(
                "F10.7 solar flux  (SFU)",
                min_value=50.0, max_value=300.0,
                value=f107_default, step=5.0,
                key="cmp_f107",
                help=(
                    f"Current value fetched from NOAA SWPC: {f107_default:.0f} SFU "
                    f"[{f107_label}]. "
                    "Scales atmospheric density — USSA76 is calibrated to 150 SFU. "
                    "Solar min ≈ 70 SFU, solar max ≈ 200–250 SFU."
                ),
            )

        if st.button("Run Propagator Comparison", key="btn_compare", type="primary"):
            with st.spinner(
                "Integrating equations of motion (RK45) — this may take 30–60 s on cloud…"
            ):
                try:
                    fig_cmp = _compare_propagators(
                        name, line1, line2,
                        days=7.0, Cd=cmp_cd, Am=cmp_am, f107=cmp_f107,
                    )
                    st.session_state.compare_bytes = fig_to_bytes(fig_cmp)
                except Exception as _e:
                    st.error(f"Propagator comparison failed: {_e}")

        if st.session_state.compare_bytes:
            st.image(st.session_state.compare_bytes, use_container_width=True)
            dl_cmp, sv_cmp, _ = st.columns([1, 1, 5])
            with dl_cmp:
                st.download_button(
                    "Download PNG",
                    data=st.session_state.compare_bytes,
                    file_name=f"{run_id}_propagator_comparison.png",
                    mime="image/png",
                )
            with sv_cmp:
                if st.button("Save to disk", key="save_cmp"):
                    out = f"figures/{run_id}_propagator_comparison.png"
                    os.makedirs("figures", exist_ok=True)
                    with open(out, "wb") as f:
                        f.write(st.session_state.compare_bytes)
                    st.success(f"Saved → `{out}`")
        else:
            st.markdown(
                "<div class='plot-placeholder'>"
                "Set Cd and A/m above, then click Run Propagator Comparison"
                "</div>",
                unsafe_allow_html=True,
            )

    # --- Tab 2: Ground Track ---
    with tab_gt:
        st.caption(
            "ECI → geodetic via skyfield WGS84  |  "
            "Each orbit coloured with the plasma colourmap  |  "
            "Coastlines: Natural Earth 110 m"
        )
        if st.button("Generate Ground Track", key="btn_gt", type="primary"):
            with st.spinner("Computing ground track..."):
                fig = _run_groundtrack(name, line1, line2, run_id, show=False)
            st.session_state.groundtrack_bytes = fig_to_bytes(fig)

        if st.session_state.groundtrack_bytes:
            st.image(st.session_state.groundtrack_bytes, use_container_width=True)
            dl_col, sv_col, _ = st.columns([1, 1, 5])
            with dl_col:
                st.download_button(
                    "Download PNG",
                    data=st.session_state.groundtrack_bytes,
                    file_name=f"{run_id}_groundtrack.png",
                    mime="image/png",
                )
            with sv_col:
                if st.button("Save to disk", key="save_gt"):
                    out = f"figures/{run_id}_groundtrack.png"
                    os.makedirs("figures", exist_ok=True)
                    with open(out, "wb") as f:
                        f.write(st.session_state.groundtrack_bytes)
                    st.success(f"Saved → `{out}`")
        else:
            st.markdown(
                "<div class='plot-placeholder'>Click Generate Ground Track to run the analysis</div>",
                unsafe_allow_html=True,
            )

    # --- Tab 3: 3D Interactive Globe ---
    with tab_globe:
        st.caption(
            "Drag the globe freely with your mouse · "
            "time slider grows the ground track trail"
        )
        if st.button("Generate Globe", key="btn_globe", type="primary"):
            with st.spinner("Computing 24-hour ground track..."):
                lats, lons, epoch_dt, period_min = _compute_track(name, line1, line2)
            st.session_state.globe_data = {
                "lats":       list(lats),
                "lons":       list(lons),
                "epoch_dt":   epoch_dt,
                "period_min": period_min,
                "n":          len(lats),
            }

        if st.session_state.globe_data:
            gd = st.session_state.globe_data

            # Slider on the left, globe on the right — always in view together
            sl_col, globe_col = st.columns([1, 5])
            with sl_col:
                st.markdown(
                    "<div style='margin-top:180px'></div>",
                    unsafe_allow_html=True,
                )
                time_h = st.slider(
                    "Time elapsed (hours)",
                    0.0, 24.0, 1.0, step=0.1,
                    key="globe_time",
                )
            with globe_col:
                n         = gd["n"]
                pts       = max(1, int(round(time_h / 24.0 * n)))
                trail_lat = gd["lats"][:pts]
                trail_lon = gd["lons"][:pts]
                dot_lat   = gd["lats"][pts - 1]
                dot_lon   = gd["lons"][pts - 1]
                # Auto-follow: centre the globe on the current satellite position
                lon_rot = dot_lon
                lat_rot = dot_lat
                start_lat = gd["lats"][0]
                start_lon = gd["lons"][0]
                epoch_str = gd["epoch_dt"].strftime("%Y-%m-%d %H:%M UTC")

                fig_globe = go.Figure()
                fig_globe.add_trace(go.Scattergeo(
                    lat=trail_lat, lon=trail_lon,
                    mode="lines",
                    line=dict(color="#00b4d8", width=2),
                    name="Ground track",
                    hoverinfo="skip",
                ))
                fig_globe.add_trace(go.Scattergeo(
                    lat=[start_lat], lon=[start_lon],
                    mode="markers",
                    marker=dict(color="#00ff88", size=9, symbol="circle",
                                line=dict(color="#ffffff", width=1.5)),
                    name=f"Start  {epoch_str}",
                    hovertemplate="Start<br>Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>",
                ))
                fig_globe.add_trace(go.Scattergeo(
                    lat=[dot_lat], lon=[dot_lon],
                    mode="markers",
                    marker=dict(color="#ffffff", size=12, symbol="circle",
                                line=dict(color="#00b4d8", width=3)),
                    name="Satellite",
                    hovertemplate="Satellite<br>Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>",
                ))
                fig_globe.update_geos(
                    projection_type="orthographic",
                    projection_rotation=dict(lon=lon_rot, lat=lat_rot, roll=0),
                    showland=True,        landcolor="#2e4a1e",
                    showocean=True,       oceancolor="#0d2a4a",
                    showlakes=True,       lakecolor="#0d2a4a",
                    showrivers=False,
                    showcoastlines=True,  coastlinecolor="#6aadcc", coastlinewidth=0.8,
                    showcountries=True,   countrycolor="#1e3f3f",   countrywidth=0.4,
                    showsubunits=False,
                    lataxis=dict(showgrid=True, gridcolor="#1a2e40", dtick=30),
                    lonaxis=dict(showgrid=True, gridcolor="#1a2e40", dtick=30),
                    showframe=False,
                    bgcolor="#0b0c10",
                )
                fig_globe.update_layout(
                    paper_bgcolor="#0b0c10",
                    margin=dict(l=0, r=0, t=48, b=0),
                    height=560,
                    title=dict(
                        text=(
                            f"<b>{name}</b>  ·  "
                            f"{time_h:.1f} h elapsed  ·  "
                            f"{int(round(time_h * 60 / gd['period_min']))} orbits"
                        ),
                        font=dict(color="#c9cdd8", size=13,
                                  family="Space Grotesk, sans-serif"),
                        x=0.5,
                    ),
                    showlegend=True,
                    legend=dict(
                        bgcolor="#13151f", bordercolor="#1e2130", borderwidth=1,
                        font=dict(color="#c9cdd8", size=11,
                                  family="Space Grotesk, sans-serif"),
                        x=0.01, y=0.01,
                    ),
                )
                st.plotly_chart(fig_globe, use_container_width=True)
        else:
            st.markdown(
                "<div class='plot-placeholder'>Click Generate Globe to load the globe</div>",
                unsafe_allow_html=True,
            )

    # --- Tab 4: Stationkeeping Simulation ---
    with tab_sk:
        st.caption(
            "Deadband stationkeeping · SGP4 decay rate · Hohmann raises · "
            "Tsiolkovsky propellant budget"
        )

        # ── Parameters — each control on its own full-width row ───────────
        alt_default = int(round(info["alt_km"]))
        alt_min     = max(200, alt_default - 50)
        alt_max     = alt_default + 50

        # Seed session state from specs on first render (before sliders are drawn)
        if "sk_isp" not in st.session_state:
            st.session_state.sk_isp = specs["isp_s"]
        if "sk_wet_mass" not in st.session_state:
            st.session_state.sk_wet_mass = specs["wet_mass_kg"]
        if "sk_thrust_n" not in st.session_state:
            st.session_state.sk_thrust_n = specs.get("thrust_n", 1.0)
        if "sk_solar_factor" not in st.session_state:
            st.session_state.sk_solar_factor = 1.0

        if st.button("Reset to defaults", key="btn_sk_reset"):
            st.session_state.sk_target_alt   = float(alt_default)
            st.session_state.sk_half_width   = 5.0
            st.session_state.sk_duration     = 90
            st.session_state.sk_isp          = specs["isp_s"]
            st.session_state.sk_wet_mass     = specs["wet_mass_kg"]
            st.session_state.sk_raise_to     = "top"
            st.session_state.sk_thrust_n     = specs.get("thrust_n", 1.0)
            st.session_state.sk_solar_factor = 1.0

        target_alt = st.slider(
            "Target altitude (km)",
            min_value=float(alt_min), max_value=float(alt_max),
            value=float(alt_default), step=1.0,
            key="sk_target_alt",
            help="Centre of the deadband. Defaults to the satellite's current altitude.",
        )
        half_width = st.slider(
            "Deadband half-width (km)",
            min_value=0.5, max_value=15.0,
            value=5.0, step=0.5,
            key="sk_half_width",
            help="Allowed drift either side of target. Narrower = more burns.",
        )
        duration = st.slider(
            "Simulation duration (days)",
            min_value=30, max_value=365,
            value=90, step=5,
            key="sk_duration",
            help="How many days to simulate.",
        )
        isp = st.slider(
            "Specific impulse — Isp (s)",
            min_value=50, max_value=400,
            value=specs["isp_s"], step=5,
            key="sk_isp",
            help=(
                f"Thruster efficiency. Default: {specs['prop_type']} ({specs['isp_s']} s).  "
                "Cold gas ≈ 55–70 s · Hydrazine ≈ 220 s · Green propellant ≈ 250 s · "
                "Hall-effect electric ≈ 1500 s."
            ),
        )

        adv_col, strat_col, thrust_col = st.columns([1, 1, 1])
        with adv_col:
            wet_mass = st.number_input(
                "Wet mass (kg)",
                min_value=0.1, max_value=500000.0,
                value=specs["wet_mass_kg"], step=0.1,
                key="sk_wet_mass",
                help=(
                    f"Total spacecraft mass including propellant. "
                    f"Default: {specs['wet_mass_kg']} kg ({specs['prop_type']})."
                ),
            )
        with strat_col:
            raise_to = st.selectbox(
                "Raise strategy",
                options=["top", "target"],
                format_func=lambda x: (
                    "Raise to top of deadband" if x == "top"
                    else "Raise to target altitude"
                ),
                key="sk_raise_to",
                help="Top of band = fewer future burns. Target = simpler ops.",
            )
        with thrust_col:
            thrust_n = st.number_input(
                "Thrust (N)",
                min_value=0.01, max_value=100000.0,
                value=specs.get("thrust_n", 1.0), step=0.01,
                key="sk_thrust_n",
                help=(
                    "Thruster output force. Used to compute realistic burn duration "
                    f"(not orbital mechanics). Default: {specs.get('thrust_n', 1.0)} N "
                    f"({specs['prop_type']})."
                ),
            )

        solar_factor = st.slider(
            "Solar activity multiplier",
            min_value=0.3, max_value=3.0,
            value=1.0, step=0.1,
            key="sk_solar_factor",
            help=(
                "Scales the measured SGP4 decay rate to simulate different solar conditions. "
                "Solar minimum ≈ 0.3× (low drag),  mean solar = 1.0×,  solar maximum ≈ 3.0× (high drag)."
            ),
        )

        st.caption(
            f"Isp, wet-mass and thrust defaults from **{specs['source']}**. "
            "Adjust sliders to explore different configurations."
        )

        # Track whether params changed since last run
        sk_params = (target_alt, half_width, duration, isp, wet_mass, raise_to, thrust_n, solar_factor)
        params_changed = (
            st.session_state.sk_results is not None
            and st.session_state.get("sk_last_params") != sk_params
        )

        if st.button("Run Simulation", key="btn_sk", type="primary"):
            with st.spinner("Fetching 30-day TLE history and simulating maneuvers..."):
                hist_rate = _spacetrack_decay_rate(info["norad_id"])
                decay_src = "30-day TLE history" if hist_rate else "7-day SGP4 fit"
                fig_sk, res_sk = _run_sk(
                    name, line1, line2,
                    target_alt=target_alt,
                    half_width=half_width,
                    duration=float(duration),
                    isp=float(isp),
                    wet_mass=wet_mass,
                    raise_to=raise_to,
                    thrust_n=float(thrust_n),
                    solar_factor=float(solar_factor),
                    decay_rate_override=hist_rate,
                    run_id=run_id,
                    show=False,
                )
            st.session_state.sk_bytes       = fig_to_bytes(fig_sk)
            st.session_state.sk_results     = res_sk
            st.session_state.sk_last_params = sk_params
            st.session_state["sk_decay_src"] = decay_src

        if params_changed:
            st.warning("Parameters changed — click Run Simulation to update results.")

        if st.session_state.sk_results:
            res = st.session_state.sk_results

            # ── Summary metric cards ───────────────────────────────────────
            st.divider()
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Maneuvers",     str(res["n_maneuvers"]))
            m2.metric("Total ΔV",      f"{res['total_dv_ms']:.3f} m/s")
            m3.metric("ΔV / year",     f"{res['dv_per_year_ms']:.2f} m/s/yr")
            m4.metric("Propellant",    f"{res['prop_consumed']:.3f} kg")
            m5.metric("Avg burn dur.", f"{res['avg_burn_dur_s']:.0f} s")

            decay_src = st.session_state.get("sk_decay_src", "7-day SGP4 fit")
            st.caption(
                f"Decay rate: {res['decay_rate']:.4f} km/day  [{decay_src}]  ·  "
                f"Solar: {res['solar_factor']:.1f}×  ·  "
                f"Isp: {res['isp']:.0f} s  ·  "
                f"Thrust: {res['thrust_n']:.2f} N  ·  "
                f"Wet mass: {res['wet_mass']:.1f} kg  ·  "
                f"Run ID: {run_id}"
            )

            # ── Staircase figure ───────────────────────────────────────────
            st.image(st.session_state.sk_bytes, use_container_width=True)

            dl_col, sv_col, _ = st.columns([1, 1, 5])
            with dl_col:
                st.download_button(
                    "Download PNG",
                    data=st.session_state.sk_bytes,
                    file_name=f"{run_id}_stationkeeping.png",
                    mime="image/png",
                )
            with sv_col:
                if st.button("Save to disk", key="save_sk"):
                    out = f"figures/{run_id}_stationkeeping.png"
                    os.makedirs("figures", exist_ok=True)
                    with open(out, "wb") as f:
                        f.write(st.session_state.sk_bytes)
                    st.success(f"Saved → `{out}`")

            # ── Maneuver log ───────────────────────────────────────────────
            if res["maneuvers"]:
                with st.expander(f"Maneuver log  ({res['n_maneuvers']} burns)"):
                    st.table(res["maneuvers"])

        else:
            st.markdown(
                "<div class='plot-placeholder'>Set parameters above and click Run Simulation</div>",
                unsafe_allow_html=True,
            )

    # ── Tab: How it works ──────────────────────────────────────────────────
    with tab_docs:
        st.markdown(
            "<h2 style='color:#e8eaf6;margin-bottom:4px;'>How it works</h2>"
            "<p style='color:#5c6280;font-size:0.85rem;margin-top:0;'>Physics pipeline, data sources, and key assumptions</p>",
            unsafe_allow_html=True,
        )

        # ── Overview card ────────────────────────────────────────────────
        st.markdown(
            "<div style='background:#13151f;border:1px solid #1e2130;border-radius:10px;"
            "padding:18px 24px;margin-bottom:20px;'>"
            "<p style='color:#a8adc0;font-size:0.92rem;line-height:1.65;margin:0;'>"
            "This app propagates a real satellite's Two-Line Element (TLE) set forward in time, "
            "measures the orbital decay rate, and simulates a deadband stationkeeping strategy "
            "— the same technique operators use to decide when and how hard to fire thrusters. "
            "All physics runs locally in Python; no external APIs are called during simulation.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Section helper ───────────────────────────────────────────────
        def _section(title):
            st.markdown(
                f"<p style='font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;"
                f"color:#5c6280;margin:20px 0 10px 0;border-bottom:1px solid #1e2130;"
                f"padding-bottom:6px;'>{title}</p>",
                unsafe_allow_html=True,
            )

        # ── Data pipeline ────────────────────────────────────────────────
        _section("1 · Data pipeline")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                "<div style='background:#13151f;border:1px solid #1e2130;border-radius:8px;padding:16px;'>"
                "<p style='color:#00b4d8;font-weight:700;font-size:0.88rem;margin:0 0 8px 0;'>TLE fetch</p>"
                "<p style='color:#a8adc0;font-size:0.82rem;line-height:1.6;margin:0;'>"
                "Two-Line Element sets fetched live from <strong style='color:#e8eaf6;'>Space-Track.org</strong> "
                "(USSPACECOM authoritative catalogue). "
                "A 30-day epoch filter removes deorbited objects. "
                "Fallback: hardcoded ÑuSat TLE if the network is unavailable.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                "<div style='background:#13151f;border:1px solid #1e2130;border-radius:8px;padding:16px;'>"
                "<p style='color:#00b4d8;font-weight:700;font-size:0.88rem;margin:0 0 8px 0;'>SGP4 propagation</p>"
                "<p style='color:#a8adc0;font-size:0.82rem;line-height:1.6;margin:0;'>"
                "The <code style='color:#c8cfe0;background:#0d0f1a;padding:1px 4px;border-radius:3px;'>sgp4</code> "
                "Python library (Vallado implementation) propagates the TLE through 7 days at 60-second steps. "
                "Altitude = |<b>r</b>| − 6371 km (spherical Earth). "
                "Orbit-averaged altitude gives the smooth decay curve. "
                "(The ground track uses skyfield's WGS-84 ellipsoid for lat/lon conversion.)</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                "<div style='background:#13151f;border:1px solid #1e2130;border-radius:8px;padding:16px;'>"
                "<p style='color:#00b4d8;font-weight:700;font-size:0.88rem;margin:0 0 8px 0;'>Decay rate</p>"
                "<p style='color:#a8adc0;font-size:0.82rem;line-height:1.6;margin:0;'>"
                "Primary: linear fit to 30 days of TLE history from Space-Track <em>gp_history</em> — "
                "semi-major axis derived from each TLE's mean motion "
                "(n → a = (μ/n²)^⅓). "
                "Fallback: 7-day SGP4 linear fit. "
                "Both methods return km/day.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Simulation physics ───────────────────────────────────────────
        _section("2 · Simulation physics")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "<div style='background:#13151f;border:1px solid #1e2130;border-radius:8px;padding:16px;margin-bottom:12px;'>"
                "<p style='color:#00b4d8;font-weight:700;font-size:0.88rem;margin:0 0 8px 0;'>Stationkeeping (Phase 2)</p>"
                "<p style='color:#a8adc0;font-size:0.82rem;line-height:1.6;margin:0;'>"
                "Deadband strategy: the simulation analytically computes the exact time until "
                "altitude hits the lower deadband boundary, then fires a Hohmann transfer to raise it back. "
                "Each burn uses the exact two-impulse ΔV formula (vis-viva). "
                "Propellant is computed via the Tsiolkovsky rocket equation (Δm = m₀·(1 − e^(−ΔV/v_e))). "
                "The trajectory change is instantaneous (impulsive approximation); burn duration "
                "t = Δm·Isp·g₀ / F is calculated from thrust and shown in the maneuver log for reference. "
                "A solar-activity multiplier scales the decay rate across the solar cycle.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='background:#13151f;border:1px solid #1e2130;border-radius:8px;padding:16px;'>"
                "<p style='color:#00b4d8;font-weight:700;font-size:0.88rem;margin:0 0 8px 0;'>Hohmann ΔV</p>"
                "<p style='color:#a8adc0;font-size:0.82rem;line-height:1.6;margin:0;'>"
                "Transfer from circular orbit r₁ to r₂:"
                "<br><code style='color:#c8cfe0;background:#0d0f1a;padding:2px 6px;border-radius:3px;font-size:0.80rem;'>"
                "Δv₁ = √(μ/r₁)·(√(2r₂/(r₁+r₂)) − 1)"
                "</code><br>"
                "<code style='color:#c8cfe0;background:#0d0f1a;padding:2px 6px;border-radius:3px;font-size:0.80rem;'>"
                "Δv₂ = √(μ/r₂)·(1 − √(2r₁/(r₁+r₂)))"
                "</code></p>"
                "</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                "<div style='background:#13151f;border:1px solid #1e2130;border-radius:8px;padding:16px;margin-bottom:12px;'>"
                "<p style='color:#00b4d8;font-weight:700;font-size:0.88rem;margin:0 0 8px 0;'>Custom RK45 propagator</p>"
                "<p style='color:#a8adc0;font-size:0.82rem;line-height:1.6;margin:0;'>"
                "A numerical propagator integrates the equations of motion with three force terms: "
                "<strong style='color:#e8eaf6;'>two-body gravity</strong>, "
                "<strong style='color:#e8eaf6;'>J2 oblateness</strong> (Earth's equatorial bulge), and "
                "<strong style='color:#e8eaf6;'>atmospheric drag</strong> (USSA76 density model, F10.7-scaled). "
                "Integrator: <code style='color:#c8cfe0;background:#0d0f1a;padding:1px 4px;border-radius:3px;'>scipy.integrate.solve_ivp</code> "
                "(RK45, rtol=10⁻⁶). "
                "The Propagator tab lets you tune C_D·A/m and solar flux to match SGP4.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='background:#13151f;border:1px solid #1e2130;border-radius:8px;padding:16px;'>"
                "<p style='color:#00b4d8;font-weight:700;font-size:0.88rem;margin:0 0 8px 0;'>F10.7 atmospheric scaling</p>"
                "<p style='color:#a8adc0;font-size:0.82rem;line-height:1.6;margin:0;'>"
                "Solar EUV heats the upper atmosphere, expanding it and increasing drag. "
                "The USSA76 model is calibrated at F10.7 ≈ 150 SFU (mean solar). "
                "Scaling applied: ρ = ρ_USSA76 · exp(β · (F10.7 − 150)), "
                "where β ranges from 0.006 at 300 km to 0.020 at 700 km. "
                "Live F10.7 is fetched from NOAA SWPC on first use. "
                "β = 0.020 for alt ≥ 600 km.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Satellite specs ──────────────────────────────────────────────
        _section("3 · Satellite specifications")
        specs_html = (
            "<div style='overflow-x:auto;'>"
            "<table style='border-collapse:collapse;width:100%;font-size:0.82rem;'>"
            "<thead><tr>"
            + "".join(
                f"<th style='text-align:left;padding:7px 12px;color:#5c6280;"
                f"font-size:0.68rem;letter-spacing:1.5px;text-transform:uppercase;"
                f"border-bottom:1px solid #1e2130;'>{h}</th>"
                for h in ["Constellation", "Wet mass", "Propulsion", "Isp", "Thrust", "Geometry", "A/m"]
            )
            + "</tr></thead><tbody>"
        )
        sat_rows = [
            ("Satellogic ÑuSat", "40 kg", "Cold gas (N₂)", "65 s", "0.5 N", "0.45 × 0.45 × 0.70 m", "0.010 m²/kg"),
            ("Planet Dove", "5.8 kg", "Cold gas (Dove+)", "60 s", "0.1 N", "0.10 × 0.10 × 0.30 m (3U)", "0.006 m²/kg"),
            ("Spire LEMUR-2", "4.5 kg", "Cold gas", "55 s", "0.1 N", "0.10 × 0.10 × 0.30 m (3U)", "0.008 m²/kg"),
            ("ISS", "420 000 kg", "UDMH/N₂O₄", "310 s", "400 N", "~73 m wingspan", "0.006 m²/kg"),
        ]
        for i, row in enumerate(sat_rows):
            bg = "#13151f" if i % 2 == 0 else "#0d0f1a"
            specs_html += (
                f"<tr style='background:{bg};'>"
                + "".join(
                    f"<td style='padding:7px 12px;color:#a8adc0;border-bottom:1px solid #1a1c2b;'>{v}</td>"
                    for v in row
                )
                + "</tr>"
            )
        specs_html += "</tbody></table></div>"
        st.markdown(specs_html, unsafe_allow_html=True)

        # ── Key assumptions ──────────────────────────────────────────────
        _section("4 · Key assumptions & limitations")
        a1, a2 = st.columns(2)
        with a1:
            st.markdown(
                "<ul style='color:#a8adc0;font-size:0.83rem;line-height:1.75;padding-left:18px;margin:0;'>"
                "<li>Decay rate assumed constant over the simulation window (valid for weeks–months)</li>"
                "<li>Trajectory change at each burn is instantaneous (impulsive ΔV approximation); "
                "actual burn duration is estimated from thrust and shown in the maneuver log</li>"
                "<li>Only altitude (semi-major axis) is controlled — no inclination or RAAN corrections</li>"
                "<li>USSA76 density model with single-parameter F10.7 scaling; NRLMSISE-00 would be more accurate</li>"
                "</ul>",
                unsafe_allow_html=True,
            )
        with a2:
            st.markdown(
                "<ul style='color:#a8adc0;font-size:0.83rem;line-height:1.75;padding-left:18px;margin:0;'>"
                "<li>Satellite mass assumed constant between burns (propellant mass is small)</li>"
                "<li>J2 precession changes RAAN and argument of perigee — not fed back into decay</li>"
                "<li>No conjunction screening or exclusion zones modelled</li>"
                "<li>Thruster specs are from public documentation; actual values may differ</li>"
                "</ul>",
                unsafe_allow_html=True,
            )

        # ── Links ────────────────────────────────────────────────────────
        _section("5 · Source & further reading")
        st.markdown(
            "<p style='color:#a8adc0;font-size:0.85rem;line-height:1.8;'>"
            "Full source code, derivations, and session notes: "
            "<a href='https://github.com/falahi1/leo-stationkeeping' target='_blank' "
            "style='color:#00b4d8;text-decoration:none;font-weight:600;'>github.com/falahi1/leo-stationkeeping</a>"
            "<br>TLE data: "
            "<a href='https://www.space-track.org' target='_blank' style='color:#00b4d8;text-decoration:none;'>Space-Track.org</a>"
            " (USSPACECOM / 18th Space Control Squadron)"
            "<br>F10.7 solar flux: "
            "<a href='https://www.swpc.noaa.gov' target='_blank' style='color:#00b4d8;text-decoration:none;'>NOAA Space Weather Prediction Center</a>"
            "<br>SGP4 reference: Vallado et al., <em>Revisiting Spacetrack Report #3</em> (2006)"
            "</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; padding: 12px 0 8px 0; color:#5c6280; font-size:0.78rem; font-family:'Space Grotesk',sans-serif;">
        Built by
        <a href="https://github.com/falahi1" target="_blank"
           style="color:#00b4d8; text-decoration:none; font-weight:600;">
            falahi1
        </a>
        &nbsp;·&nbsp;
        <a href="https://github.com/falahi1/leo-stationkeeping" target="_blank"
           style="color:#00b4d8; text-decoration:none;">
            GitHub repo
        </a>
        &nbsp;·&nbsp;
        LEO orbit propagation &amp; stationkeeping simulation
    </div>
    """,
    unsafe_allow_html=True,
)
