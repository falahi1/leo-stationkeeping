# LEO Orbit Propagation and Stationkeeping

**Author:** [falahi1](https://github.com/falahi1)
**Date:** September 2026  

> Live interactive demo → **[leo-stationkeeping.streamlit.app](https://leo-stationkeeping.streamlit.app)**

---

## What this does

| Phase | Description |
|---|---|
| **1a** | Fetch a live TLE from Space-Track, propagate with SGP4, plot 7-day altitude decay (perigee/apogee oscillation + orbit-averaged trend) |
| **1b** | Plot the 24-hour ground track on a world map; interactive 3D globe with time slider |
| **2** | Deadband stationkeeping simulation — detect altitude violations, fire Hohmann raises, compute ΔV/year and propellant budget |
| **Propagator** | Custom RK45 integrator (two-body + J2 + USSA76 drag) compared against SGP4 side-by-side |

The Streamlit app supports four satellite families live: **Satellogic ÑuSat**, **Planet Labs Dove**, **Spire LEMUR**, and the **ISS** — select any individual satellite from the fetched TLE catalogue and run all analyses on it.

---

## Results — NUSAT-26 (M Somerville), epoch 2026-09-26

### Orbit characterisation (Phase 1)

| Quantity | Value |
|---|---|
| Altitude (mean) | 437 km, sun-synchronous (i = 97.22°) |
| Orbital period | 93.2 min  (15.46 rev/day) |
| **Orbital speed** | **7.652 km/s  (27 547 km/h)** |
| Perigee altitude | 434 km |
| Apogee altitude | 440 km |
| Eccentricity | 0.000 44  (near-circular) |
| Altitude decay rate (SGP4, 7-day fit) | **0.143 km/day** |
| Annual altitude loss | 52.2 km/yr |

### Stationkeeping budget (Phase 2, ±5 km deadband, Isp = 65 s, 40 kg — cold gas N₂)

| Quantity | Value |
|---|---|
| ΔV per burn (5 km raise) | ~2.8 m/s |
| Burn frequency | ~10 burns/year |
| Propellant per burn | ~0.175 kg |
| **ΔV/year (steady-state)** | **~29 m/s/yr** |
| Propellant/year | ~1.75 kg/yr  (4.4% of wet mass) |
| Mission life (10% prop budget) | ~2–3 years |

### Propagator comparison

| Propagator | Decay rate | Force model |
|---|---|---|
| SGP4 (operational) | 0.143 km/day | Calibrated B* from tracking data |
| Custom RK45 | 0.103 km/day | Two-body + J2 + USSA76 drag (A/m = 0.010 m²/kg) |

The 30% gap reflects that SGP4's B* is fitted to actual observed tracking, while the RK45 uses an assumed area-to-mass ratio and a mean-solar-activity atmosphere.

---

## Run locally

```bash
git clone https://github.com/falahi1/leo-stationkeeping.git
cd leo-stationkeeping
pip install -r requirements.txt

# Interactive Streamlit app (recommended)
streamlit run src/app.py

# Or run individual phase scripts
python src/phase1_altitude_plot.py    # SGP4 altitude decay plot
python src/phase1_groundtrack.py      # 24-hour ground track
python src/phase2_stationkeeping.py   # Stationkeeping simulation
```

Figures are saved to `figures/` (versioned: `phase1_altitude_v1.png`, `v2.png`, …).  
On first run, `skyfield` will download two small data files (~1 MB); these are cached locally.

---

## Method

### Phase 1 — SGP4 propagation
- **TLE source:** Space-Track.org GP catalogue (requires free account; credentials passed via `st.secrets`); `data/nusat26_tle.txt` contains a saved TLE for offline reproducibility
- **Propagator:** `sgp4` library (Vallado's implementation of SGP4/SDP4)
- **Coordinate frame:** ECI; altitude = |**r**| − 6371 km
- **Ground track:** `skyfield` converts ECI → geodetic lat/lon via WGS84; coastlines from Natural Earth 110 m GeoJSON

### Phase 2 — Stationkeeping simulation
- **Decay rate:** 30-day TLE history from Space-Track `gp_history` endpoint (semi-major axis extracted from mean motion of each historical TLE, linear trend fitted); falls back to 7-day SGP4 orbit-averaged fit if history unavailable
- Deadband control: impulsive Hohmann raise fires when altitude drops below the lower limit
- ΔV: exact two-impulse Hohmann formula from vis-viva equation
- Propellant: Tsiolkovsky rocket equation `dm = m(1 − exp(−ΔV / (Isp g₀)))`
- **Burn duration:** `t_burn = Δm × Isp × g₀ / F` — finite burn time shown in maneuver log
- Steady-state ΔV/year: `ṙ × 365.25 × v_circ / (2a)` — independent of window size
- **Solar activity multiplier:** scales decay rate from 0.3× (solar min) to 3.0× (solar max)

Simulation defaults (Isp, wet mass, thrust) are drawn from public mission documentation for each satellite family:

| Satellite family | Wet mass | Propulsion | Isp | Thrust | Source |
|---|---|---|---|---|---|
| Satellogic ÑuSat | 40 kg | Cold gas (N₂) | 65 s | ~0.5 N | Satellogic public mission docs |
| Planet Labs Dove | 5.8 kg | Cold gas (Dove+/Pelican) | 60 s | ~0.1 N | Planet Labs public specs |
| Spire LEMUR-2 | 4.5 kg | Cold gas | 55 s | ~0.1 N | Spire Global public specs |
| ISS | 420 000 kg | UDMH/N₂O₄ (Progress/Zvezda) | 310 s | ~400 N | NASA/Roscosmos ops data |

Target altitude defaults to the satellite's current altitude from the TLE. All parameters are adjustable via sliders.

### Custom numerical propagator (`src/propagator.py`)
- State vector: `[x, y, z, vx, vy, vz]` (ECI, SI units), seeded from SGP4 at TLE epoch
- Forces: two-body gravity · J2 oblateness (Vallado eq. 8-20) · atmospheric drag with rotating atmosphere
- Atmosphere: USSA76 piecewise exponential (11 layers, 200–700 km), **scaled by live F10.7 solar flux** fetched from NOAA SWPC; sensitivity β increases with altitude (0.006/SFU at 300 km → 0.017/SFU at 600 km)
- Integration: `scipy.integrate.solve_ivp` RK45, `rtol=1e-6`, `atol=1e-7`
- A/m slider seeded from satellite geometry estimate (bus dimensions → mean projected area = SA/4); **B\*-implied Cd·A/m** shown alongside as a reference target

---

## Physics and derivations

Everything in this project connects through one chain:

**TLE → SGP4 → altitude decay rate → Hohmann ΔV → propellant budget**

Each step is derived below, using NUSAT-26 at 437 km as the worked example throughout.

---

### 1. Circular orbit velocity and period

For a circular orbit of radius r around Earth (μ = 398 600.4 km³/s²), the vis-viva equation reduces to:

```
v = √(μ / r)
```

At 437 km altitude, r = 6371 + 437 = 6808 km:

```
v = √(398 600.4 / 6808) = 7.652 km/s
```

The orbital period follows from the circumference:

```
T = 2π r / v = 2π × 6808 / 7.652 = 5592 s ≈ 93.2 min
```

This matches NUSAT-26's TLE mean motion of 15.46 rev/day exactly — confirming the TLE is self-consistent.

---

### 2. Reading a TLE — annotated NUSAT-26 example

A Three-Line Element set (TLE) is the standard format published by the US Space Force for every tracked object. Here is NUSAT-26's TLE with every field decoded:

```
NUSAT-26 (M SOMERVILLE)                         ← Line 0: satellite name
1 52184U 22033AD  26269.60424588  .00024559  00000+0  50917-3 0  9991
2 52184  97.2277 348.1020 0004438 222.9360 137.1544 15.45714027120359
```

**Line 1 — identity and timing:**

```
1 52184U 22033AD  26269.60424588  .00024559  00000+0  50917-3 0  9991
  ─────                                                              ↑
  52184 → NORAD catalog number (unique ID for every tracked object)  checksum

         22033AD → International Designator
           22    = launched in 2022
             033 = 33rd launch of that year
                AD = piece "AD" from that launch (ÑuSat rideshares carry many sats)

                   26269.60424588 → Epoch (when the TLE was measured)
                     26           = year 2026
                       269        = day 269 of 2026 = 26 September
                          .60424588 × 24 h = 14:30 UTC
                                    → Epoch: 2026-09-26 14:30 UTC

                                     .00024559 → Ṅ: first time-derivative of mean motion
                                                   (rev/day²) — positive means orbit is
                                                   shrinking (drag is increasing mean motion)

                                                  50917-3 → B* drag term
                                                    = 0.50917 × 10⁻³ = 5.09 × 10⁻⁴ /Rₑ
```

**Line 2 — orbital elements:**

```
2 52184  97.2277 348.1020 0004438 222.9360 137.1544 15.45714027120359
         ───────                                                      ↑
         97.2277° → inclination i                           checksum (9)
                  (> 90° means retrograde — required for sun-synchronous)

                  348.1020° → RAAN Ω (Right Ascension of the Ascending Node)
                  (the "longitude" of where the orbit crosses the equator going north)

                             0004438 → eccentricity e = 0.0004438
                             (implied decimal: the TLE omits the "0.")
                             Near-zero → nearly circular

                                     222.9360° → argument of perigee ω
                                     (where in the orbit the closest point is)

                                              137.1544° → mean anomaly M
                                              (where in the orbit the satellite is at epoch)

                                                       15.45714027 → mean motion n
                                                       (rev/day — the most important number)
                                                                   120359 → revolution count
                                                                   since launch
```

**Deriving physical quantities from the TLE:**

*Orbital period* — directly from mean motion:
```
T = 1440 min/day ÷ 15.45714027 rev/day = 93.17 min  ✓ (matches 93.2 min measured)
```

*Semi-major axis* — from Kepler's third law (n in rad/s):
```
n = 15.45714027 × 2π / 86400 = 1.1248 × 10⁻³ rad/s

a = (μ / n²)^(1/3) = (398600.4 / (1.1248 × 10⁻³)²)^(1/3) = 6808 km

altitude = a − Rₑ = 6808 − 6371 = 437 km  ✓
```

*Perigee and apogee altitudes* — from eccentricity:
```
e = 0.0004438

perigee = a(1 − e) − Rₑ = 6808 × 0.9995562 − 6371 = 434.0 km
apogee  = a(1 + e) − Rₑ = 6808 × 1.0004438 − 6371 = 440.0 km
```

The orbit oscillates between 434 and 440 km — exactly the ±3 km oscillation visible in the left panel of the altitude decay plot.

**The B* drag term:**

```
B* = ρ₀ × Cd × A / (2m)       [units: 1/Rₑ]
```

where ρ₀ = 2.461 × 10⁻⁵ kg/m²/Rₑ is SGP4's reference density. B* encodes the satellite's aerodynamic properties in a single number that is calibrated against real tracking data. It implicitly captures Cd, the cross-sectional area A, the mass m, and the actual atmospheric density at epoch — all in one observed quantity.

The inverse relationship gives the implied Cd·A/m:
```
Cd · A/m ≈ B* × 2 / ρ₀
```

> **Key fact:** TLE elements are *mean* elements — short-period oscillations are averaged out and the numbers are tuned specifically for SGP4. They cannot be substituted directly into vis-viva. Always propagate through SGP4.

---

### 3. Atmospheric drag and altitude decay

The drag acceleration on a satellite is:

```
a_drag = ½ × Cd × (A/m) × ρ(h) × v²
```

Atmospheric density ρ(h) falls off exponentially with altitude. At 437 km, ρ ≈ 3 × 10⁻¹⁰ kg/m³ (USSA76). As drag removes energy, the semi-major axis shrinks at a rate proportional to ρ and B*.

Rather than solving this analytically (which requires knowing ρ precisely), **Phase 1 measures the decay rate empirically** from SGP4:

1. Propagate 7 days at 60 s steps → 10 080 ECI position vectors
2. Convert each to altitude: h = |**r**| − 6371 km
3. Average h over each orbital period → one point per orbit
4. Fit a linear trend to the orbit-averaged altitude vs time

For NUSAT-26 this gives:

```
ḣ = −0.143 km/day  (the slope of the linear fit)
```

This measured ḣ is the input to everything else in Phase 2.

---

### 4. Hohmann transfer ΔV

A Hohmann transfer raises a circular orbit from radius r₁ to r₂ using two tangential (prograde) burns. From vis-viva, the velocities on the transfer ellipse at perigee and apogee are:

```
v_t1 = √( 2μ r₂ / (r₁ (r₁ + r₂)) )     ← speed at r₁ on the ellipse
v_t2 = √( 2μ r₁ / (r₂ (r₁ + r₂)) )     ← speed at r₂ on the ellipse
```

The two burns are:

```
Δv₁ = v_t1 − √(μ/r₁)      ← burn at r₁: raise apogee to r₂
Δv₂ = √(μ/r₂) − v_t2      ← burn at r₂: circularise
ΔV  = Δv₁ + Δv₂
```

For a small raise Δh ≪ r this linearises to the useful planning approximation:

```
ΔV ≈ v × Δh / (2a)
```

**Worked example — NUSAT-26, stationkeeping raise from 432 → 437 km (Δh = 5 km):**

```
r₁ = 6803 km,  r₂ = 6808 km,  μ = 398 600.4 km³/s²

v_t1 = √(2 × 398 600.4 × 6808 / (6803 × 13 611)) = 7.6576 km/s
Δv₁  = 7.6576 − √(398 600.4 / 6803)
      = 7.6576 − 7.6548 = 0.0028 km/s = 2.8 m/s

v_t2 = √(2 × 398 600.4 × 6803 / (6808 × 13 611)) = 7.6491 km/s  (approx)
Δv₂  ≈ 0  (circularisation is near-zero for a small raise)

ΔV ≈ 2.8 m/s  (dominated by the first burn)
```

This is the ΔV that Phase 2 computes exactly for every simulated burn.

---

### 5. Steady-state annual ΔV budget

The satellite decays continuously at ḣ = 0.143 km/day. Over one year it loses:

```
Δh_annual = 0.143 × 365.25 = 52.2 km
```

Every km lost must be recovered. The cost per km from the linearised Hohmann formula is:

```
ΔV/km = v / (2a) = 7652 m/s / (2 × 6 808 000 m) × 1000 m/km = 0.562 m/s per km
```

Therefore:

```
ΔV/year = ḣ × 365.25 × v / (2a)
         = 0.143 × 365.25 × 0.562
         ≈ 29 m/s/yr
```

This result is **independent of deadband width**: wider deadbands mean fewer, larger burns; narrower deadbands mean more, smaller burns; the total ΔV/year is identical either way, because the satellite always recovers the same total altitude loss.

---

### 6. Propellant budget — Tsiolkovsky rocket equation

The rocket equation relates ΔV to the propellant mass consumed:

```
ΔV = Isp × g₀ × ln(m_wet / m_dry)
```

Rearranging for the propellant used in a single burn:

```
Δm = m_current × ( 1 − exp(−ΔV / (Isp × g₀)) )
```

**Worked example — NUSAT-26, 5 km raise, Isp = 65 s (cold gas N₂), m = 40 kg:**

```
Isp × g₀ = 65 × 9.807 = 637.5 m/s

Δm = 40 × (1 − exp(−2.8 / 637.5))
   = 40 × (1 − exp(−0.004392))
   = 40 × 0.004382
   ≈ 0.175 kg per burn
```

With ~10 burns/year (at ±5 km deadband, raise-to-target strategy):

```
propellant/year ≈ 10 × 0.175 = 1.75 kg/yr  ≈ 4.4% of wet mass
```

---

### 7. J2 perturbation and the sun-synchronous condition

Earth's equatorial bulge (J2 = 1.0826 × 10⁻³) causes the orbital plane to precess. The RAAN drift rate is:

```
dΩ/dt = −(3/2) × n × J2 × (Rₑ/a)² × cos(i) / (1 − e²)²
```

A **sun-synchronous orbit** requires the plane to drift eastward at exactly +0.9856°/day — matching Earth's annual orbit around the Sun. This keeps the local solar crossing time fixed, giving consistent lighting for optical imaging.

Setting dΩ/dt = +0.9856°/day and solving for inclination at a = 6808 km:

```
cos(i) = −0.00482   →   i ≈ 97.3°
```

NUSAT-26's TLE gives i = 97.22° — consistent with SSO design. The slightly retrograde orbit (i > 90°) makes the J2 precession eastward instead of westward, which is what SSO requires.

---

### 8. Custom propagator — equations of motion

`propagator.py` integrates the full 6-DOF state **r** = [x, y, z, ẋ, ẏ, ż] in ECI (SI units). The total acceleration has three terms:

**Two-body gravity:**
```
a_grav = −(μ / r³) × r⃗
```

**J2 oblateness** (Vallado eq. 8-20), where z is along the polar axis:
```
a_J2x = −(3/2) (μ J2 Rₑ²/ r⁵) x (1 − 5z²/r²)
a_J2y = −(3/2) (μ J2 Rₑ²/ r⁵) y (1 − 5z²/r²)
a_J2z = −(3/2) (μ J2 Rₑ²/ r⁵) z (3 − 5z²/r²)
```

**Atmospheric drag** with rotating atmosphere (ωₑ = 7.292 × 10⁻⁵ rad/s):
```
v_rel = v_sat − ωₑ × r⃗            ← velocity relative to atmosphere
a_drag = −½ Cd (A/m) ρ(r) |v_rel| v_rel
```

ρ(r) comes from the USSA76 piecewise exponential model (11 altitude layers between 200–700 km).

The integrator is seeded from SGP4 at the TLE epoch. From that point it propagates independently — any divergence from SGP4 reflects genuine differences between the two force models (mainly: SGP4's calibrated B* vs the assumed Cd·A/m).

---

## Assumptions and limitations

- SGP4 is optimised for short-arc propagation (days to weeks); accuracy degrades over months as B* is fixed at epoch
- Stationkeeping simulation uses a constant decay rate — valid for short windows (~weeks)
- Custom propagator uses a mean-solar-activity atmosphere; actual density varies by up to 10× with the solar cycle
- Only J2 gravitational harmonic modelled; J3–J6, lunar/solar third-body, and SRP are omitted
- Trajectory change at each burn is instantaneous (impulsive ΔV approximation); burn duration is estimated from thrust and shown in the maneuver log for reference
- Isp and wet mass default to values from public mission docs; actual values are proprietary and may differ

---

## Possible next steps

- Replace USSA76 with NRLMSISE-00 (operational standard; the current scaling is a single-parameter approximation)
- Integrate between burns with the custom propagator — remove the constant-decay assumption
- Annual ΔV budget heatmap: sweep altitude × deadband, visualise the design trade space
- J2 RAAN drift: propagate the orbital plane over 90 days, show SSO maintenance requirement
- Conjunction screening: minimum approach distance to catalogued debris over a 7-day window

---

## Repository layout

```
data/           saved TLEs with retrieval date (fixed epoch → reproducible results)
src/            source code — one module per phase + Streamlit app
  app.py                  Streamlit web app (About, Altitude Decay, Ground Track, Globe, Stationkeeping)
  phase1_altitude_plot.py SGP4 altitude decay figure
  phase1_groundtrack.py   24-hour ground track figure
  phase2_stationkeeping.py stationkeeping simulation
  propagator.py           custom RK45 numerical integrator
figures/        output plots (git-ignored; reproduced by running the scripts)
requirements.txt pip dependencies
```
