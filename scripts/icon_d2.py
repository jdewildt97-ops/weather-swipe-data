#!/usr/bin/env python3
"""
Haalt de nieuwste ICON-EU-run van de DWD (open data) op en zet windstoten + wind om
naar kleine PNG-bestanden per uur, voor de kaart in Weather Swipe.

Per uur één PNG (RGB) op het volle ICON-EU-raster, 0,0625° (≈ 7 km):
  R = windstoot in km/u (0–254), 255 = geen data
  G = u-wind (m/s) * 4 + 128      (west → oost)
  B = v-wind (m/s) * 4 + 128      (zuid → noord)
Plus per uur een zon-PNG (grijs, sNN.png):
  0–60 = zonneschijn in minuten in het uur dat eindigt op dat tijdstip
  200  = nacht (zon onder de horizon), 255 = geen data
  Afgeleid uit ASWDIR_S (directe straling, gemiddelde sinds modelstart): per uur
  terugrekenen, omzetten naar straling loodrecht op de zon en vergelijken met heldere hemel.
Plus out/latest.json met run, tijden, gebied en schaal.

Bron: Deutscher Wetterdienst, ICON-EU (https://opendata.dwd.de), CC BY 4.0.
"""
import bz2, datetime as dt, io, json, os, sys, time, urllib.request
import numpy as np
from PIL import Image
import eccodes

BASE = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
STEPS = range(0, 49)        # 0 t/m +48 uur
DOWN = 1                    # volle resolutie (0,0625°)

def url(run, var, step):
    hh = run.strftime("%H"); stamp = run.strftime("%Y%m%d%H")
    return f"{BASE}/{hh}/{var}/icon-eu_europe_regular-lat-lon_single-level_{stamp}_{step:03d}_{var.upper()}.grib2.bz2"

def fetch(u, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(u, timeout=60) as r:
                return bz2.decompress(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(3)
        except Exception:
            time.sleep(3)
    return None

def decode(raw):
    """GRIB2 → (2D array noord-boven, west-links, lat_noord, lat_zuid, lon_west, lon_oost)."""
    gid = eccodes.codes_new_from_message(raw)
    try:
        ni = eccodes.codes_get(gid, "Ni"); nj = eccodes.codes_get(gid, "Nj")
        la1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"); la2 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
        lo1 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"); lo2 = eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees")
        miss = eccodes.codes_get(gid, "missingValue")
        v = np.array(eccodes.codes_get_values(gid), dtype=np.float32).reshape(nj, ni)
    finally:
        eccodes.codes_release(gid)
    v[v == miss] = np.nan
    if lo1 > 180: lo1 -= 360
    if lo2 > 180: lo2 -= 360
    if la1 < la2: v = v[::-1]; la1, la2 = la2, la1          # noorden boven
    return v, la1, la2, lo1, lo2

def down(a):
    nj, ni = a.shape; nj -= nj % DOWN; ni -= ni % DOWN
    a = a[:nj, :ni].reshape(nj // DOWN, DOWN, ni // DOWN, DOWN)
    with np.errstate(invalid="ignore"):
        return np.nanmean(a, axis=(1, 3))

def latest_run():
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    for back in range(0, 30):
        run = now - dt.timedelta(hours=back)
        if run.hour % 6: continue          # 00/06/12/18 UTC lopen tot +120 uur
        # run is pas compleet (voor onze 48 uur) als stap 48 er is
        if fetch_head(url(run, "vmax_10m", 48)): return run
    raise SystemExit("geen complete ICON-EU-run gevonden")

def fetch_head(u):
    try:
        req = urllib.request.Request(u, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r: return r.status == 200
    except Exception: return False

def sun_elev(t, lat, lon):
    """Zonnehoogte (graden) op tijdstip t (UTC) voor rasters lat/lon (NOAA-benadering)."""
    doy = t.timetuple().tm_yday; hr = t.hour + t.minute / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    ha = np.radians((hr * 60 + eqt + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    return np.degrees(np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)))

def sun_minutes(F, t_mid, t_end, lat, lon):
    """F = gemiddelde directe straling op een horizontaal vlak in het uur (W/m²) → minuten zon, 200 = nacht."""
    h = sun_elev(t_mid, lat, lon)
    s = np.sin(np.radians(np.clip(h, 0.5, 90)))
    dni = F / s                                                # straling loodrecht op de zon
    am = 1 / s
    dni_cs = 1361 * 0.7 ** (am ** 0.678)                       # heldere hemel (Meinel)
    frac = np.clip(dni / (0.8 * dni_cs), 0, 1)
    frac = np.where(dni_cs < 120, np.clip(dni / 120, 0, 1) * frac, frac)   # lage zon: WMO-drempel 120 W/m²
    m = np.rint(frac * 60)
    night = (h < -0.5) & (sun_elev(t_end, lat, lon) < 0) & (sun_elev(t_mid - dt.timedelta(minutes=30), lat, lon) < 0)
    out = np.where(night, 200, m)
    return np.where(np.isnan(F), 255, out).astype(np.uint8)

def main():
    run = latest_run()
    print("run", run.isoformat())
    os.makedirs(OUT, exist_ok=True)
    times, box = [], None
    prev_a, has_sun, LAT, LON = None, False, None, None
    for step in STEPS:
        g = fetch(url(run, "vmax_10m", step)); u = fetch(url(run, "u_10m", step)); v = fetch(url(run, "v_10m", step))
        if not (g and u and v): print("stap ontbreekt", step); continue
        G, la1, la2, lo1, lo2 = decode(g); U = decode(u)[0]; V = decode(v)[0]
        G, U, V = down(G), down(U), down(V)
        r = np.where(np.isnan(G), 255, np.clip(np.rint(G * 3.6), 0, 254)).astype(np.uint8)
        gg = np.where(np.isnan(U), 128, np.clip(np.rint(U * 4 + 128), 0, 255)).astype(np.uint8)
        bb = np.where(np.isnan(V), 128, np.clip(np.rint(V * 4 + 128), 0, 255)).astype(np.uint8)
        Image.fromarray(np.dstack([r, gg, bb]), "RGB").save(os.path.join(OUT, f"{step:02d}.png"), optimize=True)
        t = run + dt.timedelta(hours=step)
        times.append({ "step": step, "t": int(t.timestamp()) })
        if box is None:
            # na samenvoegen van DOWN×DOWN cellen ligt het midden van cel k op la1 - (k·DOWN + (DOWN-1)/2)·Δ
            nj0, ni0 = decode(g)[0].shape
            dlat0 = (la1 - la2) / (nj0 - 1); dlon0 = (lo2 - lo1) / (ni0 - 1)
            box = { "n": round(la1 - (DOWN - 1) / 2 * dlat0, 5), "w": round(lo1 + (DOWN - 1) / 2 * dlon0, 5),
                    "dlat": round(DOWN * dlat0, 6), "dlon": round(DOWN * dlon0, 6), "ni": int(r.shape[1]), "nj": int(r.shape[0]) }
        # zon: ASWDIR_S is het gemiddelde sinds de start → gemiddelde over het afgelopen uur terugrekenen
        a = fetch(url(run, "aswdir_s", step))
        A = down(decode(a)[0]) if a else None
        if LAT is None:
            LAT = (box["n"] - np.arange(box["nj"]) * box["dlat"])[:, None] * np.ones((1, box["ni"]))
            LON = (box["w"] + np.arange(box["ni"]) * box["dlon"])[None, :] * np.ones((box["nj"], 1))
        if A is not None and prev_a is not None and A.shape == r.shape:
            F = np.maximum(0, step * A - (step - 1) * prev_a)
            sm = sun_minutes(F, t - dt.timedelta(minutes=30), t, LAT, LON)
            Image.fromarray(sm, "L").save(os.path.join(OUT, f"s{step:02d}.png"), optimize=True)
            times[-1]["sun"] = 1; has_sun = True
        prev_a = A
        print("stap", step, "ok")
    if not times: raise SystemExit("geen data")
    meta = { "model": "ICON-EU", "source": "Deutscher Wetterdienst (CC BY 4.0)", "run": int(run.timestamp()),
             "grid": box, "scale": { "gust": "R = km/u, 255 = geen data", "uv": "(G|B - 128) / 4 = m/s",
             "sun": "sNN.png: 0-60 = minuten zon, 200 = nacht, 255 = geen data" }, "sun": has_sun, "times": times }
    with open(os.path.join(OUT, "latest.json"), "w") as f: json.dump(meta, f)
    with open(os.path.join(OUT, "index.html"), "w") as f: f.write("<p>Weather Swipe kaartdata (ICON-EU, DWD)</p>")
    print("klaar", len(times), "uren")

if __name__ == "__main__":
    main()
