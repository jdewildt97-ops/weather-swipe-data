#!/usr/bin/env python3
"""
Haalt de nieuwste ICON-EU-run van de DWD (open data) op en zet windstoten + wind om
naar kleine PNG-bestanden per uur, voor de kaart in Weather Swipe.

Per uur één PNG (RGB) op het volle ICON-EU-raster, 0,0625° (≈ 7 km):
  R = windstoot in km/u (0–254), 255 = geen data
  G = u-wind (m/s) * 4 + 128      (west → oost)
  B = v-wind (m/s) * 4 + 128      (zuid → noord)
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

def main():
    run = latest_run()
    print("run", run.isoformat())
    os.makedirs(OUT, exist_ok=True)
    times, box = [], None
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
        print("stap", step, "ok")
    if not times: raise SystemExit("geen data")
    meta = { "model": "ICON-EU", "source": "Deutscher Wetterdienst (CC BY 4.0)", "run": int(run.timestamp()),
             "grid": box, "scale": { "gust": "R = km/u, 255 = geen data", "uv": "(G|B - 128) / 4 = m/s" }, "times": times }
    with open(os.path.join(OUT, "latest.json"), "w") as f: json.dump(meta, f)
    with open(os.path.join(OUT, "index.html"), "w") as f: f.write("<p>Weather Swipe kaartdata (ICON-EU, DWD)</p>")
    print("klaar", len(times), "uren")

if __name__ == "__main__":
    main()
