"""
Netherlands Weather Map – KNMI data source
============================================
Fetches the latest 10-minute observations from the KNMI Open Data Platform
(dataset: 10-minute-in-situ-meteorological-observations v1.0, NetCDF format)
and renders interpolated weather maps for the Netherlands.

Install once:
    pip install requests matplotlib cartopy numpy scipy pillow netCDF4

Optional QML colour-table files (same folder as script):
    temperature_color_table_high.qml
    wind_gust_color_table.qml
    pressure_color_table.qml
    precipitation_color_table.qml

Run:
    python main_nl.py

You will be prompted for your KNMI API key.
A free anonymous key (valid until 2026-07-01) is pre-filled – just press Enter.

Output PNGs (saved next to the script):
    map_1_temperature.png   map_2_wind_speed.png   map_3_wind_gust.png
    map_4_pressure.png      map_5_humidity.png     map_6_precipitation.png
    map_7_dewpoint.png

Data source: KNMI (knmi.nl) – CC BY 4.0
"""

import os, io, time, datetime, tempfile, threading, warnings
import requests
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union

# netCDF4 is required for reading KNMI NetCDF files
try:
    import netCDF4 as nc
except ImportError:
    raise SystemExit(
        "netCDF4 is not installed. Run:  pip install netCDF4"
    )

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

UPDATE_INTERVAL = 300        # seconds between auto-refreshes
DPI             = 200

# Netherlands bounding box  [lon_min, lon_max, lat_min, lat_max]
EXTENT = [3.2, 7.3, 50.7, 53.7]

GRID_STEP = 0.01             # degrees (~1 km)
SIGMA     = 2.5              # Gaussian smoothing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# KNMI Open Data API
KNMI_API_BASE   = "https://api.dataplatform.knmi.nl/open-data/v1"
DATASET_NAME    = "10-minute-in-situ-meteorological-observations"
DATASET_VERSION = "1.0"

# Public anonymous key valid until 2026-07-01 (replace with your registered key
# for higher rate limits: https://developer.dataplatform.knmi.nl/)
ANON_API_KEY = (
    "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6ImVlNDFjMW"
    "I0MjlkODQ2MThiNWI4ZDViZDAyMTM2YTM3IiwiaCI6Im11cm11cjEyOCJ9"
)

# ── Colour maps ───────────────────────────────────────────────────────────────

def parse_qml(filename, vmin, vmax):
    path = os.path.join(SCRIPT_DIR, filename)
    root = ET.parse(path).getroot()
    vals, cols = [], []
    for item in root.findall(".//colorrampshader/item"):
        v = float(item.get("value"))
        h = item.get("color").lstrip("#")
        r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
        vals.append(v); cols.append((r, g, b))
    vals = np.array(vals, dtype=float)
    cols = np.array(cols, dtype=float)
    idx  = np.argsort(vals)
    vals, cols = vals[idx], cols[idx]
    pos  = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list("qml", list(zip(pos, cols)), N=2048)
    return cmap, Normalize(vmin=vmin, vmax=vmax)


def _bc(stops, v0, v1):
    pos = [max(0., min(1., (v - v0) / (v1 - v0))) for v, _ in stops]
    cm  = LinearSegmentedColormap.from_list(
        "fb", list(zip(pos, [c for _, c in stops])), N=2048
    )
    return cm, Normalize(vmin=v0, vmax=v1)


def load_cmaps():
    c = {}
    try:    c["temp"] = parse_qml("temperature_color_table_high.qml", -40, 50)
    except: c["temp"] = _bc([(-40,"#ff6eff"),(-20,"#32007f"),(-10,"#259aff"),
                               (0,"#d9ecff"),(10,"#52ca0b"),(20,"#f4bd0b"),
                               (30,"#af0f14"),(45,"#c5c5c5")], -40, 45)
    try:    c["wind"] = parse_qml("wind_gust_color_table.qml", 0, 50)
    except: c["wind"] = _bc([(0,"#ffffff"),(10,"#3c96f5"),(20,"#ffa000"),
                               (30,"#e11400"),(50,"#8c645a")], 0, 50)
    c["gust"] = c["wind"]
    try:    c["pres"] = parse_qml("pressure_color_table.qml", 890, 1064)
    except: c["pres"] = _bc([(965,"#32007f"),(990,"#91ccff"),(1000,"#07a127"),
                               (1013,"#f3fb01"),(1030,"#f4520b"),(1050,"#f0a0a0")],
                              960, 1055)
    try:    c["prec"] = parse_qml("precipitation_color_table.qml", 0, 125)
    except: c["prec"] = _bc([(0,"#f0f0f0"),(1,"#0482ff"),(5,"#1acf05"),
                               (10,"#ff7f27"),(20,"#bf0000"),(50,"#64007f")], 0, 50)
    c["hum"]  = _bc([(0,"#ffffff"),(20,"#d0eaff"),(50,"#55a3e0"),
                     (80,"#084a90"),(100,"#021860")], 0, 100)
    try:    c["dewp"] = parse_qml("temperature_color_table_high.qml", -40, 50)
    except: c["dewp"] = c["temp"]
    return c


# ── KNMI data fetch ───────────────────────────────────────────────────────────

def _dewpoint(t, rh):
    if np.isnan(t) or np.isnan(rh) or rh <= 0:
        return np.nan
    a, b = 17.625, 243.04
    g = np.log(rh / 100.0) + a * t / (b + t)
    return round(b * g / (a - g), 1)


def _get_latest_nc_url(api_key: str) -> tuple[str, str]:
    """Return (temporary_download_url, filename) for the latest 10-min file."""
    headers = {"Authorization": api_key}
    # List the most-recently-modified file
    list_url = (
        f"{KNMI_API_BASE}/datasets/{DATASET_NAME}"
        f"/versions/{DATASET_VERSION}/files"
    )
    r = requests.get(
        list_url,
        headers=headers,
        params={"maxKeys": 1, "orderBy": "lastModified", "sorting": "desc"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"KNMI list error: {data['error']}")
    filename = data["files"][0]["filename"]

    # Get signed download URL
    url_resp = requests.get(
        f"{list_url}/{filename}/url",
        headers=headers,
        timeout=20,
    )
    url_resp.raise_for_status()
    url_data = url_resp.json()
    return url_data["temporaryDownloadUrl"], filename


def _read_nc_stations(nc_bytes: bytes) -> tuple[list[dict], str]:
    """Parse a KNMI 10-min NetCDF file into a list of station dicts."""
    # Write to a temp file because netCDF4 needs a real path
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(nc_bytes)
        tmp_path = tmp.name

    try:
        ds = nc.Dataset(tmp_path)

        def _arr(name):
            try:
                v = ds.variables[name][:]
                arr = np.ma.filled(v, np.nan).astype(float).flatten()
                return arr
            except KeyError:
                return None

        lats     = _arr("lat")
        lons     = _arr("lon")
        station_names = None
        try:
            raw = ds.variables["station"][:]
            station_names = ["".join(row.compressed().astype(str)).strip()
                             for row in raw]
        except Exception:
            station_names = [f"S{i}" for i in range(len(lats))]

        # Variable mapping: NetCDF name → our key
        var_map = {
            "ta":  "temp",    # Ambient Temperature 1.5m
            "ff":  "wind",    # Wind Speed 10m average
            "dd":  "wind_dir",
            "fx":  "gust",    # Gust max last 10 min
            "pp":  "pres",    # Sea-level pressure
            "rh":  "hum",     # Relative humidity
            "rg":  "prec",    # Precipitation intensity mm/h
            "td":  "dewp",    # Dew-point temperature
        }

        arrays = {}
        for nc_var, key in var_map.items():
            arrays[key] = _arr(nc_var)

        # Timestamp
        try:
            time_var = ds.variables["time"]
            t_val = nc.num2date(time_var[:][0], units=time_var.units)
            time_str = t_val.strftime("%Y-%m-%d  %H:%M UTC")
        except Exception:
            time_str = datetime.datetime.utcnow().strftime("%Y-%m-%d  %H:%M UTC")

        n = len(lats)
        stations = []
        for i in range(n):
            lat = float(lats[i]) if lats is not None else np.nan
            lon = float(lons[i]) if lons is not None else np.nan
            if np.isnan(lat) or np.isnan(lon):
                continue

            def _v(key):
                a = arrays.get(key)
                if a is None or i >= len(a):
                    return np.nan
                v = float(a[i])
                return np.nan if np.isnan(v) else v

            t  = _v("temp")
            rh = _v("hum")
            # Prefer stored dewpoint, fall back to computed
            dp = _v("dewp")
            if np.isnan(dp):
                dp = _dewpoint(t, rh)

            stations.append({
                "name"    : station_names[i] if i < len(station_names) else f"S{i}",
                "country" : "NL",
                "lat"     : lat,
                "lon"     : lon,
                "temp"    : t,
                "wind"    : _v("wind"),
                "wind_dir": _v("wind_dir"),
                "gust"    : _v("gust"),
                "hum"     : rh,
                "prec"    : _v("prec"),
                "pres"    : _v("pres"),
                "dewp"    : dp,
            })

        ds.close()
    finally:
        os.unlink(tmp_path)

    return stations, time_str


def fetch(api_key: str) -> tuple[list[dict], str]:
    print("  Fetching latest KNMI file …")
    dl_url, filename = _get_latest_nc_url(api_key)
    print(f"  Downloading: {filename}")
    r = requests.get(dl_url, timeout=60)
    r.raise_for_status()
    stations, time_str = _read_nc_stations(r.content)
    print(f"  Parsed {len(stations)} stations  |  {time_str}")
    return stations, time_str


# ── Grid & interpolation ──────────────────────────────────────────────────────

def make_grid():
    lons = np.arange(EXTENT[0], EXTENT[1] + GRID_STEP, GRID_STEP)
    lats = np.arange(EXTENT[2], EXTENT[3] + GRID_STEP, GRID_STEP)
    return np.meshgrid(lons, lats)


def interpolate(stations, key, gx, gy):
    ok = [s for s in stations if not np.isnan(s.get(key, np.nan))]
    if len(ok) < 4:
        return None
    pts = np.array([(s["lon"], s["lat"]) for s in ok])
    vs  = np.array([s[key] for s in ok])
    zi  = griddata(pts, vs, (gx, gy), method="linear")
    znn = griddata(pts, vs, (gx, gy), method="nearest")
    zi  = np.where(np.isnan(zi), znn, zi)
    return gaussian_filter(zi, sigma=SIGMA)


# ── Shapefile cache ───────────────────────────────────────────────────────────

_SHPCACHE = {}

def _get_country_geoms(iso_keep_tuple):
    if iso_keep_tuple in _SHPCACHE:
        return _SHPCACHE[iso_keep_tuple]
    iso_keep = set(iso_keep_tuple)
    shp_path = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shp_path)
    keep_geoms, other_geoms = [], []
    for rec in reader.records():
        iso  = rec.attributes.get("ISO_A2", "")
        geom = rec.geometry
        if iso in iso_keep:
            keep_geoms.append(geom)
        else:
            other_geoms.append(geom)
    keep_union  = unary_union(keep_geoms)  if keep_geoms  else None
    other_union = unary_union(other_geoms) if other_geoms else None
    _SHPCACHE[iso_keep_tuple] = (keep_union, other_union)
    return keep_union, other_union


# ── Rendering ─────────────────────────────────────────────────────────────────

_PE  = [mpe.withStroke(linewidth=3, foreground="white")]
_PE2 = [mpe.withStroke(linewidth=4, foreground="white")]


def render_one(stations, key, cmap, norm, title, unit, fmt,
               time_str, gx, gy, outfile,
               wind_arrows=False, isobars=False):

    grid = interpolate(stations, key, gx, gy)
    ok   = [s for s in stations if not np.isnan(s.get(key, np.nan))]
    if ok:
        vmin_obs = min(s[key] for s in ok)
        vmax_obs = max(s[key] for s in ok)
        s_min    = min(ok, key=lambda s: s[key])
        s_max    = max(ok, key=lambda s: s[key])
    else:
        vmin_obs = vmax_obs = 0
        s_min = s_max = None

    keep_union, other_union = _get_country_geoms(("NL",))

    fig = plt.figure(figsize=(12, 12), facecolor="white", dpi=DPI)
    ax  = fig.add_axes([0.04, 0.06, 0.82, 0.88],
                       projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor("white")

    if grid is not None:
        masked = np.ma.masked_invalid(grid)
        ax.pcolormesh(gx, gy, masked, cmap=cmap, norm=norm,
                      shading="auto", transform=ccrs.PlateCarree(),
                      rasterized=True, zorder=2)

    ax.add_feature(cfeature.OCEAN.with_scale("10m"),
                   facecolor="#cce5ff", zorder=3)
    if other_union is not None:
        ax.add_geometries([other_union], ccrs.PlateCarree(),
                          facecolor="white", edgecolor="none", zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("10m"),
                   facecolor="#cce5ff", edgecolor="#6699bb",
                   linewidth=0.5, zorder=4)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"),
                   edgecolor="#6699bb", linewidth=0.4, zorder=4)

    if isobars and grid is not None:
        pmin   = np.floor(np.nanmin(grid) / 2) * 2
        pmax   = np.ceil(np.nanmax(grid)  / 2) * 2
        levels = np.arange(pmin, pmax + 2, 2)
        cs = ax.contour(gx, gy, grid, levels=levels,
                        colors="#222222", linewidths=0.7,
                        transform=ccrs.PlateCarree(), zorder=6)
        ax.clabel(cs, inline=True, fontsize=6.5, fmt="%d",
                  inline_spacing=4)

    ax.coastlines(resolution="10m", linewidth=1.2, color="#222222", zorder=7)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                   linestyle="-", edgecolor="#333333", linewidth=1.0, zorder=7)
    admin1 = cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "10m",
        facecolor="none", edgecolor="#777777", linewidth=0.6
    )
    ax.add_feature(admin1, zorder=7)

    lon0, lon1, lat0, lat1 = EXTENT
    if wind_arrows:
        for s in ok:
            slon, slat = s["lon"], s["lat"]
            if not (lon0-0.1 <= slon <= lon1+0.1 and
                    lat0-0.1 <= slat <= lat1+0.1):
                continue
            wd = s.get("wind_dir", np.nan)
            if np.isnan(wd) or s[key] < 0.5:
                continue
            wr = np.radians(wd)
            u  = -s[key] * np.sin(wr) * 0.04
            v  = -s[key] * np.cos(wr) * 0.03
            ax.annotate(
                "", xy=(slon+u, slat+v), xytext=(slon, slat),
                arrowprops=dict(arrowstyle="-|>", color="#111111",
                                lw=0.7, mutation_scale=6),
                transform=ccrs.PlateCarree(), zorder=9
            )

    margin = 0.05
    for s in ok:
        slon, slat = s["lon"], s["lat"]
        if not (lon0-margin <= slon <= lon1+margin and
                lat0-margin <= slat <= lat1+margin):
            continue
        ax.plot(slon, slat, "s", color="black", ms=3.0,
                transform=ccrs.PlateCarree(), zorder=10)
        ax.text(slon+0.02, slat+0.015, fmt.format(s[key]),
                fontsize=6, fontweight="bold", color="black",
                path_effects=_PE, transform=ccrs.PlateCarree(), zorder=11)

    if s_min and s_max:
        ax.text(0.012, 0.17, f"{fmt.format(vmax_obs)}{unit}",
                transform=ax.transAxes, fontsize=14, fontweight="bold",
                color="#cc0000", path_effects=_PE2, zorder=20)
        ax.text(0.012, 0.12, s_max["name"],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color="#cc0000", path_effects=_PE2, zorder=20)
        ax.text(0.012, 0.07, f"{fmt.format(vmin_obs)}{unit}",
                transform=ax.transAxes, fontsize=14, fontweight="bold",
                color="#0055cc", path_effects=_PE2, zorder=20)
        ax.text(0.012, 0.02, s_min["name"],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color="#0055cc", path_effects=_PE2, zorder=20)

    cax = fig.add_axes([0.875, 0.06, 0.022, 0.88])
    cb  = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=cax, orientation="vertical", extend="both"
    )
    cb.set_label(unit, fontsize=9, rotation=270, labelpad=14)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)

    ax.set_title(
        f"Nederland  •  {title}\n{time_str}",
        fontsize=13, fontweight="bold", pad=10,
        loc="center", color="#111111"
    )
    fig.text(
        0.5, 0.005,
        "Data source: KNMI (Royal Netherlands Meteorological Institute)  •  "
        "dataplatform.knmi.nl  •  CC BY 4.0",
        ha="center", fontsize=8, color="#888888"
    )

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved  {os.path.basename(outfile)}")


# ── Panel definitions ─────────────────────────────────────────────────────────

PANELS = [
    # (key,  title,                               unit,  fmt,      arrows, isobars, filename)
    ("temp", "Luchttemperatuur 1.5m (°C)",        "°C",  "{:.1f}", False,  False, "map_1_temperature.png"),
    ("wind", "Windsnelheid (m/s)",                "m/s", "{:.1f}", True,   False, "map_2_wind_speed.png"),
    ("gust", "Windstoot max (m/s)",               "m/s", "{:.1f}", True,   False, "map_3_wind_gust.png"),
    ("pres", "Luchtdruk zeeniveau (hPa)",         "hPa", "{:.1f}", False,  True,  "map_4_pressure.png"),
    ("hum",  "Relatieve vochtigheid (%)",         "%",   "{:.0f}", False,  False, "map_5_humidity.png"),
    ("prec", "Neerslag intensiteit (mm/h)",       "mm",  "{:.1f}", False,  False, "map_6_precipitation.png"),
    ("dewp", "Dauwpunt 1.5m (°C)",               "°C",  "{:.1f}", False,  False, "map_7_dewpoint.png"),
]


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_once(api_key, cmaps, gx, gy):
    try:
        stations, time_str = fetch(api_key)
    except Exception as e:
        print(f"  Fetch error: {e}")
        return
    for (key, title, unit, fmt, arrows, isobars, fname) in PANELS:
        outfile = os.path.join(SCRIPT_DIR, fname)
        try:
            render_one(
                stations, key, *cmaps[key],
                title, unit, fmt, time_str, gx, gy, outfile,
                wind_arrows=arrows, isobars=isobars
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  Error {fname}: {e}")


def loop(api_key, cmaps, gx, gy):
    while True:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now}] Updating …")
        run_once(api_key, cmaps, gx, gy)
        print(f"  Next update in {UPDATE_INTERVAL // 60} min.")
        time.sleep(UPDATE_INTERVAL)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    Netherlands Weather Map  –  KNMI 10-min observations  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Data: KNMI Open Data Platform (CC BY 4.0)               ║")
    print("║  An anonymous API key is pre-filled; press Enter to use  ║")
    print("║  it, or paste your own registered key for higher limits.  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    try:
        user_key = input(
            f"  API key [{ANON_API_KEY[:20]}…]: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nStopped."); raise SystemExit

    api_key = user_key if user_key else ANON_API_KEY
    print()
    print("  Auto-updates every 5 minutes.  Press Ctrl+C to stop.")
    print()

    cmaps  = load_cmaps()
    gx, gy = make_grid()
    run_once(api_key, cmaps, gx, gy)

    t = threading.Thread(target=loop, args=(api_key, cmaps, gx, gy), daemon=True)
    t.start()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped.")