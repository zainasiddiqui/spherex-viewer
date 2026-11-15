"""
SPHEREx Time-Lapse Viewer
NASA Space Apps Challenge – Planet X and SPHEREx

A public web tool that shows SPHEREx images of any sky patch as a short
time-lapse sequence so moving objects (brown dwarfs, asteroids, comets…)
stand out by changing position between frames.
"""

import concurrent.futures
import hashlib
import http.client
import logging
import time
import urllib.error
import warnings
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils.data import conf
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from matplotlib import animation
from matplotlib.patches import Circle
from reproject import reproject_interp

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPHEREx Time-Lapse Viewer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global config ────────────────────────────────────────────────────────────
logging.getLogger("astropy").setLevel(logging.ERROR)
conf.remote_timeout = 180
plt.rcParams["animation.embed_limit"] = 80

CACHE_DIR = Path("cached_frames")
CACHE_DIR.mkdir(exist_ok=True)

# ── Built-in targets (same as the notebook) ──────────────────────────────────
TARGETS = {
    "WISE 0855": dict(
        coord="08h55m10.831s -07d14m42.53s",
        epoch=2010.4,
        pm=(-8118.9, 679.3),
        note="Very cold brown dwarf – use detector 5 or 6",
    ),
    "Barnard's Star": dict(
        name="Barnard's Star",
        epoch=2000.0,
        pm=(-802.8, 10362.5),
        note="Highest proper-motion star",
    ),
    "Epsilon Indi A": dict(
        coord="22h03m21.65363s -56d47m09.5228s",
        epoch=2000.0,
        pm=(3966.661, -2536.192),
        note="Nearby K dwarf",
    ),
    "Epsilon Indi B": dict(
        # Position of the Ba/Bb brown-dwarf pair (~6.7 arcmin from Eps Ind A)
        coord="22h04m10.52s -56d46m57.7s",
        epoch=2000.0,
        pm=(3966.661, -2536.192),
        note="Brown-dwarf binary companion (~6.7' from Eps Ind A)",
    ),
    "RW Aur": dict(
        coord="05h07m49.456s +30d24m04.78s",
        epoch=2000.0,
        pm=(2.64, -27.4),
        note="Variable T Tauri star",
    ),
    "R Dra": dict(
        coord="16h32m40.23s +66d45m17.9s",
        epoch=2000.0,
        pm=(-20.6, 6.15),
        note="Mira variable star",
    ),
}

BAND_COLORS = {
    1: "#0072B2", 2: "#56B4E9", 3: "#009E73",
    4: "#E69F00", 5: "#D55E00", 6: "#CC79A7",
}
STRETCH = dict(vmin=-2, vmax=8, cmap="gray_r", origin="lower", interpolation="nearest")


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions (ported from the hardened notebook)
# ══════════════════════════════════════════════════════════════════════════════

def get_target(cfg):
    if "name" in cfg:
        coord = SkyCoord.from_name(cfg["name"])
    elif cfg.get("unit") == "deg":
        ra, dec = [float(v) for v in cfg["coord"].split()]
        coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    else:
        coord = SkyCoord(cfg["coord"], frame="icrs")
    return dict(coord=coord, epoch=cfg["epoch"], pm=cfg["pm"])


def mjd_to_year(mjd):
    return Time(np.asarray(mjd, dtype=float), format="mjd").decimalyear


def position_at(coord, epoch, pm, years):
    dt = np.asarray(years, dtype=float) - epoch
    return coord.spherical_offsets_by(pm[0] * dt * u.mas, pm[1] * dt * u.mas)


@st.cache_data(ttl=3600, show_spinner=False)
def query_images(ra_deg, dec_deg, detectors=None, tries=6, base_wait=12):
    """Query IRSA TAP for SPHEREx images covering the position."""
    det_clause = ""
    if detectors:
        names = ", ".join(f"'SPHEREx-D{int(d)}'" for d in detectors)
        det_clause = f"AND p.energy_bandpassname IN ({names})"

    query = f"""
    SELECT a.uri, p.time_bounds_lower, p.energy_bandpassname
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra_deg:.6f}, {dec_deg:.6f}), p.poly)
    {det_clause}
    ORDER BY p.time_bounds_lower
    """

    last = None
    for k in range(tries):
        try:
            t = Irsa.query_tap(query).to_table()
            break
        except Exception as e:
            last = e
            if k < tries - 1:
                time.sleep(base_wait * (1.6 ** k))
    else:
        raise RuntimeError(
            f"IRSA TAP query failed after {tries} attempts.\nLast error: {last}\n"
            "This is usually a temporary IRSA issue. Wait a few minutes and try again."
        )

    df = t.to_pandas()
    df = df.rename(columns={"time_bounds_lower": "mjd", "energy_bandpassname": "band"})
    df["band"] = df["band"].astype(str).str.extract(r"D(\d)")[0].astype(float).astype("Int64")
    df["uri"] = df["uri"].astype(str)
    df = df.drop_duplicates("uri").sort_values("mjd").reset_index(drop=True)
    return df


def thin_evenly(df, n):
    if len(df) <= n:
        return df.reset_index(drop=True)
    idx = np.unique(np.linspace(0, len(df) - 1, n).round().astype(int))
    return df.iloc[idx].reset_index(drop=True)


def build_cutout_url(base, center, size_deg):
    return (
        f"https://irsa.ipac.caltech.edu/{str(base).lstrip('/')}"
        f"?center={center.ra.deg:.7f},{center.dec.deg:.7f}d&size={size_deg:.5f}"
    )


def _get_hdu(hdul, name, index):
    try:
        return hdul[name]
    except KeyError:
        return hdul[index]


def _wavelength_at(header, hdul, x, y):
    spectral_wcs = WCS(header, fobj=hdul, key="W")
    spectral_wcs.sip = None
    wavelength, bandwidth = spectral_wcs.pixel_to_world(x, y)
    try:
        bw = float(bandwidth.to(u.micrometer).value)
    except Exception:
        bw = np.nan
    return float(wavelength.to(u.micrometer).value), bw


def fetch_one(url, target, tries=4):
    for k in range(tries):
        try:
            with fits.open(url, cache=False, memmap=False) as hdul:
                for hdu in hdul:
                    _ = hdu.data
                img_hdu = _get_hdu(hdul, "IMAGE", 1)
                flag_hdu = _get_hdu(hdul, "FLAGS", 2)
                header = img_hdu.header.copy()
                x, y = WCS(header).world_to_pixel(target)
                try:
                    wl, bw = _wavelength_at(header, hdul, x, y)
                except Exception:
                    wl, bw = np.nan, np.nan
                return dict(
                    image=np.array(img_hdu.data, dtype=np.float32),
                    flags=np.array(flag_hdu.data),
                    header=header,
                    wavelength_um=wl,
                    bandwidth_um=bw,
                    version=str(hdul[0].header.get("VERSION", "")),
                )
        except (TimeoutError, urllib.error.HTTPError, http.client.IncompleteRead,
                OSError, Exception):
            if k == tries - 1:
                return None
            time.sleep(6 * (k + 1))
    return None


def fetch_all(df, center, size_deg, track, workers=8):
    results = [None] * len(df)

    def job(i):
        return i, fetch_one(build_cutout_url(df["uri"][i], center, size_deg), track[i])

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(job, i) for i in range(len(df))]
        for f in concurrent.futures.as_completed(futures):
            i, r = f.result()
            results[i] = r
    return results


def make_grid(center, fov_arcmin, pix_arcsec=6.2):
    n = int(np.ceil(fov_arcmin * 60 / pix_arcsec))
    n += 1 - n % 2
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.cdelt = [-pix_arcsec / 3600, pix_arcsec / 3600]
    return w, (n, n)


def robust_sigma(a):
    return 1.4826 * np.nanmedian(np.abs(a - np.nanmedian(a)))


def remove_background(img, n_iter=5, clip=3.0):
    yy, xx = np.indices(img.shape)
    good = np.isfinite(img)
    A = np.column_stack([np.ones(good.sum()), xx[good], yy[good]]).astype(float)
    z = img[good].astype(float)
    keep = np.ones(z.size, bool)
    coef = np.array([np.nanmedian(z), 0.0, 0.0])
    for _ in range(n_iter):
        coef = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
        resid = z - A @ coef
        s = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        if not np.isfinite(s) or s <= 0:
            break
        keep = np.abs(resid) < clip * s
    out = img - (coef[0] + coef[1] * xx + coef[2] * yy).astype(np.float32)
    return out - np.nanmedian(out)


def align_frames(df, raw, grid, shape, track, mask_bits=(), min_coverage=0.85):
    frames, dropped = [], []
    bitmask = int(sum(1 << int(b) for b in mask_bits))
    for i, r in enumerate(raw):
        if r is None:
            dropped.append((i, "download failed"))
            continue
        img = r["image"].astype(np.float32).copy()
        flags = r["flags"]
        if bitmask:
            img[(flags.astype(np.int64) & bitmask) > 0] = np.nan
        out = reproject_interp(
            (img, WCS(r["header"])), grid, shape_out=shape,
            order="bilinear", return_footprint=False,
        )
        coverage = float(np.isfinite(out).mean())
        if coverage < min_coverage:
            dropped.append((i, f"only {coverage:.0%} coverage"))
            continue
        out = remove_background(out)
        sig = robust_sigma(out)
        if not np.isfinite(sig) or sig <= 0:
            dropped.append((i, "bad noise estimate"))
            continue
        x, y = grid.world_to_pixel(track[i])
        mjd = float(df["mjd"][i])
        frames.append(dict(
            i=i, mjd=mjd, date=Time(mjd, format="mjd").iso[:10],
            band=int(df["band"][i]), wavelength_um=r["wavelength_um"],
            flux=out, snr=out / sig, sigma=float(sig),
            coverage=coverage, flagged_frac=float((flags > 0).mean()),
            x=float(x), y=float(y), version=r["version"],
        ))
    frames.sort(key=lambda f: f["mjd"])
    return frames, dropped


def build_frames(images, tgt, fov_arcmin=3.0, max_frames=40,
                 mask_bits=(), pix_arcsec=6.2, workers=8):
    df = thin_evenly(images, max_frames)
    years = mjd_to_year(df["mjd"].to_numpy())
    track = position_at(tgt["coord"], tgt["epoch"], tgt["pm"], years)
    center = position_at(tgt["coord"], tgt["epoch"], tgt["pm"], float(np.median(years)))
    size_deg = fov_arcmin * 1.5 / 60
    grid, shape = make_grid(center, fov_arcmin, pix_arcsec)
    raw = fetch_all(df, center, size_deg, track, workers)
    frames, dropped = align_frames(df, raw, grid, shape, track, mask_bits)
    return dict(
        frames=frames, grid=grid, shape=shape, center=center,
        dropped=dropped, df=df, pix_arcsec=pix_arcsec,
    )


# ── Caching helpers ──────────────────────────────────────────────────────────

def _cache_key(name, fov, detectors, max_frames):
    det_str = ",".join(map(str, sorted(detectors or [])))
    raw = f"{name}|{fov:.2f}|{det_str}|{max_frames}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def save_frames_to_disk(key, frames):
    path = CACHE_DIR / f"{key}.npz"
    np.savez_compressed(
        path,
        snr=np.stack([f["snr"] for f in frames]),
        mjd=np.array([f["mjd"] for f in frames]),
        band=np.array([f["band"] for f in frames]),
        wavelength_um=np.array([f["wavelength_um"] for f in frames]),
        x=np.array([f["x"] for f in frames]),
        y=np.array([f["y"] for f in frames]),
        date=np.array([f["date"] for f in frames]),
    )
    return path


def load_frames_from_disk(key):
    path = CACHE_DIR / f"{key}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    frames = []
    for i in range(len(data["mjd"])):
        frames.append(dict(
            snr=data["snr"][i],
            mjd=float(data["mjd"][i]),
            band=int(data["band"][i]),
            wavelength_um=float(data["wavelength_um"][i]),
            x=float(data["x"][i]),
            y=float(data["y"][i]),
            date=str(data["date"][i]),
        ))
    return frames


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _label(f):
    wl = "?" if not np.isfinite(f.get("wavelength_um", np.nan)) else f"{f['wavelength_um']:.2f}"
    return f"{f['date']}  D{f['band']}  {wl} µm"


def plot_contact_sheet(frames, ncols=6, max_frames=18):
    show = frames[:: max(1, len(frames) // max_frames)][:max_frames]
    nrows = int(np.ceil(len(show) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, f in zip(axes.ravel(), show):
        ax.imshow(f["snr"], **STRETCH)
        ax.add_patch(Circle((f["x"], f["y"]), 2.8, fill=False, ec="tab:red", lw=1))
        ax.set_title(_label(f), fontsize=7)
    fig.suptitle("Frame grid (red ring = predicted position)", y=1.01, fontsize=11)
    fig.tight_layout()
    return fig


def plot_single_frame(f, size=5.0):
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(f["snr"], **STRETCH)
    ax.add_patch(Circle((f["x"], f["y"]), 2.8, fill=False, ec="tab:red", lw=1.6))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(_label(f), fontsize=10)
    ax.text(0.98, 0.02, "N ↑   E ←", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8,
            bbox=dict(fc="white", ec="none", alpha=0.7))
    fig.tight_layout()
    return fig


def make_flipbook_html(frames, interval_ms=450, size=5.5):
    """Return HTML string of a playable animation."""
    f0 = frames[0]
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(f0["snr"], **STRETCH)
    ring = Circle((f0["x"], f0["y"]), 2.8, fill=False, ec="tab:red", lw=1.6)
    ax.add_patch(ring)
    if len(frames) > 1:
        ax.plot([f["x"] for f in frames], [f["y"] for f in frames],
                "-", color="tab:red", lw=0.8, alpha=0.5)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.text(0.98, 0.02, "N ↑   E ←", transform=ax.transAxes, va="bottom", ha="right",
            fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.7))

    def update(k):
        f = frames[k]
        im.set_data(f["snr"])
        ring.center = (f["x"], f["y"])
        txt.set_text(f"{_label(f)}\nframe {k+1}/{len(frames)}")
        return im, ring, txt

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=interval_ms, blit=False)
    html = anim.to_jshtml()
    plt.close(fig)
    return html


def coadd(frs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stack = np.nanmean([f["snr"] for f in frs], axis=0)
    stack = stack - np.nanmedian(stack)
    return stack / robust_sigma(stack)


def plot_difference(frames, fraction=0.34):
    n = max(1, int(len(frames) * fraction))
    early, late = frames[:n], frames[-n:]
    A, B = coadd(early), coadd(late)
    D = coadd([dict(snr=B - A)])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, img, title, grp in zip(
        axes[:2],
        [A, B],
        [f"EARLY  {early[0]['date']} → {early[-1]['date']}",
         f"LATE   {late[0]['date']} → {late[-1]['date']}"],
        [early, late],
    ):
        ax.imshow(img, **STRETCH)
        ax.add_patch(Circle(
            (np.mean([f["x"] for f in grp]), np.mean([f["y"] for f in grp])),
            2.8, fill=False, ec="tab:red", lw=1.2,
        ))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[2].imshow(D, vmin=-6, vmax=6, cmap="RdBu_r", origin="lower", interpolation="nearest")
    axes[2].set_title("LATE − EARLY", fontsize=9)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar UI
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("SPHEREx Time-Lapse")
st.sidebar.markdown(
    "Choose a target, then build a time-lapse of SPHEREx observations "
    "to look for objects that change position between visits."
)

# --- Target selection ---
mode = st.sidebar.radio(
    "Target input",
    ["Challenge targets", "Custom coordinates", "Object name"],
    index=0,
)

tgt = None
target_label = ""

if mode == "Challenge targets":
    choice = st.sidebar.selectbox(
        "Target",
        list(TARGETS.keys()),
        format_func=lambda k: f"{k}  –  {TARGETS[k].get('note', '')}",
    )
    cfg = TARGETS[choice]
    tgt = get_target(cfg)
    target_label = choice

elif mode == "Custom coordinates":
    coord_unit = st.sidebar.radio(
        "Coordinate format",
        ["Degrees", "h/m/s  +  d/m/s"],
        index=0,
        horizontal=True,
        help="Degrees match the SPHEREx tutorials. Sexagesimal is the classic format.",
    )
    if coord_unit == "Degrees":
        ra_deg = st.sidebar.number_input(
            "RA (degrees)", value=133.7951, min_value=0.0, max_value=360.0,
            step=0.0001, format="%.4f",
            help="Right Ascension in decimal degrees (0–360)",
        )
        dec_deg = st.sidebar.number_input(
            "Dec (degrees)", value=-7.2451, min_value=-90.0, max_value=90.0,
            step=0.0001, format="%.4f",
            help="Declination in decimal degrees (−90 to +90)",
        )
        try:
            coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
        except Exception as e:
            st.sidebar.error(f"Could not create coordinates: {e}")
            coord = None
    else:
        ra_str = st.sidebar.text_input(
            "RA (h/m/s)", value="08h55m10.831s",
            help="Example: 08h55m10.831s or 8:55:10.831",
        )
        dec_str = st.sidebar.text_input(
            "Dec (d/m/s)", value="-07d14m42.53s",
            help="Example: -07d14m42.53s or -7:14:42.53",
        )
        try:
            coord = SkyCoord(f"{ra_str} {dec_str}", frame="icrs")
        except Exception as e:
            st.sidebar.error(f"Could not parse coordinates: {e}")
            coord = None

    epoch = st.sidebar.number_input("Epoch of coordinates", value=2010.4, step=0.1)
    pm_ra = st.sidebar.number_input("pmRA×cosDec (mas/yr)", value=-8118.9, step=1.0)
    pm_dec = st.sidebar.number_input("pmDec (mas/yr)", value=679.3, step=1.0)

    if coord is not None:
        tgt = dict(coord=coord, epoch=epoch, pm=(pm_ra, pm_dec))
        target_label = f"{coord.ra.deg:.4f}, {coord.dec.deg:.4f}"

else:  # Object name
    name = st.sidebar.text_input("Object name (SIMBAD)", "Barnard's Star")
    if name:
        try:
            coord = SkyCoord.from_name(name)
            tgt = dict(coord=coord, epoch=2000.0, pm=(0.0, 0.0))
            target_label = name
            st.sidebar.success(f"Resolved: RA {coord.ra.deg:.4f}°, Dec {coord.dec.deg:.4f}°")
        except Exception as e:
            st.sidebar.error(f"Name lookup failed: {e}")

# --- Observation settings ---
st.sidebar.markdown("---")
st.sidebar.subheader("Observation settings")

fov = st.sidebar.slider("Field of view (arcmin)", 1.5, 8.0, 3.0, 0.5)
detectors = st.sidebar.multiselect(
    "Detectors (leave empty = all)",
    options=[1, 2, 3, 4, 5, 6],
    default=[6],
    help="WISE 0855 is very red → prefer 5 or 6",
)
max_frames = st.sidebar.slider("Max frames", 10, 80, 30, 5)
query_epoch = st.sidebar.number_input("Query epoch (year)", value=2026.0, step=0.1)

# --- Demo / cache mode ---
st.sidebar.markdown("---")
use_cache = st.sidebar.checkbox(
    "Prefer cached frames (demo mode)",
    value=False,
    help="If checked, load previously saved frames for this target instead of querying IRSA.",
)

run_button = st.sidebar.button("Build time-lapse", type="primary", use_container_width=True)

st.sidebar.markdown(
    """
    ---
    **Getting started**
    - For a fast-moving brown dwarf, try **WISE 0855** with detector **6** and a 3′ field of view
    - Larger fields of view help track faster objects but take longer to load
    - Once frames are loaded, enable **Prefer cached frames** for instant replay
    - The difference image often makes motion clearer than the time-lapse alone
    """
)


# ══════════════════════════════════════════════════════════════════════════════
# Main area
# ══════════════════════════════════════════════════════════════════════════════

st.title("SPHEREx Time-Lapse Viewer")
st.markdown(
    "Explore how any region of the SPHEREx infrared sky changes over time. "
    "Enter coordinates, search by name, or pick a preset target to view a "
    "time-lapse, frame grid, and difference image that highlight moving objects."
)

if tgt is None:
    st.info("← Choose a target in the sidebar and click **Build time-lapse**.")
    st.stop()

# Show predicted position (decimal degrees, matching SPHEREx tutorials)
pos_now = position_at(tgt["coord"], tgt["epoch"], tgt["pm"], query_epoch)
st.markdown(
    f"**Target:** {target_label}  ·  "
    f"Predicted {query_epoch:.1f} position: "
    f"**RA {pos_now.ra.deg:.4f}°, Dec {pos_now.dec.deg:.4f}°**  ·  "
    f"Moved {tgt['coord'].separation(pos_now).arcsec:.1f}″ since epoch {tgt['epoch']}"
)

# ── Build or load frames ─────────────────────────────────────────────────────
frames = None
key = _cache_key(target_label, fov, detectors, max_frames)

if run_button or use_cache:
    # Try cache first when requested
    if use_cache:
        frames = load_frames_from_disk(key)
        if frames:
            st.success(f"Loaded {len(frames)} cached frames.")
        else:
            st.warning("No cache found for these settings – will query IRSA.")

    if frames is None and run_button:
        progress = st.progress(0, text="🔭 Querying NASA IRSA archive…")
        status = st.empty()

        try:
            status.info("Step 1/3 – Searching SPHEREx image catalog…")
            images = query_images(
                pos_now.ra.deg, pos_now.dec.deg,
                detectors=detectors or None,
            )
            progress.progress(15, text=f"Found {len(images)} SPHEREx images")
            status.success(f"Found {len(images)} SPHEREx images covering this position.")

            if len(images) == 0:
                st.error(
                    "No SPHEREx images cover this position with the selected detectors. "
                    "Try clearing the detector filter or a different target."
                )
                st.stop()

            status.info("Step 2/3 – Downloading image cutouts from orbit…")
            progress.progress(25, text="Downloading cutouts from SPHEREx…")

            # We need a non-cached build so we can show progress
            df = thin_evenly(images, max_frames)
            years = mjd_to_year(df["mjd"].to_numpy())
            track = position_at(tgt["coord"], tgt["epoch"], tgt["pm"], years)
            center = position_at(
                tgt["coord"], tgt["epoch"], tgt["pm"], float(np.median(years))
            )
            size_deg = fov * 1.5 / 60
            grid, shape = make_grid(center, fov, 6.2)

            # Download with a simple progress callback
            raw = [None] * len(df)
            done = 0

            def job(i):
                return i, fetch_one(
                    build_cutout_url(df["uri"][i], center, size_deg), track[i]
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                futures = [ex.submit(job, i) for i in range(len(df))]
                for f in concurrent.futures.as_completed(futures):
                    i, r = f.result()
                    raw[i] = r
                    done += 1
                    pct = 25 + int(55 * done / len(df))
                    progress.progress(pct, text=f"📡 Received {done}/{len(df)} cutouts")

            status.info("Step 3/3 – Aligning frames & removing sky background…")
            frames, dropped = align_frames(df, raw, grid, shape, track)
            progress.progress(100, text="✨ Time-lapse ready")

            if not frames:
                st.error(
                    "All downloads failed or frames were dropped. "
                    "IRSA may be having trouble – try again in a few minutes, "
                    "or reduce Max frames / FOV."
                )
                st.stop()

            # Save to disk for future demo mode
            save_frames_to_disk(key, frames)
            st.success(
                f"Ready: **{len(frames)}** frames "
                f"({len(dropped)} dropped). Cached for instant reload."
            )

        except Exception as e:
            st.error(f"Failed to build frames:\n\n```\n{e}\n```")
            st.info(
                "Most failures are temporary IRSA 502s. Wait a few minutes and click "
                "**Build time-lapse** again. You can also lower Max frames or FOV."
            )
            st.stop()

# ── Display results ──────────────────────────────────────────────────────────
if frames:
    tab_flip, tab_sheet, tab_diff, tab_info = st.tabs(
        ["Time-lapse", "Frame grid", "Difference image", "Frame info"]
    )

    with tab_flip:
        st.subheader("Time-lapse sequence")
        st.caption(
            "Press play. Fixed stars stay put; a moving object slides along the red track. "
            "The red ring marks the predicted position of the target."
        )
        speed = st.slider("Frame interval (ms)", 150, 1200, 450, 50)
        try:
            html = make_flipbook_html(frames, interval_ms=speed)
            st.components.v1.html(html, height=620, scrolling=False)
        except Exception as e:
            st.warning(f"Could not build interactive animation ({e}). Showing slider instead.")
            idx = st.slider("Frame", 0, len(frames) - 1, 0)
            fig = plot_single_frame(frames[idx])
            st.pyplot(fig)
            plt.close(fig)

    with tab_sheet:
        st.subheader("Frame grid")
        st.caption(
            "All selected frames arranged in time order. "
            "The red ring marks the predicted position of the target in each frame."
        )
        fig = plot_contact_sheet(frames)
        st.pyplot(fig)
        plt.close(fig)

    with tab_diff:
        st.subheader("Early vs Late difference")
        st.caption(
            "Left/middle: co-added early and late frames. "
            "Right: late − early. A mover appears as a blue/red pair; "
            "a star that only changed brightness shows a single blob."
        )
        if len(frames) >= 4:
            fig = plot_difference(frames)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Need at least 4 frames for a useful difference image.")

    with tab_info:
        st.subheader("Frame metadata")
        meta = pd.DataFrame([
            {
                "date": f["date"],
                "band": f["band"],
                "wavelength_um": f.get("wavelength_um", np.nan),
                "x_pred": f["x"],
                "y_pred": f["y"],
            }
            for f in frames
        ])
        st.dataframe(meta.round(3), use_container_width=True)

        st.download_button(
            "Download frame list (CSV)",
            meta.to_csv(index=False),
            file_name=f"{target_label.replace(' ', '_')}_frames.csv",
            mime="text/csv",
        )

else:
    st.info("Click **Build time-lapse** in the sidebar to start.")
