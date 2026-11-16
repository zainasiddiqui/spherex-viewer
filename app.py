"""
SPHEREx Time-Lapse Viewer
NASA Space Apps Challenge – Planet X and SPHEREx

A public web tool that shows SPHEREx images of any sky patch as a short
time-lapse sequence so moving objects (brown dwarfs, asteroids, comets…)
stand out by changing position between frames.

Data: NASA SPHEREx spectral images, served by the NASA/IPAC Infrared Science
Archive (IRSA). Optional overlays: Gaia DR3 (via VizieR) and Aladin Lite (CDS).

Shareable links (all optional):
  ?target=WISE 0855                          a built-in target
  ?ra=133.76&dec=-7.24                       custom coordinates (degrees)
  ?name=Barnard's Star&pmra=-802.8&pmdec=10362.5
  &fov=3  &det=6  or  det=5,6  or  det=all   &frames=30  &gaia=1
  &epoch=2000  &pmra=..  &pmdec=..           (mas/yr, for moving targets)
"""

import concurrent.futures
import io
import logging
import time
import warnings
from pathlib import Path
from urllib.parse import quote, urlencode

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
from scipy.ndimage import gaussian_filter

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

BASE_URL = "https://spherex-sky-viewer.streamlit.app/"   # used to build share links
DEMO_DIR = Path(__file__).parent / "demo_data"            # optional saved views (.npz) in the repo
PIX_ARCSEC = 6.2                                          # SPHEREx pixel size is ~6.15"
GAIA_EPOCH = 2016.0

# ── Built-in targets ─────────────────────────────────────────────────────────
# coord = a position you trust, epoch = the year that position is valid for,
# pm = (pmRA*cos(Dec), pmDec) in milli-arcseconds per year.
TARGETS = {
    "WISE 0855": dict(
        coord="08h55m10.831s -07d14m42.53s",
        epoch=2010.4,
        pm=(-8118.9, 679.3),
        note="Very cold brown dwarf – use detector 5 or 6",
    ),
    "Barnard's Star": dict(
        # Fixed position (approx. J2000, check on SIMBAD) so no name lookup is needed.
        coord="17h57m48.50s +04d41m36.2s",
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
BAND_RANGES = {
    1: "0.75–1.09 µm", 2: "1.10–1.62 µm", 3: "1.63–2.41 µm",
    4: "2.42–3.82 µm", 5: "3.83–4.41 µm", 6: "4.42–5.00 µm",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data access and processing (the parts that worked on live data are unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class NoImagesError(Exception):
    pass


def get_target(cfg):
    coord = SkyCoord(cfg["coord"], frame="icrs")
    return dict(coord=coord, epoch=cfg["epoch"], pm=cfg["pm"])


def mjd_to_year(mjd):
    return Time(np.asarray(mjd, dtype=float), format="mjd").decimalyear


def position_at(coord, epoch, pm, years):
    dt = np.asarray(years, dtype=float) - epoch
    return coord.spherical_offsets_by(pm[0] * dt * u.mas, pm[1] * dt * u.mas)


@st.cache_data(ttl=86400, show_spinner=False)
def resolve_name(name):
    """Look a name up in SIMBAD/NED/VizieR through the CDS Sesame service."""
    c = SkyCoord.from_name(name)
    return float(c.ra.deg), float(c.dec.deg)


@st.cache_data(ttl=3600, show_spinner=False)
def query_images(ra_deg, dec_deg, detectors=None, tries=5, base_wait=10):
    """Ask IRSA which SPHEREx images cover the position. Cached for one hour,
    so new data shows up automatically."""
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
            # First try a normal query. If IRSA answers with a 502, switch to an
            # asynchronous job (submit, then poll), which avoids long-held connections.
            t = Irsa.query_tap(query, async_job=(k >= 1)).to_table()
            break
        except Exception as e:
            last = e
            if k < tries - 1:
                time.sleep(base_wait * (1.5 ** k))
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


def thin_per_band(df, max_frames):
    """Keep at most max_frames in total, split evenly between the detector bands present."""
    bands = sorted(df["band"].dropna().unique())
    if len(bands) <= 1:
        return thin_evenly(df, max_frames)
    per = max(4, max_frames // len(bands))
    parts = [thin_evenly(df[df["band"] == b], per) for b in bands]
    return pd.concat(parts).sort_values("mjd").reset_index(drop=True)


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
        except Exception:
            if k == tries - 1:
                return None
            time.sleep(6 * (k + 1))
    return None


def make_grid(center, fov_arcmin, pix_arcsec=PIX_ARCSEC):
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


def build_view(tgt, pos_now, fov, detectors, max_frames, label="", progress=None):
    """The whole pipeline: query, download cutouts, align. `progress` is an st.progress object."""
    def say(pct, text):
        if progress is not None:
            progress.progress(pct, text=text)

    say(5, "🔭 Searching the SPHEREx catalogue at IRSA…")
    images = query_images(
        round(float(pos_now.ra.deg), 6), round(float(pos_now.dec.deg), 6),
        tuple(detectors) if detectors else None,
    )
    if len(images) == 0:
        raise NoImagesError("No SPHEREx images cover this position with the selected detectors.")
    say(15, f"Found {len(images)} images – choosing {min(len(images), max_frames)}")

    df = thin_per_band(images, max_frames)
    years = mjd_to_year(df["mjd"].to_numpy())
    track = position_at(tgt["coord"], tgt["epoch"], tgt["pm"], years)
    center = position_at(tgt["coord"], tgt["epoch"], tgt["pm"], float(np.median(years)))
    size_deg = fov * 1.5 / 60
    grid, shape = make_grid(center, fov)

    raw = [None] * len(df)
    done = 0

    def job(i):
        return i, fetch_one(build_cutout_url(df["uri"][i], center, size_deg), track[i])

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(job, i) for i in range(len(df))]
        for f in concurrent.futures.as_completed(futures):
            i, r = f.result()
            raw[i] = r
            done += 1
            say(15 + int(65 * done / len(df)), f"📡 Received {done}/{len(df)} cutouts")

    say(85, "Aligning frames and removing sky background…")
    frames, dropped = align_frames(df, raw, grid, shape, track)
    say(100, "✨ Ready")
    return dict(
        frames=frames, grid=grid, shape=shape, center=center, dropped=dropped,
        n_found=int(len(images)), first_mjd=float(images["mjd"].min()),
        last_mjd=float(images["mjd"].max()), label=label, fov=float(fov),
        detectors=list(detectors or []), built_at=time.time(), source="IRSA (live)",
    )


# ── Results shared between visitors (so a repeat view loads instantly) ───────

@st.cache_resource
def _shared_results():
    return {}


def _remember(key, res, max_entries=24):
    store = _shared_results()
    store[key] = res
    while len(store) > max_entries:
        oldest = min(store, key=lambda k: store[k]["built_at"])
        store.pop(oldest)


# ── Saving and loading views (used for a backup demo that needs no internet) ─

def result_to_npz_bytes(res):
    fr = res["frames"]
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        snr=np.stack([f["snr"] for f in fr]).astype(np.float32),
        flux=np.stack([f["flux"] for f in fr]).astype(np.float32),
        mjd=np.array([f["mjd"] for f in fr]),
        date=np.array([f["date"] for f in fr]),
        band=np.array([f["band"] for f in fr]),
        wavelength_um=np.array([f["wavelength_um"] for f in fr], dtype=float),
        sigma=np.array([f["sigma"] for f in fr]),
        coverage=np.array([f["coverage"] for f in fr]),
        flagged_frac=np.array([f["flagged_frac"] for f in fr]),
        x=np.array([f["x"] for f in fr]),
        y=np.array([f["y"] for f in fr]),
        version=np.array([str(f["version"]) for f in fr]),
        wcs_header=np.array(res["grid"].to_header().tostring(sep="\n")),
        center=np.array([res["center"].ra.deg, res["center"].dec.deg]),
        label=np.array(res["label"]),
        fov=np.array(res["fov"]),
        n_found=np.array(res["n_found"]),
        first_mjd=np.array(res["first_mjd"]),
        last_mjd=np.array(res["last_mjd"]),
    )
    return buf.getvalue()


def load_demo_npz(path):
    d = np.load(path, allow_pickle=False)
    frames = []
    for i in range(len(d["mjd"])):
        frames.append(dict(
            i=i, mjd=float(d["mjd"][i]), date=str(d["date"][i]), band=int(d["band"][i]),
            wavelength_um=float(d["wavelength_um"][i]), flux=d["flux"][i], snr=d["snr"][i],
            sigma=float(d["sigma"][i]), coverage=float(d["coverage"][i]),
            flagged_frac=float(d["flagged_frac"][i]), x=float(d["x"][i]), y=float(d["y"][i]),
            version=str(d["version"][i]),
        ))
    grid = WCS(fits.Header.fromstring(str(d["wcs_header"]), sep="\n"))
    n = frames[0]["snr"].shape
    return dict(
        frames=frames, grid=grid, shape=n,
        center=SkyCoord(float(d["center"][0]) * u.deg, float(d["center"][1]) * u.deg),
        dropped=[], n_found=int(d["n_found"]), first_mjd=float(d["first_mjd"]),
        last_mjd=float(d["last_mjd"]), label=str(d["label"]), fov=float(d["fov"]),
        detectors=sorted({f["band"] for f in frames}), built_at=time.time(),
        source=f"saved demo file ({Path(path).name})", share=None,
    )


# ── Gaia DR3 overlay ─────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gaia(ra, dec, radius_arcmin, gmax=19.0, max_rows=200):
    """Gaia DR3 stars around the field, from VizieR (catalogue I/355/gaiadr3)."""
    from astroquery.vizier import Vizier
    v = Vizier(columns=["RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Gmag"],
               column_filters={"Gmag": f"<{gmax}"}, row_limit=3000)
    tabs = v.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=radius_arcmin * u.arcmin,
                          catalog="I/355/gaiadr3")
    if len(tabs) == 0:
        return pd.DataFrame(columns=["ra", "dec", "pmra", "pmdec", "g"])
    t = tabs[0].to_pandas().rename(columns={"RA_ICRS": "ra", "DE_ICRS": "dec", "pmRA": "pmra",
                                            "pmDE": "pmdec", "Gmag": "g"})
    t[["pmra", "pmdec"]] = t[["pmra", "pmdec"]].fillna(0.0)
    return t.sort_values("g").head(max_rows).reset_index(drop=True)


def gaia_pixels(gaia, grid, shape, frames):
    """Gaia positions in each frame's pixels, moved to that frame's date using proper motion."""
    sc = SkyCoord(gaia["ra"].to_numpy() * u.deg, gaia["dec"].to_numpy() * u.deg)
    pmra, pmdec = gaia["pmra"].to_numpy(), gaia["pmdec"].to_numpy()
    out = []
    for f in frames:
        dt = Time(f["mjd"], format="mjd").decimalyear - GAIA_EPOCH
        pos = sc.spherical_offsets_by(pmra * dt * u.mas, pmdec * dt * u.mas)
        x, y = grid.world_to_pixel(pos)
        ok = (x >= -0.5) & (x < shape[1] - 0.5) & (y >= -0.5) & (y < shape[0] - 0.5)
        out.append(np.column_stack([np.asarray(x)[ok], np.asarray(y)[ok]]))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Display, stacking and measurement helpers
# ══════════════════════════════════════════════════════════════════════════════

def stretch(vmax=6.0, smooth=0.0):
    return dict(vmin=-2, vmax=vmax, cmap="gray_r", origin="lower",
                interpolation="bicubic" if smooth > 0 else "nearest")


def _label(f):
    wl = "?" if not np.isfinite(f.get("wavelength_um", np.nan)) else f"{f['wavelength_um']:.2f}"
    extra = f"  (avg of {f['n_stack']})" if f.get("n_stack", 1) > 1 else ""
    return f"{f['date']}  D{f['band']}  {wl} µm{extra}"


def coadd(frs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stack = np.nanmean([f["snr"] for f in frs], axis=0)
    stack = stack - np.nanmedian(stack)
    return stack / robust_sigma(stack)


def smooth_image(img, sigma):
    if sigma <= 0:
        return img
    bad = ~np.isfinite(img)
    out = gaussian_filter(np.where(bad, 0.0, img), sigma)
    s = robust_sigma(out)
    if np.isfinite(s) and s > 0:
        out = out / s
    out[bad] = np.nan
    return out


def make_display_frames(frames, stack=1, smooth=0.0):
    """Frames ready to show: optionally averaged with neighbours in time (same detector only,
    which improves signal-to-noise) and optionally smoothed."""
    out, half = [], stack // 2
    for i, f in enumerate(frames):
        g = dict(f)
        grp = [h for h in frames[max(0, i - half): i + half + 1] if h["band"] == f["band"]]
        if stack > 1 and len(grp) > 1:
            g["snr"] = coadd(grp)
            g["n_stack"] = len(grp)
        g["snr"] = smooth_image(g["snr"], smooth)
        out.append(g)
    return out


def make_timelapse_html(frames, interval_ms=450, vmax=6.0, smooth=0.0, gaia_xy=None, size=5.5):
    """Return HTML for a playable animation (play, pause, step, speed controls)."""
    st_kw = stretch(vmax, smooth)
    f0 = frames[0]
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(f0["snr"], **st_kw)
    ring = Circle((f0["x"], f0["y"]), 2.8, fill=False, ec="tab:red", lw=1.6)
    ax.add_patch(ring)
    if len(frames) > 1:
        ax.plot([f["x"] for f in frames], [f["y"] for f in frames],
                "-", color="tab:red", lw=0.8, alpha=0.5)
    scat = None
    if gaia_xy is not None:
        scat = ax.scatter([], [], s=55, facecolors="none", edgecolors="#0072B2", linewidths=0.9)
        scat.set_offsets(gaia_xy[0] if len(gaia_xy[0]) else np.empty((0, 2)))
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.text(0.98, 0.02, "N ↑   E ←", transform=ax.transAxes, va="bottom", ha="right",
            fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.7))
    if scat is not None:
        ax.text(0.02, 0.02, "○ Gaia DR3 star", transform=ax.transAxes, va="bottom", ha="left",
                fontsize=8, color="#0072B2", bbox=dict(fc="white", ec="none", alpha=0.7))

    def update(k):
        f = frames[k]
        im.set_data(f["snr"])
        ring.center = (f["x"], f["y"])
        if scat is not None:
            scat.set_offsets(gaia_xy[k] if len(gaia_xy[k]) else np.empty((0, 2)))
        txt.set_text(f"{_label(f)}\nframe {k + 1}/{len(frames)}   flagged px: {f.get('flagged_frac', 0):.1%}")
        return im, ring, txt

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=False)
    html = anim.to_jshtml()
    plt.close(fig)
    return html


def plot_contact_sheet(frames, ncols=6, max_frames=18, vmax=6.0, smooth=0.0):
    show = frames[:: max(1, int(np.ceil(len(frames) / max_frames)))][:max_frames]
    nrows = int(np.ceil(len(show) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, f in zip(axes.ravel(), show):
        ax.imshow(f["snr"], **stretch(vmax, smooth))
        ax.add_patch(Circle((f["x"], f["y"]), 2.8, fill=False, ec="tab:red", lw=1))
        ax.set_title(_label(f), fontsize=7)
    fig.suptitle("Frame grid (red ring = predicted position)", y=1.01, fontsize=11)
    fig.tight_layout()
    return fig


def plot_detector_grid(frames, n_windows=4, vmax=6.0):
    """Rows = detector, columns = time window. Each panel adds together every frame in that cell."""
    mjds = np.array([f["mjd"] for f in frames])
    edges = np.linspace(mjds.min(), mjds.max() + 1, n_windows + 1)
    bands = sorted({f["band"] for f in frames})
    fig, axes = plt.subplots(len(bands), n_windows, figsize=(3.0 * n_windows, 3.1 * len(bands)),
                             squeeze=False)
    for r, b in enumerate(bands):
        for c in range(n_windows):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            sel = [f for f in frames if f["band"] == b and edges[c] <= f["mjd"] < edges[c + 1]]
            if not sel:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                continue
            ax.imshow(coadd(sel), **stretch(vmax))
            ax.add_patch(Circle((np.mean([f["x"] for f in sel]), np.mean([f["y"] for f in sel])),
                                2.8, fill=False, ec="tab:red", lw=1))
            d0 = Time(min(f["mjd"] for f in sel), format="mjd").iso[:7]
            d1 = Time(max(f["mjd"] for f in sel), format="mjd").iso[:7]
            ax.set_title(f"D{b} ({BAND_RANGES.get(b, '')})\n{d0}" + ("" if d0 == d1 else f" → {d1}")
                         + f"  n={len(sel)}", fontsize=8)
    fig.suptitle("Co-added images: one row per detector, one column per time window", y=1.0)
    fig.tight_layout()
    return fig


def plot_difference(frames, fraction=0.34, vmax=6.0):
    n = max(1, int(len(frames) * fraction))
    early, late = frames[:n], frames[-n:]
    A, B = coadd(early), coadd(late)
    D = coadd([dict(snr=B - A)])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, img, title, grp in zip(
        axes[:2], [A, B],
        [f"EARLY  {early[0]['date']} → {early[-1]['date']}  (n={len(early)})",
         f"LATE   {late[0]['date']} → {late[-1]['date']}  (n={len(late)})"],
        [early, late],
    ):
        ax.imshow(img, **stretch(vmax))
        ax.add_patch(Circle((np.mean([f["x"] for f in grp]), np.mean([f["y"] for f in grp])),
                            2.8, fill=False, ec="tab:red", lw=1.2))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[2].imshow(D, vmin=-6, vmax=6, cmap="RdBu_r", origin="lower", interpolation="nearest")
    axes[2].set_title("LATE − EARLY  (blue = fainter, red = brighter)", fontsize=9)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.tight_layout()
    return fig


def aperture_photometry(f, r_ap=2.0, r_in=4.0, r_out=6.5, x=None, y=None):
    """Flux density (µJy) in a circle at (x, y), minus the sky level in a ring around it.
    No aperture or colour correction, so the absolute scale is approximate."""
    img = f["flux"]
    x = f["x"] if x is None else x
    y = f["y"] if y is None else y
    yy, xx = np.indices(img.shape)
    rr = np.hypot(xx - x, yy - y)
    ap = (rr <= r_ap) & np.isfinite(img)
    ann = (rr >= r_in) & (rr <= r_out) & np.isfinite(img)
    if ap.sum() < 4 or ann.sum() < 12:
        return np.nan, np.nan
    bkg = np.median(img[ann])
    sig = robust_sigma(img[ann])
    omega = (PIX_ARCSEC / 206265.0) ** 2            # pixel solid angle (sr)
    flux = np.sum(img[ap] - bkg) * omega * 1e12     # MJy -> µJy
    err = sig * np.sqrt(ap.sum()) * omega * 1e12    # ignores pixel-to-pixel correlation
    return float(flux), float(err)


def measure(frames, r_ap=2.0, dx=0.0, dy=0.0):
    rows = []
    for f in frames:
        flux, err = aperture_photometry(f, r_ap=r_ap, x=f["x"] + dx, y=f["y"] + dy)
        rows.append(dict(date=f["date"], mjd=f["mjd"], detector=f["band"],
                         wavelength_um=f["wavelength_um"], flux_uJy=flux, err_uJy=err,
                         flagged_frac=f["flagged_frac"]))
    return pd.DataFrame(rows)


def bin_spectrum(phot, width=0.1):
    """Weighted average of the measurements in wavelength bins, separately for each detector."""
    rows = []
    for b, g in phot.dropna(subset=["wavelength_um", "flux_uJy", "err_uJy"]).groupby("detector"):
        g = g[g["err_uJy"] > 0]
        if g.empty:
            continue
        idx = np.floor((g["wavelength_um"] - g["wavelength_um"].min()) / width).astype(int)
        for _, h in g.groupby(idx):
            w = 1.0 / h["err_uJy"] ** 2
            rows.append(dict(detector=b, wavelength_um=float(np.average(h["wavelength_um"], weights=w)),
                             flux_uJy=float(np.average(h["flux_uJy"], weights=w)),
                             err_uJy=float(np.sqrt(1.0 / w.sum())), n=len(h)))
    return pd.DataFrame(rows, columns=["detector", "wavelength_um", "flux_uJy", "err_uJy", "n"])


def plot_spectrum(phot, binned=True, width=0.1):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
    for b, g in phot.groupby("detector"):
        col = BAND_COLORS.get(b, "k")
        g1 = g[np.isfinite(g["wavelength_um"])]
        a1.errorbar(g1["wavelength_um"], g1["flux_uJy"], g1["err_uJy"], fmt="o", ms=3.5, lw=0.8,
                    alpha=0.45, color=col, label=f"D{b}")
        dates = Time(g["mjd"].to_numpy(), format="mjd").to_datetime()
        a2.errorbar(dates, g["flux_uJy"], g["err_uJy"], fmt="o", ms=3.5, lw=0.8, alpha=0.6,
                    color=col, label=f"D{b}")
    if binned:
        bs = bin_spectrum(phot, width)
        for b, g in bs.groupby("detector"):
            a1.errorbar(g["wavelength_um"], g["flux_uJy"], g["err_uJy"], fmt="s", ms=6, lw=1.6,
                        color=BAND_COLORS.get(b, "k"), mec="k", mew=0.6)
    for a in (a1, a2):
        a.axhline(0, color="gray", lw=0.6)
    a1.set(xlabel="wavelength (µm)", ylabel="flux density (µJy)", title="Spectrum at the red ring")
    a2.set(xlabel="date", ylabel="flux density (µJy)", title="Light curve (all frames)")
    a2.tick_params(axis="x", rotation=30)
    a1.legend(title="detector", fontsize=8)
    fig.tight_layout()
    return fig


def show_html(html, height):
    """Show an HTML string in the page. st.components.v1.html is deprecated, so use st.iframe when present."""
    html = html.strip()
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        st.components.v1.html(html, height=height, scrolling=False)


def aladin_html(ra, dec, fov_deg, survey, height=480):
    return f"""
    <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js" charset="utf-8"></script>
    <div id="aladin-lite-div" style="width:100%;height:{height}px;"></div>
    <script>
      A.init.then(() => {{
        A.aladin('#aladin-lite-div', {{survey: "{survey}", fov: {fov_deg:.4f},
                                      target: "{ra:.5f} {dec:.5f}"}});
      }});
    </script>
    """


# ══════════════════════════════════════════════════════════════════════════════
# URL parameters (shareable / embeddable links)
# ══════════════════════════════════════════════════════════════════════════════

MODES = ["Challenge targets", "Custom coordinates", "Object name"]


def read_url_params():
    """Turn ?ra=..&dec=..&fov=.. into widget defaults. Returns (settings, warnings)."""
    qp = st.query_params
    out, warns = {}, []

    def num(key, lo=None, hi=None):
        if key not in qp:
            return None
        try:
            v = float(qp[key])
        except (TypeError, ValueError):
            warns.append(f"Ignored '{key}={qp[key]}' in the link (not a number).")
            return None
        if not np.isfinite(v) or (lo is not None and not (lo <= v <= hi)):
            warns.append(f"Ignored '{key}={qp[key]}' in the link (out of range).")
            return None
        return v

    t = qp.get("target")
    ra, dec = num("ra", 0, 360), num("dec", -90, 90)
    name = qp.get("name")
    if t:
        match = next((k for k in TARGETS if k.lower() == t.lower()), None)
        if match:
            out.update(mode=MODES[0], target=match, has_target=True)
        else:
            warns.append(f"Unknown target '{t}' in the link. Choose one of: {', '.join(TARGETS)}.")
    if "has_target" not in out:
        if ra is not None and dec is not None:
            out.update(mode=MODES[1], ra=ra, dec=dec, has_target=True)
        elif ra is not None or dec is not None:
            warns.append("The link needs both 'ra' and 'dec'.")
        elif name:
            out.update(mode=MODES[2], name=name, has_target=True)
    for key, dest, lo, hi in [("epoch", "epoch", 1900, 2100), ("pmra", "pm_ra", -1e5, 1e5),
                              ("pmdec", "pm_dec", -1e5, 1e5)]:
        v = num(key, lo, hi)
        if v is not None:
            out[dest] = v
    fov = num("fov", 0.5, 20)
    if fov is not None:
        out["fov"] = float(min(8.0, max(1.5, round(fov * 2) / 2)))
    frames = num("frames", 1, 1000)
    if frames is not None:
        out["frames"] = int(min(80, max(10, round(frames / 5) * 5)))
    det = qp.get("det")
    if det:
        if det.lower() == "all":
            out["det"] = []
        else:
            try:
                vals = sorted({int(x) for x in det.split(",") if x.strip()})
                if vals and all(1 <= v <= 6 for v in vals):
                    out["det"] = vals
                else:
                    raise ValueError
            except ValueError:
                warns.append(f"Ignored 'det={det}' in the link (use 1–6, e.g. det=5,6 or det=all).")
    if qp.get("gaia") in ("1", "true", "yes"):
        out["gaia"] = True
    return out, warns


def make_share_url(mode, params):
    q = {k: v for k, v in params.items() if v is not None}
    return BASE_URL + "?" + urlencode(q, quote_via=quote)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar UI
# ══════════════════════════════════════════════════════════════════════════════

url, url_warnings = read_url_params()
# Default for "Custom coordinates": WISE 0855's position in 2026, with no proper motion.
_d = position_at(get_target(TARGETS["WISE 0855"])["coord"], 2010.4, TARGETS["WISE 0855"]["pm"], 2026.0)

st.sidebar.title("SPHEREx Time-Lapse")
st.sidebar.markdown(
    "Choose a target, then build a time-lapse of SPHEREx observations "
    "to look for objects that change position between visits."
)
for w in url_warnings:
    st.sidebar.warning(w)

mode = st.sidebar.radio("Target input", MODES, index=MODES.index(url.get("mode", MODES[0])))

tgt = None
target_label = ""
share_params = {}

if mode == MODES[0]:
    keys = list(TARGETS.keys())
    choice = st.sidebar.selectbox(
        "Target", keys, index=keys.index(url.get("target", keys[0])),
        format_func=lambda k: f"{k}  –  {TARGETS[k].get('note', '')}",
    )
    tgt = get_target(TARGETS[choice])
    target_label = choice
    share_params = dict(target=choice)

elif mode == MODES[1]:
    coord_unit = st.sidebar.radio(
        "Coordinate format", ["Degrees", "h/m/s  +  d/m/s"], index=0, horizontal=True,
        help="Degrees match the SPHEREx tutorials. Sexagesimal is the classic format.",
    )
    coord = None
    if coord_unit == "Degrees":
        ra_deg = st.sidebar.number_input(
            "RA (degrees)", value=float(url.get("ra", round(float(_d.ra.deg), 4))),
            min_value=0.0, max_value=360.0, step=0.0001, format="%.4f",
            help="Right Ascension in decimal degrees (0–360)")
        dec_deg = st.sidebar.number_input(
            "Dec (degrees)", value=float(url.get("dec", round(float(_d.dec.deg), 4))),
            min_value=-90.0, max_value=90.0, step=0.0001, format="%.4f",
            help="Declination in decimal degrees (−90 to +90)")
        coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    else:
        ra_str = st.sidebar.text_input("RA (h/m/s)", value="08h55m10.831s",
                                       help="Example: 08h55m10.831s or 8:55:10.831")
        dec_str = st.sidebar.text_input("Dec (d/m/s)", value="-07d14m42.53s",
                                        help="Example: -07d14m42.53s or -7:14:42.53")
        try:
            coord = SkyCoord(f"{ra_str} {dec_str}", frame="icrs")
        except Exception as e:
            st.sidebar.error(f"Could not parse coordinates: {e}")

    epoch = st.sidebar.number_input("Year these coordinates are valid for", value=float(url.get("epoch", 2026.0)), step=0.1,
                                    help="Catalogue positions are often for the year 2000.")
    pm_ra = st.sidebar.number_input("Proper motion RA×cosDec (mas/yr)", value=float(url.get("pm_ra", 0.0)), step=1.0,
                                    help="0 for distant stars and galaxies. A nearby moving object needs its real value.")
    pm_dec = st.sidebar.number_input("Proper motion Dec (mas/yr)", value=float(url.get("pm_dec", 0.0)), step=1.0)
    if coord is not None:
        tgt = dict(coord=coord, epoch=epoch, pm=(pm_ra, pm_dec))
        target_label = f"{coord.ra.deg:.4f}, {coord.dec.deg:.4f}"
        share_params = dict(ra=f"{coord.ra.deg:.5f}", dec=f"{coord.dec.deg:.5f}", epoch=epoch,
                            pmra=pm_ra or None, pmdec=pm_dec or None)

else:  # Object name
    name = st.sidebar.text_input("Object name (SIMBAD)", url.get("name", "Barnard's Star"))
    epoch = st.sidebar.number_input("Year of the catalogue position", value=float(url.get("epoch", 2000.0)), step=0.1)
    pm_ra = st.sidebar.number_input("Proper motion RA×cosDec (mas/yr)", value=float(url.get("pm_ra", 0.0)), step=1.0)
    pm_dec = st.sidebar.number_input("Proper motion Dec (mas/yr)", value=float(url.get("pm_dec", 0.0)), step=1.0)
    st.sidebar.caption("The name lookup gives a position only. For a fast-moving object, "
                       "enter its proper motion (from SIMBAD), or use a built-in target.")
    if name:
        try:
            ra0, dec0 = resolve_name(name)
            tgt = dict(coord=SkyCoord(ra0 * u.deg, dec0 * u.deg, frame="icrs"), epoch=epoch, pm=(pm_ra, pm_dec))
            target_label = name
            st.sidebar.success(f"Resolved: RA {ra0:.4f}°, Dec {dec0:.4f}°")
            share_params = dict(name=name, epoch=epoch, pmra=pm_ra or None, pmdec=pm_dec or None)
        except Exception as e:
            st.sidebar.error(f"Name lookup failed: {e}")

# --- Observation settings (changing these needs a new download) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Observation settings")
fov = st.sidebar.slider("Field of view (arcmin)", 1.5, 8.0, float(url.get("fov", 3.0)), 0.5,
                        help="Larger fields can follow faster objects but take longer to download.")
detectors = st.sidebar.multiselect(
    "Detectors (leave empty = all six)", options=[1, 2, 3, 4, 5, 6], default=url.get("det", [6]),
    format_func=lambda d: f"{d}  ({BAND_RANGES[d]})",
    help="Each detector sees a different range of infrared colours. WISE 0855 is very red: prefer 5 or 6.")
max_frames = st.sidebar.slider("Max frames", 10, 80, int(url.get("frames", 30)), 5)

with st.sidebar.expander("Advanced"):
    query_epoch = st.number_input("Year used to search the archive", value=2026.0, step=0.1)
    fresh = st.checkbox("Re-query the archive (ignore saved results)", value=False,
                        help="Use this to pick up newly released SPHEREx data right away.")

build_btn = st.sidebar.button("Build time-lapse", type="primary", key="build_btn")

# --- Display options (applied instantly, no new download) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Display options")
stack = st.sidebar.select_slider(
    "Average neighbouring frames", options=[1, 3, 5, 7], value=1,
    help="Averages each frame with its neighbours in time (same detector) to reduce noise.")
smooth = st.sidebar.slider("Smooth image (pixels)", 0.0, 1.5, 0.0, 0.25,
                           help="Display only. Makes faint sources easier to see.")
vmax = st.sidebar.slider("Brightness limit (σ)", 3.0, 15.0, 6.0, 0.5,
                         help="Lower values make faint sources look darker.")
show_gaia = st.sidebar.checkbox("Overlay Gaia DR3 stars", value=bool(url.get("gaia", False)),
                                help="Shows known stars, moved to each frame's date with their proper motion.")
max_flag = st.sidebar.slider("Hide frames with more flagged pixels than (%)", 0, 100, 100, 5,
                             help="Flags mark pixels SPHEREx marked as questionable.")

# --- Saved demo views from the repository (works without the archive) ---
demo_files = sorted(DEMO_DIR.glob("*.npz")) if DEMO_DIR.exists() else []
if demo_files:
    with st.sidebar.expander("Saved demo views"):
        demo_choice = st.selectbox("File", [p.name for p in demo_files], label_visibility="collapsed")
        if st.button("Load saved view", key="demo_btn"):
            try:
                st.session_state["result"] = load_demo_npz(DEMO_DIR / demo_choice)
            except Exception as e:
                st.error(f"Could not load that file: {e}")

st.sidebar.markdown(
    """
    ---
    **Tips**
    - Frames are real SPHEREx images, about 6″ per pixel, so objects are only a pixel or two wide.
    - Objects move slowly: the **Difference** and **Detector co-adds** tabs often show motion better than the time-lapse.
    - Larger fields of view follow faster objects but take longer to load.
    """
)


# ══════════════════════════════════════════════════════════════════════════════
# Main area
# ══════════════════════════════════════════════════════════════════════════════

st.title("SPHEREx Time-Lapse Viewer")
st.markdown(
    "Explore how any region of the SPHEREx infrared sky changes over time. "
    "Enter coordinates, search by name, or pick a preset target to view a "
    "time-lapse, co-added images, a difference image and a spectrum."
)

if tgt is not None:
    pos_now = position_at(tgt["coord"], tgt["epoch"], tgt["pm"], query_epoch)
    st.markdown(
        f"**Target:** {target_label}  ·  Expected {query_epoch:.1f} position: "
        f"**RA {pos_now.ra.deg:.4f}°, Dec {pos_now.dec.deg:.4f}°**  ·  "
        f"moved {tgt['coord'].separation(pos_now).arcsec:.1f}″ since {tgt['epoch']:.1f}"
    )

# Open a shared link: build right away, once per visit.
autorun = False
if url.get("has_target") and tgt is not None and not st.session_state.get("_url_autorun_done"):
    st.session_state["_url_autorun_done"] = True
    autorun = True

if (build_btn or autorun) and tgt is not None:
    share = make_share_url(mode, dict(
        **share_params, fov=fov, det=",".join(map(str, sorted(detectors))) if detectors else "all",
        frames=max_frames))
    key = (round(float(tgt["coord"].ra.deg), 5), round(float(tgt["coord"].dec.deg), 5), tgt["epoch"],
           tuple(tgt["pm"]), fov, tuple(sorted(detectors)), max_frames, query_epoch)
    store = _shared_results()
    if fresh:
        query_images.clear()
        store.pop(key, None)
    hit = store.get(key)
    if hit is not None and time.time() - hit["built_at"] < 3600:
        res = dict(hit, share=share)
        st.session_state["result"] = res
        st.info(f"Showing a view built {int((time.time() - hit['built_at']) / 60)} min ago. "
                "Tick 'Re-query the archive' under Advanced to refresh it.")
    else:
        progress = st.progress(0, text="Starting…")
        st.session_state.pop("result", None)
        try:
            res = build_view(tgt, pos_now, fov, detectors, max_frames, label=target_label, progress=progress)
            res["share"] = share
            if not res["frames"]:
                st.error("All downloads failed or every frame was rejected. IRSA may be having trouble: "
                         "try again in a few minutes, or reduce Max frames or the field of view.")
            else:
                _remember(key, res)
                st.session_state["result"] = res
        except NoImagesError as e:
            st.error(f"{e} Try clearing the detector filter, or check the coordinates.")
        except Exception as e:
            st.error(f"Failed to build the time-lapse:\n\n```\n{e}\n```")
            st.info("Most failures are temporary IRSA errors (502). Wait a few minutes and click "
                    "**Build time-lapse** again, or lower Max frames or the field of view.")
        finally:
            progress.empty()

res = st.session_state.get("result")

if not res:
    st.info("← Choose a target in the sidebar and click **Build time-lapse**.")
    st.stop()

# ── Summary line ─────────────────────────────────────────────────────────────
all_frames = res["frames"]
bands_present = sorted({f["band"] for f in all_frames})
first = Time(res["first_mjd"], format="mjd").iso[:10]
last = Time(res["last_mjd"], format="mjd").iso[:10]
st.success(
    f"**{res['label']}** · {len(all_frames)} frames shown "
    f"({len(res['dropped'])} rejected) from {res['n_found']} SPHEREx images in the archive "
    f"({first} → {last}) · detectors {', '.join(map(str, bands_present))} · source: {res['source']}"
)
if res.get("share"):
    with st.expander("🔗 Share or embed this view"):
        st.code(res["share"], language=None)
        st.caption("Anyone who opens this link sees the same view. To embed it in another page, "
                   "use it in an iframe and add &embed=true to hide the menu.")

frames = [f for f in all_frames if f.get("flagged_frac", 0.0) <= max_flag / 100.0]
if len(frames) < 1:
    st.warning("All frames are hidden by the flagged-pixel limit. Raise it in the sidebar.")
    st.stop()

tab_names = ["Time-lapse", "Frame grid", "Detector co-adds", "Difference", "Spectrum",
             "Sky context", "Frame info"]
t_lapse, t_grid, t_det, t_diff, t_spec, t_sky, t_info = st.tabs(tab_names)

# ── Time-lapse ───────────────────────────────────────────────────────────────
with t_lapse:
    st.subheader("Time-lapse sequence")
    st.caption("Press play. Fixed stars stay put; a moving object slides along the red track. "
               "The red ring marks the predicted position of the target. Dark = bright.")
    c1, c2 = st.columns(2)
    speed = c1.slider("Frame interval (ms)", 150, 1200, 450, 50)
    band_opts = ["All detectors"] + [f"Detector {b}  ({BAND_RANGES[b]})" for b in bands_present]
    band_sel = c2.selectbox("Show", band_opts, index=0 if len(bands_present) == 1 else 1,
                            help="Each detector sees a different colour, so one detector at a time gives the "
                                 "cleanest time-lapse.")
    shown = frames if band_sel == band_opts[0] else [f for f in frames if f["band"] == bands_present[band_opts.index(band_sel) - 1]]
    disp = make_display_frames(shown, stack, smooth)

    gaia_xy = None
    if show_gaia:
        try:
            g = fetch_gaia(round(float(res["center"].ra.deg), 5), round(float(res["center"].dec.deg), 5),
                           round(res["fov"] * 0.75, 2))
            gaia_xy = gaia_pixels(g, res["grid"], res["shape"], disp) if len(g) else None
            if len(g) == 0:
                st.info("Gaia returned no stars for this field.")
            else:
                st.caption(f"{len(g)} Gaia DR3 stars (G < 19) overlaid. A bright source that is not circled "
                           "is not in Gaia, which makes it more interesting.")
        except Exception as e:
            st.warning(f"Could not get Gaia data right now ({str(e)[:120]}). The time-lapse is shown without it.")
    try:
        html = make_timelapse_html(disp, interval_ms=speed, vmax=vmax, smooth=smooth, gaia_xy=gaia_xy)
        show_html(html, 680)
    except Exception as e:
        st.warning(f"Could not build the animation ({e}). Showing a slider instead.")
        idx = st.slider("Frame", 0, len(disp) - 1, 0)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(disp[idx]["snr"], **stretch(vmax, smooth))
        ax.set_title(_label(disp[idx]), fontsize=10)
        st.pyplot(fig)
        plt.close(fig)

# ── Frame grid ───────────────────────────────────────────────────────────────
with t_grid:
    st.subheader("Frame grid")
    st.caption("Selected frames in time order. The red ring marks the predicted position of the target.")
    fig = plot_contact_sheet(make_display_frames(frames, stack, smooth), vmax=vmax, smooth=smooth)
    st.pyplot(fig)
    plt.close(fig)

# ── Detector co-adds ─────────────────────────────────────────────────────────
with t_det:
    st.subheader("Co-added images by detector and time")
    st.caption("All frames from one detector in one time window are added together, which lowers the "
               "noise. A moving object shifts from one column to the next. Pick several detectors in the "
               "sidebar to compare colours.")
    n_win = st.slider("Time windows", 2, 6, 4)
    if len(frames) >= n_win:
        fig = plot_detector_grid(frames, n_win, vmax)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need more frames than time windows. Raise Max frames or lower the windows.")

# ── Difference ───────────────────────────────────────────────────────────────
with t_diff:
    st.subheader("Early vs late difference")
    st.caption("Earlier frames added together, later frames added together, then subtracted. A mover appears "
               "as a blue/red pair; a star that only changed brightness shows a single blob.")
    d1, d2 = st.columns(2)
    dband = d1.selectbox("Detector", bands_present, format_func=lambda b: f"{b}  ({BAND_RANGES[b]})",
                         help="Frames are only compared within one detector, because colour changes brightness.")
    frac = d2.slider("Share of frames in each group", 0.2, 0.5, 0.34, 0.02)
    dframes = [f for f in frames if f["band"] == dband]
    if len(dframes) >= 4:
        fig = plot_difference(dframes, frac, vmax)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need at least 4 frames from this detector for a useful difference image.")

# ── Spectrum ─────────────────────────────────────────────────────────────────
with t_spec:
    st.subheader("Spectrum and light curve")
    st.caption("Each frame sees a different wavelength at the target's position, so measuring the brightness "
               "at the red ring in every frame builds up a spectrum. Use the offsets to measure a "
               "neighbouring spot instead.")
    s1, s2, s3 = st.columns(3)
    r_ap = s1.slider("Aperture radius (pixels)", 1.0, 4.0, 2.0, 0.5)
    dx = s2.slider("Offset right (pixels)", -10.0, 10.0, 0.0, 0.5)
    dy = s3.slider("Offset up (pixels)", -10.0, 10.0, 0.0, 0.5)
    binned = st.checkbox("Show averages in wavelength bins (squares)", value=True)
    if not any(np.isfinite(f.get("wavelength_um", np.nan)) for f in frames):
        st.warning("Wavelength information was not available for these frames, so only the light curve is meaningful.")
    phot = measure(frames, r_ap, dx, dy)
    fig = plot_spectrum(phot, binned)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Approximate: no aperture or colour correction, and the errors ignore correlated noise between "
               "pixels. Single frames of faint objects are mostly noise, so look at the binned squares.")
    st.download_button("Download measurements (CSV)", phot.to_csv(index=False),
                       file_name=f"{res['label'].replace(' ', '_')}_spectrum.csv", mime="text/csv")

# ── Sky context (Aladin Lite) ────────────────────────────────────────────────
with t_sky:
    st.subheader("Sky context (Aladin Lite, CDS)")
    st.caption("The same patch of sky in other surveys, including the WISE infrared sky, to see what else is there.")
    surveys = {"WISE (AllWISE colour)": "P/allWISE/color", "Optical (DSS2 colour)": "P/DSS2/color",
               "Near-infrared (2MASS colour)": "P/2MASS/color"}
    sv = st.selectbox("Survey", list(surveys))
    c = res["center"]
    fov_deg = max(res["fov"] / 60.0 * 1.5, 0.05)
    show_html(aladin_html(float(c.ra.deg), float(c.dec.deg), fov_deg, surveys[sv]), 500)
    st.markdown(f"[Open this view in Aladin Lite](https://aladin.cds.unistra.fr/AladinLite/?target="
                f"{c.ra.deg:.5f}%20{c.dec.deg:+.5f}&fov={fov_deg:.3f}&survey={quote(surveys[sv], safe='')})")

# ── Frame info ───────────────────────────────────────────────────────────────
with t_info:
    st.subheader("Frame metadata")
    meta = pd.DataFrame([
        {
            "date": f["date"], "detector": f["band"],
            "wavelength_um": f.get("wavelength_um", np.nan),
            "flagged_pixels_%": 100 * f.get("flagged_frac", np.nan),
            "coverage_%": 100 * f.get("coverage", np.nan),
            "noise": f.get("sigma", np.nan),
            "data_version": f.get("version", ""),
            "x_pred": f["x"], "y_pred": f["y"],
        }
        for f in all_frames
    ])
    st.dataframe(meta.round(3))
    st.caption("flagged_pixels_% is the share of pixels in the cutout that SPHEREx marked in its quality "
               "flags. data_version is the processing version stored in each file.")
    e1, e2 = st.columns(2)
    e1.download_button("Download frame list (CSV)", meta.to_csv(index=False),
                       file_name=f"{res['label'].replace(' ', '_')}_frames.csv", mime="text/csv")
    try:
        e2.download_button("Download this view (.npz)", result_to_npz_bytes(res),
                           file_name=f"{res['label'].replace(' ', '_').replace(chr(39), '')}.npz",
                           help="Save it into a demo_data folder in the repository to get a backup demo "
                                "that works without the archive.")
    except Exception as e:
        e2.caption(f"Could not prepare the download: {e}")

with st.expander("About the data and tools"):
    st.markdown(
        "- **Images:** NASA SPHEREx spectral images, from the NASA/IPAC Infrared Science Archive (IRSA). "
        "The tool asks the archive live and keeps results for an hour, so newly released data appears "
        "automatically.\n"
        "- **Processing here:** cutouts are re-projected onto one grid, the sky background is removed, and "
        "each frame is scaled to its own noise level.\n"
        "- **Extras:** Gaia DR3 via VizieR; Aladin Lite from the Strasbourg Astronomical Data Center (CDS); "
        "Astropy, Astroquery, reproject and Streamlit."
    )
