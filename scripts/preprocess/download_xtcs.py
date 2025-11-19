import re
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# --- Config ---
CSV_PATH = "traj_metadata.csv"              # your metadata file
ID_COLUMN = "Dynamic id"                    # column with dynamics IDs
OUT_ROOT = Path("/oak/stanford/groups/rbaltman/ziyiw23/GPCR_trajectories")  # root download dir
SLEEP_BETWEEN_REQUESTS = 1.0                # polite delay (seconds)

BASE_DYN_URL = "https://www.gpcrmd.org/dynadb/dynamics/id/{dyn_id}/"
BASE_FILE_URL = "https://www.gpcrmd.org/dynadb/file/id/{file_id}/download"
BASE_FILES_DIR = "https://www.gpcrmd.org/dynadb/files/Dynamics"

# Regex patterns to find file IDs in the Simulation report HTML
traj_re = re.compile(r"Trajectory file\s*\(ID:\s*(\d+)\)")
pdb_re = re.compile(r"(?:Model|Structure)\s*file\s*\(ID:\s*(\d+)\)", re.I)
# Capture any direct links embedded on the page
direct_href_re = re.compile(r'href="(?P<h>https?://[^"]*?/dynadb/files/Dynamics/[^"]+)"', re.I)

# Thread-local session for thread-safe reuse of connections/cookies
_TLS = threading.local()
FAILED_DYNS = set()
FAILED_LOCK = threading.Lock()


def _build_session() -> requests.Session:
    s = requests.Session()
    # Be resilient to transient 5xx/connection errors
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def get_session() -> requests.Session:
    sess = getattr(_TLS, "session", None)
    if sess is None:
        sess = _build_session()
        _TLS.session = sess
    return sess


def get_file_ids_for_dyn(dyn_id: int):
    """Return (trajectory_ids, pdb_ids, direct_map, extra_pdb_urls) for a given dynamics id.
    direct_map: dict[file_id] = [list of direct URLs found on the page]
    extra_pdb_urls: list of PDB URLs without a numeric file-id (e.g., tmp_dyn_*.pdb)
    """
    url = BASE_DYN_URL.format(dyn_id=dyn_id)
    print(f"[INFO] Fetching simulation report for dynamics {dyn_id}: {url}")
    r = get_session().get(url)
    r.raise_for_status()
    text = r.text

    traj_ids = traj_re.findall(text)
    pdb_ids = pdb_re.findall(text)

    if not traj_ids:
        print(f"  [WARN] No trajectory file IDs found for dynamics {dyn_id}")
    if not pdb_ids:
        print(f"  [WARN] No PDB/model file IDs found for dynamics {dyn_id}")

    # Build mapping from file_id to any direct links embedded
    direct_map = {}
    extra_pdb_urls = []
    for m in direct_href_re.finditer(text):
        h = m.group("h")
        # Try to extract a file_id token from the filename prefix
        m_id = re.search(r"/(\d+)[^/]*\.(?:xtc|pdb|zip|gz)$", h)
        if m_id:
            fid = m_id.group(1)
            direct_map.setdefault(fid, []).append(h)
        else:
            # If it's a PDB without numeric id (e.g., tmp_dyn_*.pdb), keep it as extra
            if re.search(r"\.pdb$", h, re.I):
                extra_pdb_urls.append(h)

    return traj_ids, pdb_ids, direct_map, extra_pdb_urls


def download_file(file_id: str, out_path: Path, referer: str = None, direct_candidates=None, force: bool = False):
    """Download a GPCRmd file by its file ID with progress if available."""
    base = BASE_FILE_URL.format(file_id=file_id)
    candidates = [
        base,
        f"{base}/" if not base.endswith("/") else base,
        f"{base}?download=1",
    ]
    # Respect existing files unless forcing
    if out_path.exists() and not force:
        print(f"  [SKIP] {out_path} already exists")
        return
    if out_path.exists() and force:
        try:
            out_path.unlink()
        except Exception:
            pass

    last_err: Exception = None  # type: ignore
    # Browser-like headers + optional Referer (helps some gateways)
    common_headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    if referer:
        common_headers["Referer"] = referer

    # Try landing page first; some servers set cookies/redirect chains here
    direct_from_landing = None
    try:
        landing = base.rsplit("/download", 1)[0]
        lr = get_session().get(landing, timeout=30, headers=common_headers)
        # Try to discover a direct Dynamics link on the landing page (may include dynXXXX folders)
        if lr.ok:
            m = direct_href_re.search(lr.text)
            if m:
                direct_from_landing = m.group("h")
    except Exception:
        direct_from_landing = None

    # Try direct file paths (if provided)
    if direct_candidates or direct_from_landing:
        if direct_from_landing:
            # Prepend discovered direct link
            if direct_candidates is None:
                direct_candidates = [direct_from_landing]
            else:
                direct_candidates = [direct_from_landing] + list(direct_candidates)
        for url in direct_candidates:
            try:
                print(f"  [DL] {url} -> {out_path}")
                with get_session().get(url, stream=True, timeout=120, headers=common_headers, allow_redirects=True) as r:
                    r.raise_for_status()
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    total = int(r.headers.get("content-length", 0))
                    chunk_size = 1024 * 64
                    if tqdm is not None:
                        with open(out_path, "wb") as f, tqdm(
                            total=total if total > 0 else None,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=out_path.name,
                            leave=False,
                        ) as pbar:
                            for buf in r.iter_content(chunk_size=chunk_size):
                                if not buf:
                                    continue
                                f.write(buf)
                                pbar.update(len(buf))
                    else:
                        with open(out_path, "wb") as f:
                            for buf in r.iter_content(chunk_size=chunk_size):
                                if not buf:
                                    continue
                                f.write(buf)
                print(f"  [OK] Saved {out_path}")
                return
            except Exception as e:
                last_err = e
                continue

    for url in candidates:
        try:
    print(f"  [DL] {url} -> {out_path}")
            with get_session().get(url, stream=True, timeout=120, headers=common_headers, allow_redirects=True) as r:
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
                total = int(r.headers.get("content-length", 0))
                chunk_size = 1024 * 64
                if tqdm is not None:
                    with open(out_path, "wb") as f, tqdm(
                        total=total if total > 0 else None,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=out_path.name,
                        leave=False,
                    ) as pbar:
                        for buf in r.iter_content(chunk_size=chunk_size):
                            if not buf:
                                continue
                            f.write(buf)
                            pbar.update(len(buf))
                else:
        with open(out_path, "wb") as f:
                        for buf in r.iter_content(chunk_size=chunk_size):
                            if not buf:
                                continue
                            f.write(buf)
            print(f"  [OK] Saved {out_path}")
            return
        except Exception as e:  # keep trying other URL variants
            last_err = e
            continue
    raise last_err if last_err is not None else RuntimeError("Unknown download error")


def download_direct_url(url: str, out_path: Path, referer: str = None, force: bool = False):
    """Download a file from a direct URL (no numeric file-id available)."""
    # Respect existing files unless forcing
    if out_path.exists() and not force:
        print(f"  [SKIP] {out_path} already exists")
        return
    if out_path.exists() and force:
        try:
            out_path.unlink()
        except Exception:
            pass

    common_headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    if referer:
        common_headers["Referer"] = referer

    try:
        print(f"  [DL] {url} -> {out_path}")
        with get_session().get(url, stream=True, timeout=120, headers=common_headers, allow_redirects=True) as r:
            r.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            total = int(r.headers.get("content-length", 0))
            chunk_size = 1024 * 64
            if tqdm is not None:
                with open(out_path, "wb") as f, tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=out_path.name,
                    leave=False,
                ) as pbar:
                    for buf in r.iter_content(chunk_size=chunk_size):
                        if not buf:
                            continue
                        f.write(buf)
                        pbar.update(len(buf))
            else:
                with open(out_path, "wb") as f:
                    for buf in r.iter_content(chunk_size=chunk_size):
                        if not buf:
                            continue
                        f.write(buf)
        print(f"  [OK] Saved {out_path}")
    except Exception as e:
        raise e


def process_one_dynamics(dyn_id: int, out_root: Path, sleep_between: float, force: bool, pdb_only: bool):
    OUT_XTC = out_root / "xtc"
    # Save PDBs into the same folder as XTCs for downstream scripts that expect a unified directory
    OUT_PDB = OUT_XTC
        try:
        traj_ids, pdb_ids, direct_map, extra_pdb_urls = get_file_ids_for_dyn(dyn_id)
        except Exception as e:
            print(f"[ERROR] Failed to fetch dynamics {dyn_id}: {e}")
        with FAILED_LOCK:
            FAILED_DYNS.add(dyn_id)
        return

    # Download all trajectories for this dynamics (skip if pdb_only)
    if not pdb_only:
        for t_id in traj_ids:
            xtc_path = OUT_XTC / f"d{dyn_id}_traj_{t_id}.xtc"
            try:
                direct = []
                # Page-embedded direct links for this file id
                if t_id in direct_map:
                    direct.extend(direct_map[t_id])
                # Common filename patterns (both trj and traj)
                direct.extend([
                    f"{BASE_FILES_DIR}/{t_id}_trj_{dyn_id}.xtc",
                    f"{BASE_FILES_DIR}/{t_id}_traj_{dyn_id}.xtc",
                ])
                download_file(t_id, xtc_path, referer=BASE_DYN_URL.format(dyn_id=dyn_id), direct_candidates=direct, force=force)
            except Exception as e:
                print(f"  [ERROR] Failed to download trajectory {t_id}: {e}")
                with FAILED_LOCK:
                    FAILED_DYNS.add(dyn_id)

    # Download the model PDB (one per dynamics); prefer direct URLs from the page.
    # Save with the page basename to preserve the _dyn_{dyn_id}.pdb pattern expected downstream.
    try:
        # Collect all pdb links discovered on the page
        page_pdb_urls = []
        for _fid, urls in (direct_map or {}).items():
            for u in urls:
                if u.lower().endswith(".pdb"):
                    page_pdb_urls.append(u)
        for u in extra_pdb_urls or []:
            if u.lower().endswith(".pdb"):
                page_pdb_urls.append(u)
        # Prefer names containing _dyn_{dyn_id}.pdb
        preferred = [u for u in page_pdb_urls if f"_dyn_{dyn_id}.pdb" in u]
        ordered = preferred + [u for u in page_pdb_urls if u not in preferred]
        downloaded_direct = False
        for url in ordered:
            try:
                basename = Path(url).name
                pdb_path_direct = OUT_PDB / basename
                download_direct_url(url, pdb_path_direct, referer=BASE_DYN_URL.format(dyn_id=dyn_id), force=force)
                downloaded_direct = True
                break
            except Exception:
                continue
        if not downloaded_direct and pdb_ids:
            # Fallback to file-id based patterns if no direct URL succeeded
            p_id = pdb_ids[0]
            pdb_path = OUT_PDB / f"{p_id}_dyn_{dyn_id}.pdb"
            direct = []
            if p_id in direct_map:
                direct.extend(direct_map[p_id])
            direct.extend([
                f"{BASE_FILES_DIR}/{p_id}_dyn_{dyn_id}.pdb",
                f"{BASE_FILES_DIR}/{p_id}_model_{dyn_id}.pdb",
                f"{BASE_FILES_DIR}/{p_id}_structure_{dyn_id}.pdb",
            ])
            download_file(p_id, pdb_path, referer=BASE_DYN_URL.format(dyn_id=dyn_id), direct_candidates=direct, force=force)
    except Exception as e:
        print(f"  [ERROR] Failed to download model PDB for dyn {dyn_id}: {e}")
        with FAILED_LOCK:
            FAILED_DYNS.add(dyn_id)

    # Download any extra PDBs linked directly on the page without numeric file-id (e.g., tmp_dyn_*.pdb)
    if 'extra_pdb_urls' in locals() and extra_pdb_urls:
        for url in extra_pdb_urls:
            try:
                basename = Path(url).name
                pdb_extra_path = OUT_PDB / f"d{dyn_id}_extra_{basename}"
                download_direct_url(url, pdb_extra_path, referer=BASE_DYN_URL.format(dyn_id=dyn_id), force=force)
            except Exception as e:
                print(f"  [ERROR] Failed to download extra PDB {url}: {e}")
                with FAILED_LOCK:
                    FAILED_DYNS.add(dyn_id)

    # politeness delay per dynamics
    time.sleep(sleep_between)


def main():
    parser = argparse.ArgumentParser(description="Download GPCRmd trajectories and PDBs.")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to traj metadata CSV")
    parser.add_argument("--id_column", default=ID_COLUMN, help="Column name for dynamics IDs")
    parser.add_argument("--out_root", default=str(OUT_ROOT), help="Output root directory")
    parser.add_argument("--max_workers", type=int, default=8, help="Max parallel dynamics downloads")
    parser.add_argument("--sleep_between_requests", type=float, default=SLEEP_BETWEEN_REQUESTS, help="Polite delay per dynamics")
    parser.add_argument("--dyn_ids", type=str, default="", help="Comma-separated subset of dynamics IDs to download")
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing files")
    parser.add_argument("--pdb_only", action="store_true", help="Only download PDB files (skip XTCs)")
    parser.add_argument("--retry_passes", type=int, default=1, help="Number of retry passes over failed dynamics")
    parser.add_argument("--retry_sleep", type=float, default=90.0, help="Sleep seconds between retry passes")
    parser.add_argument("--retry_max_workers", type=int, default=None, help="Max workers for retry passes (default: half of main)")
    args = parser.parse_args()
    csv_path = args.csv
    id_column = args.id_column
    out_root = Path(args.out_root)
    sleep_between = float(args.sleep_between_requests)
    force = bool(args.force)
    OUT_XTC = OUT_ROOT / "xtc"
    OUT_PDB = OUT_ROOT / "pdb"

    if args.dyn_ids.strip():
        dyn_ids = [int(x) for x in args.dyn_ids.split(",") if x.strip()]
    else:
        df = pd.read_csv(csv_path)
        dyn_ids = df[id_column].dropna().astype(int).tolist()

    print(f"[INFO] Found {len(dyn_ids)} dynamics IDs")

    # tqdm does not play nicely with multithreading; show a simple counter instead
    if args.max_workers <= 1:
        iterator = tqdm(dyn_ids, desc="Dynamics", unit="dyn") if tqdm is not None else dyn_ids
        for dyn_id in iterator:
            process_one_dynamics(dyn_id, out_root, sleep_between, force, args.pdb_only)
    else:
        total = len(dyn_ids)
        completed = 0
        print(f"[INFO] Using max_workers={args.max_workers}")
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(process_one_dynamics, did, out_root, sleep_between, force, args.pdb_only): did for did in dyn_ids}
            for fut in as_completed(futures):
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"[INFO] Progress: {completed}/{total} dynamics done")
    # Retry passes over failed dynamics with fewer workers
    passes = max(0, int(args.retry_passes))
    for r in range(passes):
        with FAILED_LOCK:
            retry_dyns = list(FAILED_DYNS)
            FAILED_DYNS.clear()
        if not retry_dyns:
            break
        mw = int(args.retry_max_workers) if args.retry_max_workers else max(1, int(args.max_workers) // 2)
        print(f"[INFO] Retry pass {r+1}/{passes}: {len(retry_dyns)} dynamics with max_workers={mw}")
        if float(args.retry_sleep) > 0:
            time.sleep(float(args.retry_sleep))
        if mw <= 1:
            for did in (tqdm(retry_dyns, desc="Retry", unit="dyn") if tqdm is not None else retry_dyns):
                process_one_dynamics(did, out_root, sleep_between, True, args.pdb_only)
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                for _ in as_completed({ex.submit(process_one_dynamics, did, out_root, sleep_between, True, args.pdb_only): did for did in retry_dyns}):
                    pass


if __name__ == "__main__":
    main()
