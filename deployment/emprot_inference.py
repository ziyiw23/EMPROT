from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import shlex
import shutil

DEPLOY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = PROJECT_ROOT
SCRIPT_PATH = ROOT / "scripts" / "attn_rollout_min.py"
PNG_EXTENSIONS = {".png", ".jpg", ".jpeg"}
NPZ_EXTENSIONS = {".npz"}
JSON_EXTENSIONS = {".json"}

# Preferred default Python for external collapse embedder if none provided
DEFAULT_COLLAPSE_PYTHON = "/oak/stanford/groups/rbaltman/ziyiw23/venv/emprot/bin/python"
DEFAULT_COLLAPSE_MODULE_CMD = "module purge && module load openblas/0.3.10 devel cuda/11.7.1 gcc/12.4.0 python/3.12.1"
DEMO_EMBED_BASE = ROOT / "output" / "demo_embeddings" / "latest"


def infer_protein_id_from_path(pdb_path: Optional[str]) -> Optional[str]:
    if not pdb_path:
        return None
    name = Path(pdb_path).name
    match = re.search(r"d(\d+)", name, re.IGNORECASE)
    if match:
        return match.group(1)
    numeric = re.findall(r"(\d+)", name)
    if numeric:
        return numeric[0]
    return None


def _default_device() -> str:
    override = os.environ.get("EMPROT_DEVICE")
    if override:
        return override.strip()
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                return "cuda"
            except Exception:
                return "cpu"
    except Exception:
        return "cpu"
    return "cpu"


def _ensure_script_exists() -> None:
    if not SCRIPT_PATH.is_file():
        raise FileNotFoundError(f"Expected pipeline script at {SCRIPT_PATH}")


def _collect_artifacts(directory: Path) -> Dict[str, List[str]]:
    pngs: List[str] = []
    jsons: List[str] = []
    npzs: List[str] = []
    for path in directory.iterdir():
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in PNG_EXTENSIONS:
                pngs.append(str(path))
            elif suffix in JSON_EXTENSIONS:
                jsons.append(str(path))
            elif suffix in NPZ_EXTENSIONS:
                npzs.append(str(path))
    pngs.sort()
    jsons.sort()
    npzs.sort()
    return {"plots": pngs, "json": jsons, "npz": npzs}


def _load_metrics(directory: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    dist_path = directory / "distribution_metrics.json"
    hist_path = directory / "histogram_summary.json"
    if dist_path.is_file():
        try:
            with dist_path.open() as handle:
                metrics["distribution"] = json.load(handle)
        except Exception:
            metrics["distribution_read_error"] = str(dist_path)
    if hist_path.is_file():
        try:
            with hist_path.open() as handle:
                metrics["histogram"] = json.load(handle)
        except Exception:
            metrics["histogram_read_error"] = str(hist_path)
    return metrics


def run_emprot_pipeline(
    pdb_source: Optional[str],
    *,
    ckpt_path: str,
    data_root: str,
    split: str = "test",
    protein_id: Optional[str] = None,
    time_start: int = 500,
    time_steps: int = 100,
    recent_full_frames: int = 8,
    k_residues: int = 5,
    residue_select: str = "most_change",
    temperature: float = 1.0,
    top_p: float = 1.0,
    hist_topk: int = 30,
    plot_hist: bool = True,
    plot_step_attn: bool = False,
    attn_step: int = 0,
    device: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_script_exists()
    if not ckpt_path:
        raise ValueError("Checkpoint path is required.")
    if not data_root:
        raise ValueError("Data root is required.")
    if not Path(ckpt_path).expanduser().resolve().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    data_root_path = Path(data_root).expanduser().resolve()
    if not data_root_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_root_path}")
    resolved_device = device or _default_device()
    inferred_id = infer_protein_id_from_path(pdb_source)
    protein_arg = protein_id or inferred_id
    temp_dir = tempfile.mkdtemp(prefix="emprot_ui_", dir=work_dir)
    out_dir = Path(temp_dir).resolve()
    ui_logs: List[str] = []
    ui_logs.append(
        f"[LMDB PIPELINE] ckpt={Path(ckpt_path).name} data_root={data_root_path} split={split} device={resolved_device}"
    )
    ui_logs.append(
        f"[LMDB PIPELINE] time_start={int(time_start)} time_steps={int(time_steps)} recent_full_frames={int(recent_full_frames)}"
    )
    args: List[str] = [
        sys.executable,
        str(SCRIPT_PATH),
        "--ckpt",
        ckpt_path,
        "--data_root",
        str(data_root_path),
        "--split",
        split,
        "--time_start",
        str(int(time_start)),
        "--time_steps",
        str(int(time_steps)),
        "--recent_full_frames",
        str(int(recent_full_frames)),
        "--k_residues",
        str(int(k_residues)),
        "--residue_select",
        residue_select,
        "--temperature",
        str(float(temperature)),
        "--top_p",
        str(float(top_p)),
        "--hist_topk",
        str(int(hist_topk)),
        "--output_dir",
        str(out_dir),
        "--device",
        resolved_device,
    ]
    if protein_arg:
        args.extend(["--protein_id", str(protein_arg)])
    if plot_hist:
        args.append("--plot_hist")
    if plot_step_attn:
        args.extend(["--plot_step_attn", "--attn_step", str(int(attn_step))])
    if extra_args:
        args.extend(extra_args)
    env = os.environ.copy()
    start = time.time()
    result = subprocess.run(
        args,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    duration = time.time() - start
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed with code {result.returncode}:\n{stdout}\n{stderr}"
        )
    artifacts = _collect_artifacts(out_dir)
    metrics = _load_metrics(out_dir)
    summary: Dict[str, Any] = {
        "status": "ok",
        "output_dir": str(out_dir),
        "duration_sec": duration,
        "split": split,
        "time_start": int(time_start),
        "time_steps": int(time_steps),
        "recent_full_frames": int(recent_full_frames),
        "k_residues": int(k_residues),
        "residue_select": residue_select,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "hist_topk": int(hist_topk),
        "device": resolved_device,
        "protein_id": str(protein_arg) if protein_arg is not None else None,
        "artifacts": artifacts,
    }
    if metrics:
        summary["metrics"] = metrics
    logs_pieces: List[str] = []
    if ui_logs:
        logs_pieces.append("\n".join(ui_logs))
    core_logs = stdout
    if stderr.strip():
        core_logs = f"{stdout}\n{stderr}".strip()
    if core_logs.strip():
        logs_pieces.append(core_logs.strip())
    logs = "\n\n".join(p for p in logs_pieces if p).strip()
    result: Dict[str, Any] = {
        "summary": summary,
        "plots": artifacts.get("plots", []),
        "logs": logs,
        "output_dir": str(out_dir),
    }
    # Include rollout arrays path (npz) if present
    npzs = artifacts.get("npz", [])
    if npzs:
        # Heuristic: prefer file ending with 'rollout_arrays.npz' if present
        chosen = None
        for p in npzs:
            if str(p).endswith("rollout_arrays.npz"):
                chosen = p
                break
        result["rollout_npz"] = chosen or npzs[0]
    return result


# -----------------------------
# PDB-only quick pipeline (no LMDB dataset required)
# -----------------------------

def _which_python_for_collapse(override: Optional[str] = None) -> str:
    """Resolve Python executable for running gen_embed.py.

    Priority: explicit override → env var COLLAPSE_PYTHON → DEFAULT_COLLAPSE_PYTHON → current interpreter.
    Returns the first existing path.
    """
    for cand in [
        (override or "").strip(),
        (os.environ.get("COLLAPSE_PYTHON") or "").strip(),
        DEFAULT_COLLAPSE_PYTHON,
    ]:
        if cand:
            try:
                if Path(cand).exists():
                    return cand
            except Exception:
                continue
    return sys.executable


def _collapse_dir(override: Optional[str] = None) -> str:
    p = override if override is not None else os.environ.get("COLLAPSE_DIR")
    if not p or not Path(p).is_dir():
        raise FileNotFoundError(
            "COLLAPSE_DIR env var not set or invalid; required to run gen_embed.py"
        )
    return str(Path(p).resolve())


def _collapse_module_command() -> Optional[str]:
    cmd = os.environ.get("COLLAPSE_MODULE_CMD")
    if cmd is not None:
        cmd = cmd.strip()
        return cmd or None
    return DEFAULT_COLLAPSE_MODULE_CMD


def _collapse_activate_command(python_exec: str) -> Optional[str]:
    override = os.environ.get("COLLAPSE_ACTIVATE")
    if override:
        override = override.strip()
        return override or None
    activate_path = Path(python_exec).resolve().parent / "activate"
    if activate_path.is_file():
        return f"source {shlex.quote(str(activate_path))}"
    return None


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _locate_generated_traj_dir(data_out: Path, traj_name: str) -> Path:
    candidate = data_out / traj_name
    if candidate.exists():
        return candidate
    data_mdb = data_out / "data.mdb"
    if data_mdb.is_file():
        return data_out
    subdirs = [p for p in data_out.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    raise RuntimeError(f"Could not locate generated trajectory directory under {data_out}")


def _normalize_demo_dataset(data_out: Path, traj_dir: Path) -> Path:
    if traj_dir != data_out:
        return traj_dir
    new_dir = data_out / "demo_single_000"
    if new_dir.exists():
        shutil.rmtree(new_dir)
    new_dir.mkdir(parents=True, exist_ok=True)
    for item in list(data_out.iterdir()):
        if item == new_dir:
            continue
        target = new_dir / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        item.rename(target)
    return new_dir


def _cluster_model_path(override: Optional[str] = None) -> str:
    p = override or os.environ.get("CLUSTER_MODEL_PATH") or \
        "/oak/stanford/groups/rbaltman/aderry/collapse-motifs/data/pdb100_cluster_fit_50000.pkl"
    if not Path(p).is_file():
        raise FileNotFoundError(
            f"Cluster model not found. Set CLUSTER_MODEL_PATH env var (got: {p})"
        )
    return str(Path(p).resolve())


def _duplicate_pdb_frames(src_pdb: str, dst_dir: Path, k_frames: int) -> Path:
    traj_dir = dst_dir / "single_000"
    traj_dir.mkdir(parents=True, exist_ok=True)
    src = Path(src_pdb)
    for i in range(int(k_frames)):
        out = traj_dir / f"frame_{i:04d}.pdb"
        shutil.copyfile(src, out)
    return traj_dir


def _run_gen_embed(input_root: Path, output_root: Path, collapse_dir: Optional[str] = None, collapse_python: Optional[str] = None) -> Tuple[str, str]:
    collapse = _collapse_dir(collapse_dir)
    python = _which_python_for_collapse(collapse_python)
    gen_script = Path(collapse) / "gen_embed.py"
    if not gen_script.is_file():
        raise FileNotFoundError(f"gen_embed.py not found in {collapse}")
    cmd_parts: List[str] = []
    module_cmd = _collapse_module_command()
    if module_cmd:
        cmd_parts.append(module_cmd)
    activate_cmd = _collapse_activate_command(python)
    if activate_cmd:
        cmd_parts.append(activate_cmd)
    quoted_script = " ".join(
        [
            shlex.quote(str(python)),
            shlex.quote(str(gen_script)),
            shlex.quote(str(input_root)),
            shlex.quote(str(output_root)),
            "--filetype",
            "pdb",
            "--num_workers",
            "1",
        ]
    )
    cmd_parts.append(quoted_script)
    shell_cmd = " && ".join(cmd_parts)
    # Ensure local source packages inside COLLAPSE_DIR are importable (e.g., gvp inside a subfolder)
    env = os.environ.copy()
    extra_paths: List[str] = [collapse]
    # Heuristics: add any parent directory that contains a 'gvp' package within a few levels
    try:
        root_path = Path(collapse)
        # Depth-limited walk to avoid huge trees
        max_depth = 3
        stack = [(root_path, 0)]
        seen: set[str] = set()
        while stack:
            cur, depth = stack.pop()
            if str(cur) in seen:
                continue
            seen.add(str(cur))
            if depth > max_depth:
                continue
            try:
                for p in cur.iterdir():
                    if p.is_dir():
                        # If this directory directly contains a 'gvp' package folder, add its path
                        if (p / "gvp").is_dir():
                            extra_paths.append(str(p))
                        # If this directory itself is the 'gvp' package, add its parent
                        if p.name == "gvp":
                            extra_paths.append(str(p.parent))
                        stack.append((p, depth + 1))
            except Exception:
                continue
    except Exception:
        pass
    existing = env.get("PYTHONPATH", "").split(os.pathsep) if env.get("PYTHONPATH") else []
    # Prepend extras to take precedence
    env["PYTHONPATH"] = os.pathsep.join([p for p in (extra_paths + existing) if p])
    result = subprocess.run(
        ["/bin/bash", "-lc", shell_cmd],
        cwd=collapse,
        env=env,
        capture_output=True,
        text=True,
    )
    # Attach the PYTHONPATH used to stderr for easier debugging upstream
    stderr = (result.stderr or "").strip()
    stderr += ("\nPYTHONPATH=" + env.get("PYTHONPATH", ""))
    return result.stdout or "", stderr


def _load_seed_cluster_ids_from_emb(out_traj_dir: Path, cluster_model: str, device: str) -> Tuple[Any, Any, Any]:
    # Read embeddings for a single frame, assign clusters
    from scripts.preprocess.cluster_lookup import ClusterCentroidLookup  # type: ignore
    from emprot.data.data_loader import LMDBLoader  # reuse loader to read embeddings
    with LMDBLoader(str(out_traj_dir), read_only=True) as loader:
        meta = loader.get_metadata()
        N = int(meta['num_residues'])
        first = loader.load_frame(0)
        emb = first.get('embeddings')
        if emb is None:
            raise KeyError("Embeddings not found in generated LMDB for PDB quick path")
        import torch  # local import
        emb_t = torch.as_tensor(emb, dtype=torch.float32, device=device)
    lookup = ClusterCentroidLookup(num_clusters=50000, embedding_dim=emb_t.shape[1], device=device)
    lookup.load_centroids_from_sklearn(cluster_model)
    ids = lookup.batch_assign_to_clusters(emb_t)
    import numpy as np
    seed = ids.detach().cpu().numpy().astype(np.int32)  # (N,)
    return seed, N, emb_t.shape[1]


def generate_demo_embeddings(
    pdb_source: str,
    *,
    recent_full_frames: int = 8,
    collapse_dir: Optional[str] = None,
    collapse_python: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    if not Path(pdb_source).is_file():
        raise FileNotFoundError(f"PDB not found: {pdb_source}")
    logs: List[str] = []
    logs.append("Preparing demo workspace…")
    data_in = DEMO_EMBED_BASE / "pdb_frames"
    data_out = DEMO_EMBED_BASE / "embeddings"
    _reset_dir(data_in)
    _reset_dir(data_out)

    logs.append("Duplicating PDB frames…")
    traj_in = _duplicate_pdb_frames(pdb_source, data_in, int(recent_full_frames))

    logs.append("Running collapse embedding (gen_embed.py)…")
    stdout_e, stderr_e = _run_gen_embed(data_in, data_out, collapse_dir=collapse_dir, collapse_python=collapse_python)
    try:
        out_traj_dir = _locate_generated_traj_dir(data_out, traj_in.name)
    except Exception as exc:
        logs.extend([stdout_e.strip(), stderr_e.strip()])
        raise RuntimeError(f"gen_embed.py did not produce expected output at {data_out}") from exc
    out_traj_dir = _normalize_demo_dataset(data_out, out_traj_dir)

    embed_meta: Dict[str, Any] = {}
    try:
        from emprot.data.data_loader import LMDBLoader as _LMDBLoader

        with _LMDBLoader(str(out_traj_dir), read_only=True) as loader:
            meta = loader.get_metadata()
        embed_meta = {
            "traj_path": meta.get("path", str(out_traj_dir)),
            "traj_name": meta.get("traj_name"),
            "num_frames": int(meta.get("num_frames", 0)),
            "num_residues": int(meta.get("num_residues", 0)),
            "embedding_dim": int(meta.get("embedding_dim", 0)),
            "sorted_lmdb_keys_sample": (meta.get("sorted_lmdb_keys") or [])[:5],
        }
        logs.append(
            f"Embeddings ready: traj={embed_meta.get('traj_name')} frames={embed_meta.get('num_frames')} "
            f"residues={embed_meta.get('num_residues')} dim={embed_meta.get('embedding_dim')}"
        )
    except Exception as exc:
        embed_meta = {
            "traj_path": str(out_traj_dir),
            "read_error": str(exc),
        }
        logs.append(f"Warning: failed to read LMDB metadata ({exc})")

    if stdout_e.strip():
        logs.append("gen_embed.py stdout:")
        logs.extend(stdout_e.strip().splitlines())
    if stderr_e.strip():
        logs.append("gen_embed.py stderr:")
        logs.extend(stderr_e.strip().splitlines())

    summary: Dict[str, Any] = {
        "status": "ok",
        "mode": "embed_only",
        "embed_data_root": str(data_out),
        "embed_traj_path": str(out_traj_dir),
        "embed_metadata": embed_meta,
        "recent_full_frames": int(recent_full_frames),
    }
    state: Dict[str, Any] = {
        "pdb_source": pdb_source,
        "traj_dir": str(out_traj_dir),
        "data_root": str(out_traj_dir.parent),
        "traj_name": Path(out_traj_dir).name,
        "recent_full_frames": int(recent_full_frames),
        "embed_metadata": embed_meta,
        "cluster_ids_ready": False,
    }
    return state, summary, logs


def add_cluster_ids_to_demo(
    traj_dir: str,
    *,
    cluster_model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_device = device or _default_device()
    cluster_model = _cluster_model_path(cluster_model_path)
    from scripts.preprocess.cluster_lookup import ClusterCentroidLookup  # type: ignore
    from emprot.data.data_loader import LMDBLoader
    import torch
    import numpy as np

    traj_path = Path(traj_dir)
    if not traj_path.exists():
        raise FileNotFoundError(f"LMDB trajectory not found: {traj_dir}")

    with LMDBLoader(str(traj_path), read_only=False) as loader:
        meta = loader.get_metadata()
        num_frames = int(meta["num_frames"])
        first = loader.load_frame(0)
        emb = np.asarray(first.get("embeddings"))
        if emb.ndim != 2:
            raise ValueError("Embeddings missing or malformed in LMDB.")
        emb_dim = emb.shape[1]
        lookup = ClusterCentroidLookup(num_clusters=50000, embedding_dim=emb_dim, device=resolved_device)
        lookup.load_centroids_from_sklearn(cluster_model)
        for idx in range(num_frames):
            frame = loader.load_frame(idx)
            embeddings = np.asarray(frame.get("embeddings"))
            if embeddings is None:
                raise KeyError(f"Embeddings missing in frame {idx}.")
            emb_t = torch.as_tensor(embeddings, dtype=torch.float32, device=resolved_device)
            cluster_ids = lookup.batch_assign_to_clusters(emb_t).detach().cpu().numpy().astype(np.int32)
            frame["cluster_ids"] = cluster_ids
            loader.add_frame(idx, frame)
    return {
        "num_frames": int(meta["num_frames"]),
        "num_residues": int(meta["num_residues"]),
        "device": resolved_device,
    }


def run_emprot_pipeline_quick_from_pdb(
    pdb_source: str,
    *,
    ckpt_path: str,
    time_steps: int = 100,
    recent_full_frames: int = 8,
    device: Optional[str] = None,
    work_dir: Optional[str] = None,
    collapse_dir: Optional[str] = None,
    cluster_model_path: Optional[str] = None,
    collapse_python: Optional[str] = None,
) -> Dict[str, Any]:
    """PDB-only quick pipeline: embed → clusterize first frame → duplicate K frames → rollout.

    Requires env vars:
    - COLLAPSE_DIR: directory containing gen_embed.py
    - CLUSTER_MODEL_PATH: sklearn k-means pickle for the residue clustering
    """
    if not ckpt_path:
        raise ValueError("Checkpoint path is required.")
    if not Path(ckpt_path).expanduser().resolve().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not Path(pdb_source).is_file():
        raise FileNotFoundError(f"PDB not found: {pdb_source}")

    # For quick-from-PDB, reuse the demo embed helper to produce an LMDB
    state, _summary, _logs = generate_demo_embeddings(
        pdb_source,
        recent_full_frames=int(recent_full_frames),
        collapse_dir=collapse_dir,
        collapse_python=collapse_python,
    )
    return run_emprot_pipeline_quick_from_lmdb(
        traj_dir=state["traj_dir"],
        ckpt_path=ckpt_path,
        time_steps=time_steps,
        recent_full_frames=recent_full_frames,
        device=device,
        work_dir=work_dir,
    )


def run_emprot_pipeline_quick_from_lmdb(
    traj_dir: str,
    *,
    ckpt_path: str,
    time_steps: int = 100,
    recent_full_frames: int = 8,
    device: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """LMDB-only quick pipeline: seed from first frame's cluster_ids → duplicate K frames → rollout."""
    if not ckpt_path:
        raise ValueError("Checkpoint path is required.")
    if not Path(ckpt_path).expanduser().resolve().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    traj_path = Path(traj_dir)
    if not traj_path.exists():
        raise FileNotFoundError(f"LMDB trajectory not found: {traj_dir}")

    resolved_device = device or _default_device()
    out_root = Path(tempfile.mkdtemp(prefix="emprot_quick_lmdb_", dir=work_dir)).resolve()
    ui_logs: List[str] = []
    ui_logs.append(
        f"[LMDB QUICK] ckpt={Path(ckpt_path).name} traj={traj_path.name} device={resolved_device} time_steps={time_steps} recent_full_frames={recent_full_frames}"
    )

    import numpy as np
    from emprot.data.data_loader import LMDBLoader

    ui_logs.append("Loading LMDB metadata and seed cluster IDs…")
    with LMDBLoader(str(traj_path), read_only=True) as loader:
        meta = loader.get_metadata()
        N = int(meta["num_residues"])
        first = loader.load_frame(0)
        if "cluster_ids" not in first:
            raise KeyError("cluster_ids missing in first frame; run add_cluster_ids_to_demo first.")
        seed_ids = np.asarray(first["cluster_ids"], dtype=np.int32)
        if seed_ids.shape[0] != N:
            raise ValueError("cluster_ids length does not match num_residues.")

    K = int(recent_full_frames)
    T = int(time_steps)
    ui_logs.append(f"Preparing quick rollout tensors (K={K}, T={T}, residues={N})…")
    Y_all = np.full((K + T, int(N)), -1, dtype=np.int32)
    for i in range(K):
        Y_all[i, :] = seed_ids

    from scripts.autoregressive_eval import load_model, rollout_autoregressive, select_residues
    import torch

    ui_logs.append("Loading EMPROT checkpoint and running autoregressive rollout…")
    model, _cfg, id2col, col2id, col2id_array = load_model(
        ckpt_path, torch.device(resolved_device), use_sparse_logits=True
    )
    eval_out = rollout_autoregressive(
        model=model,
        Y_all=Y_all,
        time_start=K,
        time_steps=T,
        device=torch.device(resolved_device),
        recent_full_frames=K,
        col2id_array=col2id_array,
        col2id=col2id or {},
        decode_mode="sample",
        temperature=1.0,
        top_p=0.98,
        simple_nucleus=True,
    )

    ridxs = select_residues(eval_out.pred, min(5, eval_out.pred.shape[1]), "uniform", 42)
    times_abs = np.arange(0, T, dtype=np.float32)
    npz_path = out_root / "quick_rollout_arrays.npz"
    try:
        np.savez(
            npz_path,
            gt=np.full_like(eval_out.pred, -1),
            pred=eval_out.pred.astype(np.int32),
            times_abs=times_abs,
            times_ns=times_abs * 0.2,
            ridxs=np.asarray(ridxs, dtype=np.int32),
        )
    except Exception:
        npz_path = None

    import matplotlib.pyplot as _plt

    fig, ax = _plt.subplots(figsize=(8, 4))
    colors = [f"C{i%10}" for i in range(len(ridxs))]
    for i, r in enumerate(ridxs):
        ax.plot(times_abs * 0.2, eval_out.pred[:, int(r)], color=colors[i], label=f"Residue {int(r)}")
    ax.set_title("Quick Rollout — Predicted clusters (LMDB demo)")
    ax.set_xlabel("Time (ns; 0.2/frame)")
    ax.set_ylabel("Cluster ID")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    plot_path = out_root / "quick_rollout.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    _plt.close(fig)
    ui_logs.append(f"Saved rollout artifacts to {out_root}")

    summary: Dict[str, Any] = {
        "status": "ok",
        "mode": "lmdb_quick",
        "output_dir": str(out_root),
        "traj_dir": str(traj_path),
        "recent_full_frames": K,
        "time_steps": T,
        "num_residues": int(N),
    }
    logs = "\n".join(ui_logs)
    result: Dict[str, Any] = {
        "summary": summary,
        "plots": [str(plot_path)] if plot_path.exists() else [],
        "logs": logs,
        "output_dir": str(out_root),
    }
    if npz_path and Path(npz_path).is_file():
        result["rollout_npz"] = str(npz_path)
    return result
