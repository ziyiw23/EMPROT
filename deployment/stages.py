"""Staged workflow callbacks for the EMPROT Gradio UI."""

from pathlib import Path
import sys
from typing import Any, Dict, Generator, List, Optional

import gradio as gr

from .ui_components import (
    _format_logs_block,
    _render_pipeline_outputs,
    _summary_to_table,
)

DEPLOY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Import the real pipeline
    from .emprot_inference import (  # type: ignore
        add_cluster_ids_to_demo,
        generate_demo_embeddings,
        run_emprot_pipeline_quick_from_lmdb,
    )
except Exception:  # pragma: no cover - fallback is for dev only
    raise


def _stage_embed(pdb_file: Optional[str], history_k: int, progress=gr.Progress(track_tqdm=True)) -> Generator:
    logs: List[str] = []
    if not pdb_file:
        raise gr.Error("Please upload a PDB file before embedding.")
    logs.append("Starting collapse embedding…")
    progress(0.0, desc="Queued…")
    status_text = "**Status:** Preparing to embed PDB."
    yield (
        gr.update(visible=False),
        status_text,
        None,
        "",
        _format_logs_block(logs),
        {},
        False,
        gr.update(interactive=False),
    )
    try:
        progress(0.05, desc="Preparing demo workspace…")
        status_text = "**Status:** Embedding frames with COLLAPSE <span class='dots'></span>"
        yield (
            gr.update(visible=False),
            status_text,
            None,
            "",
            _format_logs_block(logs),
            {},
            False,
            gr.update(interactive=False),
        )
        progress(0.2, desc="Running collapse embedding (gen_embed.py)…")
        state, summary_data, stage_logs = generate_demo_embeddings(
            pdb_source=pdb_file,
            recent_full_frames=int(history_k),
        )
        logs.extend(stage_logs)
        logs.append("Embedding complete. Adding cluster IDs…")
        stats = add_cluster_ids_to_demo(state["traj_dir"])
        logs.append(f"Cluster IDs added (device={stats.get('device', 'unknown')}).")
        state["cluster_ids_ready"] = True
        ids_ready = True
        combined_summary = dict(summary_data)
        combined_summary["cluster_ids"] = stats
        progress(1.0, desc="Embedding complete")
        summary_table = _summary_to_table(combined_summary)
        status_text = "**Status:** Embedding complete."
        yield (
            gr.update(value=summary_table, visible=True),
            status_text,
            None,
            "",
            _format_logs_block(logs),
            state,
            ids_ready,
            gr.update(interactive=True, visible=True),
        )
    except Exception as exc:
        logs.append(f"ERROR: {exc}")
        try:
            progress(1.0, desc="Failed")
        except Exception:
            pass
        status_text = "**Status:** Error during embedding — see Logs for details."
        yield (
            gr.update(value=_summary_to_table({"error": str(exc)}), visible=True),
            status_text,
            None,
            "",
            _format_logs_block(logs),
            {},
            False,
            gr.update(interactive=False),
        )
        raise


def _stage_run_inference(
    embed_state: Dict[str, Any],
    ids_ready: bool,
    ckpt_path: str,
    time_steps: int,
    delta_t: float,
    history_k: int,
) -> Generator:
    logs: List[str] = []
    if not embed_state or not embed_state.get("traj_dir"):
        raise gr.Error("Please embed the PDB first.")
    if not ids_ready or not embed_state.get("cluster_ids_ready"):
        raise gr.Error("Please add cluster IDs before running inference.")
    if not ckpt_path:
        raise gr.Error("Please provide a checkpoint (.pt) path.")
    logs.append("Launching EMPROT quick rollout from demo LMDB…")
    status_text = "**Status:** Running EMPROT rollout & generating plots <span class='dots'></span>"
    yield (
        gr.update(value=_summary_to_table({"status": "running_inference"}), visible=True),
        status_text,
        None,
        "",
        _format_logs_block(logs),
        embed_state,
        ids_ready,
        gr.update(interactive=False),
    )
    try:
        result = run_emprot_pipeline_quick_from_lmdb(
            traj_dir=embed_state["traj_dir"],
            ckpt_path=ckpt_path,
            time_steps=int(time_steps),
            recent_full_frames=int(history_k),
        )
        summary_table, rollout_fig, gallery_html, pipeline_logs = _render_pipeline_outputs(result, delta_t)
        logs.append("Inference complete.")
        combined_logs = "\n".join(logs + ["", pipeline_logs])
        status_text = "**Status:** Inference complete, results below."
        yield (
            gr.update(value=summary_table, visible=True),
            status_text,
            rollout_fig,
            gallery_html,
            f"```\n{combined_logs}\n```",
            embed_state,
            ids_ready,
            gr.update(interactive=True, visible=True),
        )
    except Exception as exc:
        logs.append(f"ERROR: {exc}")
        status_text = "**Status:** Error during inference — see Logs for details."
        yield (
            gr.update(value=_summary_to_table({"error": str(exc)}), visible=True),
            status_text,
            None,
            "",
            _format_logs_block(logs),
            embed_state,
            ids_ready,
            gr.update(interactive=True, visible=True),
        )
        raise

