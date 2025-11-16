"""EMPROT Protein Dynamics Explorer (Gradio UI)

This file provides a small, clean Gradio interface that lets a user upload a
PDB file and run the EMPROT inference pipeline, then view a JSON summary,
plots, and optional logs. It is self-contained and minimizes dependencies.

Note:
    - The real inference entrypoint is emprot_inference.run_emprot_pipeline.
    - If that import fails in your environment, there is a commented fallback
      stub below to illustrate the expected return format.
"""

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union
import sys

import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

DEPLOY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

custom_css = """
:root {
  --emprot-bg: #05060a;
  --emprot-card-bg: #151821;
  --emprot-border: #262a33;
  --emprot-primary: #b1040e;
  --emprot-primary-hover: #d52b1e;
  --emprot-text-muted: #9ea4b3;
}

body {
  background: radial-gradient(circle at top, #141824 0, #05060a 55%);
}

.emprot-card {
  background: var(--emprot-card-bg);
  border-radius: 12px;
  border: 1px solid var(--emprot-border);
  box-shadow: 0 16px 34px rgba(0,0,0,0.65);
  padding: 20px 24px;
  margin-top: 20px;
}

.emprot-card h3 {
  margin-top: 0;
  margin-bottom: 4px;
  font-size: 1.05rem;
}

.emprot-step-desc {
  color: var(--emprot-text-muted);
  font-size: 0.9rem;
  margin-bottom: 12px;
}

.emprot-primary-btn button {
  width: 70%;
  max-width: 520px;
  font-weight: 600;
  font-size: 1rem;
}

#emprot-pdb-viewer {
  border-radius: 10px;
  border: 1px solid var(--emprot-border);
  box-shadow: 0 10px 30px rgba(0,0,0,0.7);
}

.emprot-status {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 0.9rem;
  color: var(--emprot-text-muted);
  margin-top: 8px;
}

.emprot-status strong {
  color: #e5e9f0;
}

.emprot-status .dots::after {
  content: "…";
  animation: emprot-dots 1.2s steps(4, end) infinite;
}

@keyframes emprot-dots {
  0%, 20%   { opacity: 0; }
  40%       { opacity: 0.4; }
  60%       { opacity: 0.7; }
  80%, 100% { opacity: 1; }
}

#emprot-logs pre {
  background: #050608;
  color: #e5e9f0;
  border-radius: 8px;
  padding: 12px 14px;
  max-height: 320px;
  overflow-y: auto;
  font-family: "JetBrains Mono", "Fira Code", Menlo, monospace;
  font-size: 0.8rem;
}

.emprot-plot-gallery {
  margin-top: 8px;
}

.emprot-plot-tile {
  display: inline-block;
  margin: 8px;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 10px 24px rgba(0,0,0,0.45);
  transition: transform 0.15s ease-out, box-shadow 0.15s ease-out;
}

.emprot-plot-tile img {
  max-width: 340px;
  height: auto;
  display: block;
}

.emprot-plot-tile:hover {
  transform: translateY(-3px) scale(1.01);
  box-shadow: 0 18px 34px rgba(0,0,0,0.7);
}

.emprot-footer {
  color: var(--emprot-text-muted);
  font-size: 0.8rem;
  text-align: center;
  margin-top: 18px;
}
"""

try:  # Import the real pipeline
    from emprot_inference import (
        add_cluster_ids_to_demo,
        generate_demo_embeddings,
        run_emprot_pipeline_quick_from_lmdb,
    )
except Exception:  # pragma: no cover - fallback is for dev only
    # Fallback placeholder for development only — DO NOT USE IN PRODUCTION.
    # def run_emprot_pipeline(pdb_source: Optional[str], *args, **kwargs) -> Dict[str, Any]:
    #     fig, ax = plt.subplots(figsize=(4, 3))
    #     ax.plot([0, 1, 2], [0, 1, 0]); ax.set_title("Demo Plot")
    #     return {
    #         "summary": {"demo": True, "pdb": pdb_source},
    #         "plots": [fig],
    #         "logs": "Ran demo fallback pipeline.",
    #     }
    raise


# -----------------------------
# Utilities
# -----------------------------

ImageLike = Union[str, os.PathLike]
FigureOrPath = Union[Figure, ImageLike]

_TEMP_FILES: List[str] = []

# Set cluster-specific defaults so users don't need to export env vars
_DEFAULT_ENV = {
    "COLLAPSE_DIR": "/oak/stanford/groups/rbaltman/ziyiw23/opt_collapse",
    "CLUSTER_MODEL_PATH": "/oak/stanford/groups/rbaltman/aderry/collapse-motifs/data/pdb100_cluster_fit_50000.pkl",
    "COLLAPSE_PYTHON": "/oak/stanford/groups/rbaltman/ziyiw23/venv/collapse/bin/python",
    # Used only for LMDB fallback mode; safe default from scripts
    "EMPROT_DATA_ROOT": "/scratch/groups/rbaltman/ziyiw23/traj_embeddings",
    "EMPROT_SPLIT": "test",
}
for _k, _v in _DEFAULT_ENV.items():
    os.environ.setdefault(_k, _v)

# Work around Pydantic v2 + Gradio schema generation crash in some environments
try:  # pragma: no cover - defensive monkeypatch
    from gradio.components.base import Component as _GrComponent  # type: ignore

    _orig_api_info = getattr(_GrComponent, "api_info", None)

    if callable(_orig_api_info):
        def _safe_api_info(self):  # type: ignore
            try:
                return _orig_api_info(self)
            except Exception:
                # Return minimal info to avoid failing Blocks config build
                return {}

        _GrComponent.api_info = _safe_api_info  # type: ignore
except Exception:
    pass


def _cleanup_temp_files() -> None:
    while _TEMP_FILES:
        p = _TEMP_FILES.pop()
        try:
            os.remove(p)
        except OSError:
            pass


def _fig_to_temp_png(fig: Figure, idx: int) -> str:
    tmp = tempfile.NamedTemporaryFile(prefix=f"emprot_plot_{idx}_", suffix=".png", delete=False)
    try:
        fig.savefig(tmp.name, format="png", bbox_inches="tight")
    finally:
        try:
            tmp.flush(); tmp.close()
        except Exception:
            pass
        plt.close(fig)
    _TEMP_FILES.append(tmp.name)
    return tmp.name


def _to_gallery_items(plots: Iterable[FigureOrPath]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for i, p in enumerate(plots):
        label = f"Plot {i + 1}"
        if isinstance(p, Figure):
            items.append((_fig_to_temp_png(p, i), label))
        else:
            items.append((os.fspath(p), label))
    return items


def _normalize_summary(summary: Any) -> str:
    if summary is None:
        return "{}"
    if isinstance(summary, str):
        try:
            # It's already JSON string?
            json.loads(summary)
            return summary
        except Exception:
            return json.dumps({"summary": summary}, indent=2)
    try:
        return json.dumps(summary, indent=2, ensure_ascii=False)
    except Exception:
        return json.dumps({"summary": str(summary)}, indent=2)


def _normalize_logs(logs: Any) -> str:
    if logs is None:
        return ""
    if isinstance(logs, str):
        return logs
    try:
        return json.dumps(logs, indent=2, ensure_ascii=False)
    except Exception:
        return str(logs)


# -----------------------------
# Helpers for staged workflow
# -----------------------------

def _summary_to_table(data: Any) -> List[List[Any]]:
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except Exception:
            return [["summary", data]]
    elif isinstance(data, dict):
        parsed = data
    else:
        return [["summary", str(data)]]
    rows: List[List[Any]] = []
    for k, v in parsed.items():
        if isinstance(v, (dict, list)):
            try:
                v_str = json.dumps(v, indent=2, ensure_ascii=False)
            except Exception:
                v_str = str(v)
            rows.append([k, v_str])
        else:
            rows.append([k, v])
    return rows


def _format_logs_block(lines: List[str]) -> str:
    text = "\n".join(line for line in lines if line is not None)
    return f"```\n{text}\n```" if text else "``` ```"


def _render_pdb_html(pdb_file: Optional[Union[str, Dict[str, Any]]]) -> str:
    """Render an uploaded PDB file using 3Dmol.js inside an iframe, showing only Cα atoms."""
    if not pdb_file:
        return "<div style='color:#888'>Upload a PDB file to view the structure.</div>"

    from pathlib import Path as _Path

    path: Optional[_Path] = None
    try:
        if isinstance(pdb_file, dict):
            name = pdb_file.get("name") or pdb_file.get("path")
            if isinstance(name, str):
                path = _Path(name)
        elif isinstance(pdb_file, str):
            path = _Path(pdb_file)
        elif hasattr(pdb_file, "name"):
            path = _Path(getattr(pdb_file, "name"))
    except Exception:
        path = None

    if path is None or not path.is_file():
        return "<div style='color:#c33'>PDB file could not be located on disk.</div>"

    try:
        pdb_text = path.read_text()
    except Exception as exc:
        return f"<div style='color:#c33'>Failed to read PDB: {exc}</div>"
    pdb_preview = pdb_text

    pdb_js = (
        pdb_preview
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("</script>", "<\\/script>")
    )

    inner_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head>
<body style="margin:0; background:#000;">
  <div id="pdb_viewer" style="width:100%; height:400px; position:relative;"></div>
  <script>
  (function init() {{
    if (typeof $3Dmol === "undefined") {{
      setTimeout(init, 200);
      return;
    }}
    var element = document.getElementById("pdb_viewer");
    if (!element) return;

    var viewer = $3Dmol.createViewer(element, {{ backgroundColor: "black" }});
    var pdbData = `{pdb_js}`;
    viewer.addModel(pdbData, "pdb");

    var model = viewer.getModel();
    if (!model || !model.selectedAtoms({{}}).length) {{
      element.innerHTML = "<div style='color:#f66;padding:1rem'>PDB parsed but no atoms were found.</div>";
      return;
    }}

    viewer.setStyle({{}}, {{ cartoon: {{ color: 'spectrum' }} }});
    viewer.zoomTo();
    viewer.render();
  }})();
  </script>
</body>
</html>"""

    srcdoc = inner_html.replace("'", "&#39;")

    iframe = (
        "<iframe style='width:100%;height:420px;border:1px solid #333;"
        "border-radius:4px;' "
        f"srcdoc='{srcdoc}'></iframe>"
    )
    return iframe


def _render_pipeline_outputs(result: Dict[str, Any], delta_t: float) -> Tuple[List[List[Any]], str, str]:
    summary_json = _normalize_summary(result.get("summary"))
    try:
        summary_data = json.loads(summary_json)
    except Exception:
        summary_data = {"summary": summary_json}
    summary_table = _summary_to_table(summary_data)
    plots: List[FigureOrPath] = list(result.get("plots", []) or [])
    import base64

    tiles: List[str] = []
    npz_path = result.get("rollout_npz")
    if isinstance(npz_path, str) and os.path.isfile(npz_path):
        try:
            import numpy as _np
            data = _np.load(npz_path)
            pred = _np.asarray(data.get("pred"))
            times_abs = _np.asarray(data.get("times_abs"))
            ridxs = _np.asarray(data.get("ridxs")) if "ridxs" in data else _np.arange(min(5, pred.shape[1]))
            try:
                dt = float(delta_t)
            except Exception:
                dt = 0.2
            t_plot = times_abs * float(dt)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_title("Rollout: time vs cluster id (selected residues)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Cluster ID")
            colors = [f"C{i%10}" for i in range(len(ridxs))]
            lines = []
            for i, r in enumerate(ridxs):
                line, = ax.plot([], [], color=colors[i], label=f"Residue {int(r)}")
                lines.append(line)
            ax.legend(loc="upper right", ncol=2, fontsize=8)
            ax.grid(alpha=0.25)

            def _snap(step: int) -> str:
                for i, r in enumerate(ridxs):
                    y = pred[: step + 1, int(r)]
                    x = t_plot[: step + 1]
                    lines[i].set_data(x, y)
                ax.relim(); ax.autoscale_view()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                buf.close()
                return f'<div class="emprot-plot-tile"><img src="data:image/png;base64,{b64}" alt="rollout step {step+1}"></div>'

            steps = list(range(pred.shape[0]))
            max_snaps = 12
            if len(steps) > max_snaps:
                import numpy as _np2
                idxs = _np2.linspace(0, len(steps) - 1, num=max_snaps).astype(int)
                steps = [int(s) for s in idxs]
            for s in steps:
                tiles.append(_snap(s))
            plt.close(fig)
        except Exception:
            pass
    for i, p in enumerate(plots):
        if isinstance(p, Figure):
            buf = io.BytesIO()
            p.savefig(buf, format="png", bbox_inches="tight")
            plt.close(p)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            tiles.append(f'<div class="emprot-plot-tile"><img src="data:image/png;base64,{b64}" alt="plot {i+1}"></div>')
        else:
            path = Path(os.fspath(p))
            try:
                with open(path, "rb") as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode("ascii")
                suffix = path.suffix.lower()
                if suffix == ".png":
                    mime = "image/png"
                elif suffix in {".jpg", ".jpeg"}:
                    mime = "image/jpeg"
                else:
                    mime = "application/octet-stream"
                tiles.append(f'<div class="emprot-plot-tile"><img src="data:{mime};base64,{b64}" alt="plot {i+1}"></div>')
            except Exception:
                tiles.append(f'<div class="emprot-plot-tile"><pre>{path}</pre></div>')

    gallery_html = "<div class='emprot-plot-gallery'>" + "".join(tiles) + "</div>"
    logs_text = _normalize_logs(result.get("logs"))
    return summary_table, gallery_html, logs_text


def _stage_embed(pdb_file: Optional[str], history_k: int, progress=gr.Progress(track_tqdm=True)) -> Generator:
    logs: List[str] = []
    if not pdb_file:
        raise gr.Error("Please upload a PDB file before embedding.")
    logs.append("Starting collapse embedding…")
    progress(0.0, desc="Queued…")
    status_text = "**Status:** Preparing to embed PDB."
    yield (
        gr.update(visible=False),              # summary
        status_text,                           # status
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
        # Add cluster IDs as part of the embed step
        stats = add_cluster_ids_to_demo(state["traj_dir"])
        logs.append("Cluster IDs added.")
        state["cluster_ids_ready"] = True
        ids_ready = True
        # Merge embedding summary with cluster-id stats
        combined_summary = dict(summary_data)
        combined_summary["cluster_ids"] = stats
        progress(1.0, desc="Embedding complete")
        summary_table = _summary_to_table(combined_summary)
        status_text = "**Status:** Embedding complete."
        yield (
            gr.update(value=summary_table, visible=True),
            status_text,
            "",
            _format_logs_block(logs),
            state,
            ids_ready,
            gr.update(interactive=True, visible=True),   # enable Run EMPROT
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
        summary_table, gallery_html, pipeline_logs = _render_pipeline_outputs(result, delta_t)
        logs.append("Inference complete.")
        combined_logs = "\n".join(logs + ["", pipeline_logs])
        status_text = "**Status:** Inference complete, results below."
        yield (
            gr.update(value=summary_table, visible=True),
            status_text,
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
            "",
            _format_logs_block(logs),
            embed_state,
            ids_ready,
            gr.update(interactive=True, visible=True),
        )
        raise


# -----------------------------
# UI definition
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = "/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/checkpoints/st_gumbel_F8_v3/best.pt"

with gr.Blocks(title="EMPROT Protein Dynamics Explorer", css=custom_css) as ui:
    gr.Markdown(
        """
        # EMPROT Protein Dynamics Explorer
        Run a three-step workflow to embed a PDB and roll out EMPROT predictions.
        """.strip()
    )

    # Step 1 – Upload & visualize PDB
    with gr.Column(elem_classes=["emprot-card"]):
        gr.Markdown("### Step 1 — Upload & visualize PDB")
        gr.Markdown(
            "Upload a single-frame PDB structure. The viewer will display a cartoon representation of the uploaded model.",
            elem_classes=["emprot-step-desc"],
        )
        pdb_input = gr.File(
            label="Upload PDB (.pdb)",
            file_types=[".pdb"],
            type="filepath",
        )
        pdb_view = gr.HTML(label="Uploaded PDB structure", elem_id="emprot-pdb-viewer")

    # Step 2 – Compute embeddings (Collapse)
    with gr.Column(elem_classes=["emprot-card"]):
        gr.Markdown("### Step 2 — Compute embeddings (COLLAPSE)")
        gr.Markdown(
            "Run the COLLAPSE pipeline to generate residue-level embeddings from the uploaded PDB, "
            "then assign cluster IDs used as input to EMPROT.",
            elem_classes=["emprot-step-desc"],
        )
        with gr.Row():
            embed_btn = gr.Button("Embed PDB", variant="primary", elem_classes=["emprot-primary-btn"])
        status_out = gr.Markdown("", label="Status", elem_classes=["emprot-status"])
        summary_out = gr.Dataframe(
            headers=["Key", "Value"],
            datatype=["str", "str"],
            label="Summary",
            visible=False,
        )

    # Step 3 – Run EMPROT inference
    with gr.Column(elem_classes=["emprot-card"]):
        gr.Markdown("### Step 3 — Run EMPROT inference")
        gr.Markdown(
            "Configure the rollout horizon and run the EMPROT transformer to predict future conformational states.",
            elem_classes=["emprot-step-desc"],
        )
        with gr.Row():
            ckpt_path = gr.Textbox(
                label="Checkpoint (.pt) path",
                value=DEFAULT_CKPT if os.path.isfile(DEFAULT_CKPT) else "",
                placeholder=str(PROJECT_ROOT / "output/checkpoints/<run>/best.pt"),
            )
        with gr.Row():
            steps = gr.Slider(label="Time steps", minimum=10, maximum=1000, step=10, value=100)
            dt = gr.Number(label="Delta t (display only)", value=0.2)
            history_k = gr.Slider(label="History K (frames)", minimum=1, maximum=32, step=1, value=1)
        with gr.Row():
            run_btn = gr.Button("Run EMPROT", interactive=False, visible=False, elem_classes=["emprot-primary-btn"])

    # Outputs
    plots_out = gr.HTML(label="Plots")
    with gr.Accordion("View Logs (advanced)", open=False):
        logs_out = gr.Markdown(label="Logs", elem_id="emprot-logs")

    # Info + footer
    with gr.Accordion("What is EMPROT?", open=False):
        gr.Markdown(
            "EMPROT predicts future protein conformational states by combining residue-level "
            "embeddings from the COLLAPSE pipeline with a transformer-like temporal model over "
            "cluster IDs. The model rolls out likely future cluster trajectories from a single PDB frame."
        )
    gr.Markdown(
        "EMPROT Protein Dynamics Explorer — Built at Stanford Biomedical Data Science / Altman Lab",
        elem_classes=["emprot-footer"],
    )

    embed_state = gr.State({})
    ids_state = gr.State(False)

    embed_btn.click(
        fn=_stage_embed,
        inputs=[pdb_input, history_k],
        outputs=[summary_out, status_out, plots_out, logs_out, embed_state, ids_state, run_btn],
    )
    pdb_input.change(
        fn=_render_pdb_html,
        inputs=[pdb_input],
        outputs=[pdb_view],
    )
    run_btn.click(
        fn=_stage_run_inference,
        inputs=[embed_state, ids_state, ckpt_path, steps, dt, history_k],
        outputs=[summary_out, status_out, plots_out, logs_out, embed_state, ids_state, run_btn],
    )


if __name__ == "__main__":
    ui.queue()  # enable queuing with default settings
    ui.launch(share=True, max_threads=1)
