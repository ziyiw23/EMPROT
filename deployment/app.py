"""EMPROT Protein Dynamics Explorer (Gradio UI)

This file provides a small, clean Gradio interface that lets a user upload a
PDB file and run the EMPROT inference pipeline, then view a JSON summary,
plots, and optional logs. It is self-contained and minimizes dependencies.

Note:
    - The real inference entrypoint is emprot_inference.run_emprot_pipeline.
    - If that import fails in your environment, there is a commented fallback
      stub below to illustrate the expected return format.
"""

import os
import sys
from pathlib import Path

import gradio as gr

from .stages import _stage_embed, _stage_run_inference
from .ui_components import AA_CHOICES, custom_css, _render_pdb_html, _summarize_pdb_with_meta

DEPLOY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
                return {}

        _GrComponent.api_info = _safe_api_info  # type: ignore
except Exception:
    pass

DEFAULT_CKPT = "/oak/stanford/groups/rbaltman/ziyiw23/EMPROT/output/checkpoints/res_centric_F3_v1/best.pt"

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
        pdb_summary = gr.HTML(label="PDB summary")

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
            dt = gr.Number(label="Delta t (ns)", value=0.2)
            history_k = gr.Slider(label="History K (frames)", minimum=1, maximum=32, step=1, value=1)
        with gr.Row():
            residue_filter = gr.Textbox(
                label="Residues (0-based indices or ranges)",
                placeholder="e.g., 0-5, 12, 15",
            )
            aa_filter = gr.Dropdown(
                label="Amino acids (optional)",
                choices=AA_CHOICES,
                multiselect=True,
            )
        with gr.Row():
            run_btn = gr.Button("Run EMPROT", interactive=False, visible=False, elem_classes=["emprot-primary-btn"])
        inference_status = gr.Markdown("", label="Inference status", elem_classes=["emprot-status"])

    # Outputs
    rollout_plot = gr.Plot(label="", show_label=False)
    plots_out = gr.HTML(label="Plots (PNG gallery)")
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

    residue_meta_state = gr.State([])
    embed_state = gr.State({})
    ids_state = gr.State(False)

    embed_btn.click(
        fn=_stage_embed,
        inputs=[pdb_input, history_k, residue_meta_state],
        outputs=[summary_out, status_out, inference_status, rollout_plot, plots_out, logs_out, embed_state, ids_state, run_btn],
    )
    pdb_input.change(
        fn=_render_pdb_html,
        inputs=[pdb_input],
        outputs=[pdb_view],
    )
    pdb_input.change(
        fn=_summarize_pdb_with_meta,
        inputs=[pdb_input],
        outputs=[pdb_summary, residue_meta_state],
    )
    run_btn.click(
        fn=_stage_run_inference,
        inputs=[embed_state, ids_state, ckpt_path, steps, dt, history_k, residue_filter, aa_filter, residue_meta_state],
        outputs=[summary_out, status_out, inference_status, rollout_plot, plots_out, logs_out, embed_state, ids_state, run_btn],
    )


if __name__ == "__main__":
    ui.queue()  # enable queuing with default settings
    ui.launch(share=True, max_threads=1)
