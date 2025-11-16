"""UI helper utilities for the EMPROT Gradio application."""

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go

ImageLike = Union[str, os.PathLike]
FigureOrPath = Union[Figure, ImageLike]

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

.emprot-summary {
  margin-top: 16px;
  font-size: 0.9rem;
  color: var(--emprot-text-muted);
}

.emprot-summary-title {
  font-weight: 600;
  font-size: 0.95rem;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  color: #e5e9f0;
  margin-bottom: 8px;
}

.emprot-summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
}

.emprot-summary-card {
  background: #090b12;
  border-radius: 10px;
  border: 1px solid var(--emprot-border);
  padding: 10px 14px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.6);
}

.emprot-summary-card h4 {
  margin: 0 0 6px 0;
  font-size: 0.9rem;
  font-weight: 600;
  color: #e5e9f0;
}

.emprot-summary-item {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin: 2px 0;
}

.emprot-summary-key {
  color: var(--emprot-text-muted);
}

.emprot-summary-value {
  color: #f3f4f6;
  font-weight: 500;
  text-align: right;
}

.emprot-summary-subtext {
  margin-top: 6px;
  font-size: 0.8rem;
  line-height: 1.3;
}
"""

_TEMP_FILES: List[str] = []


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


def _fmt_int(n: Optional[int]) -> str:
    return f"{n:,}" if n is not None else "—"


def _summarize_pdb(pdb_file: Optional[Union[str, Dict[str, Any]]]) -> str:
    """Compute a small textual summary/metadata block for an uploaded PDB."""
    if not pdb_file:
        return "Upload a PDB file to see a summary."

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
        return "PDB file could not be located on disk."

    try:
        pdb_text = path.read_text()
    except Exception as exc:
        return f"Failed to read PDB: {exc}"

    try:
        lines = pdb_text.splitlines()
        aa_resnames = {
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        }
        water_resnames = {"HOH", "WAT"}
        lipid_resnames = {"POPC", "POPE", "DPPC", "DPPE", "CHOL", "DLPE", "DLPC", "DSPC", "DOPC"}
        ion_resnames = {"NA", "K", "CL", "CA", "MG", "ZN", "MN", "FE"}

        total_atom_records = 0
        total_hetatm_records = 0
        protein_like_residues: set = set()
        ca_count = 0
        chains_seen: set = set()
        chain_res_ranges: Dict[str, List[int]] = {}
        coords: List[Tuple[float, float, float]] = []

        protein_residues: set = set()
        water_residues: set = set()
        lipid_residues: set = set()
        ion_residues: set = set()
        ligand_residues: set = set()

        for line in lines:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            rec_type = line[0:6].strip()
            is_atom = rec_type == "ATOM"
            if is_atom:
                total_atom_records += 1
            else:
                total_hetatm_records += 1

            resname = line[17:20].strip() or "UNK"
            chain = line[21].strip() or "?"
            resid_str = line[22:26].strip()
            try:
                resid = int(resid_str)
            except Exception:
                resid = None

            if is_atom:
                chains_seen.add(chain)
                if resid is not None:
                    key_basic = (chain, resid)
                    protein_like_residues.add(key_basic)
                    if chain not in chain_res_ranges:
                        chain_res_ranges[chain] = [resid, resid]
                    else:
                        lo, hi = chain_res_ranges[chain]
                        if resid < lo:
                            chain_res_ranges[chain][0] = resid
                        if resid > hi:
                            chain_res_ranges[chain][1] = resid

                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    ca_count += 1
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except Exception:
                    pass

                if resid is not None and resname in aa_resnames:
                    protein_residues.add((chain, resid, resname))
            else:
                if resid is None:
                    continue
                key_comp = (chain, resid, resname)
                if resname in water_resnames:
                    water_residues.add(key_comp)
                elif resname in lipid_resnames:
                    lipid_residues.add(key_comp)
                elif resname in ion_resnames:
                    ion_residues.add(key_comp)
                else:
                    ligand_residues.add(key_comp)

        total_atoms = total_atom_records + total_hetatm_records
        if total_atoms == 0:
            return "No ATOM/HETATM records found in this file."

        residue_count = len(protein_like_residues)

        chain_ids_clean = sorted({c for c in chains_seen if c and c != "?"})
        if chain_ids_clean:
            chains_label = ", ".join(chain_ids_clean)
        elif chains_seen:
            chains_label = "(no chain IDs; treating as single chain)"
        else:
            chains_label = "N/A"

        chain_range_parts: List[str] = []
        for c in sorted(chain_res_ranges.keys()):
            lo, hi = chain_res_ranges[c]
            label = c if c and c != "?" else "?"
            chain_range_parts.append(f"{label}: {lo}–{hi}")
        chain_ranges_str = "; ".join(chain_range_parts) if chain_range_parts else "N/A"

        protein_res_count = len(protein_residues)
        water_res_count = len(water_residues)
        lipid_res_count = len(lipid_residues)
        ion_res_count = len(ion_residues)
        ligand_res_count = len(ligand_residues)

        bbox_ok = False
        size_x = size_y = size_z = 0.0
        center_x = center_y = center_z = 0.0
        if coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            zs = [c[2] for c in coords]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            zmin, zmax = min(zs), max(zs)
            size_x = xmax - xmin
            size_y = ymax - ymin
            size_z = zmax - zmin
            center_x = (xmin + xmax) / 2.0
            center_y = (ymin + ymax) / 2.0
            center_z = (zmin + zmax) / 2.0
            bbox_ok = True

        ranges_detail = "; ".join(
            f"{(c if c and c != '?' else '?')}: {lo}-{hi}"
            for c, (lo, hi) in sorted(chain_res_ranges.items())
        )
        if not ranges_detail:
            ranges_detail = "No residue ranges available."

        html = f"""
<div class="emprot-summary">
  <div class="emprot-summary-title">PDB Summary</div>

  <div class="emprot-summary-grid">

    <div class="emprot-summary-card">
      <h4>Basic</h4>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">Total atoms</span>
        <span class="emprot-summary-value">{_fmt_int(total_atoms)}</span>
      </div>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">ATOM records</span>
        <span class="emprot-summary-value">{_fmt_int(total_atom_records)}</span>
      </div>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">HETATM records</span>
        <span class="emprot-summary-value">{_fmt_int(total_hetatm_records)}</span>
      </div>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">Protein-like residues</span>
        <span class="emprot-summary-value">{_fmt_int(residue_count)}</span>
      </div>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">Cα atoms</span>
        <span class="emprot-summary-value">{_fmt_int(ca_count)}</span>
      </div>
    </div>

    <div class="emprot-summary-card">
      <h4>Chains & ranges</h4>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">Chains</span>
        <span class="emprot-summary-value">{", ".join(chain_ids_clean) if chain_ids_clean else "—"}</span>
      </div>
      <div class="emprot-summary-subtext">
        {ranges_detail}
      </div>
    </div>

    <div class="emprot-summary-card">
      <h4>System composition (by residues)</h4>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">Protein</span>
        <span class="emprot-summary-value">{_fmt_int(protein_res_count)}</span>
      </div>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">Waters</span>
        <span class="emprot-summary-value">{_fmt_int(water_res_count)}</span>
      </div>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">Lipids</span>
        <span class="emprot-summary-value">{_fmt_int(lipid_res_count)}</span>
      </div>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">Ions</span>
        <span class="emprot-summary-value">{_fmt_int(ion_res_count)}</span>
      </div>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">Ligands</span>
        <span class="emprot-summary-value">{_fmt_int(ligand_res_count)}</span>
      </div>
    </div>

    <div class="emprot-summary-card">
      <h4>Bounding box (Å)</h4>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">Size</span>
        <span class="emprot-summary-value">
          {f"{size_x:.1f} × {size_y:.1f} × {size_z:.1f}" if bbox_ok else "N/A"}
        </span>
      </div>
      <div class="emprot-summary-item">
        <span class="emprot-summary-key">Center</span>
        <span class="emprot-summary-value">
          {f"({center_x:.1f}, {center_y:.1f}, {center_z:.1f})" if bbox_ok else "N/A"}
        </span>
      </div>
    </div>

  </div>
</div>
"""
        return html
    except Exception as exc:
        return f"Failed to summarize PDB: {exc}"


def _render_pipeline_outputs(result: Dict[str, Any], delta_t: float) -> Tuple[List[List[Any]], "go.Figure", str, str]:
    summary_json = _normalize_summary(result.get("summary"))
    try:
        summary_data = json.loads(summary_json)
    except Exception:
        summary_data = {"summary": summary_json}
    summary_table = _summary_to_table(summary_data)
    plots: List[FigureOrPath] = list(result.get("plots", []) or [])
    import base64

    tiles: List[str] = []
    rollout_fig = go.Figure()
    npz_path = result.get("rollout_npz")
    if isinstance(npz_path, str) and os.path.isfile(npz_path):
        try:
            import numpy as _np
            data = _np.load(npz_path)
            pred = _np.asarray(data.get("pred"))
            times_abs = _np.asarray(data.get("times_abs"))
            ridxs = _np.asarray(data.get("ridxs")) if "ridxs" in data else _np.arange(min(5, pred.shape[1]))
            try:
                dt_val = float(delta_t)
            except Exception:
                dt_val = 0.2
            t_plot = times_abs * float(dt_val)
            x_min = float(t_plot.min()) if t_plot.size else 0.0
            x_max = float(t_plot.max()) if t_plot.size else 1.0
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
                    x_vals = t_plot[: step + 1]
                    lines[i].set_data(x_vals, y)
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

            x_vals = t_plot.astype(float).tolist()
            traces = []
            for i, r in enumerate(ridxs):
                y_vals = pred[:, int(r)].astype(float).tolist()
                traces.append(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines+markers",
                        name=f"Residue {int(r)}",
                        customdata=[int(r)] * pred.shape[0],
                        hovertemplate="t=%{x}<br>cluster=%{y}<br>residue=%{customdata}<extra></extra>",
                        marker=dict(size=6),
                    )
                )
            rollout_fig = go.Figure(data=traces)
            try:
                y_min = float(pred.min())
                y_max = float(pred.max())
            except Exception:
                y_min, y_max = 0.0, 1.0
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
            rollout_fig.update_layout(
                xaxis=dict(title="Time", range=[x_min, x_max]),
                yaxis=dict(title="Cluster ID", range=[y_min, y_max]),
                showlegend=True,
            )
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
    return summary_table, rollout_fig, gallery_html, logs_text

