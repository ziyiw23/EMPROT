"""UI helper utilities for the EMPROT Gradio application."""

import html
import io
import json
import os
import re
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

.emprot-plot-button {
  background: none;
  border: 0;
  padding: 0;
  width: 100%;
  cursor: pointer;
}

.emprot-plot-button img {
  width: 100%;
  display: block;
}

.emprot-lightbox {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.85);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.emprot-lightbox.open {
  display: flex;
}

.emprot-lightbox-inner {
  position: relative;
  max-width: 92%;
  max-height: 92%;
}

.emprot-lightbox-inner img {
  max-width: 100%;
  max-height: 100%;
  border-radius: 10px;
  box-shadow: 0 25px 60px rgba(0,0,0,0.65);
}

.emprot-lightbox-close {
  position: absolute;
  top: -18px;
  right: -18px;
  background: rgba(0,0,0,0.8);
  color: #fff;
  border: none;
  width: 38px;
  height: 38px;
  border-radius: 50%;
  font-size: 1.4rem;
  cursor: pointer;
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

AA_CHOICES: List[str] = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

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


def _image_tile(data_uri: str, alt: str) -> str:
    escaped_alt = html.escape(alt or "")
    return (
        "<div class='emprot-plot-tile'>"
        f"<button type='button' class='emprot-plot-button' data-src='{data_uri}' data-alt='{escaped_alt}'>"
        f"<img src='{data_uri}' alt='{escaped_alt}'>"
        "</button>"
        "</div>"
    )


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


def _compute_pdb_summary(pdb_file: Optional[Union[str, Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Return the summary HTML plus residue metadata for an uploaded PDB."""
    if not pdb_file:
        return "Upload a PDB file to see a summary.", []

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
        return "PDB file could not be located on disk.", []

    try:
        pdb_text = path.read_text()
    except Exception as exc:
        return f"Failed to read PDB: {exc}", []

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

        residue_meta: List[Dict[str, Any]] = []
        seen_residues: set = set()
        seq_index = 0
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
                    if key_basic not in seen_residues:
                        seen_residues.add(key_basic)
                        residue_meta.append(
                            {
                                "index": seq_index,
                                "chain": chain,
                                "resid": resid,
                                "resname": resname,
                            }
                        )
                        seq_index += 1
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

        residue_count = len(residue_meta) or len(protein_like_residues)

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
        return html, residue_meta
    except Exception as exc:
        return f"Failed to summarize PDB: {exc}", []


def _summarize_pdb(pdb_file: Optional[Union[str, Dict[str, Any]]]) -> str:
    summary, _ = _compute_pdb_summary(pdb_file)
    return summary


def _summarize_pdb_with_meta(pdb_file: Optional[Union[str, Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
    return _compute_pdb_summary(pdb_file)


def _parse_residue_filter_text(text: Optional[str]) -> List[int]:
    if not text:
        return []
    values: List[int] = []
    tokens = re.split(r"[,\s]+", text.strip())
    for tok in tokens:
        if not tok:
            continue
        if "-" in tok:
            parts = tok.split("-", 1)
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            values.extend(list(range(start, end + 1)))
        else:
            try:
                values.append(int(tok))
            except ValueError:
                continue
    return values


def _format_residue_label(meta: Optional[Dict[str, Any]], default: str) -> str:
    if not meta:
        return default
    chain = meta.get("chain") or "?"
    resid = meta.get("resid")
    resname = meta.get("resname") or "UNK"
    idx = meta.get("index")
    if resid is None:
        return f"{resname} chain {chain} (idx {idx})"
    return f"{resname} {chain}{resid} (idx {idx})"


def _resolve_residue_selection(
    pred: Any,
    default_indices: Optional[Iterable[int]],
    residue_meta: Optional[List[Dict[str, Any]]],
    manual_ids: Optional[List[int]],
    aa_filter: Optional[Iterable[str]],
) -> Tuple[List[int], str]:
    R = pred.shape[1]
    selected: Optional[List[int]] = None
    notes: List[str] = []
    aa_set = {aa.strip().upper() for aa in (aa_filter or []) if aa}
    filters_requested = bool(manual_ids) or bool(aa_set)
    manual = sorted({idx for idx in (manual_ids or []) if 0 <= idx < R})
    if manual:
        selected = manual
        notes.append(f"manual={manual}")
    elif manual_ids:
        notes.append("manual filter ignored (no valid indices)")
    if aa_set:
        if residue_meta:
            aa_indices: List[int] = []
            limit = min(R, len(residue_meta))
            for pos in range(limit):
                meta = residue_meta[pos] or {}
                resname = (meta.get("resname") or "").strip().upper()
                if resname not in aa_set:
                    continue
                candidate_idx = meta.get("index")
                if isinstance(candidate_idx, int) and 0 <= candidate_idx < R:
                    target_idx = candidate_idx
                else:
                    target_idx = pos
                if 0 <= target_idx < R:
                    aa_indices.append(target_idx)
            aa_indices = list(dict.fromkeys(aa_indices))
            if selected is None:
                selected = aa_indices
            else:
                selected = [idx for idx in selected if idx in aa_indices]
            match_note = f"aa={'/'.join(sorted(aa_set))}"
            match_note += f" ({len(aa_indices)} match{'es' if len(aa_indices) == 1 else 'es'})"
            notes.append(match_note)
            if not aa_indices:
                notes.append("aa filter yielded zero residues")
        else:
            notes.append("aa filter ignored (metadata unavailable)")
    if not selected:
        if filters_requested:
            selected = []
            notes.append("filters matched 0 residues")
            return selected, "; ".join(notes)
        if default_indices:
            selected = [int(idx) for idx in default_indices]
        else:
            selected = list(range(min(10, R)))
        notes.append("using default residue subset")
    return selected, "; ".join(notes)


def _render_pipeline_outputs(
    result: Dict[str, Any],
    delta_t: float,
    residue_meta: Optional[List[Dict[str, Any]]] = None,
    residue_ids: Optional[List[int]] = None,
    aa_filter: Optional[List[str]] = None,
) -> Tuple[List[List[Any]], "go.Figure", str, str]:
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
    filter_note = ""
    volatility_rows: List[List[str]] = []
    npz_path = result.get("rollout_npz")
    if isinstance(npz_path, str) and os.path.isfile(npz_path):
        try:
            import numpy as _np
            data = _np.load(npz_path)
            pred = _np.asarray(data.get("pred"))
            times_abs = _np.asarray(data.get("times_abs"))
            ridxs = _np.asarray(data.get("ridxs")) if "ridxs" in data else _np.arange(min(5, pred.shape[1]))
            manual_ids = residue_ids or []
            aa_filters = [aa.strip().upper() for aa in (aa_filter or []) if aa]
            filters_active = bool(manual_ids) or bool(aa_filters)
            residue_meta_safe = residue_meta or []
            T, R = pred.shape

            default_indices = (
                ridxs.tolist() if hasattr(ridxs, "tolist") else list(ridxs)
            )
            filtered_indices, filter_note = _resolve_residue_selection(
                pred, default_indices, residue_meta_safe, manual_ids, aa_filters
            )
            analysis_indices = filtered_indices[:] if (filtered_indices or filters_active) else list(range(R))
            if not analysis_indices and not filters_active:
                analysis_indices = list(range(min(10, R)))
            plot_indices = analysis_indices[:]
            max_plot_lines = 20
            if len(plot_indices) > max_plot_lines:
                extra_note = f"plot limited to first {max_plot_lines} residues of selection"
                filter_note = f"{filter_note}; {extra_note}" if filter_note else extra_note
                plot_indices = plot_indices[:max_plot_lines]

            # ---- Volatility statistics ----
            try:
                if analysis_indices:
                    stats_indices = analysis_indices
                elif filters_active:
                    stats_indices = []
                else:
                    stats_indices = list(range(R))
                stats_pred = pred[:, stats_indices] if stats_indices else pred
                T_stats = stats_pred.shape[0]
                R_stats = stats_pred.shape[1]
                if T_stats > 1:
                    changes = (stats_pred[1:, :] != stats_pred[:-1, :])
                    changes_count = _np.sum(changes, axis=0)
                    volatility = changes_count / float(T_stats - 1)
                else:
                    volatility = _np.zeros(R_stats, dtype=float)
                mean_vol = float(volatility.mean()) if volatility.size else 0.0
                median_vol = float(_np.median(volatility)) if volatility.size else 0.0
                num_frozen = int(_np.sum(volatility == 0.0))
                num_high = int(_np.sum(volatility > 0.3))
                k = min(10, volatility.shape[0])
                top_labels: List[str] = []
                top_vol_list: List[float] = []
                if k > 0:
                    top_idx = _np.argsort(-volatility)[:k]
                    for idx in top_idx:
                        global_idx = stats_indices[idx]
                        label = _format_residue_label(
                            residue_meta_safe[global_idx] if residue_meta_safe and global_idx < len(residue_meta_safe) else None,
                            f"Residue {global_idx}",
                        )
                        top_labels.append(label)
                        top_vol_list.append(float(volatility[idx]))
                volatility_rows = [
                    ["num_residues_considered", str(len(stats_indices))],
                    ["volatility_mean", f"{mean_vol:.3f}"],
                    ["volatility_median", f"{median_vol:.3f}"],
                    ["num_frozen_residues", str(num_frozen)],
                    ["num_highly_dynamic_residues", str(num_high)],
                    ["top_dynamic_residues", ", ".join(top_labels)],
                    ["top_dynamic_residue_volatility", ", ".join(f"{v:.2f}" for v in top_vol_list)],
                ]
            except Exception:
                volatility_rows = []

            try:
                dt_val = float(delta_t)
            except Exception:
                dt_val = 0.2
            t_plot = times_abs * float(dt_val)
            x_min = float(t_plot.min()) if t_plot.size else 0.0
            x_max = float(t_plot.max()) if t_plot.size else 1.0

            def _res_label(idx: int) -> str:
                if residue_meta_safe and idx < len(residue_meta_safe):
                    return _format_residue_label(residue_meta_safe[idx], f"Residue {idx}")
                return f"Residue {idx}"

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_title("Rollout: time vs cluster id (selected residues)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Cluster ID")
            colors = [f"C{i%10}" for i in range(len(plot_indices))]
            lines = []
            for i, r in enumerate(plot_indices):
                line, = ax.plot([], [], color=colors[i], label=_res_label(int(r)))
                lines.append(line)
            ax.legend(loc="upper right", ncol=2, fontsize=8)
            ax.grid(alpha=0.25)

            def _snap(step: int) -> str:
                for i, r in enumerate(plot_indices):
                    y = pred[: step + 1, int(r)]
                    x_vals = t_plot[: step + 1]
                    lines[i].set_data(x_vals, y)
                ax.relim(); ax.autoscale_view()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
                buf.close()
                return _image_tile(data_uri, f"rollout step {step+1}")

            steps = list(range(pred.shape[0]))
            max_snaps = 12
            if len(steps) > max_snaps:
                import numpy as _np2
                idxs = _np2.linspace(0, len(steps) - 1, num=max_snaps).astype(int)
                steps = [int(s) for s in idxs]
            if plot_indices:
                for s in steps:
                    tiles.append(_snap(s))
            plt.close(fig)

            x_vals = t_plot.astype(float).tolist()
            traces = []
            for i, r in enumerate(plot_indices):
                y_vals = pred[:, int(r)].astype(float).tolist()
                label = _res_label(int(r))
                traces.append(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines+markers",
                        name=label,
                        customdata=[label] * pred.shape[0],
                        hovertemplate="t=%{x}<br>cluster=%{y}<br>%{customdata}<extra></extra>",
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
            margin = max(1.0, 0.05 * (y_max - y_min if (y_max - y_min) != 0 else 1.0))
            y_low = y_min - margin
            y_high = y_max + margin
            rollout_fig.update_layout(
                xaxis=dict(title="Time", range=[x_min, x_max]),
                yaxis=dict(title="Cluster ID", range=[y_low, y_high]),
                showlegend=True,
            )
        except Exception:
            pass
    for i, p in enumerate(plots):
        if isinstance(p, Figure):
            buf = io.BytesIO()
            p.savefig(buf, format="png", bbox_inches="tight")
            plt.close(p)
            data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            buf.close()
            tiles.append(_image_tile(data_uri, f"plot {i+1}"))
        else:
            path = Path(os.fspath(p))
            try:
                with open(path, "rb") as f:
                    data = f.read()
                suffix = path.suffix.lower()
                if suffix in {".jpg", ".jpeg"}:
                    mime = "image/jpeg"
                elif suffix == ".png":
                    mime = "image/png"
                else:
                    mime = "application/octet-stream"
                data_uri = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
                tiles.append(_image_tile(data_uri, f"plot {i+1}"))
            except Exception:
                tiles.append(f'<div class="emprot-plot-tile"><pre>{path}</pre></div>')

    if volatility_rows:
        summary_table.extend(volatility_rows)
    if residue_ids or aa_filters:
        desc_parts: List[str] = []
        if residue_ids:
            desc_parts.append(f"indices={','.join(str(i) for i in residue_ids)}")
        if aa_filters:
            aa_clean = sorted({aa for aa in aa_filters if aa})
            if aa_clean:
                desc_parts.append(f"aa={'/'.join(aa_clean)}")
        if desc_parts:
            summary_table.append(["residue_filters", "; ".join(desc_parts)])
    if filter_note:
        summary_table.append(["residue_filter_note", filter_note])

    lightbox_script = """
<script>
(function(){
  if (!window.__emprotLightboxSetup) {
    window.__emprotLightboxSetup = true;
    const overlay = document.createElement('div');
    overlay.className = 'emprot-lightbox';
    overlay.innerHTML = '<div class="emprot-lightbox-inner"><button type="button" class="emprot-lightbox-close" aria-label="Close">×</button><img src="" alt=""></div>';
    document.body.appendChild(overlay);
    const img = overlay.querySelector('img');
    const closeBtn = overlay.querySelector('.emprot-lightbox-close');
    const close = () => {
      overlay.classList.remove('open');
      img.src = '';
      img.alt = '';
    };
    closeBtn.addEventListener('click', close);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) close();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') close();
    });
    window.__emprotLightbox = { overlay, img, open(src, alt) {
      img.src = src;
      img.alt = alt || '';
      overlay.classList.add('open');
    }};
  }
  document.querySelectorAll('.emprot-plot-button').forEach((btn) => {
    if (!btn.__emprotBound) {
      btn.__emprotBound = true;
      btn.addEventListener('click', () => {
        const src = btn.getAttribute('data-src');
        const alt = btn.getAttribute('data-alt') || '';
        window.__emprotLightbox.open(src, alt);
      });
    }
  });
})();
</script>
"""

    gallery_html = "<div class='emprot-plot-gallery'>" + "".join(tiles) + "</div>" + lightbox_script
    logs_text = _normalize_logs(result.get("logs"))
    return summary_table, rollout_fig, gallery_html, logs_text

