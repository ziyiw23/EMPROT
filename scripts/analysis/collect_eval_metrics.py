#!/usr/bin/env python3
import argparse, json, os, csv

def read_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description='Collect eval metrics into a CSV')
    ap.add_argument('--root', type=str, required=True, help='Sweep root containing run subdirs')
    ap.add_argument('--out_csv', type=str, required=True, help='Path to write CSV')
    args = ap.parse_args()

    rows = []
    for name in sorted(os.listdir(args.root)):
        run_dir = os.path.join(args.root, name)
        if not os.path.isdir(run_dir):
            continue
        params = read_json(os.path.join(run_dir, 'params.json')) or {}
        hs = read_json(os.path.join(run_dir, 'histogram_summary.json')) or {}
        dm = read_json(os.path.join(run_dir, 'distribution_metrics.json')) or {}
        per = (dm or {}).get('per_residue', {})
        rows.append({
            'run_dir': name,
            'temperature': params.get('temperature'),
            'top_p': params.get('top_p'),
            'sample_topk': params.get('sample_topk'),
            'prior_w': params.get('context_prior_weight'),
            'restrict_hist': params.get('restrict_to_history_support'),
            'k_hist': params.get('history_support_k'),
            'min_dwell': params.get('min_dwell'),
            'js_hist': hs.get('js'),
            'l1_hist': hs.get('l1'),
            'gt_only': hs.get('gt_only_count'),
            'pred_only': hs.get('pred_only_count'),
            'js_vis': dm.get('visitation_js'),
            'per_res_js_mean': per.get('per_residue_js_mean'),
            'cov_gt_mean': per.get('per_residue_coverage_gt_mean'),
            'cov_pr_mean': per.get('per_residue_coverage_pred_mean'),
        })

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f'Wrote {len(rows)} rows to {args.out_csv}')

if __name__ == '__main__':
    main()


