#!/usr/bin/env python3
import argparse, csv, os
import matplotlib.pyplot as plt

def read_rows(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description='Plot sweep results CSV')
    ap.add_argument('--csv', required=True, help='results.csv path')
    ap.add_argument('--out', required=True, help='output directory for plots')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = read_rows(args.csv)
    if not rows:
        print('No rows found')
        return

    # Extract series
    T = [to_float(r['temperature']) for r in rows]
    P = [to_float(r['top_p']) for r in rows]
    K = [to_float(r['sample_topk']) for r in rows]
    W = [to_float(r['prior_w']) for r in rows]
    JS = [to_float(r['js_vis']) or to_float(r['js_hist']) for r in rows]
    L1 = [to_float(r['l1_hist']) for r in rows]
    Cg = [to_float(r['cov_gt_mean']) for r in rows]
    Cp = [to_float(r['cov_pr_mean']) for r in rows]
    labels = [r['run_dir'] for r in rows]

    # Scatter: JS vs top_p (marker color by prior W, size by sample_topk)
    plt.figure(figsize=(8,6))
    sizes = [(k if k is not None else 0) for k in K]
    sizes = [max(20, min(120, s/2)) for s in sizes]
    sc = plt.scatter(P, JS, c=W, s=sizes, cmap='viridis', edgecolors='k', alpha=0.85)
    plt.colorbar(sc, label='prior_w')
    plt.xlabel('top_p')
    plt.ylabel('JS (visitation)')
    plt.title('Sweep: JS vs top_p (size=topk, color=prior_w)')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'sweep_js_vs_top_p.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Scatter: JS vs temperature (color by prior_w)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(T, JS, c=W, s=sizes, cmap='viridis', edgecolors='k', alpha=0.85)
    plt.colorbar(sc, label='prior_w')
    plt.xlabel('temperature')
    plt.ylabel('JS (visitation)')
    plt.title('Sweep: JS vs temperature (size=topk, color=prior_w)')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'sweep_js_vs_temperature.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Scatter: Coverage (pred) vs Coverage (GT)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(Cg, Cp, c=JS, s=sizes, cmap='plasma', edgecolors='k', alpha=0.85)
    plt.colorbar(sc, label='JS (lower is better)')
    plt.xlabel('Coverage GT (mean)')
    plt.ylabel('Coverage Pred (mean)')
    plt.title('Per-residue coverage vs coverage (color = JS)')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'sweep_coverage_scatter.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Print top-5 by JS to console
    ranked = sorted(zip(JS, labels), key=lambda x: (x[0] if x[0] is not None else 1e9))
    print('Top 5 by JS:')
    for j, lab in ranked[:5]:
        print(f'  {lab}: JS={j}')

if __name__ == '__main__':
    main()


