import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_series(path: Path) -> np.ndarray:
    vals = []
    with path.open('r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('//'):
                continue
            try:
                sec = float(s)
                vals.append(sec)  # s -> ms
            except ValueError:
                continue
    return np.asarray(vals, dtype=float)

def smooth(arr: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1 or arr.size == 0:
        return arr
    k = np.ones(win, dtype=float) / win
    return np.convolve(arr, k, mode='same')

def resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.exists():
        return pp
    # 尝试脚本同目录
    alt = Path(__file__).parent / p
    return alt if alt.exists() else pp

def plot_series(series_list, one_based: bool, out: Path | None, show: bool):
    fig, ax = plt.subplots(figsize=(10,4), layout='constrained')
    avgs = []
    all_vals = []
    for label, arr in series_list:
        x0 = 1 if one_based else 0
        x = np.arange(x0, x0 + arr.size)
        avg = float(np.mean(arr)) if arr.size else float('nan')
        avgs.append((label, avg))
        all_vals.append(arr)
        ax.plot(x, arr, label=f'{label} (avg={avg:.3f} ms)')
    if all_vals:
        concat = np.concatenate(all_vals)
        overall = float(np.mean(concat))
        ax.axhline(overall, color='k', ls='--', lw=0.8, alpha=0.4, label=f'overall avg={overall:.3f} ms')
        ax.set_title(' | '.join([f'{n}: {v:.3f} ms' for n, v in avgs]) + f' | overall: {overall:.3f} ms', fontsize=9)
    ax.set_xlabel('sample index' + (' (1-based)' if one_based else ''))
    ax.set_ylabel('Time (ms)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f'Saved: {out}')
    if show:
        plt.show()
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description='Plot LIO times (s->ms) with sample index as X')
    ap.add_argument('files', nargs='*', help='输入文件，默认: lio_time0.txt lio_time1.txt')
    ap.add_argument('-s','--smooth', type=int, default=0, help='滑动平均窗口(>1启用)')
    ap.add_argument('-o','--out', default='', help='输出图片路径')
    ap.add_argument('--one-based', action='store_true', help='横轴从 1 开始')
    ap.add_argument('--no-show', action='store_true', help='不显示图窗口')
    args = ap.parse_args()

    files = args.files if args.files else ['lio_time0.txt', 'lio_time1.txt']
    series_list = []
    for fp in files:
        p = resolve_path(fp)
        if not p.exists():
            raise SystemExit(f'文件不存在: {p}')
        arr = load_series(p)
        if args.smooth and args.smooth > 1:
            arr = smooth(arr, args.smooth)
        series_list.append((p.stem, arr))

    out = Path(args.out) if args.out else None
    plot_series(series_list, one_based=args.one_based, out=out, show=not args.no_show)

if __name__ == '__main__':
    main()