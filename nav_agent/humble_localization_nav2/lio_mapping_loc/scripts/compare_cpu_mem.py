import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ['total_cpu_percent']
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f'{path} 缺少列: {miss}')
    return df

def apply_smooth(df: pd.DataFrame, win: int) -> pd.DataFrame:
    if win and win > 1:
        df = df.copy()
        for c in ['total_cpu_percent']:
            df[c] = df[c].rolling(win, min_periods=1).mean()
    return df

# ...existing code...
def plot_many(datasets, smooth: int, one_based: bool, out: Path|None, show: bool):
    fig, (ax_cpu) = plt.subplots(1, 1, figsize=(11,6), sharex=True, layout='constrained')

    for i, (label, df) in enumerate(datasets):
        df = apply_smooth(df, smooth)
        n = len(df)
        if n == 0:
            continue
        x = np.arange(n) + (1 if one_based else 0)
        cpu = df['total_cpu_percent'].to_numpy()

        # 计算均值
        mean_cpu = float(np.nanmean(cpu))

        # 原曲线，图例里直接带上均值
        line, = ax_cpu.plot(x, cpu, label=f'CPU {label} (avg {mean_cpu:.1f}%)')

        # 画均值虚线（不进图例）
        ax_cpu.hlines(mean_cpu, x[0], x[-1],
                      colors=line.get_color(), linestyles='--', linewidth=1.0)

    ax_cpu.set_ylabel('CPU (%)')
    ax_cpu.grid(alpha=0.3)
    ax_cpu.axhline(100, color='tab:red', ls='--', lw=0.6)

    lines = ax_cpu.get_lines()
    ax_cpu.legend(lines, [l.get_label() for l in lines], loc='upper left')

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f'Saved: {out}')
    if show:
        plt.show()
    plt.close(fig)
# ...existing code...
# ...existing code...

def main():
    ap = argparse.ArgumentParser(description='Plot cpu_percent/mem_percent/mem_gb from mem logs (x=sample index)')
    ap.add_argument('csv', nargs='+', help='一个或多个日志文件，如 mem0.csv mem1.csv')
    ap.add_argument('-s','--smooth', type=int, default=0, help='滑动平均窗口(>1启用)')
    ap.add_argument('-o','--out', default='', help='输出图片路径')
    ap.add_argument('--one-based', action='store_true', help='横轴从 1 开始')
    ap.add_argument('--no-show', action='store_true', help='不显示图窗口')
    args = ap.parse_args()

    datasets = []
    for p in args.csv:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f'文件不存在: {path}')
        df = load_df(path)
        datasets.append((path.stem, df))

    out = Path(args.out) if args.out else None
    plot_many(datasets, smooth=args.smooth, one_based=args.one_based, out=out, show=not args.no_show)

if __name__ == '__main__':
    main()