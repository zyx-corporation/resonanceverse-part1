"""
Phase II-A 実データ JSON（v7_phase2a_empirical_run 出力）からグラフを出力する。

- 通常: PNG（画面・スライド向け）。macOS では Hiragino 等で和文ラベル可。
- **論文向け**: ``--paper`` で英語ラベル・単欄相当サイズ・Okabe–Ito 色・パネル (A)(B)(C)・**PNG 300 dpi + PDF**。

論文用キャプション案: ``docs/planning/v7_phase2a_paper_figures.md``

例::

    python experiments/v7_phase2a_tau_plots.py path/to/*_with_contrib.json
    python experiments/v7_phase2a_tau_plots.py path/to/*_with_contrib.json --paper
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# macOS に同梱の日本語対応フォント（優先順）
_MACOS_JP_FONT_CANDIDATES = (
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Hiragino Maru Gothic Pro",
    ".Hiragino Kaku Gothic Interface",
    "Arial Unicode MS",
    "PingFang HK",
)

# Okabe–Ito（色覚バリアフリー、白黒印刷では線種で区別済み）
_PAPER_COLORS = {
    "r_mean": "#0072B2",
    "r_var": "#E69F00",
    "r_var_smooth": "#D55E00",
    "n_fill": "#009E73",
    "n_line": "#006D5B",
    "aux_raw": "#0072B2",
    "aux_smooth": "#CC79A7",
}

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.v7_phase2a_tau_summary import rolling_mean  # noqa: E402


def apply_matplotlib_japanese_fonts(
    *,
    prefer_family: str | None = None,
) -> tuple[bool, str]:
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    _log_fm = logging.getLogger("matplotlib.font_manager")
    _prev_level = _log_fm.level
    _log_fm.setLevel(logging.ERROR)
    try:
        candidates: list[str] = []
        if prefer_family and prefer_family.strip():
            candidates.append(prefer_family.strip())
        if sys.platform == "darwin":
            candidates.extend(_MACOS_JP_FONT_CANDIDATES)
        else:
            candidates.extend(
                (
                    "Noto Sans CJK JP",
                    "Noto Sans JP",
                    "Yu Gothic",
                    "Meiryo",
                    "MS Gothic",
                )
            )

        verified: list[str] = []
        for name in candidates:
            path = fm.findfont(fm.FontProperties(family=name))
            if path and "dejavu" not in path.lower():
                verified.append(name)

        if not verified and sys.platform == "darwin":
            for f in fm.fontManager.ttflist:
                if "Hiragino" in f.name and f.fname.lower().endswith((".ttc", ".otf", ".ttf")):
                    verified.append(f.name)
                    break

        plt.rcParams["axes.unicode_minus"] = False
        if verified:
            primary = verified[0]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [primary, "DejaVu Sans"]
            return True, f"primary_font={primary!r}"

        return False, "no_cjk_font_fallback_english"
    finally:
        _log_fm.setLevel(_prev_level)


def apply_paper_rcparams() -> None:
    """単欄図想定のフォント・線幅。PDF は編集しやすいよう TrueType アウトライン。"""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        }
    )


def _panel_tag(ax: Any, tag: str) -> None:
    ax.text(
        -0.14,
        1.03,
        tag,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _load_by_tau(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("by_tau") or []
    if not rows:
        raise ValueError("by_tau が空")
    tau = np.array([int(r["tau"]) for r in rows], dtype=np.int64)
    r_mean = np.array([float(r["R_mean"]) for r in rows], dtype=np.float64)
    r_var = np.array([float(r["R_var"]) for r in rows], dtype=np.float64)
    n = np.array([int(r["n"]) for r in rows], dtype=np.int64)
    return tau, r_mean, r_var, n, data


def _save_figure(fig: Any, path_without_suffix: Path, *, formats: list[str], dpi: int) -> list[str]:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for fmt in formats:
        p = path_without_suffix.with_suffix(f".{fmt}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "format": fmt}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(p, **kwargs)
        out.append(str(p))
    return out


def plot_primary(
    tau: np.ndarray,
    r_mean: np.ndarray,
    r_var: np.ndarray,
    n: np.ndarray,
    *,
    smooth_window: int,
    path_without_suffix: Path,
    dpi: int,
    formats: list[str],
    title_suffix: str = "",
    japanese_ui: bool = True,
    paper: bool = False,
    show_suptitle: bool = False,
    meta: dict[str, Any] | None = None,
) -> list[str]:
    import matplotlib.pyplot as plt

    meta = meta or {}
    if japanese_ui:
        st = f"Phase II-A τ掃引（主解析）{title_suffix}"
        ylab_n = "n（有効対話数）"
        xlab = "τ"
        leg_smooth = f"移動平均（窓 {smooth_window}）"
    else:
        st = f"Phase II-A tau sweep (primary){title_suffix}"
        ylab_n = "Count"
        xlab = r"Lag $\tau$" if paper else "τ"
        leg_smooth = f"Moving avg. (window {smooth_window})"

    r_var_s = rolling_mean(r_var, max(1, smooth_window))
    if paper:
        c = _PAPER_COLORS
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(3.6, 5.9),
            sharex=True,
            layout="constrained",
            gridspec_kw={"height_ratios": [1, 1.15, 0.85], "hspace": 0.12},
        )
    else:
        c = {
            "r_mean": "C0",
            "r_var": "C1",
            "r_var_smooth": "C3",
            "n_fill": "C2",
            "n_line": "C2",
        }
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(10, 8),
            sharex=True,
            layout="constrained",
            gridspec_kw={"height_ratios": [1, 1.2, 0.9]},
        )

    if show_suptitle or not paper:
        fig.suptitle(st, fontsize=11 if paper else 12, y=1.01 if paper else 0.98)

    axes[0].plot(tau, r_mean, color=c["r_mean"], linewidth=1.1 if paper else 1.0)
    axes[0].set_ylabel(r"$\overline{R}$" if paper else "R_mean")
    if paper:
        _panel_tag(axes[0], "(A)")

    axes[1].plot(tau, r_var, color=c["r_var"], linewidth=1.0, alpha=0.95, label="Var" if paper else "R_var")
    axes[1].plot(
        tau,
        r_var_s,
        color=c["r_var_smooth"],
        linewidth=1.15 if paper else 1.1,
        linestyle="--",
        label=leg_smooth,
    )
    axes[1].set_ylabel("Var" if paper else "R_var")
    axes[1].legend(loc="upper right", framealpha=0.92)
    if paper:
        _panel_tag(axes[1], "(B)")

    axes[2].fill_between(tau, 0, n, step="mid", color=c["n_fill"], alpha=0.35)
    axes[2].plot(tau, n, color=c["n_line"], linewidth=0.9 if paper else 0.8, drawstyle="steps-mid")
    axes[2].set_ylabel(r"$n(\tau)$" if paper else ylab_n)
    axes[2].set_xlabel(xlab)
    axes[2].set_ylim(bottom=0)
    if paper:
        _panel_tag(axes[2], "(C)")

    if not paper:
        for ax in axes:
            ax.grid(True, alpha=0.3)

    if paper and meta.get("model"):
        fig.text(
            0.99,
            0.01,
            f"{meta.get('model', '')}, layer {meta.get('layer_index', '')}, D={meta.get('n_dialogues', '')}",
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.35",
        )

    paths = _save_figure(fig, path_without_suffix, formats=formats, dpi=dpi)
    plt.close(fig)
    return paths


def plot_auxiliary_r_var(
    aux: dict[str, list[dict[str, Any]]],
    *,
    smooth_window: int,
    path_without_suffix: Path,
    dpi: int,
    formats: list[str],
    title_suffix: str = "",
    japanese_ui: bool = True,
    paper: bool = False,
    show_suptitle: bool = False,
) -> list[str]:
    import matplotlib.pyplot as plt

    axes_names = sorted(aux.keys())
    if not axes_names:
        return []
    ncols = 3
    nrows = (len(axes_names) + ncols - 1) // ncols
    mosaic: list[list[str]] = []
    for r in range(nrows):
        row: list[str] = []
        for c in range(ncols):
            i = r * ncols + c
            row.append(axes_names[i] if i < len(axes_names) else ".")
        mosaic.append(row)

    if paper:
        fig, ax_dict = plt.subplot_mosaic(
            mosaic,
            figsize=(7.0, 2.35 * nrows),
            layout="constrained",
        )
    else:
        fig, ax_dict = plt.subplot_mosaic(
            mosaic,
            figsize=(11, 3.2 * nrows),
            layout="constrained",
        )

    if japanese_ui:
        st = f"Phase II-A 補助（6 軸 R_var）{title_suffix}"
        leg_smooth = "移動平均"
        xlab = "τ"
    else:
        st = f"Phase II-A auxiliary R_var (six axes){title_suffix}"
        leg_smooth = "Moving avg."
        xlab = r"Lag $\tau$" if paper else "τ"

    if show_suptitle or not paper:
        fig.suptitle(st, fontsize=11 if paper else 12)

    letters = "abcdef"
    for i, axis in enumerate(axes_names):
        ax = ax_dict[axis]
        rows = aux[axis]
        t = np.array([int(x["tau"]) for x in rows], dtype=np.int64)
        v = np.array([float(x["R_var"]) for x in rows], dtype=np.float64)
        vs = rolling_mean(v, max(1, smooth_window))
        cr, cs = _PAPER_COLORS["aux_raw"], _PAPER_COLORS["aux_smooth"]
        if not paper:
            cr, cs = "C0", "C1"
        ax.plot(t, v, linewidth=0.85, alpha=0.88, color=cr, label="Var" if paper else "R_var")
        ax.plot(t, vs, linewidth=1.05, linestyle="--", color=cs, label=leg_smooth)
        ax.set_title(axis.capitalize())
        ax.set_xlabel(xlab)
        ax.set_ylabel("Var" if paper else "R_var")
        if paper:
            ax.tick_params(axis="both", labelsize=7)
            _panel_tag(ax, f"({letters[i]})")
        else:
            ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.92)

    if paper:
        for axis in axes_names:
            ax_dict[axis].grid(True, alpha=0.35, linewidth=0.4)

    paths = _save_figure(fig, path_without_suffix, formats=formats, dpi=dpi)
    plt.close(fig)
    return paths


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")

    p = argparse.ArgumentParser(description="Phase II-A τ 掃引グラフ（PNG / PDF）")
    p.add_argument("json_path", type=Path, help="*_with_contrib.json 等（by_tau 必須）")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="出力先ディレクトリ（省略時は入力と同じディレクトリ）",
    )
    p.add_argument("--smooth", type=int, default=5, help="R_var 移動平均窓")
    p.add_argument("--dpi", type=int, default=None, help="PNG の dpi（--paper 時既定 300、それ以外既定 120）")
    p.add_argument(
        "--formats",
        type=str,
        default=None,
        help="カンマ区切り: png, pdf（--paper 時既定 png,pdf）",
    )
    p.add_argument(
        "--paper",
        action="store_true",
        help="論文向け: 英語・単欄サイズ・Okabe–Ito 色・パネル表記・高解像 PNG + PDF",
    )
    p.add_argument(
        "--show-suptitle",
        action="store_true",
        help="--paper 時も図全体のタイトルを表示（既定はオフでキャプション任せ）",
    )
    p.add_argument(
        "--font-family",
        type=str,
        default=None,
        help="日本語フォント（--paper では未使用）",
    )
    p.add_argument(
        "--english-labels",
        action="store_true",
        help="和文フォントが使えても英語ラベルのみ",
    )
    args = p.parse_args()

    paper = bool(args.paper)
    if args.dpi is None:
        dpi = 300 if paper else 120
    else:
        dpi = max(72, args.dpi)

    if args.formats:
        formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    else:
        formats = ["png", "pdf"] if paper else ["png"]

    path = args.json_path.resolve()
    tau, r_mean, r_var, n, data = _load_by_tau(path)
    out_dir = (args.out_dir or path.parent).resolve()
    stem = path.name.replace(".json", "")

    if paper:
        apply_paper_rcparams()
        font_note = "paper_style_dejavu"
        japanese_ui = False
    else:
        jp_ok, font_note = apply_matplotlib_japanese_fonts(prefer_family=args.font_family)
        japanese_ui = jp_ok and not args.english_labels

    title_suffix = ""
    if data.get("n_dialogues") is not None:
        title_suffix = f"  対話数 D={data['n_dialogues']}" if japanese_ui else f"  D={data['n_dialogues']}"

    meta = {
        "model": data.get("model"),
        "layer_index": data.get("layer_index"),
        "n_dialogues": data.get("n_dialogues"),
    }

    primary_stem = out_dir / (f"{stem}_tau_paper_primary" if paper else f"{stem}_tau_primary")
    primary_paths = plot_primary(
        tau,
        r_mean,
        r_var,
        n,
        smooth_window=max(1, args.smooth),
        path_without_suffix=primary_stem,
        dpi=dpi,
        formats=formats,
        title_suffix=title_suffix,
        japanese_ui=japanese_ui,
        paper=paper,
        show_suptitle=args.show_suptitle,
        meta=meta,
    )

    aux = data.get("auxiliary_label_delay_coherence")
    aux_paths: list[str] = []
    aux_stem = out_dir / (f"{stem}_tau_paper_auxiliary_rvar" if paper else f"{stem}_tau_auxiliary_rvar")
    if isinstance(aux, dict) and aux:
        aux_paths = plot_auxiliary_r_var(
            aux,
            smooth_window=max(1, args.smooth),
            path_without_suffix=aux_stem,
            dpi=dpi,
            formats=formats,
            title_suffix=title_suffix,
            japanese_ui=japanese_ui,
            paper=paper,
            show_suptitle=args.show_suptitle,
        )

    print(
        "v7_phase2a_tau_plots_ok",
        json.dumps(
            {
                "paper": paper,
                "primary": primary_paths,
                "auxiliary": aux_paths if aux_paths else None,
                "japanese_ui": japanese_ui,
                "font": font_note,
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
