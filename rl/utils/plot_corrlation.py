import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

# x
# epe = np.array([0.9136, 1.0243, 0.9558, 1.0540, 1.1190, 0.9187, 0.9529, 1.0117, 1.0551, 1.1276])
# cos = np.array([0.2234, 0.2051, 0.1765, 0.1451, 0.1376, 0.2195, 0.2199, 0.2101, 0.2015, 0.1835])
epe = np.array([0.9136, 1.0243, 0.9187, 0.9529, 1.0117, 1.0551, 1.1276])
cos = np.array([0.2234, 0.2051, 0.2195, 0.2199, 0.2101, 0.2015, 0.1835])

# y: 你填真实 Δsuccess（7个点）
# gain = np.array([0.138, 0.083, 0.069, 0.050, 0.041, 0.113, 0.083, 0.110, 0.0246, -0.012])
gain = np.array([0.138, 0.083, 0.113, 0.083, 0.110, 0.0246, -0.012])

labels = [
    "P-30k",
    "C-30k",
    # "I-300k",
    # "CT-30k",
    # "SVD-30k",
    "P-26k",
    "P-18k",
    "P-10k",
    "P-6k",
    "P-2k",
]

# ============================================================
# Label 位置配置 - 在这里统一调整
# 格式: (x_offset, y_offset, ha, va)
#   x_offset: 正数向右，负数向左
#   y_offset: 正数向上，负数向下
#   ha: 'left', 'center', 'right' (水平对齐)
#   va: 'top', 'center', 'bottom' (垂直对齐)
# ============================================================

# 图1 (Flow Cosine vs Gain) 的 label 位置
OFFSETS_COS = {
    "P-30k": (4, -2, 'left', 'top'),
    "C-30k": (0, 5, 'left', 'bottom'),
    # "I-300k": (4, 1, 'left', 'bottom'),
    # "CT-30k": (4, 1, 'left', 'bottom'),
    # "SVD-30k": (4, 4, 'right', 'bottom'),
    "P-26k": (0, 2, 'left', 'bottom'),
    "P-18k": (4, -4, 'left', 'top'),
    "P-10k": (0, 4, 'right', 'bottom'),
    "P-6k": (4, 0, 'left', 'bottom'),
    "P-2k": (0, 3, 'right', 'bottom'),
}

# 图2 (1/EPE vs Gain) 的 label 位置
OFFSETS_EPE = {
    "P-30k": (-25, -0, 'left', 'top'),
    "C-30k": (-25, 0, 'left', 'bottom'),
    # "I-300k": (-20, -12, 'left', 'bottom'),
    # "CT-30k": (-25, 3, 'left', 'bottom'),
    # "SVD-30k": (-25, 4, 'left', 'bottom'),
    "P-26k": (4, -10, 'left', 'bottom'),
    "P-18k": (4, 0, 'left', 'top'),
    "P-10k": (-4, 4, 'right', 'bottom'),
    "P-6k": (2, 2, 'left', 'bottom'),
    "P-2k": (6, -2, 'left', 'bottom'),
}

# ============================================================


def plot_combined_scatter(save_path="correlation_plots.pdf"):
    """绘制左右排列的两张散点图，适合双栏论文"""

    # 设置字体，更清晰
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.linewidth'] = 0.8

    # 双栏论文：稍微增加高度，让图更舒展
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0), sharey=True)

    # 字体设置 - 放大以保证清晰
    label_fontsize = 10
    tick_fontsize = 9
    annot_fontsize = 7
    eq_fontsize = 9

    inv_epe = 1.0 / epe

    # ========== 图1: Flow Cosine vs Gain ==========
    ax1.scatter(cos, gain, s=45, c='#2E75B6', alpha=0.9, edgecolors='#1a4a70', linewidth=0.6, zorder=3)

    # 线性拟合 y = ax + b
    z1 = np.polyfit(cos, gain, 1)
    a1, b1 = z1[0], z1[1]
    p1 = np.poly1d(z1)
    x_fit1 = np.linspace(cos.min() - 0.01, cos.max() + 0.01, 100)
    ax1.plot(x_fit1, p1(x_fit1), '--', color='#C65D3D', alpha=0.9, linewidth=1.2, zorder=2)

    # 在右下角显示方程
    eq1 = f"y = {a1:.2f}x + {b1:.2f}" if b1 >= 0 else f"y = {a1:.2f}x - {abs(b1):.2f}"
    ax1.text(0.95, 0.08, eq1, transform=ax1.transAxes, fontsize=eq_fontsize, color='black',
             fontweight='medium', ha='right', va='bottom')

    for i, label in enumerate(labels):
        ox, oy, ha, va = OFFSETS_COS[label]
        ax1.annotate(
            label,
            (cos[i], gain[i]),
            xytext=(ox, oy),
            textcoords='offset points',
            fontsize=annot_fontsize,
            ha=ha,
            va=va
        )

    ax1.set_xlabel(r"$\overline{\cos}$", fontsize=label_fontsize)
    ax1.set_ylabel(r"Downstream task gain $\in[0,1]$", fontsize=label_fontsize)
    ax1.tick_params(axis='both', labelsize=tick_fontsize)
    ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.3)
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=label_fontsize,
             fontweight='bold', va='top', ha='left')

    # ========== 图2: 1/EPE vs Gain ==========
    ax2.scatter(inv_epe, gain, s=45, c='#2E75B6', alpha=0.9, edgecolors='#1a4a70', linewidth=0.6, zorder=3)

    # 线性拟合 y = ax + b
    z2 = np.polyfit(inv_epe, gain, 1)
    a2, b2 = z2[0], z2[1]
    p2 = np.poly1d(z2)
    x_fit2 = np.linspace(inv_epe.min() - 0.02, inv_epe.max() + 0.02, 100)
    ax2.plot(x_fit2, p2(x_fit2), '--', color='#C65D3D', alpha=0.9, linewidth=1.2, zorder=2)

    # 在右下角显示方程
    eq2 = f"y = {a2:.2f}x + {b2:.2f}" if b2 >= 0 else f"y = {a2:.2f}x - {abs(b2):.2f}"
    ax2.text(0.95, 0.08, eq2, transform=ax2.transAxes, fontsize=eq_fontsize, color='black',
             fontweight='medium', ha='right', va='bottom')

    for i, label in enumerate(labels):
        ox, oy, ha, va = OFFSETS_EPE[label]
        ax2.annotate(
            label,
            (inv_epe[i], gain[i]),
            xytext=(ox, oy),
            textcoords='offset points',
            fontsize=annot_fontsize,
            ha=ha,
            va=va
        )

    ax2.set_xlabel(r"$1/\overline{\mathrm{EPE}}$", fontsize=label_fontsize)
    # 共享纵轴，不重复写 ylabel
    ax2.tick_params(axis='both', labelsize=tick_fontsize)
    ax2.grid(True, alpha=0.15, linestyle='-', linewidth=0.3)
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=label_fontsize,
             fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)  # 更小的子图间距

    # 保存为 PDF（矢量图，适合论文）和 PNG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")

    # 同时保存 PNG 版本
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {png_path}")

    plt.show()


if __name__ == "__main__":
    plot_combined_scatter(save_path="correlation_plots.pdf")

    # 打印相关性统计
    inv_epe = 1.0 / epe
    print("\n=== 相关性统计 ===")
    rho_cos, p_cos = spearmanr(cos, gain)
    tau_cos, p_tau_cos = kendalltau(cos, gain)
    print(f"Flow Cosine vs Gain:")
    print(f"  Spearman rho = {rho_cos:.4f}, p-value = {p_cos:.4f}")
    print(f"  Kendall tau  = {tau_cos:.4f}, p-value = {p_tau_cos:.4f}")

    rho_inv_epe, p_inv_epe = spearmanr(inv_epe, gain)
    tau_inv_epe, p_tau_inv_epe = kendalltau(inv_epe, gain)
    print(f"\n1/EPE vs Gain:")
    print(f"  Spearman rho = {rho_inv_epe:.4f}, p-value = {p_inv_epe:.4f}")
    print(f"  Kendall tau  = {tau_inv_epe:.4f}, p-value = {p_tau_inv_epe:.4f}")
