import json
import numpy as np
import matplotlib.pyplot as plt
import baselines

from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle, ConnectionPatch

bl = baselines.BaselineLogitLinearRegression.deserialize_json(
    "baseline_FlyingThings3D_Kitti15.json"
)

xx = np.array(bl.xx)
yy = np.array(bl.yy)

x_new = np.array([
    83.78538324142646,
    83.80292531276528,
    62.64760749180055,
    70.08918835427605,
    71.55789338281988,
    83.3430290897716,
    83.08609335473334,
    83.66870185322846,
    78.79486634538367,
    74.56379346659926,
    76.23556826595156,
    78.3744889920057,
]) / 100.0

y_new = np.array([
    69.21915598869323,
    69.18329704761506,
    44.47949620723725,
    53.1065297961235,
    53.91896358221769,
    68.79582102775574,
    69.11261121273041,
    69.28237431287765,
    63.389468731880186,
    60.701201629638675,
    59.944940395355225,
    63.1404313921928,
]) / 100.0

new_labels = [
    "Original",
    "Remove Motion Aggregation",
    "One Scale 16",
    "One Scale 8",
    "One Scale 8 Conv",
    "Disable Shared Parameter",
    "Remove Consolidation",
    "Align Residual Unit to Original ResNet",
    "RAFT",
    "One Scale 8 It",
    "One Scale 8 Conv It",
    "One Scale 8 Conv It Plus",
]

new_colors = [
    "red",
    "mediumpurple",
    "pink",
    "orange",
    "yellow",
    "limegreen",
    "blue",
    "turquoise",
    "gray",
    "gray",
    "gray",
    "black",
]

new_markers = [
    "*",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
    "<",
    "o",
    "D",
    "v",
    "v",
]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

x_min = min(xx.min(), x_new.min())
x_max = max(xx.max(), x_new.max())

x_curve = np.linspace(
    x_min * 0.95,
    x_max * 1.02,
    1000
)

y_curve = bl.predict(x_curve)


def draw_common(ax, decimals=0):
    ax.plot(
        x_curve,
        y_curve,
        color="cornflowerblue",
        lw=2,
        label="Robustness Baseline",
        zorder=2,
    )
    x_diag = np.linspace(0.001, 0.999, 500)
    ax.plot(
    x_diag,
    x_diag,
    color="lightseagreen",
    lw=2,
    linestyle="dashdot",
    label="Identity y = x",
    zorder=1,
    )
    ax.scatter(
        xx,
        yy,
        color="lightsteelblue",
        s=180,
        edgecolor="cornflowerblue",
        linewidths=1.0,
        zorder=3,
        alpha=0.6,
        label="Standard Models",
    )

    for x, y, label, color, marker in zip(
        x_new,
        y_new,
        new_labels,
        new_colors,
        new_markers,
    ):
        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=200,
            edgecolor="black",
            linewidths=0.5,
            zorder=4,
            label=label,
            alpha=0.8,
        )
    ax.set_xscale("logit")
    ax.set_yscale("logit")
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x * 100:.{decimals}f}")
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y * 100:.{decimals}f}")
    )

    ax.grid(
        linestyle="--",
        alpha=0.3,
    )

zoom_xmin = 0.80
zoom_xmax = 0.85
zoom_ymin = 0.665
zoom_ymax = 0.70
fig = plt.figure(figsize=(15, 7),layout="constrained")

# Overview takes more space
gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[1.0, 1.5],
    wspace=0.35,
)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

#overview
draw_common(ax1)

ax1.set_xlabel(
    "WAUC [%] on FlyingThings3D (ID)",
    fontsize=14,
)

ax1.set_ylabel(
    "WAUC [%] on KITTI (OOD)",
    fontsize=14,
)

overview_xmin = x_min * 0.95
overview_xmax = x_max * 1.02

overview_ymin = y_curve.min() * 0.95
overview_ymax = y_curve.max() * 1.02

ax1.set_xlim(
    overview_xmin,
    overview_xmax,
)

ax1.set_ylim(
    overview_ymin,
    overview_ymax,
)
#zoom-in
draw_common(ax2,decimals=1)

ax2.set_xlabel(
    "WAUC [%] on FlyingThings3D (ID)",
    fontsize=14,
)

ax2.set_ylabel(
    "WAUC [%] on KITTI (OOD)",
    fontsize=14,
)

ax2.set_xlim(
    zoom_xmin,
    zoom_xmax,
)

ax2.set_ylim(
    zoom_ymin,
    zoom_ymax,
)

zoom_rect = Rectangle(
    (
        zoom_xmin,
        zoom_ymin,
    ),
    zoom_xmax - zoom_xmin,
    zoom_ymax - zoom_ymin,
    transform=ax1.transData,
    fill=False,
    edgecolor="gray",
    linestyle="--",
    linewidth=1.5,
    zorder=10,
)
ax1.add_patch(zoom_rect)
fig.canvas.draw()
p1 = ax1.transData.transform(
    (zoom_xmin, zoom_ymin)
)

p2 = ax1.transData.transform(
    (zoom_xmax, zoom_ymax)
)

rectangle_width = abs(
    p2[0] - p1[0]
)

rectangle_height = abs(
    p2[1] - p1[1]
)

zoom_box_aspect = (
    rectangle_height / rectangle_width
)

ax2.set_box_aspect(
    zoom_box_aspect
)

con1 = ConnectionPatch(
    xyA=(zoom_xmax, zoom_ymin),
    coordsA=ax1.transData,
    xyB=(0, 0),
    coordsB=ax2.transAxes,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    #alpha=0.8,
    clip_on=False,
)
con1.set_linestyle((0, (6, 6))) 
con2 = ConnectionPatch(
    xyA=(zoom_xmax, zoom_ymax),
    coordsA=ax1.transData,
    xyB=(0, 1),
    coordsB=ax2.transAxes,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    #alpha=0.8,
    clip_on=False,
)
con2.set_linestyle((0, (6, 6))) 
fig.add_artist(con1)
fig.add_artist(con2)
#fig.tight_layout()
fig.savefig(
    "regression_things_kitti_with_zoom.png",
    dpi=300,
    bbox_inches="tight",
)

#legend
handles1, labels1 = ax1.get_legend_handles_labels()
fig3, ax3 = plt.subplots(figsize=(16, 2))
ax3.axis("off")
ax3.legend(
    handles1,
    labels1,
    ncol=3,
    loc="center",
    frameon=False,
    columnspacing=1.5,
    handletextpad=0.5,
    labelspacing=1.0,
    fontsize=15,
)

fig3.savefig(
    "legend_regression.png",
    bbox_inches="tight",
    dpi=300,
)

plt.show()
'''
import json
import numpy as np
import matplotlib.pyplot as plt
import baselines
#from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
from scipy.special import logit

bl = baselines.BaselineLogitLinearRegression.deserialize_json(
    "baseline_FlyingThings3D_Kitti15.json"
)

xx = np.array(bl.xx)
yy = np.array(bl.yy)

x_new = np.array([
    83.78538324142646,
    83.80292531276528,
    62.64760749180055,
    70.08918835427605,
    71.55789338281988,
    83.3430290897716,
    83.08609335473334,
    83.66870185322846,
    78.79486634538367,
    74.56379346659926,
    76.23556826595156,
]) / 100.0

y_new = np.array([
    69.21915598869323,
    69.18329704761506,
    44.47949620723725,
    53.1065297961235,
    53.91896358221769,
    68.79582102775574,
    69.11261121273041,
    69.28237431287765,
    63.389468731880186,
    60.701201629638675,
    59.944940395355225,
]) / 100.0

x_min = min(xx.min(), x_new.min())
x_max = max(xx.max(), x_new.max())
x_curve = np.linspace(x_min * 0.95, x_max * 1.02, 300)
y_curve = bl.predict(x_curve)

new_labels  = [
        "Original",
        "Remove Motion Aggregation",
        "One Scale 16",
        "One Scale 8",
        "One Scale 8 Conv",
        "Disable Shared Parameter",
        "Remove Consolidation",
        "Align Residual Unit to Original ResNet",
        "RAFT",
        "One Scale 8 It",
        "One Scale 8 Conv It",
    ]
new_colors  = ["red", "mediumpurple", "pink", "orange", "yellow", "limegreen","blue","turquoise","gray","gray","gray"]
new_markers = ["*", "s", "^", "D", "v", "P", "X", "<","o","D","<"]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def draw_common(ax):
    ax.plot(x_curve, y_curve, color="cornflowerblue", lw=2, label="Robustness Baseline")
    ax.scatter(xx, yy, color="lightsteelblue", s=180, edgecolor="cornflowerblue", linewidths=1.0, zorder=3, alpha=0.6,label="Standard Models")
    for x, y, label, color, marker in zip(x_new, y_new, new_labels, new_colors, new_markers):
        ax.scatter(x, y, color=color, marker=marker, s=200, edgecolor="black",linewidths=0.5, zorder=4, label=label,alpha=0.8)
    ax.set_xscale("logit")
    ax.set_yscale("logit")
    #ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    #ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 100:.0f}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * 100:.0f}'))
    ax.grid(linestyle="--",alpha=0.3)

# overview
fig1, ax1 = plt.subplots(figsize=(7, 6))
draw_common(ax1)
ax1.set_xlabel("WAUC [%] on FlyingThings3D (ID)",fontsize=14)
ax1.set_ylabel("WAUC [%] on KITTI (OOD)",fontsize=14)
#ax1.legend(loc="upper left", fontsize=11,labelspacing=0.7,framealpha=0.3)
fig1.tight_layout()
fig1.savefig("regression_things_kitti.png", dpi=300)

# zoom in
fig2, ax2 = plt.subplots(figsize=(7, 6))
draw_common(ax2)
ax2.set_xlabel("WAUC [%] on FlyingThings3D (ID)",fontsize=14)
ax2.set_ylabel("WAUC [%] on KITTI (OOD)",fontsize=14)
#ax2.legend(loc="upper left", fontsize=11,labelspacing=0.7,framealpha=0.3)

ax2.set_xlim(0.80, 0.85)
ax2.set_ylim(0.665, 0.70)

fig2.tight_layout()
fig2.savefig("regression_things_kitti_zoom.png", dpi=200)

handles, labels = ax1.get_legend_handles_labels()
dict = dict(zip(labels,handles))
b_handle = dict.pop("Robustness Baseline")
s_handle = dict.pop("Standard Models")
labels1 = list(dict.keys())
handles1 = list(dict.values())
labels1.insert(4, "Robustness Baseline")
handles1.insert(4, b_handle)
labels1.append("Standard Models")
handles1.append(s_handle)

fig3, ax3 = plt.subplots(figsize=(10,2.5))
ax3.axis('off')
ax3.legend(handles1,labels1,ncol=3,loc='center',frameon=False,columnspacing=1.5,handletextpad=0.5,labelspacing=1.0,fontsize=15)
fig3.savefig("legend_regression.png",bbox_inches='tight',dpi=300)


print("saved both")
'''
