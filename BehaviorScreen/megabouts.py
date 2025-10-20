import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cycler import cycler

from megabouts.tracking_data import TrackingConfig, FullTrackingData
from megabouts.config import TrajSegmentationConfig
from megabouts.pipeline import FullTrackingPipeline
from megabouts.utils import (
    bouts_category_name,
    bouts_category_name_short,
    bouts_category_color,
    cmp_bouts,
)
from megabouts.segmentation import Segmentation


df_recording = pd.read_csv('00_07dpf_WT_Fri_10_Oct_2025_10h04min42sec_fish_0.predictions.000_00_07dpf_WT_Fri_10_Oct_2025_10h04min42sec_fish_0.analysis.csv')
fps = 100
mm_per_unit = 1/42

thresh_score = 0.75
for kps in [
    "Head", 
    "Eye_Left_Front", "Eye_Left_Back", 
    "Eye_Right_Front", "Eye_Right_Back",
    "Tail_0", "Tail_1","Tail_2","Tail_3","Tail_4","Tail_5","Tail_6","Tail_7","Tail_8","Tail_9" 
    ]:
    df_recording.loc[df_recording["instance.score"] < thresh_score, kps + ".x"] = np.nan
    df_recording.loc[df_recording["instance.score"] < thresh_score, kps + ".y"] = np.nan
    df_recording.loc[df_recording[kps + ".score"] < thresh_score, kps + ".x"] = np.nan
    df_recording.loc[df_recording[kps + ".score"] < thresh_score, kps + ".y"] = np.nan

head_x = df_recording["Head.x"].values
head_y = df_recording["Head.y"].values

tail_x = df_recording[["Tail_0.x", "Tail_1.x", "Tail_2.x", "Tail_3.x", "Tail_4.x", "Tail_5.x", "Tail_6.x", "Tail_7.x", "Tail_8.x" , "Tail_9.x"]].values
tail_y = df_recording[["Tail_0.y", "Tail_1.y", "Tail_2.y", "Tail_3.y", "Tail_4.y", "Tail_5.y", "Tail_6.y", "Tail_7.y", "Tail_8.y" , "Tail_9.y"]].values

head_x = head_x * mm_per_unit
head_y = head_y * mm_per_unit
tail_x = tail_x * mm_per_unit
tail_y = tail_y * mm_per_unit

tracking_cfg = TrackingConfig(fps=fps, tracking="full_tracking")
tracking_data = FullTrackingData.from_keypoints(
    head_x=head_x, head_y=head_y, tail_x=tail_x, tail_y=tail_y
)
pipeline = FullTrackingPipeline(tracking_cfg, exclude_CS=True)
pipeline.segmentation_cfg = TrajSegmentationConfig(fps=tracking_cfg.fps)
ethogram, bouts, segments, tail, traj = pipeline.run(tracking_data)

# Number of bouts as function of threshold:
thresh_list = np.linspace(0.1, 5, 100)
num_bouts = np.zeros_like(thresh_list)

for i, thresh in enumerate(thresh_list):
    traj_segmentation_cfg = TrajSegmentationConfig(
        fps=tracking_cfg.fps, peak_prominence=thresh
    )
    segmentation_function = Segmentation.from_config(traj_segmentation_cfg)
    segments = segmentation_function.segment(traj.vigor)
    num_bouts[i] = len(segments.onset)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thresh_list, num_bouts)
plt.show()

# plot classification
id_b = np.unique(bouts.df.label.category[bouts.df.label.proba > 0.5]).astype("int")

fig, ax = plt.subplots(facecolor="white", figsize=(25, 4))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
G = gridspec.GridSpec(1, len(id_b))
ax0 = {}
for i, b in enumerate(id_b):
    ax0 = plt.subplot(G[i])
    ax0.set_title(bouts_category_name_short[b], fontsize=15)
    for i_sg, sg in enumerate([1, -1]):
        id = bouts.df[
            (bouts.df.label.category == b)
            & (bouts.df.label.sign == sg)
            & (bouts.df.label.proba > 0.5)
        ].index
        if len(id) > 0:
            ax0.plot(sg * bouts.tail[id, 7, :].T, color="k", alpha=0.3)
        ax0.set_xlim(0, pipeline.segmentation_cfg.bout_duration)
        ax0.set_ylim(-4, 4)
        ax0.set_xticks([])
        ax0.set_yticks([])
        for sp in ["top", "bottom", "left", "right"]:
            ax0.spines[sp].set_color(bouts_category_color[b])
            ax0.spines[sp].set_linewidth(5)

plt.show()

IdSt = 0
T = 100
Duration = T * tracking_cfg.fps
IdEd = IdSt + Duration
t = np.arange(Duration) / tracking_cfg.fps


fig = plt.figure(facecolor="white", figsize=(15, 5), constrained_layout=True)
G = gridspec.GridSpec(2, 1, height_ratios=[1, 0.2], hspace=0.5, figure=fig)
ax = plt.subplot(G[0, 0])
blue_cycler = cycler(color=plt.cm.Blues(np.linspace(0.2, 0.9, 10)))
ax.set_prop_cycle(blue_cycler)

ax.plot(t, ethogram.df["tail_angle"].values[IdSt:IdEd, :7], lw=1)
ax.set_ylim(-4, 4)
ax.set_xlim(0, T)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.get_yaxis().tick_left()
ax.get_xaxis().set_ticks([])
ax.set_ylabel("tail angle (rad)", rotation=0, labelpad=100)

ax = plt.subplot(G[1, 0])
ax.imshow(
    ethogram.df[("bout", "cat")].values[IdSt:IdEd].T,
    cmap=cmp_bouts,
    aspect="auto",
    vmin=0,
    vmax=12,
    interpolation="nearest",
    extent=(0, T, 0, 1),
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.get_yaxis().tick_left()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_xlim(0, T)
ax.set_ylim(0, 1.1)

id_b = np.unique(ethogram.df[("bout", "id")].values[IdSt:IdEd]).astype("int")
id_b = id_b[id_b > -1]
for i in id_b:
    on_ = bouts.df.iloc[i][("location", "onset")]
    b = bouts.df.iloc[i][("label", "category")]
    ax.text((on_ - IdSt) / tracking_cfg.fps, 1.1, bouts_category_name[int(b)])

ax.set_ylabel("bout category", rotation=0, labelpad=100)
plt.show()