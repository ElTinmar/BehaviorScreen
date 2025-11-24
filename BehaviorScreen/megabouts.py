import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cycler import cycler
import networkx as nx
import copy
from typing import NamedTuple, Dict, List

from megabouts.tracking_data import TrackingConfig, FullTrackingData, HeadTrackingData
from megabouts.config import TrajSegmentationConfig
from megabouts.pipeline import FullTrackingPipeline, HeadTrackingPipeline
from megabouts.utils import (
    bouts_category_name,
    bouts_category_name_short,
    bouts_category_color,
    cmp_bouts,
)
from megabouts.segmentation import Segmentation

from megabouts.pipeline.freely_swimming_pipeline import EthogramHeadTracking
from megabouts.classification.classification import TailBouts
from megabouts.segmentation.segmentation import SegmentationResult
from megabouts.preprocessing.traj_preprocessing import TrajPreprocessingResult

from .core import ROOT_FOLDER, GROUPING_PARAMETER, Stim
from .load import BehaviorData, BehaviorFiles, Directories
from .process import get_trials, get_well_coords_mm

# Force running on CPU if GPU is not compatible
CPU = True
if CPU:
    import torch
    torch.cuda.is_available = lambda: False

class MegaboutData(NamedTuple):
    timestamp: np.ndarray
    ethogram: EthogramHeadTracking
    bouts: TailBouts
    segments: SegmentationResult
    traj: TrajPreprocessingResult

def megabout_headtracking_pipeline(behavior_data: BehaviorData):

    mm_per_pix = 1/behavior_data.metadata['calibration']['pix_per_mm']
    fps = behavior_data.metadata['camera']['framerate_value']

    megabout_results = {}
    for identity, data in behavior_data.tracking.groupby('identity'):
        df = data.sort_values('timestamp').reset_index(drop=True)
        
        timestamps = df["timestamp"].values
        swimbladder_x = df["centroid_x"].values * mm_per_pix
        swimbladder_y = df["centroid_y"].values * mm_per_pix
        head_x = swimbladder_x + df['pc1_x'].values * mm_per_pix
        head_y = swimbladder_y + df['pc1_y'].values * mm_per_pix
        tracking_data = HeadTrackingData.from_keypoints(
            head_x=head_x,
            head_y=head_y,
            swimbladder_x=swimbladder_x,
            swimbladder_y=swimbladder_y,
        )

        tracking_cfg = TrackingConfig(fps=fps, tracking="head_tracking")
        pipeline = HeadTrackingPipeline(tracking_cfg, exclude_CS=True)
        pipeline.traj_segmentation_cfg = TrajSegmentationConfig(
            fps=tracking_cfg.fps, peak_prominence=0.15, peak_percentage=0.2
        )
        ethogram, bouts, segments, traj = pipeline.run(tracking_data)
        megabout_results[identity] = MegaboutData(
            timestamps, 
            ethogram, 
            bouts, 
            segments, 
            traj
        ) 

    return megabout_results

def get_bout_metrics(
        directories: Directories,
        behavior_data: BehaviorData, 
        behavior_files: BehaviorFiles,
        megabout: Dict[int, MegaboutData]
    ) -> List[Dict]:

    well_coords_mm = get_well_coords_mm(directories, behavior_files, behavior_data)
    fps = behavior_data.metadata['camera']['framerate_value']
    stim_trials = get_trials(behavior_data)
    
    rows = []
    for identity, meg_data in megabout.items():

        cx,cy,_ = well_coords_mm[identity,:]

        for stim_select, stim_data in stim_trials.groupby('stim_select'):
            
            stim = Stim(stim_select)
            if not stim in GROUPING_PARAMETER:
                break

            for condition, condition_data in stim_data.groupby(GROUPING_PARAMETER[stim]):

                for trial_idx, (trial, row) in enumerate(condition_data.iterrows()):

                    bout_start = meg_data.timestamp[meg_data.bouts.onset]
                    bout_stop = meg_data.timestamp[meg_data.bouts.offset]
                    mask = (bout_start > row.start_timestamp) & (bout_stop < row.stop_timestamp) 

                    off_previous = np.nan
                    for on, off, category, sign, proba in zip(
                        meg_data.bouts.onset[mask], 
                        meg_data.bouts.offset[mask],
                        meg_data.bouts.category[mask],
                        meg_data.bouts.sign[mask],
                        meg_data.bouts.proba[mask],
                    ):

                        # heading change
                        heading_change = meg_data.traj.yaw_smooth[off] - meg_data.traj.yaw_smooth[on]

                        # distance 
                        delta_x = meg_data.traj.x_smooth[on+1:off] - meg_data.traj.x_smooth[on:off-1]
                        delta_y = meg_data.traj.y_smooth[on+1:off] - meg_data.traj.y_smooth[on:off-1]
                        distance = np.sum(np.sqrt(delta_x**2 + delta_y**2))
                        radial_distance = np.sqrt((meg_data.traj.x_smooth[on]-cx)**2 + (meg_data.traj.y_smooth[on]-cy)**2)

                        # bout duration
                        bout_duration = (off-on)/fps

                        # interbout duration
                        interbout_duration = (on-off_previous)/fps
                        off_previous = off

                        # peak axial speed
                        axial_speed = meg_data.traj.axial_speed[on:off]
                        peak_axial_speed = axial_speed[np.argmax(np.abs(axial_speed))]

                        # peak yaw speed
                        yaw_speed = meg_data.traj.yaw_speed[on:off]
                        peak_yaw_speed = yaw_speed[np.argmax(np.abs(yaw_speed))]

                        rows.append({
                            'file': behavior_files.metadata.stem,
                            'identity': identity,
                            'stim': stim_select,
                            'stim_variable_name': GROUPING_PARAMETER[stim],
                            'stim_variable_value': str(condition),
                            'trial_num': trial_idx,
                            'heading_change': heading_change,
                            'distance': distance,
                            'distance_center': radial_distance,
                            'bout_duration': bout_duration,
                            'interbout_duration': interbout_duration,
                            'peak_axial_speed': peak_axial_speed,
                            'peak_yaw_speed': peak_yaw_speed,
                            'category': category,
                            'sign': sign,
                            'proba': proba
                        })
                        # TODO add on_edge boolean

    return rows


def transitions(bouts):
    num_categories = len(bouts_category_name)
    transition = np.zeros((num_categories, num_categories))
    for b0, b1 in zip(bouts.df.label.category[:-1], bouts.df.label.category[1:]):
        transition[int(b0), int(b1)] += 1

    row_sums = transition.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition /= row_sums
    return transition

def stationary_distribution(transition):
    eigvals, eigvecs = np.linalg.eig(transition.T)
    eigvec = eigvecs[:, np.isclose(eigvals, 1)]
    pi = eigvec[:,0].real
    pi = pi / pi.sum()
    return pi

def plot_transition_graph(P, labels=None, threshold=0.01, layout = nx.circular_layout, node_colors = bouts_category_color):

    n = P.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]

    G = nx.DiGraph()

    for i, label in enumerate(labels):
        G.add_node(i, label=label)

    for i in range(n):
        for j in range(n):
            if P[i, j] > threshold:
                G.add_edge(i, j, weight=P[i, j])

    pos = layout(G)

    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_color='w', node_size=1200, edgecolors=node_colors)
    nx.draw_networkx_labels(G, pos, labels={i: labels[i] for i in range(n)}, font_size=10, font_weight='bold')

    weights = [5 * G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', width=weights, alpha=0.7, arrowsize=10, node_size=1200)

    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.axis("off")
    plt.show()

def plot_transition_matrix(transition):
    num_categories = len(bouts_category_name)

    fig, ax = plt.subplots(facecolor="white", figsize=(6, 6))
    cax = ax.imshow(transition, cmap='Blues')

    ax.set_xticks(range(num_categories))
    ax.set_xticklabels(bouts_category_name_short, rotation=45, ha='right')
    ax.set_xlabel('Bout$_{n+1}$')

    ax.set_yticks(range(num_categories))
    ax.set_yticklabels(bouts_category_name_short)
    ax.set_ylabel('Bout$_{n}$')

    # Optional: annotate probabilities
    for i in range(num_categories):
        for j in range(num_categories):
            ax.text(j, i, f"{transition[i,j]:.2f}", ha='center', va='center', color='black', fontsize=8)

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label='Transition probability')
    plt.tight_layout()
    plt.show()

def plot_stationary_distribution(pi, labels=None, orientation="horizontal", cmap="viridis"):

    pi = np.array(pi, dtype=float)
    pi /= pi.sum()  # normalize to sum = 1
    n = len(pi)
    if labels is None:
        labels = [f"State {i}" for i in range(n)]

    # Colormap mapping
    cmap = plt.get_cmap(cmap)
    colors = cmap((pi - pi.min()) / (pi.max() - pi.min() + 1e-12))

    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), 1.5) if orientation == "horizontal"
                            else (2.0, max(4, n * 0.4)))

    if orientation == "horizontal":
        for i, (p, c) in enumerate(zip(pi, colors)):
            ax.bar(i, p, color=c, width=1.0, edgecolor='gray', align='center')
            ax.text(i, p + 0.01 * pi.max(), f"{p:.2f}", ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Probability")
        ax.set_title("Stationary Distribution (π)")
    else:
        for i, (p, c) in enumerate(zip(pi, colors)):
            ax.barh(i, p, color=c, height=1.0, edgecolor='gray', align='center')
            ax.text(p + 0.01 * pi.max(), i, f"{p:.2f}", va='center', ha='left', fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Probability")
        ax.set_title("Stationary Distribution (π)")
        ax.invert_yaxis()  # top-to-bottom order

    plt.tight_layout()
    plt.show()

    
def plot_categories(bouts, segments, traj):
    
    traj_array = segments.extract_traj_array(
        head_x=traj.df.x_smooth,
        head_y=traj.df.y_smooth,
        head_angle=traj.df.yaw_smooth,
        align_to_onset=True,
    )

    id_b = np.unique(bouts.df.label.category[bouts.df.label.proba > 0.7]).astype("int")

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
        id = bouts.df[(bouts.df.label.category == b) & (bouts.df.label.proba > 0.7)].index
        if len(id) > 0:
            for i in id:
                ax0.plot(traj_array[i, 0, :], traj_array[i, 1, :], color="k", alpha=0.3)
                ax0.arrow(
                    traj_array[i, 0, -1],
                    traj_array[i, 1, -1],
                    0.01 * np.cos(traj_array[i, 2, -1]),
                    0.01 * np.sin(traj_array[i, 2, -1]),
                    width=0.005,
                    head_width=0.2,
                    color="k",
                    alpha=0.3,
                )
        ax0.set_aspect(1)
        ax0.set(xlim=(-4, 4), ylim=(-4, 4))
        ax0.set_xticks([])
        ax0.set_yticks([])
        for sp in ["top", "bottom", "left", "right"]:
            ax0.spines[sp].set_color(bouts_category_color[b])
            ax0.spines[sp].set_linewidth(5)

    plt.show()

def plot_ethogram(ethogram, bouts, IdSt, T):
    
    Duration = T * tracking_cfg.fps
    IdEd = IdSt + Duration - 1
    t = np.arange(Duration) / tracking_cfg.fps

    data = ethogram.df.loc[IdSt:IdEd]
    x_data = data[("trajectory", "x")].values
    y_data = data[("trajectory", "y")].values
    angle_data = data[("trajectory", "angle")].values
    bout_cat_data = data[("bout", "cat")].values
    bout_id_data = data[("bout", "id")].values

    valid_data = ~np.isnan(angle_data)
    unwrapped = np.copy(angle_data)
    unwrapped[valid_data] = np.unwrap(angle_data[valid_data])
    angle_data = 180 / np.pi * unwrapped

    fig, (ax1, ax) = plt.subplots(
        2,
        1,
        figsize=(15, 5),
        gridspec_kw={"height_ratios": [1, 0.4], "hspace": 0.1},
        facecolor="white",
        constrained_layout=True,
    )
    ax2 = ax1.twinx()

    ax1.plot(t, x_data, lw=2, color="k", label="x")
    ax1.plot(t, y_data, lw=2, color="tab:gray", label="y")
    ax1.set_ylabel("(mm)")
    ax2.plot(t, angle_data, lw=2, color="tab:blue", label="angle")
    ax2.set_ylabel("(°)")

    for spine in ["top", "bottom"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax1.set_xlim(0, T)
    ax1.get_xaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])
    ax1.set_xlim(0, T)
    # Add both legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")


    ax.imshow(
        bout_cat_data.reshape(1, -1),
        cmap=cmp_bouts,
        aspect="auto",
        vmin=0,
        vmax=12,
        interpolation="nearest",
        extent=(0, T, 0, 1),
    )
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.1)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    for i in np.unique(bout_id_data[bout_id_data > -1]).astype("int"):
        on_, b = (
            bouts.df.iloc[i][("location", "onset")],
            bouts.df.iloc[i][("label", "category")],
        )
        ax.text(
            (on_ - IdSt) / tracking_cfg.fps, 1.1, bouts_category_name[int(b)], rotation=45
        )

    ax.set_ylabel("bout category", rotation=0, labelpad=100)
    plt.show()


# # Head tracking from sleap
# thresh_score = 0.75
# for kps in ["Head", "Swim_Bladder"]:
#     df_recording.loc[df_recording["instance.score"] < thresh_score, kps + ".x"] = np.nan
#     df_recording.loc[df_recording["instance.score"] < thresh_score, kps + ".y"] = np.nan
#     df_recording.loc[df_recording[kps + ".score"] < thresh_score, kps + ".x"] = np.nan
#     df_recording.loc[df_recording[kps + ".score"] < thresh_score, kps + ".y"] = np.nan

# head_x = df_recording["Head.x"].values * mm_per_unit
# head_y = df_recording["Head.y"].values * mm_per_unit
# swimbladder_x = df_recording["Swim_Bladder.x"].values * mm_per_unit
# swimbladder_y = df_recording["Swim_Bladder.y"].values * mm_per_unit

# tracking_data = HeadTrackingData.from_keypoints(
#     head_x=head_x,
#     head_y=head_y,
#     swimbladder_x=swimbladder_x,
#     swimbladder_y=swimbladder_y,
# )

# tracking_cfg = TrackingConfig(fps=fps, tracking="head_tracking")
# pipeline = HeadTrackingPipeline(tracking_cfg, exclude_CS=False)
# pipeline.traj_segmentation_cfg = TrajSegmentationConfig(
#     fps=tracking_cfg.fps, peak_prominence=0.15, peak_percentage=0.2
# )
# ethogram, bouts, segments, traj = pipeline.run(tracking_data)

# # Full tracking

# thresh_score = 0.75
# for kps in [
#     "Head", 
#     "Eye_Left_Front", "Eye_Left_Back", 
#     "Eye_Right_Front", "Eye_Right_Back",
#     "Tail_0", "Tail_1","Tail_2","Tail_3","Tail_4","Tail_5","Tail_6","Tail_7","Tail_8","Tail_9" 
#     ]:
#     df_recording.loc[df_recording["instance.score"] < thresh_score, kps + ".x"] = np.nan
#     df_recording.loc[df_recording["instance.score"] < thresh_score, kps + ".y"] = np.nan
#     df_recording.loc[df_recording[kps + ".score"] < thresh_score, kps + ".x"] = np.nan
#     df_recording.loc[df_recording[kps + ".score"] < thresh_score, kps + ".y"] = np.nan

# head_x = df_recording["Head.x"].values
# head_y = df_recording["Head.y"].values

# tail_x = df_recording[["Tail_0.x", "Tail_1.x", "Tail_2.x", "Tail_3.x", "Tail_4.x", "Tail_5.x", "Tail_6.x", "Tail_7.x", "Tail_8.x" , "Tail_9.x"]].values
# tail_y = df_recording[["Tail_0.y", "Tail_1.y", "Tail_2.y", "Tail_3.y", "Tail_4.y", "Tail_5.y", "Tail_6.y", "Tail_7.y", "Tail_8.y" , "Tail_9.y"]].values

# head_x = head_x * mm_per_unit
# head_y = head_y * mm_per_unit
# tail_x = tail_x * mm_per_unit
# tail_y = tail_y * mm_per_unit


# tracking_cfg = TrackingConfig(fps=fps, tracking="full_tracking")
# tracking_data = FullTrackingData.from_keypoints(
#     head_x=head_x, head_y=head_y, tail_x=tail_x, tail_y=tail_y
# )
# pipeline = FullTrackingPipeline(tracking_cfg, exclude_CS=True)
# pipeline.segmentation_cfg = TrajSegmentationConfig(fps=tracking_cfg.fps)
# ethogram, bouts, segments, tail, traj = pipeline.run(tracking_data)

# # Number of bouts as function of threshold:
# thresh_list = np.linspace(0.1, 5, 100)
# num_bouts = np.zeros_like(thresh_list)

# for i, thresh in enumerate(thresh_list):
#     traj_segmentation_cfg = TrajSegmentationConfig(
#         fps=tracking_cfg.fps, peak_prominence=thresh
#     )
#     segmentation_function = Segmentation.from_config(traj_segmentation_cfg)
#     segments = segmentation_function.segment(traj.vigor)
#     num_bouts[i] = len(segments.onset)

# fig, ax = plt.subplots(figsize=(6, 4))
# ax.plot(thresh_list, num_bouts)
# plt.show()

# # plot classification
# id_b = np.unique(bouts.df.label.category[bouts.df.label.proba > 0.5]).astype("int")

# fig, ax = plt.subplots(facecolor="white", figsize=(25, 4))

# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
# ax.spines["left"].set_visible(False)
# ax.set_xticks([])
# ax.set_yticks([])
# G = gridspec.GridSpec(1, len(id_b))
# ax0 = {}
# for i, b in enumerate(id_b):
#     ax0 = plt.subplot(G[i])
#     ax0.set_title(bouts_category_name_short[b], fontsize=15)
#     for i_sg, sg in enumerate([1, -1]):
#         id = bouts.df[
#             (bouts.df.label.category == b)
#             & (bouts.df.label.sign == sg)
#             & (bouts.df.label.proba > 0.5)
#         ].index
#         if len(id) > 0:
#             ax0.plot(sg * bouts.tail[id, 7, :].T, color="k", alpha=0.3)
#         ax0.set_xlim(0, pipeline.segmentation_cfg.bout_duration)
#         ax0.set_ylim(-4, 4)
#         ax0.set_xticks([])
#         ax0.set_yticks([])
#         for sp in ["top", "bottom", "left", "right"]:
#             ax0.spines[sp].set_color(bouts_category_color[b])
#             ax0.spines[sp].set_linewidth(5)

# plt.show()

# IdSt = 0
# T = 100
# Duration = T * tracking_cfg.fps
# IdEd = IdSt + Duration
# t = np.arange(Duration) / tracking_cfg.fps


# fig = plt.figure(facecolor="white", figsize=(15, 5), constrained_layout=True)
# G = gridspec.GridSpec(2, 1, height_ratios=[1, 0.2], hspace=0.5, figure=fig)
# ax = plt.subplot(G[0, 0])
# blue_cycler = cycler(color=plt.cm.Blues(np.linspace(0.2, 0.9, 10)))
# ax.set_prop_cycle(blue_cycler)

# ax.plot(t, ethogram.df["tail_angle"].values[IdSt:IdEd, :7], lw=1)
# ax.set_ylim(-4, 4)
# ax.set_xlim(0, T)

# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
# ax.get_yaxis().tick_left()
# ax.get_xaxis().set_ticks([])
# ax.set_ylabel("tail angle (rad)", rotation=0, labelpad=100)

# ax = plt.subplot(G[1, 0])
# ax.imshow(
#     ethogram.df[("bout", "cat")].values[IdSt:IdEd].T,
#     cmap=cmp_bouts,
#     aspect="auto",
#     vmin=0,
#     vmax=12,
#     interpolation="nearest",
#     extent=(0, T, 0, 1),
# )
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
# ax.get_yaxis().tick_left()
# ax.get_xaxis().set_ticks([])
# ax.get_yaxis().set_ticks([])
# ax.set_xlim(0, T)
# ax.set_ylim(0, 1.1)

# id_b = np.unique(ethogram.df[("bout", "id")].values[IdSt:IdEd]).astype("int")
# id_b = id_b[id_b > -1]
# for i in id_b:
#     on_ = bouts.df.iloc[i][("location", "onset")]
#     b = bouts.df.iloc[i][("label", "category")]
#     ax.text((on_ - IdSt) / tracking_cfg.fps, 1.1, bouts_category_name[int(b)])

# ax.set_ylabel("bout category", rotation=0, labelpad=100)
# plt.show()