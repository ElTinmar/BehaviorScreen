# adapted from  doi: 10.1016/j.cub.2024.08.008
 
import numpy as np
from scipy import signal, interpolate
from scipy.ndimage import convolve1d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import umap


# ----------------------------------------------------------------------
# 0. GENERIC UTILITIES
# ----------------------------------------------------------------------

def find_blocks(binary_vec):
    """
    Port of findblocks.m (no 'condition' argument): given a 0/1 vector,
    return (start_indices, block_lengths) for each contiguous run of 1s.
    Indices are 0-based.
    """
    x = np.asarray(binary_vec).astype(int).ravel()
    if np.any(np.abs(x) > 1):
        raise ValueError("find_blocks only accepts vectors of 0/1")
    xpad = np.concatenate([[0], x, [0]])
    d = np.diff(xpad)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    lengths = ends - starts
    # sort descending by size, as MATLAB default behaviour does
    order = np.argsort(-lengths)
    return starts[order], lengths[order]


def step_kernel(width_samples):
    """
    Step function kernel used throughout: first half = -1, second half = +1,
    normalized by width (matches `cn` in speciallowess4.m and
    `convfilter` in CDSacDetect.m).
    """
    w = int(width_samples)
    n_neg = w // 2
    n_pos = w - n_neg
    k = np.concatenate([-np.ones(n_neg), np.ones(n_pos)])
    return k / w


def gaussmf(x, sigma, center):
    """MATLAB gaussmf(x,[sigma, center])."""
    x = np.asarray(x, dtype=float)
    return np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def lowpass_filter(x, cutoff_hz, fs, order=2, pad=50):
    """Zero-phase Butterworth low-pass filter with edge-value padding."""
    x = np.asarray(x, dtype=float)
    xpad = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    b, a = signal.butter(order, cutoff_hz / (fs / 2.0), btype='low')
    y = signal.filtfilt(b, a, xpad)
    return y[pad:pad + len(x)]


def interp_to_rate(t, x, fs, t_end=None, pad_seconds=0.5):
    """
    Resample an irregularly/regularly sampled trace x(t) onto a fixed
    fs-Hz timebase from 0 to max(t) (+ padding with last value), matching
    the `interp1(...)` + end-padding logic in CDSacDetect.m.
    """
    if t_end is None:
        t_end = t[-1]
    n_pad = int(round(pad_seconds * fs))
    t_ext = np.concatenate([t, t[-1] + np.arange(1, n_pad + 1) / fs])
    x_ext = np.concatenate([x, np.full(n_pad, x[-1])])
    new_t = np.linspace(0, t_end, int(round(t_end * fs)))
    f = interpolate.interp1d(t_ext, x_ext, bounds_error=False,
                              fill_value=(x_ext[0], x_ext[-1]))
    return new_t, f(new_t)


# ----------------------------------------------------------------------
# 1. STAGE 1: COARSE DETECTION AT 100 Hz
#    (interp -> 1 Hz low-pass -> 160 ms step convolution -> peak detection)
# ----------------------------------------------------------------------

def coarse_detect_events(t, L, R, fs=100.0, lp_cutoff=1.0,
                          step_width_ms=160, min_prominence=0.8):
    """
    Reproduces the first (100 Hz) stage of CDSacDetect.m for one trial's
    left/right eye position traces.

    Returns dict with:
        timebase, L100, R100 (raw @100Hz), Lpass, Rpass (lowpassed),
        Lfil, Rfil (convolved with step kernel),
        L_events, R_events : dict(loc=index array, t=time array, sign=+/-1)
    """
    timebase, L100 = interp_to_rate(t, L, fs)
    _, R100 = interp_to_rate(t, R, fs)

    Lpass = lowpass_filter(L100, lp_cutoff, fs)
    Rpass = lowpass_filter(R100, lp_cutoff, fs)

    step_w = int(round(step_width_ms / 1000.0 * fs))  # e.g. 160ms *100Hz=16 samples
    kernel = step_kernel(step_w)

    def conv_pad(y, k, pad=50):
        ypad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
        c = np.convolve(ypad, k, mode='same')
        return c[pad:pad + len(y)]

    Lfil = conv_pad(Lpass, kernel)
    Rfil = conv_pad(Rpass, kernel)

    def peaks(fil):
        pk, props = signal.find_peaks(np.abs(fil), prominence=min_prominence)
        sign = np.sign(fil[pk])
        return pk, timebase[pk], sign

    Lloc, Lt, Lsign = peaks(Lfil)
    Rloc, Rt, Rsign = peaks(Rfil)

    return dict(timebase=timebase, L100=L100, R100=R100,
                Lpass=Lpass, Rpass=Rpass, Lfil=Lfil, Rfil=Rfil,
                L_events=dict(loc=Lloc, t=Lt, sign=Lsign),
                R_events=dict(loc=Rloc, t=Rt, sign=Rsign))


# ----------------------------------------------------------------------
# 2. STAGE 2: BINOCULAR PAIRING + 300 ms EXCLUSION WINDOW
# ----------------------------------------------------------------------

def pair_binocular_events(L_t, R_t, pair_window_s=0.1):
    """
    Pair left/right coarse-detection events that fall within `pair_window_s`
    of one another into single binocular events. Unpaired events are kept
    as monocular events (their "time" is simply their own detection time).

    Returns an array of binocular event times (mean of paired L/R times,
    or the single-eye time if unpaired), sorted ascending, plus arrays
    marking which eye(s) contributed.
    """
    L_t = np.sort(np.asarray(L_t, dtype=float))
    R_t = np.sort(np.asarray(R_t, dtype=float))

    used_L = np.zeros(len(L_t), dtype=bool)
    used_R = np.zeros(len(R_t), dtype=bool)
    events = []  # (time, has_L, has_R)

    # Greedy nearest-neighbour pairing
    for i, lt in enumerate(L_t):
        if used_L[i]:
            continue
        if len(R_t) > 0:
            diffs = np.abs(R_t - lt)
            diffs[used_R] = np.inf
            j = np.argmin(diffs) if len(diffs) else None
            if j is not None and diffs[j] <= pair_window_s:
                events.append((0.5 * (lt + R_t[j]), True, True))
                used_L[i] = True
                used_R[j] = True
                continue
        events.append((lt, True, False))
        used_L[i] = True

    for j, rt in enumerate(R_t):
        if not used_R[j]:
            events.append((rt, False, True))

    events.sort(key=lambda e: e[0])
    times = np.array([e[0] for e in events])
    has_L = np.array([e[1] for e in events])
    has_R = np.array([e[2] for e in events])
    return times, has_L, has_R


def discard_overlapping_events(event_times, refractory_s=0.3):
    """
    After pairing, discard any event that occurs within `refractory_s`
    of a *preceding retained* event (so retained events are always
    >= refractory_s apart), matching the paper's 300 ms rule.
    """
    event_times = np.sort(np.asarray(event_times, dtype=float))
    keep = []
    last_t = -np.inf
    for t in event_times:
        if t - last_t >= refractory_s:
            keep.append(t)
            last_t = t
    return np.array(keep)


# ----------------------------------------------------------------------
# 3. STAGE 3: CUSTOM LOWESS SMOOTHING AT 500 Hz (speciallowess4 port)
# ----------------------------------------------------------------------


def fast_regular_lowess(y, span):
    """
    Fast approximation to non-robust local-linear LOWESS for uniformly
    spaced data.

    It matches the interior centered LOWESS estimate. Boundary behavior
    may differ from MATLAB, so recording edges should be treated carefully.
    """
    y = np.asarray(y, dtype=float)

    window = min(max(int(round(span)), 1), len(y))

    if window % 2 == 0:
        window -= 1

    if window <= 1:
        return y.copy()

    half_window = window // 2
    offsets = np.arange(
        -half_window,
        half_window + 1,
        dtype=float,
    )

    distance = np.abs(offsets) / (half_window + 1)
    weights = (1.0 - distance**3) ** 3
    weights /= weights.sum()

    return convolve1d(
        y,
        weights,
        mode="nearest",
    )

def speciallowess4(data, wide_window, narrow_window, delta_thresh,
                    anneal_window, conv_window=None, sigma=None):
    """
    Parameters
    ----------
    data : 1D array
    wide_window : float  -> params(1)
    narrow_window : float (0 disables narrow smoothing) -> params(2)
    delta_thresh : float -> params(3)
    anneal_window : float (search/taper distance, samples) -> params(4)
    conv_window : float, step-kernel width for detecting big steps
                  (defaults to wide_window) -> params(5)
    sigma : float, gaussian taper sigma (defaults to anneal_window/4)
                  -> params(6)
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    if conv_window is None:
        conv_window = wide_window
    if sigma is None:
        sigma = anneal_window / 4.0

    pad = 50
    data2 = np.concatenate([np.full(pad, data[0]), data, np.full(pad, data[-1])])

    # wide-window smoothing (padded, then trimmed)
    y_wide = fast_regular_lowess(data2, wide_window)[pad:pad + n]
    y = y_wide.copy()

    # narrow-window smoothing (unpadded, per original)
    y2 = data.copy() if narrow_window == 0 else fast_regular_lowess(data, narrow_window)

    # detect large step-like changes
    cn = step_kernel(conv_window) * conv_window  # original cn is unnormalized +-1
    # NOTE: in the .m file `cn` is literally +1/-1 (not divided by width) -
    # reproduce that exactly:
    half_pos = int(np.floor(conv_window / 2))
    half_neg = int(np.ceil(conv_window / 2))
    cn = np.concatenate([np.ones(half_pos), -np.ones(half_neg)])

    dy = np.convolve(y2, cn, mode='same')
    dyind = (np.abs(dy) > delta_thresh).astype(int)
    onset, blocksize = find_blocks(dyind)
    # restore ascending order (find_blocks sorts by size)
    order = np.argsort(onset)
    onset, blocksize = onset[order], blocksize[order]
    offset = onset + blocksize - 1

    if len(onset) == 0:
        return y

    srchdistance = int(round(anneal_window))

    # (faithful to the .m file: thisoff is *reset* to thison, so thismp==thison)
    for on in onset:
        thison = on
        thismp = thison
        thison2 = max(0, thison - srchdistance)
        thisoff2 = min(n - 1, thison + srchdistance)

        idx_range = np.arange(thison2, thisoff2 + 1)
        srchweights = gaussmf(idx_range, sigma, thismp)

        n_on = thison - thison2 + 1
        n_off = thisoff2 - thison  # weightdexOff = 2:(thisoff2-thisoff+1) in matlab (1-based)

        w_on = srchweights[:n_on]
        seg1 = slice(thison2, thismp + 1)
        y[seg1] = y2[seg1] * w_on + y[seg1] * (1 - w_on)

        if n_off >= 1:
            w_off = srchweights[1:1 + n_off][::-1]
            seg2 = slice(thismp + 1, thisoff2 + 1)
            y[seg2] = y2[seg2] * w_off + y[seg2] * (1 - w_off)

    return y


def smooth_trace_for_metrics(x, mode='tethered'):
    """
    Apply the paper's exact span choices to a raw eye-position trace
    already resampled to 500 Hz.

    mode: 'tethered'  -> wide=33ms, narrow=0 (no smoothing at steps)
          'freeswim'  -> wide=133ms, narrow=80ms
    """
    fs = 500.0
    if mode == 'tethered':
        wide_ms, narrow_ms = 33, 0
    elif mode == 'freeswim':
        wide_ms, narrow_ms = 133, 80
    else:
        raise ValueError("mode must be 'tethered' or 'freeswim'")

    wide_samp = wide_ms / 1000.0 * fs
    narrow_samp = 0 if narrow_ms == 0 else narrow_ms / 1000.0 * fs

    # delta_thresh / anneal_window are not given numerically in the text;
    # these are reasonable defaults you should tune against your own data.
    return speciallowess4(x, wide_window=wide_samp, narrow_window=narrow_samp,
                           delta_thresh=0.5, anneal_window=50,
                           conv_window=wide_samp, sigma=None)


# ----------------------------------------------------------------------
# 4. STAGE 4: REFINED ONSET TIME (double step-convolution product)
# ----------------------------------------------------------------------

def refine_onset_time(smoothed_pos, fs, coarse_t_idx, window_ms=400,
                       wide_ms=100, narrow_ms=40, thresh_frac=0.5):
    """
    Refines a coarse onset-time estimate (index into `smoothed_pos`,
    sampled at `fs` Hz) using the product of two step-function
    convolutions (100 ms & 40 ms), thresholded within a `window_ms`
    window centred on the coarse estimate.

    `thresh_frac` (fraction of the local max of the product trace) is
    NOT given a specific numeric value in the text -> tune as needed.

    Returns refined sample index (int) within the full trace.
    """
    n = len(smoothed_pos)
    half_win = int(round(window_ms / 1000.0 * fs / 2))
    lo = max(0, coarse_t_idx - half_win)
    hi = min(n, coarse_t_idx + half_win)

    seg = smoothed_pos[lo:hi]

    k_wide = step_kernel(int(round(wide_ms / 1000.0 * fs)))
    k_narrow = step_kernel(int(round(narrow_ms / 1000.0 * fs)))

    c_wide = np.convolve(seg, k_wide, mode='same')
    c_narrow = np.convolve(seg, k_narrow, mode='same')
    product = c_wide * c_narrow

    absprod = np.abs(product)
    if absprod.max() == 0:
        return coarse_t_idx
    thresh = thresh_frac * absprod.max()
    crossing_idx = np.where(absprod >= thresh)[0]
    if len(crossing_idx) == 0:
        return coarse_t_idx

    # first crossing nearest the window centre, matching "peak provides
    # coarse estimate -> refine within window" logic
    centre = (hi - lo) // 2
    best = crossing_idx[np.argmin(np.abs(crossing_idx - centre))]
    return lo + best


# ----------------------------------------------------------------------
# 5. STAGE 5: EVENT METRICS (a-d) AND 9 OCULOMOTOR METRICS
# ----------------------------------------------------------------------

def event_position_velocity_metrics(pos, fs, onset_idx,
                                     pre_win_ms=200, post_win_ms=200,
                                     vel_win_ms=150):
    """
    Computes (a)-(d) for a single eye trace around one onset index.

    pos : smoothed eye-position trace (500 Hz recommended)
    fs  : sampling rate of `pos`
    onset_idx : refined onset sample index

    Returns dict: pre_pos, max_post_pos, max_post_idx, median_post_pos,
                  vel_cw, vel_ccw
    """
    n = len(pos)
    pre_n = int(round(pre_win_ms / 1000.0 * fs))
    post_n = int(round(post_win_ms / 1000.0 * fs))
    vel_half = int(round(vel_win_ms / 1000.0 * fs / 2))

    # (a) pre-saccadic median position
    pre_lo = max(0, onset_idx - pre_n)
    pre_pos = np.nanmedian(pos[pre_lo:onset_idx]) if onset_idx > pre_lo else np.nan

    # (b) max post-saccadic deviation from eye position AT ONSET
    post_hi = min(n, onset_idx + post_n)
    post_seg = pos[onset_idx:post_hi]
    pos_at_onset = pos[onset_idx]
    if len(post_seg) == 0:
        max_post_pos, max_post_local_idx = np.nan, 0
    else:
        dev = np.abs(post_seg - pos_at_onset)
        max_post_local_idx = int(np.nanargmax(dev))
        max_post_pos = post_seg[max_post_local_idx]
    max_post_idx = onset_idx + max_post_local_idx

    # (c) median position over 200ms window starting at max_post_idx
    med_hi = min(n, max_post_idx + post_n)
    median_post_pos = np.nanmedian(pos[max_post_idx:med_hi]) if med_hi > max_post_idx else np.nan

    # (d) velocity: gradient over 150ms window centred at onset
    v_lo = max(0, onset_idx - vel_half)
    v_hi = min(n, onset_idx + vel_half + 1)
    vel = np.gradient(pos[v_lo:v_hi], 1.0 / fs)
    vel_cw = np.nanmax(vel) if len(vel) else np.nan   # cw = max (per paper's sign convention)
    vel_ccw = np.nanmin(vel) if len(vel) else np.nan  # ccw = min

    return dict(pre_pos=pre_pos, max_post_pos=max_post_pos,
                max_post_idx=max_post_idx, median_post_pos=median_post_pos,
                vel_cw=vel_cw, vel_ccw=vel_ccw)


def compute_9_metrics(L_metrics, R_metrics):
    """
    Combine per-eye (a)-(d) metrics (from event_position_velocity_metrics)
    into the paper's 9 oculomotor metrics for one binocular event.
    """
    amp_L = L_metrics['median_post_pos'] - L_metrics['pre_pos']
    amp_R = R_metrics['median_post_pos'] - R_metrics['pre_pos']

    maxmed_L = L_metrics['max_post_pos'] - L_metrics['median_post_pos']
    maxmed_R = R_metrics['max_post_pos'] - R_metrics['median_post_pos']

    vel_cw_L, vel_ccw_L = L_metrics['vel_cw'], L_metrics['vel_ccw']
    vel_cw_R, vel_ccw_R = R_metrics['vel_cw'], R_metrics['vel_ccw']

    vergence = L_metrics['median_post_pos'] - R_metrics['median_post_pos']

    return np.array([amp_L, amp_R, maxmed_L, maxmed_R,
                      vel_cw_L, vel_ccw_L, vel_cw_R, vel_ccw_R,
                      vergence])


METRIC_NAMES = ['Amp_L', 'Amp_R', 'MaxMedAmp_L', 'MaxMedAmp_R',
                'Vel_cw_L', 'Vel_ccw_L', 'Vel_cw_R', 'Vel_ccw_R', 'Vergence']


# ----------------------------------------------------------------------
# 6. STAGE 6: WINSORIZE + Z-SCORE, UMAP EMBEDDING, DBSCAN CLUSTERING
# ----------------------------------------------------------------------

def winsorize(x, lower_pct=0.5, upper_pct=99.5):
    lo, hi = np.nanpercentile(x, [lower_pct, upper_pct], axis=0)
    return np.clip(x, lo, hi)


def winsorize_zscore_per_fish(features, fish_ids, lower_pct=0.5, upper_pct=99.5):
    """
    features : (N, D) array of the 9 metrics
    fish_ids : (N,) array identifying which animal each row belongs to

    Winsorizes and z-scores each animal's data independently, as in the
    paper ("data from each animal was first winsorized ... and z-scored").
    """
    out = np.empty_like(features, dtype=float)
    for fid in np.unique(fish_ids):
        mask = fish_ids == fid
        sub = features[mask]
        sub_w = winsorize(sub, lower_pct, upper_pct)
        mu = np.nanmean(sub_w, axis=0)
        sd = np.nanstd(sub_w, axis=0)
        sd[sd == 0] = 1.0
        out[mask] = (sub_w - mu) / sd
    return out


def fit_umap_embedding(features_z, min_dist=0.11, n_neighbors=199,
                        n_components=2, metric='euclidean', random_state=0):
    """
    Fit the UMAP model on the "training" set (tethered, non-swim-bout
    events), matching run_umap(metric=Euclidean, min_dist=0.11,
    n_neighbours=199).
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         n_components=n_components, metric=metric,
                         random_state=random_state)
    embedding = reducer.fit_transform(features_z)
    return reducer, embedding


def run_dbscan(embedding, eps=0.34, min_samples=570):
    """
    DBSCAN clustering with the paper's exact eps/minpts.
    NOTE: sklearn's `min_samples` counts the point itself as part of its
    own neighbourhood (core-point condition: neighbours >= min_samples),
    whereas the custom MATLAB dbscanCD.m uses a strict '>' on neighbour
    count *excluding* the point itself. This can shift the boundary by
    one point; for these very large min_samples values the difference is
    negligible in practice.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(embedding)
    return labels  # -1 == noise / unclustered


# ---- Border-point reassignment (unclustered points within 3 units) ----

def _density_along_line(all_points, p0, p1, n_bins=10, radius=0.15):
    """
    Sample `n_bins` locations linearly between p0 (event) and p1 (cluster
    centroid) and count how many points of `all_points` fall within
    `radius` of each sample location -> local density profile.
    """
    ts = np.linspace(0, 1, n_bins)
    pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
    dens = np.zeros(n_bins)
    for i, pt in enumerate(pts):
        d = np.linalg.norm(all_points - pt[None, :], axis=1)
        dens[i] = np.sum(d <= radius)
    return dens


def _successive_increases(density_profile):
    """Count the number of consecutive increasing steps in a 1D profile."""
    diffs = np.diff(density_profile)
    # longest run of positive diffs
    best = cur = 0
    for d in diffs:
        if d > 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def reassign_border_points(embedding, labels, radius=3.0, n_bins=10,
                            density_radius=0.15):
    """
    Implements: "Un-clustered points within 3 units of UMAP space to a
    cluster edge were assigned to a cluster within this radius; the event
    was assigned to the cluster that had the most successive increases in
    point density binned along a straight line connecting the event and
    the cluster centroid."
    """
    labels = labels.copy()
    unclustered_idx = np.where(labels == -1)[0]
    cluster_ids = np.unique(labels[labels != -1])
    if len(cluster_ids) == 0 or len(unclustered_idx) == 0:
        return labels

    clustered_points = embedding[labels != -1]
    clustered_labels = labels[labels != -1]
    centroids = {c: embedding[labels == c].mean(axis=0) for c in cluster_ids}

    nn = NearestNeighbors(n_neighbors=1).fit(clustered_points)

    for idx in unclustered_idx:
        p0 = embedding[idx]
        dist, nn_idx = nn.kneighbors(p0[None, :])
        dist = dist[0, 0]
        if dist > radius:
            continue  # too far from any cluster edge

        # candidate clusters = those with any point within `radius`
        dists_all = np.linalg.norm(clustered_points - p0[None, :], axis=1)
        candidate_clusters = np.unique(clustered_labels[dists_all <= radius])

        best_cluster, best_score = None, -1
        for c in candidate_clusters:
            profile = _density_along_line(embedding, p0, centroids[c],
                                           n_bins=n_bins, radius=density_radius)
            score = _successive_increases(profile)
            if score > best_score:
                best_score, best_cluster = score, c

        if best_cluster is not None:
            labels[idx] = best_cluster

    return labels


# ----------------------------------------------------------------------
# 7. STAGE 7: TRANSFORM HELD-OUT DATA INTO EXISTING UMAP + KNN ASSIGNMENT
# ----------------------------------------------------------------------

def assign_heldout_events(reducer, train_embedding, train_labels,
                           new_features_z, k=100, max_median_dist=0.3):
    """
    Embeds new (held-out) feature vectors into the *existing* UMAP model
    (reducer.transform), then assigns each new point the most common
    cluster label among its k=100 nearest neighbours in the training
    embedding. If the median distance to those neighbours exceeds
    `max_median_dist`, the point is left unassigned (-1).
    """
    new_embedding = reducer.transform(new_features_z)

    nn = NearestNeighbors(n_neighbors=k).fit(train_embedding)
    dist, idx = nn.kneighbors(new_embedding)

    assigned = np.full(len(new_features_z), -1, dtype=int)
    for i in range(len(new_features_z)):
        med_d = np.median(dist[i])
        if med_d > max_median_dist:
            continue
        neigh_labels = train_labels[idx[i]]
        neigh_labels = neigh_labels[neigh_labels != -1]
        if len(neigh_labels) == 0:
            continue
        vals, counts = np.unique(neigh_labels, return_counts=True)
        assigned[i] = vals[np.argmax(counts)]

    return assigned, new_embedding


# ----------------------------------------------------------------------
# 8. STAGE 8: BIPHASIC CONVERGENT (BConv) REASSIGNMENT
# ----------------------------------------------------------------------

def detect_biphasic_convergent(eye_pos_L, eye_pos_R, fs, onset_idx,
                                is_tethered=True,
                                temporal_sign_L=+1, temporal_sign_R=-1,
                                pre_win_ms=150, pre_gap_ms=100):
    """
    Flags a single binocular event as a candidate BConv (biphasic
    convergent) event if EITHER eye shows a fast, large "temporal"
    excursion prior to the main saccade.

    Parameters
    ----------
    eye_pos_L, eye_pos_R : smoothed eye-position traces (500 Hz)
    onset_idx : refined onset index (shared/binocular)
    is_tethered : selects velocity threshold (60 deg/s vs 40 deg/s free-swim)
    temporal_sign_L / temporal_sign_R : sign convention such that a
        *temporal* eye movement corresponds to velocity of that sign
        for each eye (convention-dependent on your coordinate system --
        set these based on how your Left/Right angles are defined).
    pre_win_ms : window (150ms) used to compute the eye-position std
                 baseline for the displacement threshold
    pre_gap_ms : gap (100ms) between that window and onset

    Returns
    -------
    is_bconv : bool
    details : dict with velocity/displacement values used in the decision
    """
    vel_thresh = 60.0 if is_tethered else 40.0

    n = len(eye_pos_L)
    gap_n = int(round(pre_gap_ms / 1000.0 * fs))
    win_n = int(round(pre_win_ms / 1000.0 * fs))

    win_hi = max(0, onset_idx - gap_n)
    win_lo = max(0, win_hi - win_n)

    def eye_check(pos, temporal_sign):
        vel = np.gradient(pos, 1.0 / fs)
        # displacement threshold = 1 SD of eye position in baseline window
        disp_thresh = np.nanstd(pos[win_lo:win_hi]) if win_hi > win_lo else np.nan

        # look for excursion just prior to / at onset in the temporal direction
        search_lo = max(0, onset_idx - win_n)
        seg_pos = pos[search_lo:onset_idx + 1]
        seg_vel = vel[search_lo:onset_idx + 1]

        temporal_vel_mask = (temporal_sign * seg_vel) > vel_thresh
        if not np.any(temporal_vel_mask):
            return False, dict(max_vel=np.nan, max_disp=np.nan,
                                vel_thresh=vel_thresh, disp_thresh=disp_thresh)

        idx_candidates = np.where(temporal_vel_mask)[0]
        disp = seg_pos[idx_candidates] - seg_pos[0]
        max_disp = np.nanmax(np.abs(disp))
        max_vel = np.nanmax(np.abs(seg_vel[idx_candidates]))

        passes = (max_vel > vel_thresh) and (max_disp > disp_thresh)
        return passes, dict(max_vel=max_vel, max_disp=max_disp,
                             vel_thresh=vel_thresh, disp_thresh=disp_thresh)

    passes_L, det_L = eye_check(eye_pos_L, temporal_sign_L)
    passes_R, det_R = eye_check(eye_pos_R, temporal_sign_R)

    is_bconv = bool(passes_L or passes_R)
    return is_bconv, dict(L=det_L, R=det_R)


def reassign_conv_to_bconv(labels, conv_cluster_id, bconv_cluster_ids,
                            event_bconv_flags, event_bconv_side):
    """
    Given the initial clustering `labels`, reassign events currently in
    `conv_cluster_id` that were flagged as biphasic-convergent
    (`event_bconv_flags[i] == True`) to the appropriate BConv cluster
    (`bconv_cluster_ids[0]` for left-eye-driven, `[1]` for right-eye-driven,
    based on `event_bconv_side[i] in {'L','R'}`).
    """
    labels = labels.copy()
    for i, (flag, side) in enumerate(zip(event_bconv_flags, event_bconv_side)):
        if flag and labels[i] == conv_cluster_id:
            labels[i] = bconv_cluster_ids[0] if side == 'L' else bconv_cluster_ids[1]
    return labels