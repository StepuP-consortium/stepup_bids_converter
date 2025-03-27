import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_effective_srate_xdf(lsl_stream, threshold_seconds=1, threshold_samples=500):

    stream_id = lsl_stream[0]["info"]["name"][0]
    effective_srate = 0  # will be recalculated if possible
    nsamples = len(lsl_stream[0]["time_stamps"])
    srate = float(lsl_stream[0]["info"]["nominal_srate"][0])
    tdiff = 1 / srate if srate > 0 else 0
    if nsamples > 0 and srate > 0:
        # Identify breaks in the time_stamps
        diffs = np.diff(lsl_stream[0]["time_stamps"])
        b_breaks = diffs > np.max(
            (threshold_seconds, threshold_samples * tdiff)
        )
        # find indices (+ 1 to compensate for lost sample in np.diff)
        break_inds = np.where(b_breaks)[0] + 1

        # Get indices delimiting segments without breaks
        # 0th sample is a segment start and last sample is a segment stop
        seg_starts = np.hstack(([0], break_inds))
        seg_stops = np.hstack((break_inds - 1, nsamples - 1))  # inclusive

        # Process each segment separately
        for start_ix, stop_ix in zip(seg_starts, seg_stops):
            # Calculate time stamps assuming constant intervals within each
            # segment (stop_ix + 1 because we want inclusive closing range)
            idx = np.arange(start_ix, stop_ix + 1, 1)[:, None]
            X = np.concatenate((np.ones_like(idx), idx), axis=1)
            y = lsl_stream[0]["time_stamps"][idx]
            mapping = np.linalg.lstsq(X, y, rcond=-1)[0]
            lsl_stream[0]["time_stamps"][idx] = mapping[0] + mapping[1] * idx

        # Recalculate effective_srate if possible
        counts = (seg_stops + 1) - seg_starts
        if np.any(counts):
            # Calculate range segment duration (assuming last sample
            # duration was exactly 1 * stream.tdiff)
            durations = (
                lsl_stream[0]["time_stamps"][seg_stops] + tdiff
            ) - lsl_stream[0]["time_stamps"][seg_starts]
            effective_srate = np.sum(counts) / np.sum(durations)

    if srate != 0 and np.abs(srate - effective_srate) / srate > 0.1:
        msg = (
            "Stream %d: Calculated effective sampling rate %.4f Hz is"
            " different from specified rate %.4f Hz."
        )
        logger.warning(msg, stream_id, effective_srate, srate)

    return effective_srate

def get_stream(lsl_streams, stream_name):
    """
    Retrieves a specific stream from a list of LSL streams by its name.

    Args:
        lsl_streams (list): A list of dictionaries representing LSL streams.
        stream_name (str): The name of the stream to retrieve.

    Returns:
        dict or None: The stream dictionary if found, otherwise None.
    """
    for stream in lsl_streams:
        if stream["info"]["name"][0] == stream_name:
            return stream
    return None
