import json
import logging
import os
import pickle
import warnings
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.io import loadmat
from scipy.signal import savgol_filter

# global variables, double check these match your data
from scipy.sparse import csr_matrix
# noinspection PyProtectedMember
from scipy.sparse.csgraph._traversal import connected_components

warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
logging.basicConfig()
# logging.root.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import logging
import numpy as np

from data_analysis.settings import defaults

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_pauses(data, velocity_thresh=0.0005, max_var=0.01, pause_length=0.75, track_end=defaults.track_end):
    """
    inputs
    data: the pandas dataframe data created by import_and_clean
    velocity_thresh: a reasonable threshold for what a ybinned difference between beginning of pause length and the end
    max_var: the maximum variance expected between two neighboring frames, deals with edge cases
    pause_length: the maximum length of a pause, in seconds
    track_end: the pico value where the track resets, MUST double check as this changes with track length
    outputs: outputs the data dataframe with a new column for pauses,
    if the frame is not in a pause this column = 0, if it is the number is the associated pause number
    """
    logger.debug(f'getting pauses from data with shape {data.shape}')
    logger.debug(
        f'keywords: velocity_thresh={velocity_thresh}, max_var={max_var}, pause_length={pause_length}, track_end={track_end}')
    # setup
    ybinned = data.ybinned.values
    # the minimum number of consecutive frames without much movement to be considered a pause
    num_frames = int(defaults.frame_rate * pause_length)
    # keep in mind what frame rate your imaging data was collected at as the data
    # data is adjusted to that sampling rate. This is at 15.49 (unidirectional)
    # Will make sure that num_frames is always about 1sec in real time (change the 1 to approximate diff sec lengths)

    # finds start indices where velocity is below threshold, ignoring the pause at the end of the track
    indices, = np.where(
        (np.abs(ybinned[num_frames:] - ybinned[:-num_frames]) < velocity_thresh) &
        (ybinned[:-num_frames] < track_end)
    )

    # creates a boolean of ones the size of data
    to_keep = np.ones(len(data), dtype=np.bool)

    # change the boolean of the pausing frames to false
    for idx in indices:
        if np.max(ybinned[idx:idx + num_frames]) - np.min(ybinned[idx:idx + num_frames]) > max_var:
            continue
        to_keep[idx:idx + num_frames] = False

    # Count contiguous pauses. Count continuous running down from zero
    pause = np.empty(len(data), dtype=np.int32)
    pause[~to_keep] = 1 + np.concatenate([[0], np.cumsum(np.diff(np.arange(len(data))[~to_keep]) > 1)])
    pause[to_keep] = -np.concatenate([[0], np.cumsum(np.diff(np.arange(len(data))[to_keep]) > 1)])
    return pause

import logging
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

from data_analysis.settings import defaults

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def f_to_fc3(arr, axis=1, highbound=2, lowbound=0.5):
    if axis == 0:
        arr = np.asarray(arr) - np.median(arr, axis=axis)
    else:
        arr = np.asarray(arr) - np.median(arr, axis=axis)[:, None]
    if len(arr.shape) == 1:
        arr = np.expand_dims(arr, axis=0)

    if axis == 0:
        arr = arr.T

    std = np.std(arr, axis=1)
    is_spike = arr > highbound * std[:, None]

    a, b = arr.shape
    for i in range(a):
        for j in range(b - 1):
            if not is_spike[i, j]: continue
            if is_spike[i, j + 1]: continue

            if arr[i, j + 1] > lowbound * std[i]:
                is_spike[i, j + 1] = True

    arr[~is_spike] = 0
    if axis == 0:
        arr = arr.T
    return arr


def get_velocity(behavior: pd.DataFrame, odd_frame_length=5, smooth_rate=3):
    """
    inputs
    behavior: the data structure created from load_and_clean_behavior
    odd_frame_length: the number of frames you want to be considered, must be an odd integer
    smooth_rate: how smoothed you want the data to be, lower values are MORE smoothing. Must be smaller than odd_frame_length.

    outputs
    calculated_velocity: the velocity calculated from position
    corrected:velocities in the right order/format
    """
    logger.debug(
        f'Calculating velocity for array with shape {behavior.shape} and parameters odd_frame_length={odd_frame_length}, smooth_rate={smooth_rate}')
    # this section takes the data.ybinned and stacks them end to end, to deal with teleport and concat errors
    correction: pd.DataFrame = -(
            behavior.ybinned.diff() * (behavior.lap.diff().astype(bool) | (behavior.ybinned.diff().abs() > .05)))
    correction.iloc[0] = 0
    correction = correction.cumsum()
    corrected = (behavior.ybinned + correction)

    # setup for the for loop
    num_rounds = np.ceil(behavior.index[-1] / defaults.frames_per_session)
    velocities = []

    # calculates each velocity using a savitsky golay filter
    for x in defaults.frames_per_session * np.arange(num_rounds):
        session = corrected.loc[x - 0.5:x + defaults.frames_per_session - 0.5]
        velocities.append(savgol_filter(session, odd_frame_length, smooth_rate, deriv=1, mode='nearest'))
    calculated_velocity = defaults.pico_to_cm * defaults.frame_rate * np.concatenate(
        velocities)  # converts from meters/frame to cm/second, and concatenates

    return calculated_velocity, corrected


def normalize_baseline(arr: np.ndarray, quantile=0.08, scale_window=300, log=False):
    logger.debug(
        f'Normalizing array with shape {arr.shape}, quantile={quantile}, scale_window={scale_window}, log={log}')
    baseline = pd.DataFrame(arr).T.rolling(scale_window, center=True, min_periods=0).quantile(quantile).values.T
    if log:
        return arr - baseline
    return arr / baseline


def midlap_filter(ybinned):
    diff = np.diff(ybinned)
    to_drop = []
    # .3 is needed since there's sometimes a measurement mid-drop
    for idx in np.where(diff < -defaults.track_middle)[0]:
        if idx > 0 and diff[idx - 1] < 0:
            to_drop.append(idx)
        if idx < len(diff) - 1 and diff[idx + 1] < 0:
            to_drop.append(idx + 1)
    return np.asarray(ybinned.index[to_drop])


def booleanize(df, col, threshold=0.5):
    if col not in df.columns:
        logger.debug(f"{col} missing from DataFrame columns {df.columns}")
        return
    df[col] = df[col] > threshold


def cache_json_path(mouse_path, mouse_name, name):
    return os.path.join(
        mouse_path,
        f"{mouse_name}_{name}_args.json"
    )


def cache_path(mouse_path, mouse_name, name):
    return os.path.join(
        mouse_path,
        f"{mouse_name}_{name}.pickle"
    )


def check_cache(mouse_path, mouse_name, name, kwargs, use_cache=True):
    logger.debug('Checking cache for {} with keyword args:\n{}'.format(name, '\n'.join(
        f'{k}={v}' for k, v in sorted(kwargs.items()))))
    arg_path = cache_json_path(mouse_path, mouse_name, name)
    data_path = cache_path(mouse_path, mouse_name, name)
    if use_cache and os.path.exists(arg_path) and os.path.exists(data_path):
        with open(arg_path, 'r') as f:
            if json.load(f) == kwargs:
                logger.info(f'Loaded cached data from {data_path}')
                with open(data_path, 'rb') as g:
                    return pickle.load(g)

    if os.path.exists(data_path):
        os.remove(data_path)

    with open(arg_path, 'w') as f:
        json.dump(kwargs, f)


def write_cache(mouse_path, mouse_name, name, val):
    data_path = cache_path(mouse_path, mouse_name, name)
    logger.debug(f'Writing {name} to cache at {data_path}')
    with open(data_path, 'wb') as f:
        pickle.dump(val, f)


def get_pupil(pupil_path, pupil_data):
    assert os.path.exists(pupil_path), "pupil path is incorrect or proc file does not exist"

    if pupil_data is not None:
        return pupil_data

    pupil_raw = np.load(pupil_path, allow_pickle=True).item()['pupil'][0]

    pupil_area_smoothed = pupil_raw['area_smooth']
    pupil_com = pupil_raw['com']
    pupil_xpos = np.array([seq[0] for seq in pupil_com])
    pupil_ypos = np.array([seq[1] for seq in pupil_com])

    pupil_data = pd.DataFrame(
        {'pupil_xpos': pupil_xpos, 'pupil_ypos': pupil_ypos, 'pupil_area_smoothed': pupil_area_smoothed})

    return pupil_data


def get_context(index, config, date, mouse_name):
    mouse_cfg: dict = config[mouse_name]
    day_cfg = None
    for cfg in mouse_cfg["dates"]:
        if not cfg: continue
        if cfg["date"] == date:
            day_cfg = cfg
            break
    if day_cfg is None:
        return

    context_frames: dict = day_cfg["contexts"]
    contexts = pd.Series(np.full(len(index), ""), index=index)
    for k, (start, end) in context_frames.items():
        if k not in ["shocks"]:
            assert start >= index[0] and end <= index[-1] + 1, \
                f"context {k} has frames {start},{end} in config, but data only is size {len(contexts)}"
        contexts.loc[start:end - 1] = k

    return contexts


def behavior_path(mouse_name, behavior_name):
    assert os.path.exists(path := os.path.join(
        defaults.base_path, "beh",
        mouse_name, behavior_name
    )), path
    return path


def read_behavior(path, plane=0):
    behavior = loadmat(path)

    if "behplane" in behavior:
        mat = behavior["behplane"][0, plane]
    else:
        mat = behavior["behavior"]
    return pd.DataFrame({
        k: v[0] for k, v in zip(mat.dtype.names, mat[0, 0])
        if k not in ["fr", "acceleration"]
    })


def clean_behavior(df, drop_midlap=True):
    df.index = df.index.rename("frame")
    for col in ["lick", "reward", "shock"]:
        booleanize(df, col)

    df = df.rename(
        columns={
            "t": "seconds",
            "velocity": "recorded_velocity"
        }
    ).replace([-np.inf, np.inf], np.nan)

    if drop_midlap:
        df = df.drop(midlap_filter(df.ybinned), axis=0)

    return df


def add_behavior_columns(behavior, config, mouse_name, date,
                         pauses_kwargs=None,
                         odd_frame_length=9,
                         smooth_rate=3):
    behavior['lap'] = (behavior.ybinned.diff() < -defaults.lap_size / 2).cumsum()
    if pauses_kwargs is not None:
        behavior['pause'] = get_pauses(behavior, **pauses_kwargs)

    if config is not None and date is not None:
        contexts = get_context(
            index=behavior.index,
            config=config,
            date=date,
            mouse_name=mouse_name
        )
        if contexts is not None:
            behavior["context"] = contexts

    # reorganizes the dataframe and adds in recalculated velocity information
    behavior['velocity'], behavior['total_displacement'] = get_velocity(
        behavior,
        odd_frame_length=odd_frame_length,
        smooth_rate=smooth_rate
    )
    behavior['acceleration'] = behavior.recorded_velocity.diff()
    column_order = ['seconds', 'lick', 'reward', 'ybinned', 'total_displacement', 'recorded_velocity', 'velocity',
                    'acceleration', 'lap', 'pause']
    return behavior[
        [col for col in column_order if col in behavior.columns] +
        sorted(set(behavior.columns) - set(column_order))
        ]


class DataLoader:
    """
    This class takes in a combination of suite2p output data and processed behavior data from picoscope
    and returns a processed, frame-aligned Pandas dataframe. It filters and red-subtracts (for axons)
    and normalizes (for both axons and somas) the fluorescent traces.
    If given ROI cross-day matching information, it will also handle that appropriately.
    If axon filtering is turned on, it will sub-select only the relevant traces that pass a significance
    threshold
    """

    def __init__(self,
                 mouse_name: str,
                 beh_file_name: str = None,
                 cross_day_rois_id: str = None,
                 roi_key_to_date: list[str] = None,
                 date: Union[str, int] = None,
                 pauses_kwargs: dict = None,
                 baseline_kwargs: dict = None,
                 plane: int = 0,
                 normalize: bool = True,
                 use_cache: bool = True,
                 config_name="mice.json"
                 ):
        self._class_args = locals()
        self._class_args.pop('use_cache')
        self._class_args.pop('self')

        self.use_cache = use_cache

        self._mouse_path = os.path.join(defaults.base_path, mouse_name)
        logger.info(f'Using mouse path {self._mouse_path}')
        self._mouse_name = mouse_name

        self._plane = plane
        self._suite2p_path = defaults.suite2p_fmt.format(plane=plane)
        self._normalize = normalize

        self._config = None
        if config_name is not None:
            with open(os.path.join(defaults.configs_dir, config_name), "r") as f:
                self._config = json.load(f)

        if isinstance(date, int) and self._config is not None:
            logging.debug(f"date input: {date}, looking in config under mouse {self._mouse_name}...")
            try:
                date = self._config[self._mouse_name]["dates"][date]["date"]
                logging.debug(f"got date {date} from the config")
                self._class_args["date"] = date
            except (KeyError, IndexError):
                logging.warning(f"mouse name {self._mouse_name}, date {date} not found in config")

        if isinstance(date, int) or roi_key_to_date is None:
            dates = []
            for dirname in os.listdir(self._mouse_path):
                path = os.path.join(self._mouse_path, dirname)
                if not os.path.isdir(path):
                    continue
                try:
                    dates.append(pd.to_datetime(dirname, format=defaults.date_fmt))
                except ValueError:
                    continue
            dates = [d.strftime(defaults.date_fmt) for d in sorted(dates)]
            if roi_key_to_date is None:
                roi_key_to_date = dates
            if isinstance(date, int):
                date = dates[date]
                self._class_args['date'] = date

        if beh_file_name is None:
            beh_file_name = defaults.beh_fmt.format(mouse_name=mouse_name, date=date)
            self._class_args['beh_file_name'] = beh_file_name
        self._behavior_name = beh_file_name

        self._pupil_path = os.path.join(
            defaults.base_path, mouse_name,
            date, f'{mouse_name}_{date}_pupil_proc.npy'
        ) if date is not None else None

        if cross_day_rois_id is None and date is not None:
            logger.info(f'Using date {pd.to_datetime(date, format=defaults.date_fmt).date()}')
        elif cross_day_rois_id is not None:
            logger.info('Using cross day rois at {}'.format(os.path.join(self._mouse_path, f'{cross_day_rois_id}.mat')))
            logger.info(
                f'Date order in cross-day ROIs: {pd.to_datetime(roi_key_to_date, format=defaults.date_fmt).date()}')
        self._date = date
        self._roi_to_date = roi_key_to_date
        self._roi_key = None if cross_day_rois_id is None else \
            loadmat(os.path.join(self._mouse_path, f'{cross_day_rois_id}.mat'))['roiMatchData']['allSessionMapping'][0][
                0] - 1

        self._pauses_kwargs = {} if pauses_kwargs is None else pauses_kwargs
        self._baseline_kwargs = {} if baseline_kwargs is None else baseline_kwargs

        self._behavior: Union[pd.DataFrame, None] = None
        self._pupil_data: Union[pd.DataFrame, None] = None
        self._axon: Union[np.ndarray, None] = None
        self._soma: Union[np.ndarray, None] = None

        self._axon_args: Union[dict, None] = None
        self._soma_args: Union[dict, None] = None

    @property
    def config(self):
        return deepcopy(self._config)

    def _check_cache(self, name, kwargs):
        return check_cache(
            mouse_path=self._mouse_path,
            mouse_name=self._mouse_name,
            name=name,
            kwargs=kwargs,
            use_cache=self.use_cache
        )

    def _write_cache(self, name, val):
        return write_cache(
            mouse_path=self._mouse_path,
            mouse_name=self._mouse_name,
            name=name,
            val=val
        )

    def _get_pupil(self) -> pd.DataFrame:
        return get_pupil(
            pupil_path=self._pupil_path,
            pupil_data=self._pupil_data
        )

    @staticmethod
    def _mid_lap_filter(df) -> np.ndarray:
        return midlap_filter(df.ybinned)

    def _get_context(self, df) -> Union[None, pd.Series]:
        if self._config is None:
            return
        if self._date is None:
            return

        return get_context(
            index=df.index,
            config=self._config,
            date=self._date,
            mouse_name=self._mouse_name
        )

    def get_behavior(self) -> pd.DataFrame:
        assert self._behavior_name is not None, "no behavior provided"

        if self._behavior is not None:
            return self._behavior

        name = 'behavior'
        self._behavior = self._check_cache(name, self._class_args)
        if self._behavior is not None:
            return self._behavior

        path = behavior_path(self._mouse_name, self._behavior_name)
        logger.debug(f'Getting behavior from {path}')
        behavior = read_behavior(path, self._plane)

        if os.path.exists(self._pupil_path):
            pupil_data = get_pupil(self._pupil_path, self._pupil_data)
            behavior = pd.concat([behavior, pupil_data], axis=1)

        behavior = add_behavior_columns(
            behavior=clean_behavior(behavior),
            config=self._config,
            mouse_name=self._mouse_name,
            date=self._date,
            pauses_kwargs=self._pauses_kwargs
        )

        self._write_cache(name, behavior)
        self._behavior = behavior
        return behavior

    def _axon_day(self, date: str, savgol, loc=None, normalize=True) -> Union[np.ndarray, None]:
        f_data = {name: os.path.join(self._mouse_path,
                                     date,
                                     self._suite2p_path,
                                     f'{name}.npy') for name in ['F', 'F_chan2']}
        logger.debug(
            'Getting axon data for date {} from {}'.format(pd.to_datetime(date, format=defaults.date_fmt).date(),
                                                           f_data['F']))
        for k, v in f_data.copy().items():
            if os.path.exists(v):
                f_data[k] = np.load(v)
            else:
                del f_data[k]

        if loc is not None:
            for k, v in f_data.items():
                f_data[k] = v[loc]

        # creates savitsky-golay filtered variables to use for plotting
        if 'F_chan2' in f_data:
            logger.info('Found red and green channels')
            filtered_f = savgol_filter(np.log(f_data['F'][:len(f_data['F_chan2'])]), *savgol)
            filtered_f_red = savgol_filter(np.log(f_data['F_chan2']), *savgol)

            # demeans the green channel of the red channel, then filters with a savitsky-golay, logarithmically

            covs = np.array([np.cov(green, red) for green, red in
                             zip(filtered_f[:len(filtered_f_red)], filtered_f_red)])
            demeaned = filtered_f[:len(filtered_f_red)] - filtered_f_red * (covs[:, 0, 1, None] / covs[:, 1, 1, None])
        elif 'F' in f_data:
            logger.info('Found only green channel')
            demeaned = savgol_filter(np.log(f_data['F']), *savgol)
        else:
            logger.warning('No red or green channel found')
            return
        if not self._normalize:
            return demeaned
        return normalize_baseline(demeaned, **self._baseline_kwargs, log=True)

    @staticmethod
    def _slice_axons(axons, axon_filter_thresh=None):
        sl = slice(None)
        if axon_filter_thresh is not None:
            sl = (np.std(axons, axis=1) > axon_filter_thresh)
        return sl

    def _group_axons(self, corr_thresh):
        logger.info(f'Grouping axons with correlation threshold {corr_thresh}')
        pearson_corr = np.corrcoef(self._axon)
        n_comp, group_ids = connected_components(csr_matrix(pearson_corr > corr_thresh), directed=False,
                                                 return_labels=True)

        group_axon_pcs = []
        for gid in range(n_comp):
            pca = PCA(n_components=1)
            idx = np.where(group_ids == gid)[0]
            logger.info(f'Axon group {gid} constructed from axons: {idx}')
            axon_group = self._axon[idx].T
            if axon_group.shape[-1] == 1:
                result = axon_group
            else:
                result = pca.fit_transform(axon_group)
                result /= pca.components_.sum(1)
            group_axon_pcs.append(np.squeeze(result))

        self._axon = np.vstack(group_axon_pcs)

    def get_axon(self, savgol_frames: int = 11, savgol_smoothing: int = 1,
                 axon_filter_thresh=None, corr_thresh=None) -> Union[np.ndarray, None]:
        self._axon_args, old_args = self._class_args | locals(), self._axon_args
        self._axon_args.pop('self')
        if self._axon is not None and self._axon_args == old_args:
            return self._axon

        name = 'axon'
        self._axon = self._check_cache(name, self._axon_args)
        if self._axon is not None:
            return self._axon

        savgol = [savgol_frames, savgol_smoothing]
        if self._date is not None:
            logger.debug('Getting single day axon data')
            self._axon = self._axon_day(self._date, savgol)
            if self._axon is None:
                return
            self._axon = self._axon[self._slice_axons(self._axon, axon_filter_thresh)]
        else:
            logger.debug('Getting multiday axon data')
            self._axon = []
            to_keep = None if axon_filter_thresh is None else np.zeros(self._roi_key.shape[0], dtype=np.bool)

            for i, date in sorted(enumerate(self._roi_to_date), key=lambda t: t[1]):
                fall = loadmat(os.path.join(self._mouse_path, date, self._suite2p_path, 'Fall.mat'))
                loc = np.where(fall['iscell'][:, 0])[0][self._roi_key[:, i]]
                self._axon.append(self._axon_day(date, savgol, loc))
                if to_keep is not None:
                    to_keep[self._slice_axons(self._axon[-1], axon_filter_thresh)] = True

            self._axon = np.concatenate(self._axon, axis=1)
            if to_keep is not None:
                self._axon = self._axon[to_keep]

        if corr_thresh is not None:
            self._group_axons(corr_thresh)

        self._write_cache(name, self._axon)
        return np.exp(self._axon)

    def _soma_day(self, date: str, loc=None, fc3=False):
        path = os.path.join(self._mouse_path, date, self._suite2p_path, 'F.npy')
        logger.debug('Getting soma data for date {} from {}'.format(
            pd.to_datetime(date, format=defaults.date_fmt).date(), path))
        ses = np.load(path)
        if loc is not None:
            ses = ses[loc]

        if self._normalize:
            ses = normalize_baseline(ses, **self._baseline_kwargs)
        if fc3:
            ses = f_to_fc3(ses, axis=1)

        return ses

    def get_soma(self, fc3=False):
        self._soma_args, old_args = self._class_args | locals(), self._soma_args
        self._soma_args.pop('self')
        if self._soma is not None and self._soma_args == old_args:
            return self._soma

        name = 'soma'
        self._soma = self._check_cache(name, self._soma_args)
        if self._soma is not None:
            return self._soma

        if self._date is not None:
            self._soma = self._soma_day(self._date, fc3=fc3)
        elif self._roi_key is not None:
            self._soma = []
            for i, date in sorted(enumerate(self._roi_to_date), key=lambda t: t[1]):
                fall = loadmat(os.path.join(self._mouse_path, date, self._suite2p_path, 'Fall.mat'))
                loc = np.where(fall['iscell'][:, 0])[0][self._roi_key[:, i]]
                self._soma.append(self._soma_day(date, loc, fc3=fc3))

            self._soma = np.concatenate(self._soma, axis=1)
        elif os.path.exists(path := os.path.join(
                self._mouse_path, self._suite2p_path, "F.npy")):
            logger.debug(f"Getting multiday soma from single file {path}")
            self._soma = np.load(path)
            if self._normalize:
                self._soma = normalize_baseline(self._soma, **self._baseline_kwargs)
            if fc3:
                self._soma = f_to_fc3(self._soma, axis=1)
        else:
            raise AssertionError()

        # Set first soma row to zero unless log-normalized, in which case set to 1
        self._soma[:, 0] = self._normalize and not self._baseline_kwargs.get("log", False)

        self._write_cache(name, self._soma)
        return self._soma

    def merge_behavior(self, is_axon: bool, **kwargs):
        logger.debug('Getting behavior and {} data'.format('axon' if is_axon else 'soma'))
        behavior = self.get_behavior()
        logger.debug('loaded behavior successfully')
        val = self.get_axon(**kwargs) if is_axon else self.get_soma(**kwargs)
        if val is None:
            logger.warning("Only behavior found, just use get_behavior method")
            return behavior
        logger.debug('loaded other data successfully, concatenating...')
        return pd.DataFrame(np.concatenate([behavior, val[:, np.asarray(behavior.index)].T], axis=1),
                            index=behavior.index,
                            columns=list(behavior.columns) + list(range(len(val))))

    @staticmethod
    def groupby_lap(df):
        return df.reset_index(drop=False).groupby('lap')
