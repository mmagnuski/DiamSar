import warnings
import os.path as op

import numpy as np
import pandas as pd
import mne

from borsar.csd import current_source_density
from borsar.channels import get_ch_pos
from borsar.utils import silent_mne
from sarna.events import read_rej

from . import utils, freq, analysis, pth, viz
from .utils import colors
from .pth import get_file
from .events import get_task_event_id
from .io import (read_bdi, set_or_join_annot,
                 warnings_to_ignore_when_reading_files)


def read_raw(fname, study='C', task='rest', space='avg'):
    '''
    Read a raw file and its events using the DiamSar reading pipeline.

    The DiamSar reading pipeline is following:
    * read .set file from study-task-specific directory
    * load annotations from .rej file (if .rej file exists)
    * if study C -> add Cz reference channel
    * apply average reference or CSD depending on space kwarg
    * find events (and update them with information about load for sternberg)

    Parameters
    ----------
    fname : int or string
        Name of the file to read. If int it is meant to be a subject
        identifier. It can also be int-like string for example '003'.
    study : str
        Study letter identifier. "A" (Nowowiejska), "B" (Wroński) or "C"
        (DiamSar).
    task : str
        Task to read, 'rest' or 'sternberg'.
    space : str
        Data space: average referenced channel space (``"avg"``), current
        source density (``"csd"``) or DICS beamformer-localized source space
        (``"src"``).

    Returns
    -------
    raw : instance of mne.io.Raw
        Raw data.
    events : numpy 2d array
        Events array in mne events format.

    Examples:
    ---------
    > raw, events = read_raw(23)
    > raw_rest, events = read_raw(46, study='B', task='rest')
    > raw_sternberg, events = read_raw('046', study='C', task='sternberg')
    '''

    # get relevant paths and file names
    data_path = pth.paths.get_path('eeg', study=study, task=task)
    set_file, rej_file = get_file(fname, study=study, task=task)

    # read file
    # ---------
    # read eeglab file, ignoring some expected warnings
    with warnings.catch_warnings():
        for msg in warnings_to_ignore_when_reading_files():
            warnings.filterwarnings('ignore', msg)
        # make sure we are compatible with mne 0.17 and mne 0.18:
        event_id = dict(boundary=999, empty=999)

        # FIXME - test and switch to new reading if this is the latest mne
        try:
            # read data from eeglab .set file
            raw = mne.io.read_raw_eeglab(op.join(data_path, set_file),
                                         event_id=event_id, preload=True,
                                         verbose=False)
            # read events from stim channel
            events = mne.find_events(raw, verbose=False)
        except TypeError:
            # new read_raw_eeglab does not use event_id
            raw = mne.io.read_raw_eeglab(op.join(data_path, set_file),
                                         preload=True, verbose=False)
            # get events from raw.annotations
            event_id = get_task_event_id(raw, event_id, study=study,
                                         task=task)
            events = mne.events_from_annotations(raw, event_id=event_id)[0]

    # special case for PREDiCT data (they don't differentiate
    # closed eyes start)
    if study == 'D':
        from DiamSar.events import translate_events_D
        events = translate_events_D(events)

    # FIXME: in 0.17 / 0.18 -> boundary annotations should be already present,
    #        check if this is needed
    # create annotations from boundary events
    is_boundary = events[:, -1] == 999
    n_boundaries = is_boundary.sum()
    if n_boundaries > 0:
        margin = 0.05
        onsets = events[is_boundary, 0] / raw.info['sfreq'] - margin
        duration = np.tile(2 * margin, n_boundaries)
        description = ['BAD_boundary'] * n_boundaries
        annot = mne.Annotations(onsets, duration, description)
        set_or_join_annot(raw, annot)

    # rej file annotations
    # --------------------
    if rej_file is not None:
        # FIXME: temporary fix for pandas issue #15086
        #        (check which version introduced the fix)
        if study == 'B':
            import sys
            if sys.platform == 'win32':
                sys._enablelegacywindowsfsencoding()

        # read annotations
        # FIXME: could use encoding='ANSI' for study B?
        annot = read_rej(op.join(data_path, rej_file), raw.info['sfreq'])

        # set annotations or add to those already present
        set_or_join_annot(raw, annot)

    # task specific event modification - add load info to events
    if task == 'sternberg':
        from DiamSar.events import change_events_sternberg
        events = change_events_sternberg(events)

    # channel position, reference scheme
    # ----------------------------------

    # drop stim channel and 'oko' channel if present
    drop_chan = [ch for ch in raw.ch_names
                 if 'STI' in ch or ch in ['oko', 'HEOG', 'VEOG']]
    raw.drop_channels(drop_chan)

    # add original reference channel (Cz) in study C
    if study == 'C':
        with silent_mne():
            mne.add_reference_channels(raw, 'Cz', copy=False)
            maxpos = get_ch_pos(raw).max()
            raw.info['chs'][-1]['loc'][2] = maxpos

    # make sure channel order is correct, else reorder
    chan_ord = pth.paths.get_data('chanord', study=study)
    if not (np.array(chan_ord) == np.array(raw.ch_names)).all():
        raw.reorder_channels(chan_ord)

    # rereference to average or apply CSD
    if space == 'avg':
        raw.set_eeg_reference(verbose=False, projection=False)
    elif space == 'csd':
        G, H = pth.paths.get_data('GH', study=study)
        raw = current_source_density(raw, G, H)

    return raw, events


def read_beh(fname, study='C', task='sternberg'):
    '''Read behavioral data from given study and task.

    Parameters
    ----------
    fname : str | int
        Filename or subject int identifier. Data to read.
    study : str
        Study to read data for.
    task : str
        Task to read data for.

    Returns
    -------
    beh : pandas.DataFrame
        Behavioral data for given subject, in given task from given study.

    This function is not used in "Three times NO" paper.
    '''
    if task == 'sternberg':
        beh_dir = pth.paths.get_path('beh', task='sternberg')

        if isinstance(fname, int):
            fname = '{:03d}.csv'.format(fname)

        beh = pd.read_csv(op.join(beh_dir, fname), index_col=0)
        beh.load = [len(x) for x in beh.digits.str.split()]
        return beh
    else:
        raise NotImplementedError('Currently only sternberg task is allowed.')


# coreg stuff
# -----------
# - [ ] CONSIDER removing as obsolete
def create_dig_montage(inst, montage=None, dig_ch_pos=True, hsp=True,
                       scale=1 / 1000., coords=None):
    '''
    Create DigMontage from Montage or digitization dataframe.

    This function was used to create custom digitization in earlier mne
    versions. However, since then, this has become much easier in recent mne
    versions, so this function has become obsolete. Also - it is likely not to
    work on recent mne versions.

    Parameters
    ----------
    inst : mne object (raw, epochs, info)
        Mne object containing channel names.
    montage : mne Montage object | None (default None)
        Montage to use in DigMontage creation
    dig_ch_pos : bool (default True)
        Whether to create dig_ch_pos field - that contains digitized channel
        position.
    hsp : bool (default True)
        Whether to create hsp field - that contains head shape information.
        The positions will be the same as channel positions in the montage
    scale : float (default 1 / 1000.)
        How to scale the montage points. For example if your montage is in
        meters and you need to scale it to mm that would be 1 / 1000.
    coords : DataFrame
        DataFrame containing channel names and their digitized positions.

    Returns
    -------
    dig : instance of DigMontage
        DigMontage create out of the passed `montage`.
    '''
    from copy import deepcopy

    if montage is None:
        montage = mne.channels.read_montage('easycap-M1')

    if coords is not None:
        args = dict()
        row_names = ['Nasion', 'LPA', 'RPA']
        for row_name in row_names:
            row = coords.query('name == "{}"'.format(row_name))
            xyz = np.array([row.iloc[0, i] for i in range(1, 4)])
            args[row_name.lower()] = xyz * scale
    else:
        args = dict(lpa=np.array([-75., 0., -40.]) * scale,
                    rpa=np.array([75., 0., -40.]) * scale,
                    nasion=np.array([0., 80., -10.]) * scale)

    if coords is not None:
        montage = deepcopy(montage)
        for name in inst.ch_names:
            row = coords.query('name == "{}"'.format(name))
            xyz = np.array([row.iloc[0, i] for i in range(1, 4)])
            idx = montage.ch_names.index(name)
            montage.pos[idx, :] = xyz

    if dig_ch_pos:
        dig_ch_pos = dict()
        for name in inst.ch_names:
            name_idx = montage.ch_names.index(name)
            dig_ch_pos[name] = montage.pos[name_idx] * scale
        args['dig_ch_pos'] = dig_ch_pos
        args['point_names'] = inst.ch_names

    if hsp:
        correct_montage_chans = np.in1d(montage.ch_names, inst.ch_names)
        args['hsp'] = montage.pos[correct_montage_chans] * scale
        # it seems that point_names are not necessary for hsp
        # args['point_names'] = ['hsp_' + montage.ch_names[idx] for idx in
        #                        np.where(correct_montage_chans)[0]]

    dig = mne.channels.DigMontage(**args)
    return dig


def final_subject_ids(study, verbose=True):
    '''Get final subject ids for given study.'''
    psds, _, _, subj_id = pth.paths.get_data('psd', study=study)

    if verbose:
        print('Selecting subject ids for study {}.'.format(study))
        print('{} subjects with psds.'.format(len(subj_id)))

    # exclude subjects without enough good data
    no_nans = ~np.any(np.isnan(psds), axis=(1, 2))
    subj_id = subj_id[no_nans]

    if verbose and not no_nans.all():
        print('{} subjects after removing NaN psds.'.format(len(subj_id)))

    # exclude subjects without BDI in table (very rare cases)
    bdi = pth.paths.get_data('bdi', study=study)
    has_bdi = np.in1d(subj_id, bdi.index)
    subj_id = subj_id[has_bdi]

    if verbose and not has_bdi.all():
        msg = '{} subjects after removing those without BDI score.'
        print(msg.format(len(subj_id)))

    return subj_id
