import os.path as op
import numpy as np
import mne

from borsar.utils import find_range
from borsar.channels import get_ch_names
from borsar.freq import compute_rest_psd

from . import pth
from .utils import select_channels_special
from .utils import progressbar as progress_bar


def format_psds(psds, freq, freq_range=(8, 13), average_freq=False,
                selection='asy_frontal', transform='log', div_by_sum=False,
                info=None, src=None, subjects_dir=None, subject=None):
    '''
    Format power spectral densities. This includes channel selection, log
    transform, frequency range selection, averaging frequencies and calculating
    asymmetry (difference between left-right homologous sites).

    Parameters
    ----------
    psds : numpy array
        psds should be in (subjects, channels or vertices, frequencies) or
        (channels or vertices, frequencies) shape.
    freq : numpy array of shape (n_freqs,)
        Frequency bins.
    info : mne.Info
        Info about the ``psds`` spatial dimension - if ``psds`` is in channels
        space. Used for channel selection.
    freq_range : tuple or list-like with two elements
        Lower and higher frequency limits.
    average_freq : bool
        Whether to average frequencies.
    selection : str
        Type of channel selection. See `borsar.channels.select_channels`.
    transform : str | list of str | None
        Type of transformation applied to the data. 'log': log-transform;
        'zscore': z-scoring (across channels and frequencies);
        None: no transformation. Can also be a list of transforms for example:
        ``['log', 'zscore']``.
    div_by_sum : bool
        Used only when selection implies asymmetry ('asy' is in `selection`).
        If True the asymmetry difference is divided by the sum of
        the homologous channels: (ch_right - ch_left) / (ch_right + ch_left).
        Defaults to False.
    src : mne.SourceSpaces, optional
        SourceSpaces to use if ``psds`` is in source space. If ``psds``
        contains multiple subjects then the same ``src`` is used for all of
        them. Used for vertex selection and hemisphere morphing (if 'asy' is
        in ``selection``).
    subjects_dir : str, optional
        FreeSurfer subjects directory.
    subject : str, optional
        Selected subject in subjects_dir to use.

    Returns
    -------
    psds : numpy array
        Transformed psds.
    freq : numpy array
        Frequency bins.
    ch_names : list of str
        Channel names.
    '''

    has_subjects = psds.ndim == 3
    if freq_range is not None:
        rng = find_range(freq, freq_range)
        psds = psds[..., rng]
        freq = freq[rng]
    if average_freq:
        psds = psds.mean(axis=-1)
        freq = freq.mean()
    if not isinstance(transform, list):
        transform = [transform]

    # define indices selecting regions of interest
    if src is None:
        ch_names = get_ch_names(info)
        sel = select_channels_special(info, selection)
    else:
        from .src import select_vertices, _get_vertices, morph_hemi
        if 'asy' not in selection:
            sel = select_vertices(src, hemi='both', selection=selection,
                                  subjects_dir=subjects_dir, subject=subject)
            ch_names = sel

    if 'log' in transform:
        psds = np.log(psds)

    # compute asymmetry
    # =================
    if 'asy' in selection:
        if src is None:
            # CHANNEL SPACE
            # -------------
            rgt = psds[:, sel['right']]
            lft = psds[:, sel['left']]

            # create right-left channel names
            ch_names_arr = np.array(ch_names)
            ch_names = ['{}-{}'.format(ch1, ch2) for ch1, ch2 in
                        zip(ch_names_arr[sel['left']],
                            ch_names_arr[sel['right']])]
        else:
            # SOURCE SPACE
            # ------------
            # we first need to morph one hemisphere into the other to
            # make sure the vertices in left and right hemispheres match
            # we use a special version of fsaverage brain model - one that
            # is symmetrical
            src_sym = pth.paths.get_data('src_sym')
            psds = morph_hemi(
                psds, src, morph='lh2rh', src_sym=src_sym,
                subjects_dir=subjects_dir, has_subjects=has_subjects)

            # select relevant vertices from fsaverage_sym
            sel = select_vertices(src_sym, hemi='both', selection=selection,
                                  subjects_dir=subjects_dir,
                                  subject='fsaverage_sym')

            # now lh and rh should have the same number of vertices
            vertices = _get_vertices(src_sym)
            n_vert = [len(vertices[hemi]) for hemi in [0, 1]]
            assert n_vert[0] == n_vert[1]

            # select vertices for <left> and <right morphed to left>
            lft = psds[:, sel['rh']]
            rgt = psds[:, sel['rh'] + n_vert[1]]
            ch_names = sel['rh'] + n_vert[1]

        # compute asymmetry
        psds = rgt - lft
        if div_by_sum:
            psds /= rgt + lft

    else:
        psds = psds[:, sel]
        if src is None:
            ch_names = list(np.array(ch_names)[sel])

    if 'zscore' in transform:
        dims = list(range(psds.ndim))
        dims = tuple(dims[1:]) if has_subjects else tuple(dims)
        psds = ((psds - psds.mean(axis=dims, keepdims=True))
                / psds.std(axis=dims, keepdims=True))

    return psds, freq, ch_names


def save_psd(fname, psd, freq, ch_names, subject_id):
    '''Save power spectral density to disk.'''
    from scipy.io import savemat
    savemat(fname, {'psd': psd, 'freq': freq, 'ch_names': ch_names,
                    'subj_id': subject_id})


def compute_all_rest(study='C', event_id=None, tmin=1., tmax=60., winlen=2.,
                     step=0.5, space='avg', progressbar=True, save_dir=None):
    '''
    Compute power spectral density for all subjects in given study and save
    results to disk.

    Parameters
    ----------
    study : str
        Which study to use. Studies are coded with letters in the following
        fashion:

        =====   ============   ===============
        study   study letter   study directory
        =====   ============   ===============
        I       A              Nowowiejska
        II      B              Wronski
        III     C              DiamSar
        IV      D              PREDiCT
        V       E              MODMA
        =====   ============   ===============

        Study ``'C'`` is used by default.
    event_id : int | None
        Event id that starts periods of signal to compute psd for.
        This is only relevant in study C and D where both eyes closed
        (event_id: 11) and eyes open (event_id: 10) segments are available.
    tmin : float
        Start of the signal to consider. If event_id is None the tmin is with
        respect of the signal start, else it is with respect to each event
        that is event_id.
    tmax : float
        End of the signal to consider. If event_id is None the tmax is with
        respect of the signal start, else it is with respect to each event
        that is event_id.
    winlen : float
        Welch window length in seconds. The default is ``2.``.
    step : float
        Welch window step in seconds. The default is ``0.5``.
    space : str
        Signal space to use: ``'avg'`` for average referenced, ``'csd'`` for
        Current Source Density and ``'src'`` for source space.
    progressbar : str | bool
        If bool: whether to use progressbar. If string: progressbar to use
        ('notebook' vs 'text'). Progressbar requires tqdm library to work.
    save_dir : str | None
        Directory to save the psds to. By default study C main directory is
        used.
    '''
    from . import read_raw

    if event_id is not None:
        assert event_id in [10, 11]

    if save_dir is None:
        root_dir = pth.paths.get_path('main', study='C')
        save_dir = op.join(root_dir, 'analysis', 'psd')

    # construct file name based on analysis parameters
    eyes = ('closed' if study not in ['C', 'D']
            else ['open', 'closed'][event_id - 10])
    fname = ('psd_study-{}_eyes-{}_space-{}_winlen-{}_step-{}_'
             'tmin-{}_tmax-{}.mat')
    fname = fname.format(study, eyes, space, winlen, step, tmin, tmax)
    full_fname = op.join(save_dir, fname)

    all_psds = list()
    subj_ids = pth.get_subject_ids(study=study, task='rest')
    pbar = progress_bar(progressbar, total=len(subj_ids))

    for idx, subj_id in enumerate(subj_ids):
        print('Subject #{:02d}, ID: {}'.format(idx, subj_id))
        raw, events = read_raw(subj_id, study=study, task='rest', space=space)
        events = None if study not in ['C', 'D'] else events
        psd, freq = compute_rest_psd(raw, events=events, event_id=event_id,
                                     tmin=tmin, tmax=tmax, winlen=winlen,
                                     step=step)
        all_psds.append(psd)

        if idx == 0:
            ch_names = np.array(raw.ch_names)
        pbar.update(1)

    # group all psds in one array and save
    all_psds = np.stack(all_psds, axis=0)
    save_psd(full_fname, all_psds, freq, ch_names, subj_ids)


# TODO:
# - [ ] why make_csd_rest_approx did not return NaNs for the subject that
#       has NaNs both in channel data and make_csd_morlet_raw?
def make_csd_rest_approx(raw, frequencies, events=None, event_id=None,
                         tmin=None, tmax=None, n_jobs=1, n_cycles=7.,
                         decim=4, segment_length=1.):
    '''Approximate CSD for continuous signals by segmenting and computing CSD
    on segments.

    This function is not used in the "Three time NO" paper - it servers as a
    comparison.

    Parameters
    ----------
    raw : mne Raw object
        Instance of Raw object to use in CSD computation.
    frequencies : list of float
        List of frequencies to compute CSD for.
    events :
        Events array in mne format. Optional.
    event_id :
        Events to use in segmenting. Optional.
    tmin :
        Time start. If ``events`` was passed then ``tmin`` is with respect to
        each event onset.
    tmax :
        Time end. If ``events`` was passed then ``tmax`` is with respect to
        each event onset.
    n_jobs :
        Number of jobs to use. Defaults to 1.
    n_cycles :
        Number of cycles to use when computing cross spectral density. Defaults
        to 7.
    decim :
        Decimation factor in time of the cross spectral density result.
        Defaults to 4.
    segment_length : float
        Length of segments to which the signal of interest is divided. Defaults
        to 1.
    '''
    from mne.epochs import _segment_raw

    events, tmin, tmax = _deal_with_csd_inputs(tmin, tmax, events, event_id)
    sfreq = raw.info['sfreq']

    windows_list = list()
    for event_idx in range(events.shape[0]):
        event_onset = events[event_idx, 0] / sfreq
        this_tmin = event_onset + tmin
        this_tmax = event_onset + tmax
        raw_crop = raw.copy().crop(this_tmin, this_tmax)
        windows = _segment_raw(raw_crop, segment_length=segment_length,
                               preload=True, verbose=False)
        if len(windows._data) > 0:
            windows_list.append(windows)

    windows_list = mne.concatenate_epochs(windows_list)

    return mne.time_frequency.csd_morlet(
        windows_list, frequencies=frequencies, n_jobs=n_jobs,
        n_cycles=n_cycles, verbose=False, decim=decim)


# - [ ] LATER this might go to borsar
def make_csd_morlet_raw(raw, freqs, events=None, event_id=None, tmin=0.,
                        tmax=10., n_cycles=3., decim=1):
    '''Calculate cross-spectral density on raw data.

    Parameters
    ----------
    raw : mne Raw object
        Instance of Raw object to use in CSD computation.
    frequencies : list of float
        List of frequencies to compute CSD for.
    events : ndarray
        Events array in mne format (n_events by 3). Optional.
    event_id : int FIXME?
        Events to use in segmenting. Optional.
    tmin : float | None
        Time start. If ``events`` was passed then ``tmin`` is with respect to
        each event onset.
    tmax : float | None
        Time end. If ``events`` was passed then ``tmax`` is with respect to
        each event onset.
    n_jobs : int
        Number of jobs to use. Defaults to 1.
    n_cycles : int | list of int
        Number of cycles to use when computing cross spectral density. Defaults
        to 7.

    Returns
    -------
    csd : CrossSpectralDensity
        Cross spectral density of the data.
    '''
    from mne.time_frequency.csd import CrossSpectralDensity

    sfreq = raw.info['sfreq']
    freqs = np.asarray(freqs)
    n_channels = raw._data.shape[0]
    events, tmin, tmax = _deal_with_csd_inputs(tmin, tmax, events, event_id)

    all_weights = list()
    csds = list()

    first_samp = raw.first_samp
    window_len = np.round(1 / freqs * n_cycles * sfreq / 2).astype('int')
    add_rim = int(np.max(window_len) / 2)
    add_rim_post_decim = int(add_rim / decim)
    sfreq_post_decim = sfreq / decim

    for event_idx in range(events.shape[0]):
        tmin_ts = events[event_idx, 0] + int(round(tmin * sfreq)) - first_samp
        tmax_ts = events[event_idx, 0] + int(round(tmax * sfreq)) - first_samp

        # FIXME - make sure tfr here is complex
        start, end = tmin_ts - add_rim, tmax_ts + add_rim
        data = raw._data[:, start:end][np.newaxis, ...]
        tfr = mne.time_frequency.tfr_array_morlet(
            data, sfreq, freqs, n_cycles=n_cycles, decim=decim)

        n_times = tfr.shape[-1]
        tfr = tfr[0, ..., add_rim_post_decim:n_times-add_rim_post_decim]

        # FIXME - check that weights make sense
        wgt = _apply_annot_to_tfr(
            raw.annotations, tfr, sfreq_post_decim, freqs, n_cycles,
            orig_sample=int((tmin_ts + first_samp) / decim))
        reduction = np.mean if wgt == 0. else np.nanmean
        all_weights.append(wgt)

        # compute csd
        tfr_conj = np.conj(tfr)
        csd = np.vstack([reduction(tfr[[idx]] * tfr_conj[idx:], axis=2)
                         for idx in range(n_channels)])

        # Scaling by sampling frequency for compatibility with Matlab
        # (this is copied from mne-python, it makes more sense to scale
        #  CSD by number of non-NaN samples in the tfr signal...)
        csd /= sfreq_post_decim
        csds.append(csd)

    # weighted average
    if len(csds) > 1:
        perc_correct = 1 - np.array(all_weights)
        csds = np.average(np.stack(csds, axis=0), weights=perc_correct,
                          axis=0)
    else:
        csds = csds[0]

    max_wlen = int(round(max(freqs / n_cycles) * sfreq))
    return CrossSpectralDensity(csds, raw.ch_names, freqs, max_wlen)


# - [ ] LATER - this might go to borsar...
def _apply_annot_to_tfr(annot, tfr, sfreq, freqs, n_cycles, orig_sample=0,
                        fill_value=np.nan):
    '''Fill TFR data with `fill_value` where bad annotations are present.
    Useful mostly when dealing with continuous TFR.

    Parameters
    ----------
    annot : mne.Annotations
        Annotations to apply to time-frequency data.
    tfr : np.ndarray
        Numpy array of time-frequency data.
    sfreq : float
        Sampling frequency.
    n_cycles : int
        Extend each annotation by this number of cycles.
    orig_sample : int
        ...
    fill_value : float
        What to fill the data overlapping with the annotation. Defaults to
        np.nan.

    Returns
    -------
    Does not return anything, works in-place!
    '''
    n_times = tfr.shape[-1]
    n_times_rej = 0

    bad_annot = [idx for idx, desc in enumerate(annot.description)
                 if desc.startswith('BAD_')]
    tmin_sm, tmax_sm = orig_sample, orig_sample + n_times
    annot_onset_sm = (annot.onset * sfreq).astype('int')[bad_annot]
    annot_duration_sm = (annot.duration * sfreq).astype('int')[bad_annot]
    which_annot = (annot_onset_sm < tmax_sm) & (
        (annot_onset_sm + annot_duration_sm) > tmin_sm)

    if which_annot.any():
        window_len = np.round(1 / freqs * n_cycles * sfreq).astype('int') - 1
        use_annot_idx = np.where(which_annot)[0]

        for idx in use_annot_idx:
            onset = annot_onset_sm[idx] - tmin_sm
            offset = onset + annot_duration_sm[idx] + 1
            msk1 = onset - window_len
            msk2 = offset + window_len
            n_times_rej += offset - onset

            for frq_idx in range(msk1.shape[0]):
                tfr[..., frq_idx, msk1[frq_idx]:msk2[frq_idx]] = fill_value

    return n_times_rej / n_times


# TODO - should get inst to set tmin and tmax
def _deal_with_csd_inputs(tmin, tmax, events, event_id):
    '''Helper function that checks inputs to csd functions.'''
    if tmin is None or tmax is None:
        raise ValueError('Both tmin and tmax have to be specified')

    # there should be tmin and tmax checks

    if events is None:
        event_id = [123]
        events = np.array([[0, 0, 123]])
        return events, tmin, tmax
    else:
        got_event_id = event_id is not None
        if got_event_id:
            if isinstance(event_id, int):
                event_id = [event_id]
        else:
            event_id = np.unique(events[:, -1])
        events_of_interest = np.in1d(events[:, -1], event_id)
        events = events[events_of_interest]

        return events, tmin, tmax
