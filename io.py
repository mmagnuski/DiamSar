import re
import os
import os.path as op
import warnings
import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
from scipy import sparse
import pandas as pd
import mne

from borsar.cluster import construct_adjacency_matrix


# load functions
# --------------
# most of the functions here are accessed via:
# paths.get_data(data_type, study=study)
# where ``paths`` is ``DiamSar.pth.paths``

# FIXME - use in register mode (CHECK - did I mean .get_data(),
#                               is it already done?)
def read_bdi(paths, study='C', **kwargs):
    '''Read BDI scores and diagnosis status. Can also read other, confounding
    variables, when passing ``full_table=True``.

    Parameters
    ----------
    paths : borsar.project.Paths
        DiamSar paths objects containing information about all the relevant
        paths.
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
    full_table : bool
        Whether to read full table, containing for example age and gender.
        This option is not used in "Three times NO" paper.

    Returns
    -------
    bdi : pandas.DataFrame
        Dataframe with bdi scores and diagnosis status.
    '''
    full_table = kwargs.get('full_table', False)
    base_dir = paths.get_path('main', study=study)
    beh_dir = op.join(base_dir, 'beh')
    if study == 'A':
        df = pd.read_excel(op.join(beh_dir, 'baza minimal.xlsx'))
        select_col = ['ID', 'BDI_k', 'DIAGNOZA']
        rename_col = {'BDI_k': 'BDI-I'}

        if full_table:
            select_col += ['plec_k', 'wiek']
            rename_col.update({'plec_k': 'sex', 'wiek': 'age'})

        # select relevant columns
        bdi = df[select_col]
        bdi = make_sure_diagnosis_is_boolean(bdi)

        # rename columns
        bdi = bdi.rename(columns=rename_col)

        if full_table:
            # ! TODO make sure that sex is coded this way
            relabel = {1: 'female', 2: 'male'}
            bdi.loc[:, 'sex'] = bdi.sex.replace(relabel)

    if study == 'B':
        if full_table:
            bdi = pd.read_excel(op.join(beh_dir, 'BAZA DANYCH.xlsx'))
            sel_col = ['nr osoby', 'BDI 2-pomiar wynik', 'wykształcenie',
                       'wiek', 'płeć']
            sel_col = [col for col in sel_col if col in bdi.columns]
            bdi = bdi[sel_col]

            # trim rows
            idx = np.where(bdi['nr osoby'].isnull())[0][0]
            bdi = bdi.iloc[:idx, :]

            # rename columns
            rename_col = {'wiek': 'age', 'płeć': 'sex', 'nr osoby': 'ID',
                          'BDI 2-pomiar wynik': 'BDI-I',
                          'wykształcenie': 'education'}
            bdi = bdi.rename(columns=rename_col)

            # rename sex to female / male
            bdi.loc[:, 'sex'] = bdi.sex.replace({'k': 'female', 'm': 'male'})

            # missing values should be NaN
            msk = bdi.loc[:, 'BDI-I'] == 'brak'
            bdi.loc[msk, 'BDI-I'] = np.nan

            bdi = bdi.infer_objects()
        else:
            bdi = pd.read_excel(op.join(beh_dir, 'BDI.xlsx'), header=None,
                                names=['ID', 'BDI-I'])
        bdi.loc[:, 'DIAGNOZA'] = False

    if study == 'C':
        df = pd.read_excel(op.join(beh_dir, 'BAZA_DANYCH.xlsx'))
        first_null = np.where(df.ID.isnull())[0]
        if len(first_null) > 0:
            first_null = first_null[0]
            df = df.iloc[:first_null, :]
        if full_table:
            bdi = study_C_reformat_original_beh_table(df)
        else:
            bdi = df[['ID', 'BDI-II', 'DIAGNOZA']]

        bdi = make_sure_diagnosis_is_boolean(bdi)

    if study == 'D':
        bdi = pd.read_excel(op.join(beh_dir, 'subject_data.xlsx'))
        bdi.loc[:, 'DIAGNOZA'] = bdi.MDD <= 2
        sel_col = ['id', 'DIAGNOZA', 'BDI']

        if full_table:
            # translate MDD values to more meaningful strings
            translate = {1: 'present', 2: 'past', 50: 'subclinical', 99: 'no'}
            bdi.loc[:, 'depression'] = bdi.MDD.replace(translate)

            # translate sex to female/male strings
            relabel = {1: 'female', 2: 'male'}
            bdi.loc[:, 'sex'] = bdi.sex.replace(relabel)

            sel_col = sel_col + ['depression', 'sex', 'age']

        bdi = bdi[sel_col]
        rename_col = {'id': 'ID', 'BDI': 'BDI-II'}
        bdi = bdi.rename(columns=rename_col)

    if study == 'E':
        # original database name is 'subjects_information_EEG_128channels_
        # resting_lanzhou_2015.xlsx'
        bdi = pd.read_excel(op.join(beh_dir, 'database_MODMA.xlsx'))
        bdi.loc[:, 'DIAGNOZA'] = bdi.type == 'MDD'
        sel_col = ['subject id', 'DIAGNOZA', 'PHQ-9']

        if full_table:
            sel_col = sel_col + ['sex', 'age', 'education']
            relabel = {'F': 'female', 'M': 'male'}
            bdi.loc[:, 'sex'] = bdi.gender.replace(relabel)
            bdi = bdi.rename(columns={'education（years）': 'education'})

        bdi = bdi.loc[:, sel_col]
        rename_col = {'subject id': 'ID'}
        bdi = bdi.rename(columns=rename_col)

    return bdi.set_index('ID')


def study_C_reformat_original_beh_table(df):
    '''Select and recode relevant columns from behavioral table.
    '''
    # select relevant columns
    df = df.loc[:, ['ID', 'DATA BADANIA', 'WIEK', 'PŁEĆ', 'WYKSZTAŁCENIE',
                    'DIAGNOZA', 'BDI-II']]

    # fix dates
    df.loc[0, 'DATA BADANIA'] = df.loc[1, 'DATA BADANIA']
    df.loc[7, 'WIEK'] = datetime.datetime(df.loc[7, 'WIEK'], 6, 25)
    df.loc[21, 'WIEK'] = datetime.datetime(df.loc[21, 'WIEK'], 6, 25)

    # silence false alarm SettingWithCopyWarnings:
    with warnings.catch_warnings():
        irritating_warning = pd.core.common.SettingWithCopyWarning
        warnings.simplefilter('ignore', irritating_warning)

        # age
        # ---
        for idx in df.index:
            delta = relativedelta(df.loc[idx, 'DATA BADANIA'],
                                  df.loc[idx, 'WIEK'])
            df.loc[idx, 'age'] = delta.years

        # one bad birth date (the same as study date) - use average
        # student age (participant was a student)
        avg_student_age = df.query('WYKSZTAŁCENIE == "Student"').age.mean()
        df.loc[9, 'age'] = int(avg_student_age)

        # remove 'DATA BADANIA' and 'WIEK'
        bdi = df.drop(['DATA BADANIA', 'WIEK'], axis='columns')

        # rename columns
        relabel = {'WYKSZTAŁCENIE': 'education', 'PŁEĆ': 'sex'}
        bdi = bdi.rename(columns=relabel)

        # translate płeć
        relabel = {'KOBIETA': 'female', 'MĘŻCZYZNA': 'male'}
        bdi.loc[:, 'sex'] = bdi.sex.replace(relabel)

    return bdi


def make_sure_diagnosis_is_boolean(bdi):
    # silence false alarm SettingWithCopyWarnings:
    with warnings.catch_warnings():
        irritating_warning = pd.core.common.SettingWithCopyWarning
        warnings.simplefilter('ignore', irritating_warning)
        bdi.loc[:, 'DIAGNOZA'] = bdi.DIAGNOZA.astype('bool')
    return bdi


def prepare_data(paths, study='C', contrast='cvsd', eyes='closed', space='avg',
                 freq_range=(8, 13), avg_freq=True, selection='frontal_asy',
                 div_by_sum=False, transform='log', confounds=False,
                 scale_psd=False, interaction=False, verbose=True):
    '''Get and prepare data for analysis.

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
    contrast : str
        Statistical contrast to use. Contrasts are coded in the following
        fashion:

        ===========   =============   ===============================
        contrast      contrast name   contrast description
        ===========   =============   ===============================
        ``'cvsd'``    DvsHC           diagnosed vs healthy controls
        ``'cvsc'``    SvsHC           subclinical vs healthy controls
        ``'dreg'``    DReg            linear relationship with BDI restricted
        ``'dreg'``    DReg            linear relationship with BDI restricted
                                      to depressed individuals.
        ``'creg'``    nonDReg         linear relationship with BDI restricted
                                      to non-depressed individuals.
        ``'cdreg'``   allReg          linear relationship with BDI on all
                                      participants.
        ===========   =============   ===============================

        Contrast ``'cvsd'`` is used by default. For more details on the BDI
        thresholds used to create healthy controls and subclinical groups
        see ``DiamSar.utils.group_bdi``.
    eyes : str
        Rest segment to use in the analysis:
        * ``'closed'`` - eyes closed
        * ``'open'`` - eyes open

        Only study C has eyes open rest segments.
    space : str
        Space to use in the analysis:

        ===========   =====================================================
        space         description
        ===========   =====================================================
        ``'avg'``     channel space, average reference
        ``'csd'``     channel space, Current Source Density (CSD) reference
        ``'src'``     source space
        ===========   =====================================================

    freq_range : tuple
        Lower and higher edge of the frequency space to include in the
        analysis (in Hz). ``(8, 13)`` by default.
    avg_freq : bool
        whether to average the selected frequency range. ``True`` by default.
    selection : str
        Channels/sources to select.

        =================   ==========================================
        value               description
        =================   ==========================================
        ``'all'``           all channels
        ``'frontal'``       all frontal channels
        ``'frontal_asy'``   all frontal asymmetry channel pairs
        ``'asy'``           all asymmetry channel pairs
        ``'asy_pairs'``     two selected asymmetry channel pairs
                            corresponding to F4-F3 and F8-F7
        =================   ==========================================

        Defaults to ``'frontal_asy'``.

    div_by_sum : bool
        Whether to divide R - L asymmetry by the sum (R + L).
    transform : str | list of str
        Transformation on the data:

        ============   ==========================================
        value          description
        ============   ==========================================
        ``'log'``      log-transform
        ``'zscore'``   within-participant across-channels z-score
        ============   ==========================================

        You can also group transforms: ``['log', 'zscore']`` zscores the
        log-transformed data. ``'log'`` is used by default.
    confounds : bool
        Whether to include potential confounds in the returned data. These
        include: sex, age and education (if they are available for given
        study). This option works only when full behavioral data are available
        (``paths.get_data('bdi', full_table=True)``), so it may not work for
        some of the datasets I - III hosted on Dryad (Dryad does not allow
        more than 3 variables per participant to avoid the possibility of
        individual identification).
    scale_psd : bool
        Whether to center and scale (z-score) the biological data (power
        spectra). This step is done after asymmetry computation and
        log-transform.
    interaction : bool
        Whether to return diagnosis * gender (or diagnosis * bdi) interaction
        variable in the predictors table.
    verbose : bool | int
        Verbosity level supported by mne-python. ``True`` by default.

    Returns
    -------
    data : dict
        Dictionary of data.
    '''
    from .utils import group_bdi, recode_variables
    from .freq import format_psds

    if interaction and not confounds:
        raise ValueError("Performing gender * diagnosis interaction requires "
                         "confounds=True")

    data = dict()
    # get base study name and setup stat_info dict
    stat_info = dict(avg_freq=avg_freq, freq_range=freq_range,
                     selection=selection, space=space, contrast=contrast,
                     study=study, eyes=eyes, transform=transform,
                     div_by_sum=div_by_sum)

    # load relevant data
    bdi = paths.get_data('bdi', study=study, full_table=confounds)
    psds, freq, ch_names, subj_id = paths.get_data(
        'psd', study=study, eyes=eyes, space=space)

    # some src psds have spatial dimension last,
    # while freq is assumed last in format psds
    if space == 'src' and psds.shape[-1] > psds.shape[-2]:
        psds = psds.transpose((0, 2, 1))

    # select only subjects without NaNs
    no_nans = ~np.any(np.isnan(psds), axis=(1, 2))
    if not np.all(no_nans):
        psds = psds[no_nans]
        subj_id = subj_id[no_nans]

    # get information about channel / source space
    info, src, subject, subjects_dir = _get_space_info(
        paths, study, space, ch_names, selection)

    # prepare data
    # ------------
    # select regions, average frequencies, compute asymmetry
    psd, freq, ch_names = format_psds(
        psds, freq, info=info, freq_range=freq_range, average_freq=avg_freq,
        selection=selection, transform=transform, div_by_sum=div_by_sum,
        src=src, subjects_dir=subjects_dir, subject=subject)

    if scale_psd:
        # z-score across subjects
        from scipy.stats import zscore
        psd = zscore(psd, axis=0)

    # construct adjacency matrix for clustering
    adjacency = (_get_adjacency(paths, study, space, ch_names, selection, src)
                 if 'pairs' not in selection else None)

    # group psds by chosen contrast
    grp = group_bdi(subj_id, bdi, method=contrast, full_table=confounds)

    # TODO: put this in utils.group_data
    # create contrast-specific variables
    if not confounds:
        if 'reg' not in contrast:
            hi, lo = psd[grp['high']], psd[grp['low']]
            stat_info.update(dict(N_low=lo.shape[0], N_high=hi.shape[0],
                                  N_all=lo.shape[0] + hi.shape[0]))
            data['hi'], data['lo'] = hi, lo
        else:
            bdi = grp['bdi']
            hilo = psd[grp['selection']]
            stat_info['N_all'] = grp['selection'].sum()
            data['hilo'], data['bdi'] = hilo, bdi
    else:
        # we perform linear regression, where predictor of interest
        # depends on the chosen contrast
        hilo = psd[grp['selection']]

        # we include confounds in the regression analysis
        bdi, hilo = recode_variables(grp['beh'], data=hilo,
                                     use_bdi='reg' in contrast,
                                     interaction=interaction)
        # calculate N
        N_all = bdi.shape[0]
        stat_info['N_all'] = N_all
        if 'reg' not in contrast:
            N_hi = (bdi['DIAGNOZA'] > 0).sum()
            stat_info['N_high'] = N_hi
            stat_info['N_low'] = N_all - N_hi
        subj_id = bdi.index.values
        bdi = bdi.values

        data['hilo'], data['bdi'] = hilo, bdi

    # pick info
    # FIX - this could be done not in return_data, but before - for clusters
    if not space == 'src':
        if 'asy' in selection:
            this_ch_names = [ch.split('-')[1] for ch in ch_names]
            info = info.copy().pick_channels(this_ch_names, ordered=True)
        else:
            info = info.copy().pick_channels(ch_names, ordered=True)
    else:
        info = None

    data.update(dict(freq=freq, ch_names=ch_names, src=src, info=info,
                     subj_id=subj_id, subject=subject, stat_info=stat_info,
                     subjects_dir=subjects_dir, adjacency=adjacency))
    return data


def _get_space_info(paths, study, space, ch_names, selection):
    '''Helper function for ``prepare_data``. Returns relevant source/channel
    space along with additional variables.'''
    if not space == 'src':
        info = paths.get_data('info', study=study)
        src, subjects_dir, subject = None, None, None

        # check that channel order agrees between psds and info
        msg = 'Channel order does not agree between psds and info object.'
        assert (np.array(ch_names) == np.array(info['ch_names'])).all(), msg
    else:
        # read fwd, get subjects_dir and subject
        subjects_dir = paths.get_path('subjects_dir')
        info = None
        if 'asy' in selection:
            src = paths.get_data('src_sym')
            subject = 'fsaverage_sym'
        else:
            subject = 'fsaverage'
            src = paths.get_data('fwd', study=study)['src']

    return info, src, subject, subjects_dir


def _get_adjacency(paths, study, space, ch_names, selection, src):
    '''Helper function for ``prepare_data``. Returns adjacency for given study
    and space.'''
    if not space == 'src':
        # use right-side channels in adjacency if we calculate asymmetry
        if 'asy' in selection:
            ch_names = [ch.split('-')[1] for ch in ch_names]
        neighbours = paths.get_data('neighbours', study=study)
        adjacency = construct_adjacency_matrix(
            neighbours, ch_names, as_sparse=True)
    else:
        import mne
        try:
            adjacency = mne.spatial_src_connectivity(src)
        except AttributeError:
            adjacency = mne.spatial_src_adjacency(src)

        if not selection == 'all':
            if isinstance(ch_names, dict):
                from .src import _to_data_vert
                data_vert = _to_data_vert(src, ch_names)
            else:
                data_vert = ch_names
            idx1, idx2 = np.ix_(data_vert, data_vert)
            adjacency = sparse.coo_matrix(
                adjacency.toarray()[idx1, idx2])
    return adjacency


def load_chanord(paths, study=None, **kwargs):
    '''Load channel order.'''
    ch_names_dir = paths.get_path('chanpos', study=study)
    with open(op.join(ch_names_dir, 'channel_order.txt')) as f:
        txt = f.read()
    return txt.split('\t')


def load_info(paths, study=None, **kwargs):
    from borsar.utils import silent_mne
    chanpos_dir = paths.get_path('chanpos', study=study)
    with silent_mne(full_silence=True):
        raw = mne.io.read_raw_fif(op.join(chanpos_dir, 'has_info_raw.fif'))
    return raw.info


# FIXME - make a general purpose load_neighbours function in borsar
def load_neighbours(paths, study=None, **kwargs):
    chanlocs_dir = paths.get_path('chanpos', study=study)
    if study in ['A', 'D', 'E']:
        from mne.externals import h5io
        full_fname = op.join(chanlocs_dir, 'neighbours.hdf5')
        return h5io.read_hdf5(full_fname)
    else:
        from scipy.io import loadmat
        full_fname = op.join(chanlocs_dir, 'neighbours.mat')
        return loadmat(full_fname, squeeze_me=True)['neighbours']


def load_GH(paths, study=None, **kwargs):
    from scipy.io import loadmat
    chanlocs_dir = paths.get_path('chanpos', study=study)
    data = loadmat(op.join(chanlocs_dir, 'GH.mat'))
    return data['G'], data['H']


def load_forward(paths, study=None, **kwargs):
    fwd_dir = paths.get_path('src', study=study)
    fname = op.join(fwd_dir, 'DiamSar-fsaverage-oct-6-fwd.fif')
    return mne.read_forward_solution(fname, verbose=False)


def load_src_sym(paths, **kwargs):
    src_dir = paths.get_path('src')
    full_path = op.join(src_dir, 'DiamSar-fsaverage_sym-oct-6-src.fif')
    src_sym = mne.read_source_spaces(full_path, verbose=False)
    return src_sym


def load_psd(path, study='C', eyes='closed', space='avg',
             winlen=2., step=0.5, reg='.+', weight_norm='.+',
             task=None):
    '''
    Load power spectrum density for given analysis.
    '''
    from scipy.io import loadmat

    prefix = 'psd_study-{}_eyes-{}_space-{}'.format(study, eyes, space)
    if space in ['avg', 'csd']:
        prefix = prefix + '_winlen-{}_step-{}'.format(winlen, step)
    elif space == 'src':
        reg_pattern = '_reg-{:.2f}' if not isinstance(reg, str) else '_reg-{}'
        prefix = prefix + (reg_pattern + 'weightnorm-{}')
        prefix = prefix.format(reg, weight_norm)

    # all psds are in C directory for convenience
    study_dir = path.get_path('main', study='C')
    psd_dir = op.join(study_dir, 'analysis', 'psd')
    files_with_prefix = [f for f in os.listdir(psd_dir)
                         if (f.endswith('.mat') or f.endswith('.hdf5'))
                         and len(re.findall(prefix, f)) > 0]

    num_good_files = len(files_with_prefix)
    if num_good_files == 0:
        msg = 'Could not find file matching prefix: {}'.format(prefix)
        raise FileNotFoundError(msg)

    if num_good_files > 1:
        from warnings import warn
        fls = ', '.join(files_with_prefix)
        warn('More than one psd file matching specified criteria: {}.'
             'Loading the first of the matching files.'.format(fls))

    fname = files_with_prefix[0]
    if fname.endswith('.mat'):
        psds_mat = loadmat(op.join(psd_dir, fname))

        # cleanup data
        if space == 'src':
            keys = ['psd', 'freq', 'subject_id']
            psds, *rest = [psds_mat[k] for k in keys]
            freq, subj_id = [x.ravel() for x in rest]
            return psds, freq, None, subj_id
        else:
            keys = ['psd', 'freq', 'ch_names', 'subj_id']
            psds, *rest = [psds_mat[k] for k in keys]
            freq, ch_names, subj_id = [x.ravel() for x in rest]
            ch_names = [ch.replace(' ', '') for ch in ch_names]
            return psds, freq, ch_names, subj_id
    elif fname.endswith('.hdf5') and space == 'src':
        from mne.externals import h5io
        temp = h5io.read_hdf5(op.join(psd_dir, fname))
        if isinstance(temp['subject_id'], list):
            temp['subject_id'] = np.array(temp['subject_id'])
        return temp['psd'], temp['freq'], None, temp['subject_id']


def read_linord(paths):
    '''Read linear order behavioral aggregated files.

    This function is not used in "Three times NO" paper.
    '''
    root_dir = paths.get_path('main', study='C')
    beh_dir = op.join(root_dir, 'bazy')
    df = pd.read_excel(op.join(beh_dir, 'transitive.xls'))
    sel_cols = ['easy_0', 'easy_1', 'easy_2', 'difficult_0', 'difficult_1',
                'difficult_2']

    linord = df[sel_cols]
    return linord


def set_or_join_annot(raw, annot):
    '''Add annotations to the raw object or add them to already present
    annotations.'''
    if raw.annotations is not None:
        # join annotations sorting onsets
        # (temporary fix for mne not sorting joined annotations)
        full_annot = raw.annotations + annot
        sorting = np.argsort(full_annot.onset)
        annot = mne.Annotations(full_annot.onset[sorting],
                                full_annot.duration[sorting],
                                full_annot.description[sorting])
    try:
        raw.set_annotations(annot)
    except AttributeError:
        # make sure it works with mne 0.16
        raw.annotations = annot


def warnings_to_ignore_when_reading_files():
    '''List of warnings to ignore when reading files.'''
    ignore_msg = [(r"The following EEG sensors did not have a position "
                   "specified in the selected montage: ['oko']. Their"
                   " position has been left untouched."),
                  (r"Limited [0-9]+ annotation\(s\) that were expanding "
                   "outside the data range."),
                  "invalid value encountered in less",
                  "invalid value encountered in greater",
                  ("The data contains 'boundary' events, indicating data "
                   "discontinuities. Be cautious of filtering and epoching "
                   "around these events.")]
    return ignore_msg
