import re
import os
import os.path as op
import warnings
import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import mne


# load functions
# --------------
# - [ ] add load montage?

# FIXME - use in register mode
def read_bdi(paths, study='C', **kwargs):
    '''Read BDI scores.'''
    full_table = kwargs.get('full_table', False)
    base_dir = paths.get_path('main', study=study)
    beh_dir = op.join(base_dir, 'beh')
    if study == 'A':
        df = pd.read_excel(op.join(beh_dir, 'baza minimal.xlsx'))
        select_col = ['ID', 'BDI_k', 'DIAGNOZA']
        rename_col = {'BDI_k': 'BDI-I'}

        if full_table:
            select_col += ['plec_k', 'wiek']
            rename_col['plec_k'] = 'PŁEĆ'

        bdi = df[select_col]
        bdi = make_sure_diagnosis_is_boolean(bdi)
        bdi = bdi.rename(columns=rename_col)

        if full_table:
            relabel = {1: 'KOBIETA', 2: 'MĘŻCZYZNA'}
            bdi.loc[:, 'PŁEĆ'] = df.PŁEĆ.replace(relabel)

    if study == 'B':
        # FIXME: B has weird folder structure
        beh_dir = op.join(base_dir, 'porządki liniowe dźwiekowe + rest', 'beh')
        if full_table:
            bdi = pd.read_excel(op.join(beh_dir, 'BAZA DANYCH.xlsx'))
            sel_col = ['BDI 2-pomiar wynik', 'wykształcenie', 'wiek', 'płeć']
            # has also 'problemy ze snem', 'miasto'
            bdi = bdi[sel_col]
            # ! check ID with 'BDI.xlsx' !
        else:
            bdi = pd.read_excel(op.join(beh_dir, 'BDI.xlsx'), header=None,
                                names=['ID', 'BDI-I'])
        bdi.loc[:, 'DIAGNOZA'] = False
    if study == 'C':
        df = pd.read_excel(op.join(beh_dir, 'BAZA_DANYCH.xlsx'))
        if full_table:
            bdi = study_C_reformat_beh_table(df)
        else:
            bdi = df[['ID', 'BDI-II', 'DIAGNOZA']]

        bdi = make_sure_diagnosis_is_boolean(bdi)

    return bdi.set_index('ID')


def study_C_reformat_beh_table(df):
    '''Select and recode relevant columns from behavioral table.'''
    # select relevant columns
    df = df[['ID', 'DATA BADANIA', 'WIEK', 'PŁEĆ', 'WYKSZTAŁCENIE',
             'DIAGNOZA', 'BDI-II']]
    # fix dates
    df.loc[0, 'DATA BADANIA'] = df.loc[1, 'DATA BADANIA']
    df.loc[7, 'WIEK'] = datetime.datetime(df.loc[7, 'WIEK'], 6, 25)
    df.loc[21, 'WIEK'] = datetime.datetime(df.loc[21, 'WIEK'], 6, 25)

    # age
    # ---
    # calculate age
    for idx in df.index:
        delta = relativedelta(df.loc[idx, 'DATA BADANIA'], df.loc[idx, 'WIEK'])
        df.loc[idx, 'wiek'] = delta.years

    # one bad birth date (the same as study date) - use average
    # student age (participant was a student)
    avg_student_age = df.query('WYKSZTAŁCENIE == "Student"').wiek.mean()
    df.loc[9, 'wiek'] = int(avg_student_age)

    # education
    # ---------
    # fix one missing education - insert most common answer
    df.loc[85, 'WYKSZTAŁCENIE'] = 'Średnie'

    # remove 'DATA BADANIA' and 'WIEK'
    bdi = df.drop(['DATA BADANIA', 'WIEK'], axis='columns')


def make_sure_diagnosis_is_boolean(bdi):
    # silence false alarm SettingWithCopyWarnings:
    with warnings.catch_warnings():
        irritating_warning = pd.core.common.SettingWithCopyWarning
        warnings.simplefilter('ignore', irritating_warning)
        bdi.loc[:, 'DIAGNOZA'] = bdi.DIAGNOZA.astype('bool')
    return bdi


def load_chanord(paths, study=None, **kwargs):
    '''Load channel order.'''
    ch_names_dir = paths.get_path('chanpos', study=study)
    with open(op.join(ch_names_dir, 'channel_order.txt')) as f:
        txt = f.read()
    return txt.split('\t')


# FIXME - save proper info via borsar.write_info for all studies and simplify
#         this function
def load_info(paths, study=None, **kwargs):
    chanpos_dir = paths.get_path('chanpos', study=study)
    raw = mne.io.read_raw_fif(op.join(chanpos_dir, 'has_info_raw.fif'))
    return raw.info


# FIXME - make a general purpose load_neighbours function in borsar
def load_neighbours(paths, study=None, **kwargs):
    chanlocs_dir = paths.get_path('chanpos', study=study)
    if study == 'A':
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
    import mne
    if study == 'C':
        fwd_dir = paths.get_path('fwd')
        return mne.read_forward_solution(
            op.join(fwd_dir, 'DiamSar-eeg-oct-6-fwd.fif'), verbose=False)
    elif study == 'A':
        fwd_dir = op.join(paths.get_path('eeg', study='A'), 'src')
        return mne.read_forward_solution(
            op.join(fwd_dir, 'DiamSar-fsaverage-oct-6-fwd.fif'), verbose=False)


def load_src_sym(paths, **kwargs):
    import mne
    src_dir = paths.get_path('fwd')
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
    '''Read linear order behavioral aggregated files.'''
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
