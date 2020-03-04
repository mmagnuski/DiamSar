import numpy as np


# define DiamSar colors
# ---------------------
colors = dict(diag=np.array([100, 149, 237]) / 255,
              hc=np.array([255, 0 , 102]) / 255)
col = np.stack([colors['hc'], colors['diag']], axis=0)
colors['subdiag'] = np.mean(col, axis=0)

graycol = np.array([0.65] * 3)
hc_sub_mid = np.stack([colors['hc'], colors['subdiag']], axis=0).mean(axis=0)
hc_sub_mid = np.average(np.stack([graycol, hc_sub_mid], axis=0),
                        axis=0, weights=[0.7, 0.3])
colors['mid'] = hc_sub_mid
colors['gray'] = graycol
del hc_sub_mid, graycol, col


# - [ ] move to borsar, try using autonotebook if not str
# - [ ] later allow for tqdm progressbar as first arg
def progressbar(progressbar, total=None):
    if progressbar and not progressbar == 'text':
        from tqdm import tqdm_notebook
        pbar = tqdm_notebook(total=total)
    elif progressbar == 'text':
        from tqdm import tqdm
        pbar = tqdm(total=total)
    else:
        pbar = EmptyProgressbar(total=total)
    return pbar


class EmptyProgressbar(object):
    def __init__(self, total=None):
        self.total = total

    def update(self, val):
        pass


# FIXME - add more tests
def group_bdi(subj_id, bdi, method='cvsc', lower_threshold=None,
              higher_threshold=None):
    '''Select and group subjects according to a desired contrast.

    Parameters
    ----------
    method : string
        There are five possible methods available:
        * 'cvsc'  - contrast high and low-BDI controls
        * 'cvsd'  - contrast low-BDI controls and diagnosed subjects
        * 'creg'  - regression on BDI scores, limited to controls
        * 'dreg'  - regressnion on BDI scores, limited to diagnosed
        * 'cdreg' - regression on BDI scores irrespective of diagnosis
    lower_threshold : float or string
        Allows to override the default lower_threshold setting for given
        method. Subjects with scores <= lower_threshold are selected
        for the lower group.
        The default lower threshold is ``5``.
    higher_threshold : float or string
        Allows to override the default higher_threshold setting for given
        method. Subjects with scores > higher_threshold are selected for the
        higher group.
        The default higher threshold is ``0`` for 'cvsd' contrast and ``10``
        for 'cvsc' contrast.

    Returns
    -------
    grouping : dict
        Dictionary with keys depending on `method`. Key 'selection' is always
        present and contains boolean vector of subjects selected from `subj_id`
        list. If `method` was 'cvsc' or 'cvsd' the remaining keys are:
        * 'low': indices of `subj_id` selected for the high-BDI group
        * 'high': indices of `subj_id` selected for the low-BDI group
        If `method` was 'cvsc' or 'cvsd' the keys are:
        * 'bdi': BDI values for consecutive subjects in `subj_id` selected by
        `grouping['selection']` (that is `subj_id[grouping['selection']]`)
    '''
    # select only those subjects that have eeg and are present
    # in the behavioral database
    has_subj = np.in1d(subj_id, bdi.index)
    has_subj_idx = np.where(has_subj)[0]
    sid = subj_id[has_subj]
    bdi = bdi.loc[sid, :]
    bdi_col = 'BDI-II' if 'BDI-II' in bdi.columns else 'BDI-I'

    if method == 'cvsc':
        bdi_sel = bdi.loc[~bdi.DIAGNOZA, bdi_col].values
        lower_threshold = _check_threshold(lower_threshold, bdi_sel, 5)
        higher_threshold = _check_threshold(higher_threshold, bdi_sel, 10)

        selection_low = ~bdi.DIAGNOZA & (bdi[bdi_col] <= lower_threshold)
        selection_high = ~bdi.DIAGNOZA & (bdi[bdi_col] > higher_threshold)

    elif method == 'cvsd':
        lower_threshold = 5 if lower_threshold is None else lower_threshold
        higher_threshold = 0 if higher_threshold is None else higher_threshold

        selection_low = ~bdi.DIAGNOZA & (bdi[bdi_col] <= lower_threshold)
        selection_high = bdi.DIAGNOZA & (bdi[bdi_col] > higher_threshold)
    elif 'reg' in method:
        selected = np.zeros(len(sid), dtype='bool')
        if 'c' in method:
            selected = ~bdi.DIAGNOZA.values
        if 'd' in method:
            selected = selected | bdi.DIAGNOZA.values
        bdi = bdi.loc[selected, bdi_col].values

    if 'reg' not in method:
        selected = selection_low | selection_high

    selection = np.zeros(len(subj_id), dtype='bool')
    selection[has_subj_idx] = selected

    if 'reg' not in method:
        grouping = dict(selection=selection, low=has_subj_idx[selection_low],
                        high=has_subj_idx[selection_high])
    else:
        grouping = dict(selection=selection, bdi=bdi)
    return grouping


def _check_threshold(thresh, values, default):
    thresh = default if thresh is None else thresh
    if isinstance(thresh, str) and thresh[-1] == '%':
        thresh = np.percentile(values, int(thresh[:-1]))
    return thresh


def reformat_stat_table(tbl):
    '''Change format from old (studies as A, B, C letters, contrasts as
    for example ``'cvsd'``) to the one used in the final paper (studies
    I, II, III, contrasts ``'DvsHC'``).'''
    # contrast, study, space
    # ----------------------
    columns = tbl.columns.tolist()
    has_clusters = 'min cluster p' in columns

    firstcols = ['contrast', 'study', 'space']
    for col in firstcols:
        if col in columns:
            columns.remove(col)

    col_ord = firstcols + columns[2:]

    tbl2 = tbl.loc[:, col_ord]

    # N
    # -
    tbl2 = tbl2.rename(columns={'N_all': 'N'})

    for idx in tbl2.index:
        if 'reg' in tbl2.loc[idx, 'contrast']:
            tbl2.loc[idx, 'N'] = str(tbl.loc[idx, 'N_all'])
        else:
            n_l, n_h = tbl.loc[idx, ['N_low', 'N_high']]
            tbl2.loc[idx, 'N'] = '{:0.0f} vs {:0.0f}'.format(n_h, n_l)

    # sorting
    # -------
    tbl2 = tbl2.sort_values(['contrast', 'study', 'space'])
    contrast_ord = ['cvsd', 'cvsc', 'cdreg', 'dreg', 'creg']

    all_idx = list()
    for con in contrast_ord:
        idx = tbl2.query('contrast == "{}"'.format(con)).index.tolist()
        all_idx.extend(idx)

    tbl2 = tbl2.loc[all_idx, :].reset_index(drop=True)

    # translate contrasts, other enh
    # ------------------------------
    study_trsl = {'A': 'I', 'B': 'II', 'C': 'III'}
    con_trsl = {'cvsc': 'SvsHC', 'cvsd': 'DvsHC', 'dreg': 'DReg',
                'cdreg': 'allReg', 'creg': 'nonDReg'}

    for idx in tbl2.index:
        # translate contrast
        new_con = con_trsl[tbl2.loc[idx, 'contrast']]
        tbl2.loc[idx, 'contrast'] = new_con

        # rename study A -> I etc.
        tbl2.loc[idx, 'study'] = study_trsl[tbl2.loc[idx, 'study']]

        # nan -> NA
        if has_clusters:
            for col in ['min cluster p', 'largest cluster size']:
                if np.isnan(tbl2.loc[idx, col]):
                    tbl2.loc[idx, col] = 'NA'

            # round t and p vals
            for col in ['min t', 'max t', 'min cluster p']:
                val = tbl2.loc[idx, col]
                if not isinstance(val, str):
                    tbl2.loc[idx, col] = '{:.3f}'.format(val)
        else:
            # for pairs
            for col in ['t 1', 'p 1', 't 2', 'p 2']:
                val = tbl2.loc[idx, col]
                if not isinstance(val, str):
                    tbl2.loc[idx, col] = '{:.3f}'.format(val)

    return tbl2
