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
        # check mne's progressbar...
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
