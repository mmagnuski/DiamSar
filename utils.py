import numpy as np
import pandas as pd
from sarna.utils import progressbar


# define DiamSar colors
# ---------------------
colors = dict(diag=np.array([100, 149, 237]) / 255,
              hc=np.array([255, 0, 102]) / 255)
col = np.stack([colors['hc'], colors['diag']], axis=0)
colors['subdiag'] = np.mean(col, axis=0)

graycol = np.array([0.65] * 3)
hc_sub_mid = np.stack([colors['hc'], colors['subdiag']], axis=0).mean(axis=0)
hc_sub_mid = np.average(np.stack([graycol, hc_sub_mid], axis=0),
                        axis=0, weights=[0.7, 0.3])
colors['mid'] = hc_sub_mid
colors['gray'] = graycol
del hc_sub_mid, graycol, col

translate_study = dict(A='I', B='II', C='III', D='IV', E='V')
translate_contrast = {'cvsc': 'SvsHC', 'cvsd': 'DvsHC', 'dreg': 'DReg',
                      'cdreg': 'allReg', 'creg': 'nonDReg'}


# FIXME - add more tests
def group_bdi(subj_id, bdi, method='cvsc', lower_threshold=None,
              higher_threshold=None, full_table=False):
    '''Select and group subjects according to a desired contrast.

    Parameters
    ----------
    subj_id : listlike of int
        Identifiers of the subjects to choose. Some subjects are rejected
        so we want to select a subsample of all subjects.
    bdi : pandas DataFrame
        Dataframe with columns specifying BDI (either ``BDI-I`` or ``BDI-II``)
        and diagnosis status (``DIAGNOZA`` - boolean). The rows should be
        indexed with subject identifiers.
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
    full_table : bool
        Whether to include full behavioral table in the output. This is used
        to include potential confounds in the analysis (for example sex, age
        and education). This option works only when full behavioral data
        are available (``paths.get_data('bdi', full_table=True)``), so it may
        not work for some of the datasets I - III hosted on Dryad (Dryad does
        not allow more than 3 variables per participant to avoid the
        possibility of individual identification).

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
    bdi_col = [col for col in ['BDI-II', 'BDI-I', 'PHQ-9']
               if col in bdi.columns][0]

    if method not in ['cvsd', 'cvsc', 'dreg', 'creg', 'cdreg']:
        raise ValueError('Unexpected method: {}'.format(method))

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
    if full_table:
        if method == 'cvsc':
            # we add 'fake' diagnosis grouping variable so that further
            # steps of confound regression analysis of cvsc contrast work well
            bdi.loc[:, 'DIAGNOZA'] = False
            bdi.loc[selection_high, 'DIAGNOZA'] = True
        grouping['beh'] = bdi.loc[selected, :]
    return grouping


def _check_threshold(thresh, values, default):
    thresh = default if thresh is None else thresh
    if isinstance(thresh, str) and thresh[-1] == '%':
        thresh = np.percentile(values, int(thresh[:-1]))
    return thresh


def recode_variables(beh, use_bdi=False, data=None, warn_prop_missing=0.05):
    '''Recode and rescale variables in full behavioral dataframes for
    regression analyses that take confounds into account.
    Scaling continuous variables by 2SDs and leaving dummy variables intact
    has been advocated by Gelman (2008), but specifically in the context of
    interpretting regression coefficients. In our analyses we don't
    interpret the coefficients and often want the intercept to refer to
    the grand mean (not to the mean of a specific group). Therefore we
    standardize continuous variables (age, education) and center dichotomous/
    dummy variables (sex, diagnosis).

    Parameters
    ----------
    beh : pandas DataFrame
        Full behavioral data.
    use_bdi : bool
        Whether the predictor of interest (and thus the last column in the
        output) should be BDI/PHQ-9 (``use_bdi=True``) or diagnosis status.
        The default is ``False``.
    data : numpy array
        Data array to align with the behavioral data. The first step when
        recoding the variables is to reject participants with missing data.
        This necessitates removal of corresponding rows from electro-
        physiological data. Data passed to this argument will be also returned
        as the second output of the function.
    warn_prop_missing : float
        Raise a warning higher proportion of rows are removed from the data.

    Returns
    -------
    beh : pandas DataFrame
        Recoded behavioral dataframe.
    data : numpy array
        If data argument was used, then the second output is the data array
        aligned with the behavioral data.
    '''
    from scipy.stats import zscore

    sel_cols = ['sex', 'age', 'education']
    if use_bdi:
        bdi_col = [col for col in ['BDI-I', 'BDI-II', 'PHQ-9']
                   if col in beh.columns][0]
        sel_cols.append(bdi_col)
    else:
        sel_cols.append('DIAGNOZA')

    beh = beh.copy()

    # remove participants with missing data
    has_missing = beh.isnull().any(axis=1)
    beh = beh.loc[~has_missing, :]

    # make sure biological data are also pruned
    if data is not None:
        assert data.shape[0] == beh.shape[0]
        data = data[~has_missing]

    # warn if too many missing
    prop_missing = has_missing.mean()
    if prop_missing > warn_prop_missing:
        from warnings import warn
        msg = '{:.1f}% of cases removed due to missing data.'
        warn(msg.format(prop_missing * 100))

    # recode sex to zeros and ones, then center
    sx = beh.sex.replace({'female': 1, 'male': 0})
    beh.loc[:, 'sex'] = sx - sx.mean()

    # center diagnosis
    if not use_bdi:
        diag = beh.loc[:, 'DIAGNOZA']
        beh.loc[:, 'DIAGNOZA'] = diag - diag.mean()

    # standardize age and education
    beh.loc[:, 'age'] = zscore(beh.loc[:, 'age'])
    if 'education' in beh.columns:
        if beh.education.dtype == 'O':
            # Wronski study, we construct dummy codes for education
            beh.loc[:, 'lic'] = beh.education == 'licencjat'
            beh.loc[:, 'mgr'] = ((beh.education == 'magisterskie')
                                 | (beh.education == 'podyplomowe'))

            beh.loc[:, 'lic'] = beh.loc[:, 'lic'] - beh.loc[:, 'lic'].mean()
            beh.loc[:, 'mgr'] = beh.loc[:, 'mgr'] - beh.loc[:, 'mgr'].mean()

            sel_cols = sel_cols[:2] + ['lic', 'mgr'] + sel_cols[3:]
        else:
            beh.loc[:, 'education'] = zscore(beh.loc[:, 'education'])
    else:
        sel_cols.remove('education')

    # standardize bdi
    if use_bdi:
        beh.loc[:, bdi_col] = zscore(beh.loc[:, bdi_col])

    if data is None:
        return beh.loc[:, sel_cols]
    else:
        return beh.loc[:, sel_cols], data


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
    for idx in tbl2.index:
        # translate contrast
        new_con = translate_contrast[tbl2.loc[idx, 'contrast']]
        tbl2.loc[idx, 'contrast'] = new_con

        # rename study A -> I etc.
        tbl2.loc[idx, 'study'] = translate_study[tbl2.loc[idx, 'study']]

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
            # for channel pairs
            cols = ['t', 'p', 'es', 'ci']
            cols = [c + ' 1' for c in cols] + [c + ' 2' for c in cols]
            for col in cols:
                val = tbl2.loc[idx, col]
                if not isinstance(val, str):
                    if 'ci' in col:
                        v1, v2 = val
                        tbl2.loc[idx, col] = '[{:.3f}, {:.3f}]'.format(v1, v2)
                    else:
                        tbl2.loc[idx, col] = '{:.3f}'.format(val)

    return tbl2


def psd_to_df(data1, data2):
    '''Create a dataframe out of psd_high, psd_low data.'''
    nx, ny = data1.shape[0], data2.shape[0]
    data = np.concatenate([data1, data2], axis=0)
    labels = ['diagnosed'] * nx + ['controls'] * ny
    df = pd.DataFrame(data={'FAA': data, 'group': labels})
    return df


def select_channels_special(info, selection):
    '''Handles special case of 'asy_pairs' channel selection, otherwise uses
    ``borsar.select_channels``.'''
    from borsar.channels import find_channels

    if not selection == 'asy_pairs':
        # normal borsar selection
        selection = select_channels(info, selection)
    else:
        # special DiamSar case of left-right symmetric pairs
        pairs = dict(left=['F3', 'F7'], right=['F4', 'F8'])
        selection = {k: find_channels(info, pairs[k]) for k in pairs.keys()}

        if any(idx is None for idx in selection['left'] + selection['right']):
            # 10-20 names not found, try EGI channels
            if 'E128' in info['ch_names']:
                # 128 EGI cap (MODMA)
                pairs = {'left': ['E24', 'E33'], 'right': ['E124', 'E122']}
            else:
                # 64 EGI cap (Nowowiejska)
                pairs = {'left': ['E12', 'E18'], 'right': ['E60', 'E58']}
            selection = {k: find_channels(info, pairs[k])
                         for k in pairs.keys()}

        # check if any channels are missing
        ch_list = selection['left'] + selection['right']
        channel_not_found = [idx is None for idx in ch_list]
        if any(channel_not_found):
            pairs_list = pairs['left'] + pairs['right']
            not_found = [ch for idx, ch in enumerate(pairs_list)
                         if channel_not_found[idx]]
            msg = 'The following channels were not found: {}.'
            raise ValueError(msg.format(', '.join(not_found)))
    return selection


def select_channels(inst, select='all'):
    '''
    Gives indices of channels selected by a text keyword.

    Parameters
    ----------
    inst : mne Raw | mne Epochs | mne Evoked | mne TFR | mne Info
        Mne object with `ch_names` and `info` attributes or just the mne Info
        object.
    select : str
        Can be 'all' or 'frontal'. If 'asy_' is prepended to the
        select string then selected channels are grouped by mirror positions
        on the x axis (left vs right).

    Returns
    -------
    selection : numpy int array or dict of numpy int arrays
        Indices of the selected channels. If 'asy_' was in the select string
        then selection is a dictionary of indices, where selection['left']
        gives channels on the left side of the scalp and selection['right']
        gives right-side homologues of the channels in selection['left'].
    '''
    from borsar.channels import get_ch_pos, get_ch_names

    if select == 'all':
        return np.arange(len(get_ch_names(inst)))
    elif 'asy' in select and 'all' in select:
        return homologous_pairs(inst)

    if 'frontal' in select:
        # compute radius as median distance to head center: the (0, 0, 0) point
        ch_pos = get_ch_pos(inst)
        dist = np.linalg.norm(ch_pos - np.array([[0, 0, 0]]), axis=1)
        median_dist = np.median(dist)
        frontal = ch_pos[:, 1] > 0.1 * median_dist
        not_too_low = ch_pos[:, 2] > -0.6 * median_dist
        frontal_idx = np.where(frontal & not_too_low)[0]
        if 'asy' in select:
            hmlg = homologous_pairs(inst)
            sel = np.in1d(hmlg['left'], frontal_idx)
            return {side: hmlg[side][sel] for side in ['left', 'right']}
        else:
            return frontal_idx


def homologous_pairs(inst):
    '''
    Construct homologous channel pairs based on channel names or positions.

    Parameters
    ----------
    inst : mne object instance
        Mne object like mne.Raw or mne.Epochs.

    Returns
    -------
    selection: dict of {str -> list of int} mappings
        Dictionary mapping hemisphere ('left' or 'right') to array of channel
        indices.
    '''
    from borsar.channels import get_ch_pos, get_ch_names

    ch_names = get_ch_names(inst)
    ch_pos = get_ch_pos(inst)

    labels = ['right', 'left']
    selection = {label: list() for label in labels}
    has_1020_names = 'Cz' in ch_names and 'F3' in ch_names

    if has_1020_names:
        # find homologues by channel names
        left_chans = ch_pos[:, 0] < 0
        y_ord = np.argsort(ch_pos[left_chans, 1])[::-1]
        check_chans = [ch for ch in list(np.array(ch_names)[left_chans][y_ord])
                       if 'z' not in ch]

        for ch in check_chans:
            chan_base = ''.join([char for char in ch if not char.isdigit()])
            chan_value = int(''.join([char for char in ch if char.isdigit()]))

            if (chan_value % 2) == 1:
                # sometimes homologous channels are missing in the cap
                homologous_ch = chan_base + str(chan_value + 1)
                if homologous_ch in ch_names:
                    selection['left'].append(ch_names.index(ch))
                    selection['right'].append(ch_names.index(homologous_ch))
    else:
        # channel names do not come from 10-20 system
        # constructing homologues from channel position
        # (this will not work well for digitized channel positions)
        from mne.bem import _fit_sphere

        # fit sphere to channel positions and calculate median distance
        # of the channels to the sphere origin
        radius, origin = _fit_sphere(ch_pos)
        origin_distance = ch_pos - origin[np.newaxis, :]
        dist = np.linalg.norm(origin_distance, axis=1)
        median_dist = np.median(dist)

        # find channels on the left from sphere origin
        left_x_val = origin[0] - median_dist * 0.05
        sel = ch_pos[:, 0] < left_x_val
        left_chans = ch_pos[sel, :]
        sel_idx = np.where(sel)[0]

        for idx, pos in enumerate(left_chans):
            # represent channel position with respect to the origin
            this_distance = pos - origin

            # find similar origin-relative position on the right side
            this_distance[0] *= -1
            this_simil = origin_distance - this_distance[np.newaxis, :]
            similar = np.linalg.norm(this_simil, axis=1).argmin()

            # fill selection dictionary
            selection['left'].append(sel_idx[idx])
            selection['right'].append(similar)

    selection['left'] = np.array(selection['left'])
    selection['right'] = np.array(selection['right'])
    return selection
