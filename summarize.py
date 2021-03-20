import os
import os.path as op

import numpy as np
import pandas as pd

from .analysis import esci_indep_cohens_d, load_stat
from . import pth, utils


def summarize_stats_clusters(reduce_columns=True, stat_dir='stats'):
    '''Summarize multiple analyses (saved in analysis dir) in a dataframe.

    Parameters
    ----------
    reduce_columns : bool
        Whether to remove columns with no variability from the output. Defaults
        to ``True``.
    stat_dir : str
        Subdirectory to use (``'stats'``, ``'add1'`` or ``'add2'``, unless
        additional subdirectories were created).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with summarized DiamSar analyses.
    '''
    from mne.externals import h5io

    stat_dir = op.join(pth.paths.get_path('main', 'C'), 'analysis', stat_dir)
    stat_files = [f for f in os.listdir(stat_dir) if f.endswith('.hdf5')]
    n_stat = len(stat_files)

    polarity_to_str = {True: 'pos', False: 'neg'}

    # first, create an empty dataframe
    stat_params = ['study', 'contrast', 'space', 'N_low', 'N_high', 'N_all',
                   'eyes', 'selection', 'freq_range', 'avg_freq', 'transform',
                   'div_by_sum']
    stat_summary = ['min t', 'max t', 'n signif points', 'n clusters',
                    'largest cluster size', 'min cluster p', 'eff dir',
                    'n signif clusters']
    df = pd.DataFrame(index=np.arange(1, n_stat + 1),
                      columns=stat_params + stat_summary)

    row_idx = 0
    for fname in stat_files:
        stat = h5io.read_hdf5(op.join(stat_dir, fname))

        if 'description' not in stat:
            continue

        row_idx += 1
        # fill the basic columns
        for col in stat_params:
            value = (stat['description'][col]
                     if col in stat['description'] else np.nan)
            if isinstance(value, (list, tuple, np.ndarray)):
                value = str(value)

            df.loc[row_idx, col] = value

        # summarize clusters
        n_clst = len(stat['clusters']) if stat['clusters'] is not None else 0

        min_cluster_p = stat['pvals'].min() if n_clst > 0 else np.nan
        n_below_thresh = (stat['pvals'] < 0.05).sum() if n_clst > 0 else 0
        n_signif_points = stat['clusters'].sum() if n_clst > 0 else 0
        if n_clst > 0:
            largest_cluster_size = max([c.sum() for c in stat['clusters']])
            polarity = stat['stat'][stat['clusters'][0]].mean() > 0
            polarity_str = polarity_to_str[polarity]
            largest_cluster_direction = polarity_str
        else:
            largest_cluster_size, largest_cluster_direction = np.nan, np.nan

        df.loc[row_idx, 'min t'] = stat['stat'].min()
        df.loc[row_idx, 'max t'] = stat['stat'].max()
        df.loc[row_idx, 'n signif points'] = n_signif_points
        df.loc[row_idx, 'n clusters'] = n_clst
        df.loc[row_idx, 'min cluster p'] = min_cluster_p
        df.loc[row_idx, 'n signif clusters'] = n_below_thresh
        df.loc[row_idx, 'largest cluster size'] = largest_cluster_size
        df.loc[row_idx, 'eff dir'] = largest_cluster_direction

    # reduce columns
    df = df.loc[:row_idx, :]
    if reduce_columns:
        df = remove_columns_with_no_variability(df)

    return utils.reformat_stat_table(df)


def summarize_stats_pairs(reduce_columns=True, stat_dir='stats',
                          confounds=False, progressbar='text'):
    '''Summarize multiple channel pair analyses (saved in analysis dir) in a
    dataframe.

    This takes longer than ``summarize_stats`` because it adds effect sizes and
    bootstrap confidence intervals for effect sizes.

    Parameters
    ----------
    reduce_columns : bool
        Whether to remove columns with no variability from the output. Defaults
        to ``True``.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with summarized results.
    '''
    from mne.externals import h5io
    from DiamSar.utils import progressbar as pbarobj

    stat_dir = op.join(pth.paths.get_path('main', 'C'), 'analysis', stat_dir)
    stat_files = [f for f in os.listdir(stat_dir) if f.endswith('.hdf5')
                  and 'asy_pairs' in f]
    n_stat = len(stat_files)

    # first, create an empty dataframe
    stat_params = ['study', 'contrast', 'space', 'ch_pair', 'N_low', 'N_high',
                   'N_all', 'eyes', 'selection', 'freq_range', 'avg_freq',
                   'transform', 'div_by_sum']
    stats = ['t', 'p', 'es', 'ci']
    stat_summary = [x + ' 1' for x in stats] + [x + ' 2' for x in stats]
    df = pd.DataFrame(index=np.arange(1, n_stat + 1),
                      columns=stat_params + stat_summary)

    pbar = pbarobj(progressbar, n_stat * 2)
    for idx, fname in enumerate(stat_files, start=1):
        stat = h5io.read_hdf5(op.join(stat_dir, fname))

        for col in stat_params:
            df.loc[idx, col] = stat[col] if col in stat else np.nan

        # for ES and CI we need the original data:
        params = ['study', 'space', 'contrast', 'eyes']
        study, space, contrast, eyes = [stat[param] for param in params]
        data = io.prepare_data(
            pth.paths, study=study, contrast=contrast, eyes=eyes, space=space,
            freq_range=params['freq_range'], avg_freq,
            selection, div_by_sum, transform, confounds, verbose)
        data = run_analysis(selection='asy_pairs', study=study,
                            space=space, contrast=contrast,
                            confounds=confounds, return_data=True)

        # add stats
        for ch_idx in range(2):
            stat_col = ch_idx * 4
            t, p = stat['stat'][ch_idx], stat['pval'][ch_idx]
            df.loc[idx, stat_summary[stat_col]] = t
            df.loc[idx, stat_summary[stat_col + 1]] = p

            if 'vs' in contrast and not confounds:
                psd_high, psd_low, ch_names = (data['hi'], data['lo'],
                                               data['ch_names'])
                esci = esci_indep_cohens_d(psd_high[:, ch_idx],
                                           psd_low[:, ch_idx])
            else:
                psd_sel, bdi_sel, ch_names = (data['hilo'], data['bdi'],
                                              data['ch_names'])

                if 'vs' in contrast:
                    onechan = psd_sel[:, [ch_idx]]
                    beh = data['bdi']
                    grp1 = beh[:, -1] == beh[0, -1]
                    data1 = np.concatenate([onechan[grp1, :], beh[grp1, :]],
                                           axis=1)
                    data2 = np.concatenate([onechan[~grp1, :], beh[~grp1, :]],
                                           axis=1)
                    esci = esci_indep_cohens_d(data1, data2, n_boot=5000,
                                               has_preds=True)
                else:
                    if confounds:
                        beh = data['bdi']
                        esci = esci_regression_r(beh, psd_sel[:, ch_idx])
                    else:
                        esci = esci_regression_r(bdi_sel, psd_sel[:, ch_idx])

            df.loc[idx, stat_summary[stat_col + 2]] = esci['es']
            df.loc[idx, stat_summary[stat_col + 3]] = esci['ci']
            pbar.update(1)

    pbar.close()

    # split into two dfs
    pair_rows = df['selection'].str.contains('pairs').values
    df = df.loc[pair_rows, :].reset_index(drop=True)

    df = remove_columns_with_no_variability(df)
    df = utils.reformat_stat_table(df)

    return df


def remove_columns_with_no_variability(df):
    '''Remove dataframe columns with no variability.'''
    assert isinstance(df, pd.DataFrame)

    num_cols = len(df.columns)
    to_remove = list()
    for col_idx in range(num_cols):
        try:
            no_variability = (df.iloc[0, col_idx] == df.iloc[:, col_idx]).all()
            if not no_variability:
                # test for all being null
                no_variability = df.iloc[:, col_idx].isnull().all()
        except ValueError:
            no_variability = False

        to_remove.append(not no_variability)

    df_sel = df.loc[:, to_remove]
    return df_sel
