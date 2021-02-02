import os
import os.path as op
import numpy as np
import pandas as pd
from scipy import sparse

import mne
from borsar.cluster import construct_adjacency_matrix

from . import pth
from . import utils
from .freq import format_psds


def run_analysis(study='C', contrast='cvsd', eyes='closed', space='avg',
                 freq_range=(8, 13), avg_freq=True, selection='frontal_asy',
                 div_by_sum=False, transform='log', n_permutations=10000,
                 cluster_p_threshold=0.05, confounds=False, return_data=False,
                 verbose=True):
    '''Run DiamSar analysis with specified parameters.

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
    n_permutations : int, optional
        Number of permutations to conduct in the cluster-based permutation
        test. ``10000`` by default.
    cluster_p_threshold : float, optional
        Cluster entry p value threshold. ``0.05`` by default.
    confounds : bool
        Whether to include potential confounds in the analysis. These include:
        sex, age and education (if they are available for given study). This
        option works only when full behavioral data are available
        (``paths.get_data('bdi', full_table=True)``), so it may not work for
        some of the datasets I - III hosted on Dryad (Dryad does not allow
        more than 3 variables per participant to avoid the possibility of
        individual identification).
    verbose : bool | int
        Verbosity level supported by mne-python. ``True`` by default.

    Returns
    -------
    clst : borsar.cluster.Clusters | dict
        Result of the analysis. Cluster-based permutation test result object
        in all cases except ``'asy_pairs'`` contrast. ``'asy_pairs'`` does not
        correct for multiple comparisons and a dictionary of results is
        returned for this contrast.
    '''
    # get base study name and setup stat_info dict
    stat_info = dict(avg_freq=avg_freq, freq_range=freq_range,
                     selection=selection, space=space, contrast=contrast,
                     study=study, eyes=eyes, transform=transform,
                     div_by_sum=div_by_sum)

    # load relevant data
    bdi = pth.paths.get_data('bdi', study=study, full_table=confounds)
    psds, freq, ch_names, subj_id = pth.paths.get_data(
        'psd', study=study, eyes=eyes, space=space)

    # select only subjects without NaNs
    no_nans = ~np.any(np.isnan(psds), axis=(1, 2))
    if not np.all(no_nans):
        psds = psds[no_nans]
        subj_id = subj_id[no_nans]

    info, src, subject, subjects_dir = _get_space_info(study, space, ch_names,
                                                       selection)

    # prepare data
    # ------------

    # select regions, average frequencies, compute asymmetry
    psd, freq, ch_names = format_psds(
        psds, freq, info=info, freq_range=freq_range, average_freq=avg_freq,
        selection=selection, transform=transform, div_by_sum=div_by_sum,
        src=src, subjects_dir=subjects_dir, subject=subject)

    # construct adjacency matrix for clustering
    if 'pairs' not in selection:
        adjacency = _get_adjacency(study, space, ch_names, selection, src)

    # put spatial dimension last for cluster-based test
    if not avg_freq or psd.ndim == 3:
        psd = psd.transpose((0, 2, 1))

    # group psds by chosen contrast
    grp = utils.group_bdi(subj_id, bdi, method=contrast, full_table=confounds)

    # create contrast-specific variables
    if not confounds:
        if 'reg' not in contrast:
            hi, lo = psd[grp['high']], psd[grp['low']]
            stat_info.update(dict(N_low=lo.shape[0], N_high=hi.shape[0],
                                  N_all=lo.shape[0] + hi.shape[0]))
        else:
            bdi = grp['bdi']
            hilo = psd[grp['selection']]
            stat_info['N_all'] = grp['selection'].sum()
    else:
        # we perform linear regression, where predictor of interest
        # depends on the chosen contrast
        hilo = psd[grp['selection']]

    # handle confounds
    if confounds:
        # we include confounds in the regression analysis
        bdi, hilo = utils.recode_variables(grp['beh'], data=hilo,
                                           use_bdi='reg' in contrast)
        # calculate N
        N_all = bdi.shape[0]
        stat_info['N_all'] = N_all
        if 'reg' not in contrast:
            N_hi = (bdi['DIAGNOZA'] > 0).sum()
            stat_info['N_high'] = N_hi
            stat_info['N_low'] = N_all - N_hi
        subj_id = bdi.index.values
        bdi = bdi.values

    if return_data:
        data = dict(freq=freq, ch_names=ch_names, bdi=bdi, src=src,
                    subj_id=subj_id, subject=subject,
                    subjects_dir=subjects_dir)
        if confounds or 'reg' in contrast:
            data['hilo'] = hilo
        else:
            data['hi'] = hi
            data['lo'] = lo

        # pick info
        # FIX - this could be done not in return_data, but before - for clusters
        if 'pairs' not in selection:
            if 'asy' in selection:
                this_ch_names = [ch.split('-')[1] for ch in ch_names]
                info = info.copy().pick_channels(this_ch_names, ordered=True)
            else:
                info = info.copy().pick_channels(ch_names, ordered=True)
        else:
            info = ch_names
        data['info'] = info
        return data

    # statistical analysis
    # --------------------
    if 'pairs' not in selection:
        # cluster-based permutation tests for multiple comparisons
        stat_info.update(dict(cluster_p_threshold=cluster_p_threshold))

        if 'vs' in contrast and not confounds:
            # t test in cluster-based permutation test
            from scipy.stats import t
            from sarna.stats import ttest_ind_welch_no_p
            from mne.stats.cluster_level import permutation_cluster_test

            # calculate t test threshold
            df = hi.shape[0] + lo.shape[0] - 2
            threshold = t.ppf(1 - cluster_p_threshold / 2, df)

            # run cluster-based permutation test
            args = dict(threshold=threshold, n_permutations=n_permutations,
                        stat_fun=ttest_ind_welch_no_p, verbose=verbose,
                        out_type='mask')
            try:
                stat, clusters, pval, _ = permutation_cluster_test(
                    [hi, lo], **args, connectivity=adjacency)
            except TypeError:
                stat, clusters, pval, _ = permutation_cluster_test(
                    [hi, lo], **args, adjacency=adjacency)
        else:
            # regression in cluster-based permutation test
            from borsar.cluster import cluster_based_regression

            args = dict(n_permutations=n_permutations, adjacency=adjacency,
                        alpha_threshold=cluster_p_threshold, cluster_pred=-1)

            stat, clusters, pval = cluster_based_regression(hilo, bdi, **args)

        # construct Clusters with stat_info in description
        return _construct_clusters(clusters, pval, stat, space, stat_info,
                                   info, src, subjects_dir, subject,
                                   ch_names, freq)
    else:
        # for selected pairs (two channel pairs) we don't correct for
        # multiple comparisons:
        if 'vs' in contrast and not confounds:
            from scipy.stats import t, ttest_ind
            stat, pval = ttest_ind(hi, lo, equal_var=False)
        else:
            from borsar.stats import compute_regression_t
            # compute regression and ignore intercept:
            stat, pval = compute_regression_t(hilo, bdi, return_p=True)
            stat, pval = stat[-1], pval[-1]

        stat_info.update(dict(stat=stat, pval=pval, ch_names=ch_names))
        return stat_info


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

    # first, create an empty dataframe
    stat_params = ['study', 'contrast', 'space', 'N_low', 'N_high', 'N_all',
                   'eyes', 'selection', 'freq_range', 'avg_freq', 'transform',
                   'div_by_sum']
    stat_summary = ['min t', 'max t', 'n signif points', 'n clusters',
                    'largest cluster size', 'min cluster p',
                    'n signif clusters']
    df = pd.DataFrame(index=np.arange(1, n_stat + 1),
                      columns=stat_params + stat_summary)

    row_idx = 0
    for fname in stat_files:
        stat = h5io.read_hdf5(op.join(stat_dir, fname))

        if 'description' not in stat:
            continue

        row_idx += 1
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
        largest_cluster = (max([c.sum() for c in stat['clusters']])
                           if n_clst > 0 else np.nan)

        df.loc[row_idx, 'min t'] = stat['stat'].min()
        df.loc[row_idx, 'max t'] = stat['stat'].max()
        df.loc[row_idx, 'n signif points'] = n_signif_points
        df.loc[row_idx, 'n clusters'] = n_clst
        df.loc[row_idx, 'min cluster p'] = min_cluster_p
        df.loc[row_idx, 'n signif clusters'] = n_below_thresh
        df.loc[row_idx, 'largest cluster size'] = largest_cluster

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
        params = ['study', 'space', 'contrast']
        study, space, contrast = [stat[param] for param in params]
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


def esci_indep_cohens_d(data1, data2, n_boot=5000, has_preds=False):
    '''Compute Cohen's d effect size and its bootstrap 95% confidence interval.
    (using bias corrected accelerated bootstrap).

    Parameters
    ----------
    data1 : np.ndarray
        One dimensional array of values for the "high" group (for example
        diagnosed participants).
    data2 : np.ndarray
        One dimensional array of values for the "low" group (for example
        healthy controls).
    n_boot : int
        Number of bootstraps to use.
    has_preds : bool
        Wheter array of predictors is provided in the data. If so the first
        column of data1 and data2 are data for separate groups and the
        following columns are the predictors used in regression with the
        predictor of interest (group membership) being the last one
        and the rest treated as confounds.

    Returns
    -------
    stats : dict
        Dictionary of results.
        * ``stats['es']`` contains effect size.
        * ``stats['ci']`` contains 95% confidence interval for the effect size.
        * ``stats['bootstraps']`` contains bootstrap effect size values.
    '''
    if not has_preds:
        assert data2 is not None
        import dabest
        df = utils.psd_to_df(data1, data2)
        dbst_set = dabest.load(df, idx=("controls", "diagnosed"),
                               x="group", y="FAA", resamples=n_boot)
        results = dbst_set.cohens_d.results
        cohen_d = results.difference.values[0]
        cohen_d_ci = (results.bca_low.values[0], results.bca_high.values[0])
        bootstraps = results.bootstraps[0]
    else:
        from borsar.stats import compute_regression_t
        import scikits.bootstrap as boot

        def regression_Cohens_d(data1, data2):
            data = np.concatenate([data1, data2], axis=0)
            preds = data[:, 1:]
            tvals = compute_regression_t(data[:, [0]], preds)
            return d_from_t_categorical(tvals[-1, 0], preds)

        cohen_d = regression_Cohens_d(data1, data2)
        cohen_d_ci, bootstraps = boot.ci((data1, data2), regression_Cohens_d,
                                         multi='independent', n_samples=n_boot,
                                         return_dist=True)
    stats = dict(es=cohen_d, ci=cohen_d_ci, bootstraps=bootstraps)
    return stats


def esci_regression_r(x, y, n_boot=5000):
    '''Compute Pearson's r effect size and its bootstrap 95% confidence
    interval (using bias corrected accelerated bootstrap).

    Parameters
    ----------
    x : np.ndarray
        Predictors - one or two-dimensional array of values for the
        correlation. If predictors are two-dimensional the last column is
        treated as the predictor of interest and the rest as confounds.
    y : np.ndarray
        Dependent variable. One dimensional array of values for the
        correlation.
    n_boot : int
        Number of bootstraps to use.

    Returns
    -------
    stats : dict
        Dictionary of results.
        * ``stats['es']`` contains effect size.
        * ``stats['ci']`` contains 95% confidence interval for the effect size.
        * ``stats['bootstraps']`` contains bootstrap effect size values.
    '''
    # use pearson correlation
    from scipy.stats import pearsonr
    import scikits.bootstrap as boot

    stats = dict()

    if x.ndim == 1:
        # normal correlation
        def corr(x, y):
            return pearsonr(x, y)[0]
    else:
        from borsar.stats import compute_regression_t
        # we use regression t value and then turn it to r
        def corr(x, y):
            tvals = compute_regression_t(y[:, np.newaxis], x)
            return r_from_t(tvals[-1, 0], x)

    r = corr(x, y)
    # currently this is available only on my branch of scikits-bootstrap
    # but I'll prepare a PR to the github repo, and it will be available
    # when/if it gets accepted
    r_ci, bootstraps = boot.ci((x, y), corr, multi=True, n_samples=n_boot,
                               return_dist=True)
    stats.update(bootstraps=bootstraps)
    stats.update(es=r, ci=r_ci)
    return stats


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


# TODO - return analyses as a dataframe?
def list_analyses(study=list('ABCDE'), contrast=['cvsc', 'cvsd', 'creg',
                  'cdreg', 'dreg'], eyes=['closed'], space=['avg', 'csd',
                  'src'], freq_range=[(8, 13)], avg_freq=[True, False],
                  selection=['asy_frontal', 'asy_pairs', 'all'],
                  transform=['log'], confounds=[False], verbose=True):
    '''
    List all possible analyses for given set of parameter options.
    For explanation of the arguments see ``DiamSar.analysis.run_analysis``.
    The only difference is that ``list_analyses`` takes list of values for
    each of the arguments - where each element of the list is one analysis
    option that should be combined with all the other options.

    Returns
    -------
    good_analyses : list of tuples
        List of tuple where each tuple contains all relevant analysis
        parameters. To get a more user-friendly representation use
        ``DiamSar.analysis.analyses_to_df`` function on the returned tuple.
    '''

    from itertools import product
    prod = list(product(study, contrast, eyes, space, freq_range, avg_freq,
                        selection, transform, confounds))

    if verbose:
        all_combinations = len(prod)
        print('All analysis combinations: {:d}'.format(all_combinations))

    good_analyses = list()
    for std, cntr, eye, spc, frqrng, avgfrq, sel, trnsf, conf in prod:
        # averaging alpha frequency range is ommited only for wide frequency
        # range
        if not avgfrq and not frqrng == (8, 13):
            continue

        # asymmetry pairs are always used with frequency averaging
        if sel == 'asy_pairs' and not avgfrq:
            continue

        # asymmetry pairs are not used in source space
        if sel == 'asy_pairs' and spc == 'src':
            continue

        # 'all' and 'frontal' selections are not used with frequency averaging
        if sel in ['all', 'frontal'] and avgfrq:
            continue

        # non availability
        # ----------------

        # only studies C and D contain segments with open eyes
        if std not in ['C', 'D'] and eye == 'open':
            continue
        # study B does not have a diagnosed group
        if std == 'B' and cntr in ['cdreg', 'cvsd', 'dreg']:
            continue
        # studies A and E has controls only with low BDI (no subclinical)
        if std in ['A', 'E'] and cntr in ['cvsc', 'creg']:
            continue

        # else: good analysis
        good_analyses.append((std, cntr, eye, spc, frqrng, avgfrq, sel, trnsf,
                              conf))

    if verbose:
        reduced = len(good_analyses)
        print('Number of reduced combinations: {:d}'.format(reduced))

    return good_analyses


def run_many(study=list('ABC'), contrast=['cvsc', 'cvsd', 'creg', 'cdreg',
             'dreg'], eyes=['closed'], space=['avg', 'csd', 'src'],
             freq_range=[(8, 13)], avg_freq=[True, False], confounds=False,
             selection=['asy_frontal', 'asy_pairs', 'all'], transform=['log'],
             analyses=None, n_permutations=10_000, progressbar='notebook',
             save_dir='stats'):
    '''
    Run multiple analyses parametrized by combinations of options given in
    arguments.
    For explanation of the arguments see ``DiamSar.analysis.run_analysis``.
    The only difference is that ``run_many`` takes list of values for each
    of the arguments.
    Every analysis result (statistical test result) is saved to disk to
    results subdirectory defined by ``save_dir``.
    Instead of using keyword arguments to define analysis options you can
    pass the analyses via the ``analyses`` keyword argument. The analyses
    have to be in the format used by ``DiamSar.analysis.list_analyses``:
    list of (study, contrast, eyes, space, freq_range, avg_freq, selection,
    transform, confounds) tuples.
    '''
    from borsar.utils import silent_mne
    from DiamSar.utils import progressbar as pbarobj

    if analyses is None:
        analyses = list_analyses(study, contrast, eyes, space, freq_range,
                                 avg_freq, selection, transform, confounds)

    pbar = pbarobj(progressbar, len(analyses))
    for std, cntr, eys, spc, frqrng, avgfrq, sel, trnsf, conf in analyses:
        with silent_mne(full_silence=True):
            stat = run_analysis(study=std, contrast=cntr, eyes=eys, space=spc,
                                freq_range=frqrng, avg_freq=avgfrq,
                                selection=sel, transform=trnsf, confounds=conf,
                                n_permutations=n_permutations, verbose=False)
            save_stat(stat, save_dir=save_dir)
        pbar.update(1)
    pbar.update(1)
    pbar.close()


def analyses_to_df(analyses):
    '''Turn list of tuples with analysis parameters to dataframe
    representation.'''
    df = pd.DataFrame(columns=['study', 'contrast', 'eyes', 'space', 'freq',
                               'avg_freq', 'selection', 'transform',
                               'confounds'])
    for idx, analysis in enumerate(analyses):
        if isinstance(analysis[4], (list, tuple)):
            analysis = list(analysis)
            analysis[4] = '{} - {} Hz'.format(*analysis[4])
        df.loc[idx, :] = analysis

    return df


def _get_adjacency(study, space, ch_names, selection, src):
    '''Return adjacency for given study and space.'''
    if not space == 'src':
        # use right-side channels in adjacency if we calculate asymmetry
        if 'asy' in selection:
            ch_names = [ch.split('-')[1] for ch in ch_names]
        neighbours = pth.paths.get_data('neighbours', study=study)
        adjacency = construct_adjacency_matrix(
            neighbours, ch_names, as_sparse=True)
    else:
        import mne
        adjacency = mne.spatial_src_connectivity(src)
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


def _construct_clusters(clusters, pval, stat, space, stat_info, info,
                        src, subjects_dir, subject, ch_names, freq):
    '''Construct Clusters object out of cluster-based test results.'''
    from borsar.cluster import Clusters

    if space == 'src':
        dimnames, dimcoords = ['vert'], [ch_names]
    else:
        # if ch_names contain dash then asymmetry measures were used (R - L),
        # we only retain the name of the right channel (otherwise there would
        # be an error about mismatch between Info and cluster.dimcoords)
        ch_names = [ch.split('-')[-1] if '-' in ch else ch
                    for ch in ch_names]
        dimnames, dimcoords = ['chan'], [ch_names]

    if stat.ndim > 1:
        # channels have to be the first dimension for Clusters
        stat = stat.T
        clusters = ([c.T for c in clusters] if len(clusters) > 0
                    else clusters)

        # 2nd dimension: frequency
        dimnames.append('freq')
        dimcoords.append(freq)
    else:
        stat_info['freq'] = freq

    clst = Clusters(stat, clusters, pval, dimnames, dimcoords,
                    description=stat_info, info=info, src=src,
                    subjects_dir=subjects_dir, subject=subject)
    return clst


def _get_space_info(study, space, ch_names, selection):
    '''Return relevant source/channel space along with additional variables.'''
    if not space == 'src':
        info = pth.paths.get_data('info', study=study)
        src, subjects_dir, subject = None, None, None

        # check that channel order agrees between psds and info
        msg = 'Channel order does not agree between psds and info object.'
        assert (np.array(ch_names) == np.array(info['ch_names'])).all(), msg
    else:
        # read fwd, get subjects_dir and subject
        subjects_dir = pth.paths.get_path('subjects_dir')
        info = None
        if 'asy' in selection:
            src = pth.paths.get_data('src_sym')
            subject = 'fsaverage_sym'
        else:
            subject = 'fsaverage'
            src = pth.paths.get_data('fwd', study=study)['src']

    return info, src, subject, subjects_dir


def save_stat(stat, save_dir='stats'):
    '''
    Save stat_info dictionary or Clusters object with default name and to
    default directory.
    '''
    from borsar.cluster import Clusters

    save_dir = op.join(pth.paths.get_path('main', study='C'), 'analysis',
                       save_dir)
    fname = ('stat_study-{}_eyes-{}_space-{}_contrast-{}_selection-{}'
             '_freqrange-{}_avgfreq-{}_transform-{}_divbysum-{}.hdf5')
    keys = ['study', 'eyes', 'space', 'contrast', 'selection', 'freq_range',
            'avg_freq', 'transform', 'div_by_sum']

    if isinstance(stat, Clusters):
        fname = fname.format(*[stat.description[k] for k in keys])
        full_path = op.join(save_dir, fname)
        stat.save(full_path, overwrite=True)
    else:
        from mne.externals import h5io
        fname = fname.format(*[stat[k] for k in keys])
        full_path = op.join(save_dir, fname)
        h5io.write_hdf5(full_path, stat, overwrite=True)


# TODO: add option to read source space Clusters
def load_stat(fname=None, study='C', eyes='closed', space='avg',
              contrast='cvsd', selection=None, freq_range=(8, 13),
              avg_freq=None, transform=None, div_by_sum=False,
              stat_dir=None):
    '''Read previously saved analysis result.

    Parameters
    ----------
    fname : str
        Name of the file. If ``None`` then it is constructed from other
        kwargs.
    **kwargs
        Other keyword arguments are the same as in ``run_analysis``.

    Returns
    -------
    stat : borsar.cluster.Clusters | dict
        Analysis results.
    '''

    # if fname is not specified, construct
    if fname is None:
        import re
        stat_dir = 'stats' if stat_dir is None else stat_dir
        fname = (r'stat_study-{}_eyes-{}_space-{}_contrast-{}_selection-{}'
                  '_freqrange-{}_avgfreq-{}_transform-{}_divbysum-{}')
        vars = [study, eyes, space, contrast, selection, freq_range,
                avg_freq, transform, div_by_sum]
        vars = ['.+' if v is None else v for v in vars]

        ptrn = fname.format(*vars)
        ptrn = ptrn.replace('(', '\\(').replace(')', '\\)')
        stat_dir = op.join(pth.paths.get_path('main', 'C'), 'analysis',
                           stat_dir)
        fls = os.listdir(stat_dir)
        ok_files = [f for f in fls if len(re.findall(ptrn, f)) > 0]
        if len(ok_files) == 0:
            raise FileNotFoundError('Could not find requested file.')
        elif len(ok_files) > 1:
            raise ValueError('Multiple files match your description:\n'
                             + ',\n'.join(ok_files))
        fname = op.join(stat_dir, ok_files[0])

    return _load_stat(fname)


def _load_stat(fname):
    from mne.externals import h5io

    stat = h5io.read_hdf5(fname)
    if 'clusters' in stat:
        from borsar.cluster import Clusters

        study = stat['description']['study']
        if 'src' in fname:
            info = None
            # FIXME: src should be different when 'asy'
            src = pth.paths.get_data('fwd', study=study)['src']
            selection = stat['description']['selection']
            subject = 'fsaverage_sym' if 'asy' in selection else 'fsaverage'
            subjects_dir = pth.paths.get_path('subjects_dir')
        else:
            src = None
            info = pth.paths.get_data('info', study=study)
            subject, subjects_dir = None, None

        clst = Clusters(stat['stat'], stat['clusters'], stat['pvals'],
                        dimnames=stat['dimnames'], dimcoords=stat['dimcoords'],
                        info=info, src=src, description=stat['description'],
                        subject=subject, subjects_dir=subjects_dir)
        return clst
    else:
        return stat


def sort_clst_channels(clst):
    '''Sort cluster channels from left to right.'''
    import mne
    from borsar.channels import get_ch_pos

    ch_pos = get_ch_pos(clst.info)
    sorting = np.argsort(ch_pos[:, 0])

    clst.info = mne.pick_info(clst.info, sel=sorting)
    clst.stat = clst.stat[sorting]

    if clst.clusters is not None:
        clst.clusters = clst.clusters[:, sorting]
    clst.dimcoords[0] = np.array(clst.dimcoords[0])[sorting].tolist()
    return clst


def d_from_t_categorical(tvalue, preds):
    '''Calculating Cohen's d from regression t value for categorical predictor.

    reference
    ---------
    Nakagawa, S., & Cuthill, I. C. (2007). Effect size, confidence interval
    and statistical significance: a practical guide for biologists.
    Biological Reviews of the Cambridge Philosophical Society, 82(4), 591–605.
    '''
    n_obs, n_preds = preds.shape
    df = n_obs - n_preds - 1  # -1 because we assume intercept is not provided
    categ = preds[:, -1]
    values, counts = np.unique(categ, return_counts=True)
    assert len(values) == 2
    n1, n2 = counts
    d = tvalue * (n1 + n2) / (np.sqrt(n1 * n2) * np.sqrt(df))
    return d


def r_from_t(tvalue, preds):
    '''Calculating r correlation coeffiecient from regression t value for
    continuous predictor. This is known as partial correlation coefficient
    when there is more than one predictor.

    reference
    ---------
    Nakagawa, S., & Cuthill, I. C. (2007). Effect size, confidence interval
    and statistical significance: a practical guide for biologists.
    Biological Reviews of the Cambridge Philosophical Society, 82(4), 591–605.
    '''
    # -1 because we assume intercept is not provided in the preds
    df = np.diff(preds.shape[::-1])[0] - 1
    return tvalue / np.sqrt(tvalue ** 2 + df)
