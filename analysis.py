import os
import os.path as op
import numpy as np
import pandas as pd
from scipy import sparse

from borsar.cluster import construct_adjacency_matrix

from . import pth
from . import utils
from .freq import format_psds, get_psds


def run_analysis(study='C', contrast='cvsd', eyes='closed', space='avg',
                 freq_range=(8, 13), avg_freq=True, selection='frontal_asy',
                 div_by_sum=False, transform='log', n_permutations=10000,
                 cluster_p_threshold=0.05, verbose=True):
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
    bdi = pth.paths.get_data('bdi', study=study)
    psds, freq, ch_names, subj_id = pth.paths.get_data(
        'psd', study=study, eyes=eyes, space=space)

    # TODO - re-save psds without nan-subjects?
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
    grp = utils.group_bdi(subj_id, bdi, method=contrast)
    if 'reg' not in contrast:
        hi, lo = psd[grp['high']], psd[grp['low']]
        stat_info.update(dict(N_low=lo.shape[0], N_high=hi.shape[0],
                              N_all=lo.shape[0] + hi.shape[0]))
    else:
        bdi = grp['bdi']
        hilo = psd[grp['selection']]
        stat_info['N_all'] = grp['selection'].sum()

    # statistical analysis
    # --------------------
    if 'pairs' not in selection:
        # cluster-based permutation tests for multiple comparisons
        stat_info.update(dict(cluster_p_threshold=cluster_p_threshold))

        if 'vs' in contrast:
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
            stat, clusters, pval = cluster_based_regression(
                hilo, bdi, n_permutations=n_permutations, adjacency=adjacency,
                alpha_threshold=cluster_p_threshold)

        # construct Clusters with stat_info in description
        return _construct_clusters(clusters, pval, stat, space, stat_info,
                                   info, src, subjects_dir, subject,
                                   ch_names, freq)
    else:
        # for selected pairs (two channel pairs) we don't correct for
        # multiple comparisons:
        if 'vs' in contrast:
            from scipy.stats import t, ttest_ind
            stat, pval = ttest_ind(hi, lo, equal_var=False)
        else:
            from borsar.stats import compute_regression_t
            # compute regression and ignore intercept:
            stat, pval = compute_regression_t(hilo, bdi, return_p=True)
            stat, pval = stat[1], pval[1]

        stat_info.update(dict(stat=stat, pval=pval, ch_names=ch_names))
        return stat_info


def summarize_stats(split=True, reduce_columns=True, stat_dir='stats'):
    '''Summarize multiple analyses (saved in analysis dir) in a dataframe.

    Parameters
    ----------
    split : bool
        Whether to split the results into channel pairs analyses and
        cluster-based analyses dataframes. If ``True`` returns two dataframes,
        the first one with channel pair analyses and the second one with
        cluster-based analyses. Defaults to ``True``.
    reduce_columns : bool
        Whether to remove columns with no variability from the output. Defaults
        to ``True``.
    stat_dir : str
        Subdirectory to use (``'stats'``, ``'add1'`` or ``'add2'``, unless
        additional subdirectories were created).

    Returns
    -------
    df | df1, df2 (depending on ``split``) : pandas.DataFrame
        Dataframe (or dataframes if ``split`` is ``True``) with summarized
        DiamSar analyses. If ``split=True`` returns two dataframes,
        the first one with channel pair analyses and the second one with
        cluster-based analyses.
    '''
    from mne.externals import h5io

    stat_dir = op.join(pth.paths.get_path('main', 'C'), 'analysis', stat_dir)
    stat_files = [f for f in os.listdir(stat_dir) if f.endswith('.hdf5')]
    n_stat = len(stat_files)

    # first, create an empty dataframe
    stat_params = ['study', 'contrast', 'space', 'N_low', 'N_high', 'N_all',
                   'eyes', 'selection', 'freq_range', 'avg_freq', 'transform',
                   'div_by_sum']
    stat_summary = ['min t', 'max t', 'n clusters', 'min cluster p',
                    'n signif clusters', 'n signif points',
                    'largest cluster size']
    df = pd.DataFrame(index=np.arange(1, n_stat + 1),
                      columns=stat_params + stat_summary)

    for idx, fname in enumerate(stat_files, start=1):
        stat = h5io.read_hdf5(op.join(stat_dir, fname))

        if 'description' in stat:
            for col in stat_params:
                df.loc[idx, col] = (stat['description'][col]
                                    if col in stat['description'] else np.nan)

            # summarize clusters
            n_clst = (len(stat['clusters']) if stat['clusters']
                      is not None else 0)
            df.loc[idx, stat_summary[0]] = stat['stat'].min()
            df.loc[idx, stat_summary[1]] = stat['stat'].max()
            df.loc[idx, stat_summary[2]] = n_clst
            df.loc[idx, stat_summary[3]] = (stat['pvals'].min()
                                            if n_clst > 0 else np.nan)
            df.loc[idx, stat_summary[4]] = ((stat['pvals'] < 0.05).sum()
                                            if n_clst > 0 else 0)
            df.loc[idx, stat_summary[5]] = (np.abs(stat['stat']) > 2.).sum()
            df.loc[idx, stat_summary[6]] = (
                max([c.sum() for c in stat['clusters']])
                if n_clst > 0 else np.nan)
        else:
            for col in stat_params:
                df.loc[idx, col] = stat[col] if col in stat else np.nan

            df.loc[idx, stat_summary[0]] = stat['stat'][0]
            df.loc[idx, stat_summary[1]] = stat['stat'][1]
            df.loc[idx, stat_summary[2]] = np.nan
            df.loc[idx, stat_summary[3]] = stat['pval']
            df.loc[idx, stat_summary[4]] = np.nan
            df.loc[idx, stat_summary[5]] = (stat['pval'] < 0.05).sum()
            df.loc[idx, stat_summary[6]] = np.nan

    # split into two dfs
    if split:
        pair_rows = df['selection'].str.contains('pairs').values
        df1 = df.loc[pair_rows, :].reset_index(drop=True)
        df2 = df.loc[~pair_rows, :].reset_index(drop=True)

    # reduce columns
    if reduce_columns:
        if split:
            df1 = remove_columns_with_no_variability(df1)
            df2 = remove_columns_with_no_variability(df2)
        else:
            df = remove_columns_with_no_variability(df)

    if split:
        return df1, df2
    else:
        return df


def summarize_ch_pair_stats(reduce_columns=True, stat_dir='stats',
                            progressbar='text'):
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
        loaded = get_psds(selection='asy_pairs', study=study,
                          space=space, contrast=contrast)

        # add stats
        for ch_idx in range(2):
            stat_col = ch_idx * 4
            t, p = stat['stat'][ch_idx], stat['pval'][ch_idx]
            df.loc[idx, stat_summary[stat_col]] = t
            df.loc[idx, stat_summary[stat_col + 1]] = p

            if 'vs' in contrast:
                psd_high, psd_low, ch_names = loaded
                esci = esci_indep_cohens_d(psd_high[:, ch_idx],
                                           psd_low[:, ch_idx])
            else:
                psd_sel, info_sel, bdi_sel = loaded
                esci = esci_regression_r(psd_sel[:, ch_idx], bdi_sel)

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


def esci_indep_cohens_d(data1, data2, n_boot=5000):
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

    Returns
    -------
    stats : dict
        Dictionary of results.
        * ``stats['es']`` contains effect size.
        * ``stats['ci']`` contains 95% confidence interval for the effect size.
        * ``stats['bootstraps']`` contains bootstrap effect size values.
    '''
    import dabest
    df = utils.psd_to_df(data1, data2)
    dbst_set = dabest.load(df, idx=("controls", "diagnosed"),
                           x="group", y="FAA", resamples=n_boot)
    results = dbst_set.cohens_d.results
    cohen_d = results.difference.values[0]
    cohen_d_ci = (results.bca_low.values[0], results.bca_high.values[0])
    bootstraps = results.bootstraps[0]
    stats = dict(es=cohen_d, ci=cohen_d_ci, bootstraps=bootstraps)
    return stats


def esci_regression_r(x, y, n_boot=5000):
    '''Compute Pearson's r effect size and its bootstrap 95% confidence
    interval (using bias corrected accelerated bootstrap).

    Parameters
    ----------
    x : np.ndarray
        One dimensional array of values for the correlation.
    y : np.ndarray
        One dimensional array of values for the correlation.
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

    def corr(x, y):
        return pearsonr(x, y)[0]

    r = corr(x, y)
    stats = dict(es=r)
    try:
        # currently this is available only on my branch of scikits-bootstrap
        # but I'll prepare a PR to the github repo, and it will be available
        # when/if it gets accepted
        r_ci, bootstraps = boot.ci((x, y), corr, multi=True, n_samples=n_boot,
                                   return_dist=True)
        stats.update(bootstraps=bootstraps)
    except TypeError:
        # branch of boot.ci with return_dist not available, use normal boot:
        print('Oh no!')
        r_ci = boot.ci((x, y), corr, multi=True, n_samples=n_boot)
    stats.update(ci=r_ci)
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
                  transform=['log'], verbose=True):
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
                        selection, transform))

    if verbose:
        all_combinations = len(prod)
        print('All analysis combinations: {:d}'.format(all_combinations))

    good_analyses = list()
    for std, cntr, eye, spc, frqrng, avgfrq, sel, trnsf in prod:
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
        # study A has controls only with low BDI
        if std == 'A' and cntr in ['cvsc', 'creg', 'cdreg']:
            continue
        # study E did not measure BDI
        if std == 'E' and not cntr == 'cvsd':
            continue

        # else: good analysis
        good_analyses.append((std, cntr, eye, spc, frqrng, avgfrq, sel, trnsf))

    if verbose:
        reduced = len(good_analyses)
        print('Number of reduced combinations: {:d}'.format(reduced))

    return good_analyses


def run_many(study=list('ABC'), contrast=['cvsc', 'cvsd', 'creg', 'cdreg',
             'dreg'], eyes=['closed'], space=['avg', 'csd', 'src'],
             freq_range=[(8, 13)], avg_freq=[True, False],
             selection=['asy_frontal', 'asy_pairs', 'all'], transform=['log'],
             analyses=None, progressbar='notebook', save_dir='stats'):
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
    transform) tuples.
    '''
    from borsar.utils import silent_mne
    from DiamSar.utils import progressbar as pbarobj

    if analyses is None:
        analyses = list_analyses(study, contrast, eyes, space, freq_range,
                                 avg_freq, selection, transform)

    pbar = pbarobj(progressbar, len(analyses))
    for std, cntr, eys, spc, frqrng, avgfrq, sel, trnsf in analyses:
        with silent_mne(full_silence=True):
            stat = run_analysis(study=std, contrast=cntr, eyes=eys, space=spc,
                                freq_range=frqrng, avg_freq=avgfrq,
                                selection=sel, transform=trnsf, verbose=False)
            save_stat(stat, save_dir=save_dir)
        pbar.update(1)
    pbar.update(1)


def analyses_to_df(analyses):
    '''Turn list of tuples with analysis parameters to dataframe
    representation.'''
    df = pd.DataFrame(columns=['study', 'contrast', 'eyes', 'space', 'freq',
                               'avg_freq', 'selection', 'transform'])
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
              contrast='cvsd', selection='asy_frontal', freq_range=(8, 13),
              avg_freq=True, transform='log', div_by_sum=False,
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
        stat_dir = 'stats' if stat_dir is None else stat_dir
        fname = ('stat_study-{}_eyes-{}_space-{}_contrast-{}_selection-{}'
                 '_freqrange-{}_avgfreq-{}_transform-{}_divbysum-{}.hdf5')
        vars = [study, eyes, space, contrast, selection, freq_range,
                avg_freq, transform, div_by_sum]
        fname = fname.format(*vars)
        stat_dir = op.join(pth.paths.get_path('main', 'C'), 'analysis',
                           stat_dir)
        fname = op.join(stat_dir, fname)

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
