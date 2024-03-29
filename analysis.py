import os
import os.path as op
import numpy as np
import pandas as pd
import mne

from . import pth, io, utils


def run_analysis(study='C', contrast='cvsd', eyes='closed', space='avg',
                 freq_range=(8, 13), avg_freq=True, selection='frontal_asy',
                 div_by_sum=False, transform='log', n_permutations=10000,
                 cluster_p_threshold=0.05, confounds=False, interaction=False,
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
    interaction : bool
        Whether to perform gender * diagnosis or gender * bdi interaction
        analysis. If ``interaction`` is set to True the returned results
        (t values / t value maps) represent only the interaction term.
        Defaults to False.
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
    scale_psd = False
    data = io.prepare_data(
        pth.paths, study, contrast, eyes, space, freq_range, avg_freq,
        selection, div_by_sum, transform, confounds, scale_psd, interaction,
        verbose)
    return _conduct_analysis(data, contrast, space,avg_freq, selection,
                             n_permutations, cluster_p_threshold, confounds,
                             verbose)


def _conduct_analysis(data, contrast, space, avg_freq, selection,
                      n_permutations, cluster_p_threshold, confounds, verbose):
    '''Conduct analysis.'''
    stat_info, adjacency = data['stat_info'], data['adjacency']
    if 'vs' in contrast and not confounds:
        hi, lo = data['hi'], data['lo']
        psd_ndim = hi.ndim
    else:
        hilo = data['hilo']
        bdi = data['bdi']
        psd_ndim = hilo.ndim

    # put spatial dimension last for cluster-based test
    if not avg_freq or psd_ndim == 3:
        if 'hi' in data:
            hi = hi.transpose((0, 2, 1))
            lo = lo.transpose((0, 2, 1))
        else:
            hilo = hilo.transpose((0, 2, 1))

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
        return _construct_clusters(
            clusters, pval, stat, space, stat_info, data['info'], data['src'],
            data['subjects_dir'], data['subject'], data['ch_names'],
            data['freq'])
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


def _aggregate_studies(paths, space, contrast, selection='asy_pairs',
                       confounds=False, interaction=False, studies=None):
    '''
    Read all studies that include given contrast and aggregate their data.
    Used when plotting aggregated channel pairs figures (``plot_aggregated``).

    Returns
    -------
    high : np.array
        Aggregated psds for the diagnosed or subclinical group if ``'cvsd'```
        or ``'cvsc'`` contrast is used. When a regression contrast is used this
        variable contains all psds.
    low : np.array
        Aggregated psds for the healthy controls ``'cvsd'``` or ``'cvsc'``
        contrast is used. When a regression contrast is used this variable
        contains the predictor variable.
    studies : list of str
        Translated study names included in the aggregated data.
    grouping : array of bool
        Grouping information - we need to retain this when bootstrapping
        gender x diagnosis interaction.
    data : dict
        Dictionary of data-relevant information (for example adjacency) that
        is used in ``_conduct_analysis``.
    '''
    from .utils import translate_study

    if studies is None:
        if contrast in ['cvsd', 'dreg']:
            studies = ['A', 'C', 'D', 'E']
        elif contrast in ['cvsc', 'creg']:
            studies = ['B', 'C', 'D']
        elif contrast ==  'cdreg':
            studies = ['A', 'C', 'D', 'E']

    grouping = list()
    psds = {'high': list(), 'low': list()}
    for study in studies:
        data = io.prepare_data(paths, selection=selection, study=study,
                               space=space, contrast=contrast, scale_psd=True,
                               confounds=confounds, interaction=interaction)
        # deal with confounds
        if confounds:
            from scipy.stats import zscore
            residuals = _deal_with_confounds(data)
            residuals = zscore(residuals, axis=0)
            data['hilo'] = residuals

            # make sure we retain grouping info (this is used
            # when bootstrapping interaction effect size)
            if 'reg' not in contrast:
                idx = -1 if not interaction else -2
                grp = data['bdi'][:, idx] > 0
                if not interaction:
                    data['hi'] = residuals[grp]
                    data['lo'] = residuals[~grp]
            data['bdi'] = data['bdi'][:, -1]

        if 'reg' in contrast:
            # we have to standardize depression scores
            # (different scales for BDI and PHQ-9)
            from scipy.stats import zscore
            data['bdi'] = zscore(data['bdi'])

        # package output data
        if 'reg' not in contrast and not interaction:
            psds['high'].append(data['hi'])
            psds['low'].append(data['lo'])
        else:
            psds['high'].append(data['hilo'])
            psds['low'].append(data['bdi'])
        if interaction and 'reg' not in contrast:
            grouping.append(grp)

    low = np.concatenate(psds['low'], axis=0)
    high = np.concatenate(psds['high'], axis=0)

    if interaction and 'reg' not in contrast:
        grouping = np.concatenate(grouping, axis=0)
    else:
        grouping = None

    studies = [translate_study[std] for std in studies]
    return high, low, studies, grouping, data


def _deal_with_confounds(data):
    from borsar.stats import compute_regression_t
    data, preds = data['hilo'], data['bdi']
    tvals, residuals = compute_regression_t(data, preds[:, :-1],
                                            return_residuals=True)
    return residuals


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


# There were problems bootstrapping partial eta squared (stability warnings),
# so we don't use it in the paper
def esci_partial_eta_sq(x, y, n_boot=5000, grp=None):
    '''Compute partial Eta squared effect size and its bootstrap 95% confidence
    interval (using bias corrected accelerated bootstrap).
    This function is used for estimating the effect size of the interaction.

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
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import scikits.bootstrap as boot

    def paretasq(x, y):
        # create a dataframe and fit linear model
        df = pd.DataFrame({'y': y, 'x': x})
        lm = smf.ols('y ~ x', data=df).fit()

        # we use type II sum of squares, but it doesn't matter because we have
        # only one predictor and dependent data are residuals after accounting
        # for the variation explained by the other predictors
        table = sm.stats.anova_lm(lm, typ=2)

        # we calculate partial eta squared
        SS_effect = table.loc['x', 'sum_sq']
        SS_error = table.loc['Residual', 'sum_sq']
        partial_eta_sq = SS_effect / (SS_effect + SS_error)
        return partial_eta_sq

    # when the interaction is with diagnosis-like factor we need to sample
    # from diagnosis groups (but we treat sex as random)
    if grp is None:
        eta = paretasq(x, y)
        eta_ci, bootstraps = boot.ci((x, y), paretasq, multi=True,
                                     n_samples=n_boot, return_dist=True)
    else:
        gr1 = np.stack([x[grp], y[grp]], axis=1)
        gr2 = np.stack([x[~grp], y[~grp]], axis=1)
        def statfun(gr1, gr2):
            xy = np.concatenate([gr1, gr2], axis=0)
            x, y = xy[:, 0], xy[:, 1]
            return paretasq(x, y)

        eta = paretasq(x, y)
        eta_ci, bootstraps = boot.ci((gr1, gr2), statfun, multi='independent',
                                     n_samples=n_boot,
                                     return_dist=True)

    stats = dict()
    stats.update(bootstraps=bootstraps)
    stats.update(es=eta, ci=eta_ci)
    return stats


def esci_interaction_regression_r(x, y, n_boot=5000, grp=None):
    '''Compute correlation coefficeint effect size and its bootstrap 95%
    confidence interval (using bias corrected accelerated bootstrap).
    This function is used for estimating the effect size of the interaction.
    (this function is used over ``esci_regression_r`` due to some limitations
     of current implementation of how data are handled when aggregating
     studies)

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
    grp : bool array | None
        If not ``None`` then it means some grouping should be used during
        bootstrapping - the two groups (for example diagnosed vs ).

    Returns
    -------
    stats : dict
        Dictionary of results.
        * ``stats['es']`` contains effect size.
        * ``stats['ci']`` contains 95% confidence interval for the effect size.
        * ``stats['bootstraps']`` contains bootstrap effect size values.
    '''
    from scipy.stats import pearsonr
    import scikits.bootstrap as boot

    def corr(x, y):
        return pearsonr(x, y)[0]

    # when the interaction is with diagnosis-like factor we need to sample
    # from diagnosis groups (but we treat sex as random)
    if grp is None:
        multi = True
        x_, y_ = x, y
        statfun = corr
    else:
        multi = 'independent'
        x_ = np.stack([x[grp], y[grp]], axis=1)
        y_ = np.stack([x[~grp], y[~grp]], axis=1)
        def statfun(gr1, gr2):
            xy = np.concatenate([gr1, gr2], axis=0)
            x, y = xy[:, 0], xy[:, 1]
            return corr(x, y)

    r = statfun(x_, y_)
    r_ci, bootstraps = boot.ci((x_, y_), statfun, multi=multi,
                                 n_samples=n_boot, return_dist=True)

    stats = dict()
    stats.update(bootstraps=bootstraps)
    stats.update(es=r, ci=r_ci)
    return stats


# TODO - return analyses as a dataframe?
def list_analyses(study=list('ABCDE'), contrast=['cvsc', 'cvsd', 'creg',
                  'cdreg', 'dreg'], eyes=['closed'], space=['avg', 'csd',
                  'src'], freq_range=[(8, 13)], avg_freq=[True, False],
                  selection=['asy_frontal', 'asy_pairs', 'all'],
                  transform=['log'], confounds=[False], reduce_analyses=True,
                  verbose=True):
    '''
    List all possible analyses for given set of parameter options.
    For explanation of the arguments see ``DiamSar.analysis.run_analysis``.
    The only difference is that ``list_analyses`` takes list of values for
    each of the arguments - where each element of the list is one analysis
    option that should be combined with all the other options.

    Parameters
    ----------
    reduce_analyses : bool
      Whether to reduce the analyses the same way as in the "No
      relationship between frontal alpha asymmetry and depressive disorders
      in a multiverse analysis of five studies". The reduction rules were:
      * exclude analyses where frequency other than 8 - 13 Hz is not
        averaged (frequency by fequency analyses were done only in one
        special case)
      * exclude analyses where asymmetry channel pairs are used and
        frequency range is not averaged
      * exclude analyses where 'all' or 'frontal' channel selection
        was used and frequency range is not averaged

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
        if reduce_analyses:
            if not avgfrq and not frqrng == (8, 13):
                continue

            # asymmetry pairs are always used with frequency averaging
            if sel == 'asy_pairs' and not avgfrq:
                continue

            # 'all' and 'frontal' selections are not used with frequency averaging
            if sel in ['all', 'frontal'] and avgfrq:
                continue

        # asymmetry pairs are not used in source space
        if sel == 'asy_pairs' and spc == 'src':
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
        num_comb = len(good_analyses)
        addstr = ' reduced' if reduce_analyses else ''
        print(f'Number of{addstr} analysis combinations: {num_comb:d}')

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
def load_stat(fname=None, study=None, eyes='closed', space='avg',
              contrast=None, selection=None, freq_range=(8, 13),
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
        ptrn = ptrn.replace('[', '\\[').replace(']', '\\]')
        stat_dir = op.join(pth.paths.get_path('main', 'C'), 'analysis',
                           stat_dir)
        fls = os.listdir(stat_dir)
        ok_files = [f for f in fls if len(re.findall(ptrn, f)) > 0]
        n_files = len(ok_files)
        if n_files == 0:
            raise FileNotFoundError('Could not find requested file.')
        elif n_files > 1:
            # explain which files (parts of files) cannot be disambiguated
            file_segments = [f.replace('.hdf5', '').split('_')
                             for f in ok_files]
            for idx in range(len(file_segments[0])):
                is_same = True
                compare = file_segments[0][idx]
                for pair in range(1, n_files):
                    if not file_segments[pair][idx] == compare:
                        is_same = False
                        break
                if is_same:
                    for elm in range(n_files):
                        file_segments[elm][idx] = '...'
            ok_files = ['_'.join(f) for f in file_segments]
            raise ValueError('Multiple files match your description:\n'
                             + ',\n'.join(ok_files))
        fname = op.join(stat_dir, ok_files[0])

    return _load_stat(fname)


def _load_stat(fname):
    from mne.externals import h5io

    stat = h5io.read_hdf5(fname)
    if 'clusters' in stat:
        from borsar.cluster import Clusters

        if 'src' in fname:
            info = None
            study = (stat['description']['study']
                     if 'study' in stat['description'] else 'C')
            # FIXME: src should be different when 'asy'
            src = pth.paths.get_data('fwd', study=study)['src']
            selection = stat['description']['selection']
            subject = 'fsaverage_sym' if 'asy' in selection else 'fsaverage'
            subjects_dir = pth.paths.get_path('subjects_dir')
        else:
            import sarna
            src = None
            study = stat['description']['study']
            info = pth.paths.get_data('info', study=study)
            sarna.utils.fix_channel_pos(info)
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
