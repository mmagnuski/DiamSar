import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import seaborn as sns
import pandas as pd

from borsar.stats import format_pvalue
from borsar.viz import Topo

from sarna.viz import prepare_equal_axes

from . import freq, analysis, utils, io
from .analysis import load_stat, run_analysis
from .utils import colors, translate_contrast, translate_study


# - [ ] change to use Info
# - [ ] make sure Info in .get_data() are cached


def plot_unavailable(stat, axis=None):
    '''Plot "empty" topography with 'NA' in the middle.'''
    topo = stat.plot(cluster_idx=0, outlines='skirt', extrapolate='head',
                     axes=axis)
    topo.img.remove()
    topo.chan.remove()
    for line in topo.lines.collections:
        line.remove()
    if len(topo.marks) > 0:
        topo.marks[0].remove()
    topo.axes.text(0, 0, 'NA', fontsize=48, color='gray', ha='center',
                   va='center')
    return topo


def plot_grid_cluster(stats_clst, contrast, vlim=3, show_unavailable=False):
    '''
    Plot cluster-corrected contrast results in a reference by study grid.

    Parameters
    ----------
    stats_clst : pandas.DataFrame
        DataFrame with information about all cluster-based analyses.
    contrast : str
        Statistical contrast represented as string. See
        ``DiamSar.analysis.run_analysis`` for contrast description.
    vlim : float
        Value limits for the topomap colormap (in values of the t statistic).

    Returns
    -------
    fig : matplotlib.Figure
        Matplotlib figure with visualized results.
    '''
    contrast_to_ncols = {'cvsd': 4, 'cvsc': 3, 'cdreg': 4, 'dreg': 4}
    n_cols = contrast_to_ncols[contrast] if not show_unavailable else 5
    cntrst = utils.translate_contrast[contrast]

    fig = plt.figure(figsize=(n_cols * 3, 7.5))
    ax = prepare_equal_axes(fig, [2, n_cols], space=[0.12, 0.89, 0.02, 0.65])

    example_stat = load_stat(study='C', contrast='cvsd', space='avg',
                             selection='asy_frontal')
    vmin, vmax = -vlim, vlim
    axis_limit = 0.13
    study_nums = list()
    for row_idx, space in enumerate(['avg', 'csd']):
        query_str = 'contrast=="{}" & space == "{}"'.format(cntrst, space)
        this_stat = stats_clst.query(query_str)

        col_idx = 0
        for study in list('ABCDE'):
            study_num = utils.translate_study[study]

            if study_num in this_stat.study.values:
                this_ax = ax[row_idx, col_idx]

                # read analysis result from disc
                stat = load_stat(study=study, contrast=contrast, space=space,
                                 selection='asy_frontal')

                # plot results
                # earlier: outlines='skirt', extrapolate='head'
                topo = stat.plot(cluster_idx=0, extrapolate='local',
                                 axes=this_ax, vmin=vmin, vmax=vmax)

                # correct channel marks
                topo.chan.remove()
                this_ax.scatter(*topo.chan_pos.T, s=5, c='k', linewidths=0,
                                zorder=1)

                if len(topo.marks) > 0:
                    topo.marks[0].set_markersize(8.)
                    this_ax.set_title(format_pvalue(stat.pvals[0]),
                                      fontsize=16, pad=0)
                else:
                    this_ax.set_title('no clusters', fontsize=16, pad=0)

                col_idx += 1
                if space == 'avg':
                    study_nums.append(study_num)
            elif show_unavailable:
                plot_unavaiable(example_stat, axis=this_ax)
                col_idx += 1

            # set figure limits
            if study_num in this_stat.study.values or show_unavailable:
                # this_ax.set_ylim((-axis_limit, axis_limit))
                # this_ax.set_xlim((-axis_limit, axis_limit))
                zoom_topo(topo, (-0.03, 0.13), (-0.03, 0.13))

    # add colorbar
    cbar_ax = fig.add_axes([0.92, 0.08, 0.03, 0.55])
    cbar = plt.colorbar(mappable=topo.img, cax=cbar_ax)
    cbar.set_label('t values', fontsize=14)
    for tck in cbar_ax.yaxis.get_majorticklabels():
        tck.set_fontsize(12)

    # add study labels
    # ----------------
    x_mids = list()
    for idx, study in enumerate(study_nums):
        this_ax = ax[0, idx]
        box = this_ax.get_position()
        mid_x = box.corners()[:, 0].mean()
        x_mids.append(mid_x)

        plt.text(mid_x, 0.72, study, va='center', ha='center',
                 transform=fig.transFigure, fontsize=21)

    iseven = (n_cols % 2) == 0
    hlf = int(np.floor(n_cols / 2))
    if iseven:
        mid_x = np.mean(x_mids[hlf - 1:hlf + 1])
    else:
        mid_x = x_mids[hlf]
    plt.text(mid_x, 0.8, 'STUDY', va='center', ha='center',
             transform=fig.transFigure, fontsize=21)

    # add reference labels
    # --------------------
    midys = list()
    for idx, ref in enumerate(['AVG', 'CSD']):
        this_ax = ax[idx, 0]
        box = this_ax.get_position()
        mid_y = box.corners()[:, 1].mean()
        midys.append(mid_y)

        plt.text(0.085, mid_y, ref, va='center', ha='center',
                 transform=fig.transFigure, fontsize=21, rotation=90)

    mid_y = np.mean(midys)
    plt.text(0.03, mid_y, 'REFERENCE', va='center', ha='center',
             transform=fig.transFigure, fontsize=21, rotation=90)

    fig.suptitle('{} contrast'.format(cntrst), fontsize=22, fontweight='bold')
    fig.text(0.5, 0.895, 'cluster-based permutation test results', fontsize=21,
             va='center', ha='center')

    return fig


def plot_multi_topo(psds_avg, info_frontal, info_asy):
    '''Creating combined Topo object for multiple psds.'''
    axis_limit = 2.25
    fig = plt.figure(figsize=(7, 6))
    axs = prepare_equal_axes(fig, [2, 2], space=[0.12, 0.8, 0.02, 0.85],
                             h_dist=0.05, w_dist=0.05)

    topos = list()
    topomap_args = dict(extrapolate='head', outlines='skirt', border='mean')

    for val, ax in zip(psds_avg[:2], axs[0, :]):
        topos.append(Topo(val, info_frontal, cmap='Reds', vmin=-28, vmax=-26,
                          axes=ax, **topomap_args))

    for val, ax in zip(psds_avg[2:], axs[1, :]):
        topos.append(Topo(val, info_asy, cmap='RdBu_r', vmin=-0.3, vmax=0.3,
                          axes=ax, show=False, **topomap_args))

    for tp in topos:
        tp.solid_lines()
        tp.axes.scatter(*tp.chan_pos.T, facecolor='k', s=8)

    for ax in axs.ravel():
        ax.set_ylim((-axis_limit, axis_limit))
        ax.set_xlim((-axis_limit, axis_limit))

    cbar_ax = [fig.add_axes([0.82, 0.43, 0.03, 0.3]),
               fig.add_axes([0.82, 0.08, 0.03, 0.3])]
    plt.colorbar(mappable=topos[0].img, cax=cbar_ax[0])
    plt.colorbar(mappable=topos[-1].img, cax=cbar_ax[1])

    axs[0, 0].set_title('diagnosed', fontsize=17).set_position([.5, 1.0])
    axs[0, 1].set_title('healthy\ncontrols',
                        fontsize=17).set_position([.5, 1.0])
    axs[0, 0].set_ylabel('alpha\npower', fontsize=20, labelpad=7)
    axs[1, 0].set_ylabel('alpha\nasymmetry', fontsize=20, labelpad=7)
    cbar_ax[0].set_ylabel('log(alpha power)', fontsize=16, labelpad=10)
    cbar_ax[1].set_ylabel('alpha power', fontsize=16, labelpad=10)

    # correct cbar position with respect to the topo axes
    fig.canvas.draw()
    for row in range(2):
        cbar_bounds = cbar_ax[row].get_position().bounds
        topo_bounds = axs[row, 0].get_position().bounds
        cbar_ax[row].set_position([cbar_bounds[0], topo_bounds[1] + 0.05,
                                   cbar_bounds[2], topo_bounds[3] - 0.07])

    return fig, axs


# - [ ] make a little more universal and move to sarna
# - [ ] consider adding CI per swarm
# - [ ] consider adding effsize and bootstrapped CIs
def plot_swarm(df, ax=None, ygrid=True, label_size=20, point_size=10,
               spine_offset=25, ticklabel_size=14, axline_width=2,
               tick_length=8, grid_width=2):
    '''
    Swarmplot for single channel pairs asymmetry. Used for group contrast
    visualization.
    '''

    if ax is None:
        gridspec = dict(bottom=0.2, top=0.92, left=0.2, right=0.99)
        fig, ax = plt.subplots(figsize=(8, 6), gridspec_kw=gridspec)

    # swarmplot
    # ---------
    ax = sns.swarmplot("group", "asym", data=df, size=point_size,
                       palette=[colors['diag'], colors['hc']], ax=ax, zorder=5)

    # add means and CIs
    # -----------------
    means = df.groupby('group').mean()
    x_pos = ax.get_xticks()
    x_lab = [x.get_text() for x in ax.get_xticklabels()]
    width = np.diff(x_pos)[0] * 0.2

    for this_label, this_xpos in zip(x_lab, x_pos):
        # plot mean
        this_mean = means.loc[this_label, 'asym']
        ax.plot([this_xpos - width, this_xpos + width], [this_mean, this_mean],
                color=colors[this_label], lw=2.5, zorder=4)
        # add CI (currently standard error of the mean)
        df_sel = df.query('group == "{}"'.format(this_label))
        this_sem = sem(df_sel.asym.values)
        rct = plt.Rectangle((this_xpos - width, this_mean - this_sem),
                            width * 2, this_sem * 2, zorder=3,
                            facecolor=colors[this_label], alpha=0.3)
        ax.add_artist(rct)

    # axis and tick labels
    # --------------------
    ax.set_ylabel('alpha asymmetry', fontsize=label_size)
    ax.set_xticklabels(['diagnosed', 'healthy\ncontrols'],
                       fontsize=label_size)
    ax.set_xlabel('')
    axis_frame_aes(ax, ygrid=ygrid, spine_offset=spine_offset,
                   ticklabel_size=ticklabel_size, axline_width=axline_width,
                   tick_length=tick_length, grid_width=grid_width)

    # set x tick labels again - their fontsize seems to be ignored
    ax.set_xticklabels(['diagnosed', 'healthy\ncontrols'],
                       fontsize=label_size)

    # t test value
    # ------------
    # added to figures by hand in the end
    return ax


def axis_frame_aes(ax, ygrid=True, spine_offset=25, ticklabel_size=14,
                   axline_width=2, tick_length=8, grid_width=2):
    '''Nice axis spines.'''
    sns.despine(ax=ax, trim=False, offset=spine_offset)
    _trim_y(ax)

    for tck in ax.yaxis.get_majorticklabels():
        tck.set_fontsize(ticklabel_size)

    # change width of spine lines
    ax.spines['bottom'].set_linewidth(axline_width)
    ax.spines['left'].set_linewidth(axline_width)
    ax.xaxis.set_tick_params(width=axline_width, length=tick_length)
    ax.yaxis.set_tick_params(width=axline_width, length=tick_length)

    if ygrid:
        ax.yaxis.grid(color=[0.88, 0.88, 0.88], linewidth=grid_width,
                      zorder=0, linestyle='--')


def _trim_y(ax):
    '''
    Trim only y axis.

    This is a slightly modified code copied from seaborn.
    Seaborn's despine function allows only to trim both axes so we needed a
    workaround.
    '''
    yticks = ax.get_yticks()
    if yticks.size:
        firsttick = np.compress(yticks >= min(ax.get_ylim()),
                                yticks)[0]
        lasttick = np.compress(yticks <= max(ax.get_ylim()),
                               yticks)[-1]
        ax.spines['left'].set_bounds(firsttick, lasttick)
        ax.spines['right'].set_bounds(firsttick, lasttick)
        newticks = yticks.compress(yticks <= lasttick)
        newticks = newticks.compress(newticks >= firsttick)
        ax.set_yticks(newticks)


# FIXME - maybe change to the actual grid used in the paper?
def plot_swarm_grid(study, space, contrast):
    '''Swarmplot grid for single channel pair group contrasts.'''
    psd_params = dict(study=study, space=space, contrast=contrast)
    psd_high, psd_low, ch_names = freq.get_psds(selection='asy_pairs',
                                                **psd_params)

    df_list = create_swarm_df(psd_high, psd_low)
    gridspec = dict(hspace=0.05, wspace=0.25, bottom=0.12, top=0.9,
                    left=0.07, right=0.95)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 6), gridspec_kw=gridspec)

    for idx, df in enumerate(df_list):
        plot_swarm(df, axes=axs[idx])
        ch_name = ch_names[idx].replace('-', ' - ')
        axs[idx].set_title(ch_name, fontsize=22)

    return fig


# - [ ] make a bit more universal and move to sarna as `fast_df`
# - [ ] compare and merge with the function in utils
def create_swarm_df(psd_high, psd_low):
    df_list = list()
    groups = ['diag'] * psd_high.shape[0] + ['hc'] * psd_low.shape[0]
    if psd_high.ndim == 1:
        psd_high = psd_high[:, np.newaxis]
    if psd_low.ndim == 1:
        psd_low = psd_low[:, np.newaxis]

    for ar in range(psd_high.shape[1]):
        data = np.concatenate([psd_high[:, ar], psd_low[:, ar]])
        df_list.append(pd.DataFrame(data={'asym': data, 'group': groups}))
    if len(df_list) == 1:
        df_list = df_list[0]
    return df_list


def plot_heatmap_add1(clst, ax_dict=None, scale=None):
    '''Plot results of Analyses on Standardized data (ADD1) with heatmap
    and topomap.

    Parameters
    ----------
    clst : borsar.Clusters
        Cluster-based permutation test result.

    Returns
    -------
    fig : matplotlib Figure
        Matplotlib figure object.
    obj_dict : dict
        Dictionary with axes of the figure. The dictionary contains:
        * 'heatmap': heatmap axis
        * 'colorbar': colorbar axis
        * 'topo1': lower frequency topography
        * 'topo2': higher frequency topography
    '''
    if ax_dict is None:
        fig = plt.figure(figsize=(7, 9))
        gs = fig.add_gridspec(2, 2, top=0.95, bottom=0.05, left=0.08,
                              right=0.88, height_ratios=[0.6, 0.4], hspace=0.4,
                              wspace=0.25)
        f_ax1 = fig.add_subplot(gs[0, :])
        f_ax2 = fig.add_subplot(gs[1, 0])
        f_ax3 = fig.add_subplot(gs[1, 1])
    else:
        f_ax1, f_ax2, f_ax3 = (ax_dict['heatmap'], ax_dict['topo1'],
                               ax_dict['topo2'])
        fig = f_ax1.figure

    if scale is None:
        scale = dict(heatmap_xlabel=22, heatmap_ylabel=22, cbar_label=20,
                     heatmap_xticklabels=18, cbar_yticklabels=16,
                     topo_title=18, markersize=8)

    clst_idx = [0, 1] if len(clst) > 1 else None
    clst.plot(dims=['chan', 'freq'], cluster_idx=clst_idx, axis=f_ax1,
              vmin=-4, vmax=4, alpha=0.65)
    f_ax1.set_xlabel('Frequency (Hz)', fontsize=scale['heatmap_xlabel'])
    f_ax1.set_ylabel('frontal channels', fontsize=scale['heatmap_ylabel'])

    contrast = clst.description['contrast']
    cbar_label = ('Regression t value' if 'reg' in contrast
                  else 't value')
    cbar_ax = fig.axes[-1]
    cbar_ax.set_ylabel(cbar_label, fontsize=scale['cbar_label'])

    # change ticklabels fontsize
    for tck in f_ax1.get_xticklabels():
        tck.set_fontsize(scale['heatmap_xticklabels'])

    for tck in cbar_ax.get_yticklabels():
        tck.set_fontsize(scale['cbar_yticklabels'])

    freqs1, freqs2 = (9, 10), (11.5, 12.50)
    freqlabel1, freqlabel2 = '9 - 10 Hz', '11.5 - 12.5 Hz'
    idx1, idx2 = None, None

    freq_clst = clst.clusters.sum(axis=(0, 1)) if len(clst) > 0 else None
    mark_kwargs = {'markersize': scale['markersize']}
    topo_args = dict(vmin=-4, vmax=4, mark_clst_prop=0.3,
                     mark_kwargs=mark_kwargs, border='mean',
                     extrapolate='local')

    # TODO - refactor, separte into a function
    # lower freq
    if len(clst) > 0 and freq_clst[:5].sum() > 0:
        which_clst = clst.clusters[:, :, :5].sum(axis=(1, 2)).argmax()
        # find optimal freqs
        _, freqr = clst.get_cluster_limits(which_clst)
        frql, frqh = clst.dimcoords[1][freqr][[0, -1]]
        freqs1 = (frql, frqh)
        freqlabel1 = ('{} - {} Hz'.format(frql, frqh) if not frql == frqh
                      else '{} Hz'.format(frql))

        freqlabel1 += '\np = {:.3f}'.format(clst.pvals[which_clst])
        tp1 = clst.plot(cluster_idx=which_clst, freq=freqs1, axes=f_ax2, **topo_args)
    else:
        tp1 = clst.plot(freq=freqs1, axes=f_ax2, **topo_args)
    tp1.axes.set_title(freqlabel1, fontsize=scale['topo_title'])

    if len(clst) > 0 and freq_clst[6:].sum() > 0:
        which_clst = clst.clusters[:, :, 6:].sum(axis=(1, 2)).argmax()
        # find optimal freqs
        _, freqr = clst.get_cluster_limits(which_clst)
        frql, frqh = clst.dimcoords[1][freqr][[0, -1]]
        freqs2 = (frql, frqh)
        freqlabel2 = ('{} - {} Hz'.format(frql, frqh) if not frql == frqh
                      else '{} Hz'.format(frql))

        freqlabel2 += '\np = {:.3f}'.format(clst.pvals[which_clst])
        tp2 = clst.plot(cluster_idx=which_clst, freq=freqs2, axes=f_ax3, **topo_args)
    else:
        tp2 = clst.plot(freq=freqs2, axes=f_ax3, **topo_args)
    tp2.axes.set_title(freqlabel2, fontsize=scale['topo_title'])

    obj_dict = {'heatmap': f_ax1, 'colorbar': cbar_ax, 'topo1': tp1,
                'topo2': tp2}
    return fig, obj_dict


# TODO: compare with the one in script to see which one is used in the paper
def bdi_histogram(bdi, omit_unclassified=False, bin_step=5):
    '''Plot BDI histogram from ``bdi`` dataframe of given study.'''
    msk = bdi.DIAGNOZA
    bdi_col = [c for c in bdi.columns if 'BDI' in c or 'PHQ' in c][0]
    hc = bdi.loc[~msk, bdi_col].values
    diag = bdi.loc[msk, bdi_col].values
    hc1 = hc[hc <= 5]
    hc2 = hc[(hc > 5) & (hc <= 10)]
    hc3 = hc[hc > 10]

    data = [hc1, hc2, hc3, diag]
    colors_list = [colors['hc'], colors['mid'], colors['subdiag'],
                   colors['diag']]
    if omit_unclassified:
        data.pop(1)
        colors_list.pop(1)

    max_bin = 51 if 'BDI' in bdi_col else 26
    bins = np.arange(0, max_bin, step=bin_step)
    xtcks = ([0, 10, 20, 30, 40, 50] if 'BDI' in bdi_col
             else [0, 5, 10, 15, 20, 25])

    # gridspec_kw=dict(height_ratios=[0.8, 0.2]))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    if not isinstance(ax, (list, tuple, np.ndarray)):
        ax = [ax]

    plt.sca(ax[0])

    labelsize = 26
    ticksize = 19
    plt.hist(data, bins, stacked=True, color=colors_list)
    plt.yticks([0, 5, 10, 15, 20, 25], fontsize=ticksize)
    plt.ylabel('Number of\nparticipants', fontsize=labelsize)
    plt.xticks(xtcks, fontsize=ticksize)
    plt.xlabel(bdi_col, fontsize=labelsize)
    ax[0].set_xlim((0, xtcks[-1]))
    ax[0].set_ylim((0, 28))
    return fig, ax[0]


def plot_panel(bdi, bar_h=0.6, seed=22):
    '''Plot contrast panels.'''

    # create variables
    # ----------------

    # dataframe
    random_state = np.random.RandomState(seed)
    AA_diag = random_state.uniform(low=-2, high=0.75, size=20)
    AA_control = random_state.uniform(low=-1, high=1.5, size=20)
    group = ['diag'] * 20 + ['hc'] * 20
    data = {'group': group, 'asym': np.concatenate([AA_diag, AA_control])}
    df = pd.DataFrame(data)
    df['group'] = df['group'].astype('category')

    # regression data
    noise = random_state.uniform(low=0, high=10, size=bdi.shape[0])
    y = bdi['BDI-II'].values * - 0.1 + noise
    bdi.loc[:, 'y'] = y

    # regression groups
    diag = bdi.DIAGNOZA
    hc = ~diag & (bdi['BDI-II'] <= 5)
    mid = ~diag & (bdi['BDI-II'] > 5) & (bdi['BDI-II'] <= 10)
    sub = ~diag & (bdi['BDI-II'] > 10)

    # range of BDI values
    min_diag_bdi = bdi.loc[diag, 'BDI-II'].min()
    msk2 = diag & (bdi['BDI-II'] > min_diag_bdi)
    second_min_diag_bdi = bdi.loc[msk2, 'BDI-II'].min()

    # prepare figure
    # --------------
    grid = dict(height_ratios=[0.55, 0.45], hspace=0.75, wspace=0.35,
                bottom=0.1, top=0.98, left=0.14, right=0.97)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8),
                            gridspec_kw=grid)

    # plot contrasts examples
    # -----------------------

    # contrast
    plot_swarm(df, ax=axs[0, 0])

    # regression
    axs[0, 1].set_xlim([-3, 53])
    sns.regplot(x=bdi['BDI-II'].values, y=bdi['y'].values, color=colors['mid'],
                ax=axs[0, 1], scatter=True, scatter_kws={'s': 0})

    # add group-colored scatter to regression plot
    color_list = [colors['hc'], colors['mid'], colors['subdiag'],
                  colors['diag']]
    for msk, col in zip([hc, mid, sub, diag], color_list):
        axs[0, 1].scatter(bdi[msk]['BDI-II'], bdi[msk]['y'], alpha=0.7,
                          c=col[np.newaxis, :], edgecolor='white', linewidth=0,
                          zorder=4, s=100)

    # prepare contrast legend vars
    # ----------------------------
    cntr1, cntr2 = -2, -1
    bar1y, bar2y = [c - bar_h / 2 for c in [cntr1, cntr2]]
    length_to_end = 50 - second_min_diag_bdi

    # group contrast legends
    # ----------------------
    rectanges = _create_group_rectangles(
        bar2y, bar1y, bar_h, second_min_diag_bdi, length_to_end)
    for rct in rectanges:
        axs[1, 0].add_artist(rct)

    # regression contrast legends
    # ---------------------------
    rectanges = _create_regression_rectanges(
        bar1y, bar2y, bar_h, length_to_end, second_min_diag_bdi)
    for rct in rectanges:
        axs[1, 1].add_artist(rct)

    # aesthetics
    # ----------
    axs[0, 1].set_ylabel("")
    axs[0, 1].set_xticklabels([])
    axs[0, 1].set_xlabel('deression score', fontsize=18)

    # adjust regression panel x limits
    axs[0, 1].set_xlim([-5, 55])

    for ax in axs[0, :]:
        # equal tick spacing through data range
        ylm = ax.get_ylim()
        new_ticks = np.linspace(ylm[0], ylm[1], 5)
        ax.set_yticks(new_ticks)
        ax.set_yticklabels([])

        # nice axis spines
        axis_frame_aes(ax, ygrid=True)

    for ax in axs[1, :]:
        # limits, labels and ticks
        ax.set_xlim((0, 50))
        ax.set_xlabel('depression score', fontsize=18)
        ax.set_ylabel('')
        ax.set_xticks([0, 10, 20, 30, 40, 50])

        # tick labels fontsize
        for tck in ax.xaxis.get_majorticklabels():
            tck.set_fontsize(17)
        for tck in ax.yaxis.get_majorticklabels():
            tck.set_fontsize(18)

        # add grid
        ax.xaxis.grid(color=[0.88, 0.88, 0.88], linewidth=2,
                      zorder=0, linestyle='--')

    axs[0, 0].set_xticklabels(['diagnosed', 'healthy\ncontrols'], fontsize=18)

    axs[1, 0].set_ylim((-2.5, -0.5))
    axs[1, 0].set_yticks([cntr2, cntr1])
    axs[1, 0].set_yticklabels(['DvsHC', 'SvsHC'], fontsize=19)

    axs[1, 1].set_ylim((-2.5, -0.5))
    axs[1, 1].set_yticks([cntr2, cntr1])
    axs[1, 1].set_yticklabels(['allReg', 'DReg'], fontsize=19)

    return fig


def src_plot(clst, cluster_idx=None, colorbar='mayavi', azimuth=[35, 125],
             elevation=[None, None], cluster_p=True, vmin=-3, vmax=3,
             figure_size=None, backface_culling=False):
    '''Plot source-level clusters as two-axis image.

    Parameters
    ----------
    clst : borsar.cluster.Clusters
        Cluster results to plot.
    cluster_idx : int
        Cluster to plot.
    azimuth_pos : list of int
        List of two azimuth position of the brain images.
    colorbar : bool | 'mayavi' | 'matplotlib'
        Whether to show colorbar. True by default.
    cluster_p : bool | 'mayavi' | 'matplotlib'
        Whether to show cluster p value text. True by default.
    vmin : value
        Minimum value of the colormap.
    vmax : value
        Maximum value of the colormap.
    backface_culling : bool
        Backface culling seems to help with recent transparency problem with
        mayavi (see )

    Returns
    -------
    fig : matplotlib figure
        Matplotlib figure with images.'''

    from mayavi import mlab
    if isinstance(cluster_idx, str) and cluster_idx == 'all':
        n_clusters = len(clst)
        if n_clusters == 0:
            cluster_idx = None
        elif n_clusters == 1:
            cluster_idx = 0
        else:
            cluster_idx = np.arange(n_clusters).tolist()

    # plot the 3d brain
    brain = clst.plot(cluster_idx=cluster_idx, vmin=vmin, vmax=vmax,
                      figure_size=figure_size)

    if isinstance(colorbar, bool):
        colorbar = 'matplotlib' if colorbar else ''

    if isinstance(cluster_p, bool):
        cluster_p = 'matplotlib' if cluster_p else ''

    if not colorbar == 'mayavi':
        brain.hide_colorbar()

    if not cluster_p == 'mayavi':
        # fing and hide cluster p text
        clst_txt = brain.texts_dict['time_label']['text']
        clst_txt.remove()

    if backface_culling:
        brain.data['surfaces'][0].actor.property.backface_culling = True
        for ldict in brain._label_dicts.values():
            ldict['surfaces'][0].actor.property.backface_culling = True

    imgs = list()
    for azi, ele in zip(azimuth, elevation):
        mlab.view(azimuth=azi, elevation=ele)
        img = mlab.screenshot(antialiased=True)
        imgs.append(img)
    mlab.close()

    bottom = (0.05 + 0.17 * (colorbar == 'matplotlib')
              + 0.1 * (cluster_p == 'matplotlib'))
    gridspec = {'hspace': 0.1, 'wspace': 0.1, 'left': 0.025, 'right': 0.975,
                'top': 0.95, 'bottom': bottom}
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6), gridspec_kw=gridspec)

    for this_ax, this_img in zip(ax, imgs):
        # trim image
        iswhite = this_img.mean(axis=-1) == 255
        cols = np.where(~iswhite.all(axis=0))[0][[0, -1]]
        rows = np.where(~iswhite.all(axis=1))[0][[0, -1]]
        this_img = this_img[rows[0]:rows[1] + 1, cols[0]:cols[1] + 1]

        # show image
        this_ax.imshow(this_img)
        this_ax.set_axis_off()

    if colorbar == 'matplotlib':
        import matplotlib as mpl

        c_map_ax = fig.add_axes([0.2, 0.01, 0.6, 0.05])

        cmap = plt.get_cmap('RdBu_r')
        norm = mpl.colors.Normalize(vmin, vmax)
        cbar = mpl.colorbar.ColorbarBase(c_map_ax, cmap=cmap, norm=norm,
                                         orientation='horizontal')

        cbar.ax.xaxis.set_ticks_position('top')
        cbar.set_label('t value', labelpad=-70, fontsize=18)

        # increse cbar x tick labels fontsize
        for tck in cbar.ax.xaxis.get_ticklabels():
            tck.set_fontsize(14)

    if cluster_p == 'matplotlib':
        bottom = 0.06 + 0.18 * (colorbar == 'matplotlib')
        # text_ax = fig.add_axes([0.2, 0.01, 0.6, 0.05])
        if len(clst) == 0:
            pval_txt = 'NA'
        else:
            pval_txt = ', '.join(['{:.2f}'.format(val) for val in clst.pvals])
        fig.text(0.5, bottom, 'cluster p values: {}'.format(pval_txt),
                 horizontalalignment='center', fontsize=15)

    return fig


def _create_regression_rectanges(bar1y, bar2y, bar_h, length_to_end,
                                 second_min_diag_bdi):
    # rct1 = plt.Rectangle((0, bar1y), 5, bar_h, color=colors['hc'], zorder=5)
    # rct2 = plt.Rectangle((5, bar1y), 5, bar_h, color=colors['mid'], zorder=6)
    # rct4 = plt.Rectangle((10, bar1y), 40, bar_h, color=colors['subdiag'],
    #                      zorder=5)
    rct3 = plt.Rectangle((0, bar1y), 5, bar_h, color=colors['hc'], zorder=5)
    rct5 = plt.Rectangle((10, bar1y), 40, bar_h, color=colors['subdiag'],
                         zorder=5)
    rct6 = plt.Rectangle((5, bar1y), 5, bar_h, color=colors['mid'], zorder=5)
    rct7 = plt.Rectangle((second_min_diag_bdi, bar1y), length_to_end,
                         bar_h / 2, color=colors['diag'], zorder=5)
    rct8 = plt.Rectangle((second_min_diag_bdi, bar2y), length_to_end, bar_h,
                         color=colors['diag'], zorder=5)
    # return [rct1, rct2, rct3, rct4, rct5, rct6, rct7, rct8]
    return [rct3, rct5, rct6, rct7, rct8]


def _create_group_rectangles(bar1y, bar2y, bar_h, second_min_diag_bdi,
                             length_to_end):
    rct1 = plt.Rectangle((0, bar1y), 5, bar_h, color=colors['hc'], zorder=5)
    rct2 = plt.Rectangle((second_min_diag_bdi, bar1y), length_to_end, bar_h,
                         color=colors['diag'], zorder=5)
    rct3 = plt.Rectangle((0, bar2y), 5, bar_h, color=colors['hc'], zorder=5)
    rct4 = plt.Rectangle((10, bar2y), 40, bar_h,
                         color=colors['subdiag'], zorder=5)
    return [rct1, rct2, rct3, rct4]


def plot_aggregated(agg_df, distributions, ax=None, eff='d', confounds=False):
    '''Plot aggregated effect sizes, confidence intervals and bayes factors for
    channel pairs analyses.

    Parameters
    ----------
    agg_df: TODO
    distributions: TODO
    ax : matplotlib axis
        Axis to plot to.
    eff : str
        Effect size to plot. ``'r'`` shows effects for linear relationship
        analyses with Pearson's r as the effect size. ``'d'`` shows effects for
        group contrasts with Cohen's d as the effect size. Defaults to ``'d'``.

    Returns
    -------
    ax: matplotlib axis
        Axis used to plot to.
    '''
    if ax is None:
        # create axis to plot to
        fig_size = (11, 13.5) if eff == 'r' else (11, 12)
        gridspec = ({'left': 0.25, 'bottom': 0.1, 'right': 0.85} if eff == 'r'
                    else {'left': 0.18, 'bottom': 0.1, 'right': 0.8})
        fig, ax = plt.subplots(figsize=fig_size, gridspec_kw=gridspec)

    ypos = 5
    labels_pos = list()
    labels = list()

    # 'd' vs 'r' effect size (group contrasts vs linear relationships)
    addpos = 0.09 if eff == 'd' else 0.065
    ch_names = ['F3-F4', 'F7-F8']
    contrasts = ['cvsd', 'cvsc'] if eff == 'd' else ['dreg', 'cdreg']
    distr_color = (colors['hc'] if eff == 'd' else colors['subdiag'])

    for contrast in contrasts:
        for space in ['avg', 'csd']:
            # channel pair loop
            for ch_idx in range(2):
                # get relevant data
                # TODO - from the table

                # plot distribution, ES and CI
                v = _plot_dist_esci(ax, ypos, stats, color=distr_color)
                v['bodies'][0].set_zorder(4)

                # slight y tick labeling differences
                if len(studies) > 1:
                    label_schematic = '{}, {}, {}\nstudies {}'
                    studies_str = ', '.join(studies[:-1]) + ' & ' + studies[-1]
                    label = label_schematic.format(
                        utils.translate_contrast[contrast], space.upper(),
                        ch_names[ch_idx], studies_str)
                else:
                    label = '{}, {}\n{}, study {}'.format(
                        utils.translate_contrast[contrast], space.upper(),
                        ch_names[ch_idx], studies[0])

                labels_pos.append(ypos)
                labels.append(label)

                # add bf01 text:
                bf_text = '{:.2f}'.format(stats['bf01'])
                ax.text(stats['es'], ypos + addpos, bf_text, fontsize=16,
                        horizontalalignment='center', color='w', zorder=7)

                ypos -= 0.5

    # aesthetics
    # ----------
    lims = (-1.25, 1.25) if eff == 'd' else (-0.7, 0.7)
    xticks = ([-1, -0.5, 0, 0.5, 1] if eff == 'd'
              else [-0.6, -0.3, 0, 0.3, 0.6])
    xlab = ("Effect size\n(Cohen's d)" if eff == 'd'
            else "Effect size\n(Pearson's r)")
    cntr = 'group contrasts' if eff == 'd' else 'linear relationships'

    ax.set_xlim(lims)
    plt.yticks(labels_pos, labels, fontsize=15)
    plt.xticks(xticks, fontsize=15)
    ylim = ax.get_ylim()

    ax.grid(color=[0.85, 0.85, 0.85], linewidth=1.5, linestyle='--')
    ax.vlines(0, ymin=ylim[0], ymax=ylim[1], color=[0.5] * 3, lw=2.5)
    ax.set_ylim(ylim) # make sure vlines do not change y lims

    ax.set_xlabel(xlab, fontsize=20)
    ax.set_title('Aggregated channel pair results\nfor {}'.format(cntr),
                 fontsize=24, pad=25)
    return ax


def _plot_dist_esci(ax, ypos, stats, color=None):
    '''Plots a single bootstraps distribution along with effect size and
    bootstrap confidence interval for the effect size.

    Used when plotting aggregated channel pairs figures (``plot_aggregated``).
    '''
    from dabest.plot_tools import halfviolin

    color = color if color is not None else ds.utils.colors['hc']

    v = ax.violinplot(stats['bootstraps'], positions=[ypos],
                      showextrema=False, showmedians=False,
                      widths=0.5, vert=False)
    halfviolin(v, fill_color=color, alpha=0.85, half='top')

    line_color = np.array([0] * 3) / 255
    ax.plot(stats['es'], [ypos], marker='o', color=line_color,
                       markersize=12, zorder=6)
    ax.plot(stats['ci'], [ypos, ypos], linestyle="-", color=line_color,
            linewidth=3.5, zorder=5)
    return v


def full_fig5_supplement_plot(contrast, studies):
    '''Plot whole panel plot for figure 5 supplements, for given contrast
    and studies.

    Parameters
    ----------
    contrast : str
        Statistical contrast to use. See ``DiamSar.analysis.run_analysis``
        for contrast desctription.
    studies : list of studies
        Studies to use. See ``DiamSar.analysis.run_analysis`` for study
        description.
        '''
    scale = dict(heatmap_xlabel=15, heatmap_ylabel=15, cbar_label=12,
                 heatmap_xticklabels=14, cbar_yticklabels=10,
                 topo_title=14, markersize=5)

    spaces = ['avg', 'csd']
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(4, 4, top=0.85, bottom=0.05, left=0.15,
                          right=0.9, height_ratios=[0.6, 0.4, 0.6, 0.4],
                          hspace=0.75, wspace=1.)

    for study_idx, study in enumerate(studies):
        for space_idx, space in enumerate(spaces):
            # create axes
            # -----------
            row_idx = space_idx * 2
            col_idx = [study_idx * 2, study_idx * 2 + 1]
            f_ax1 = fig.add_subplot(gs[row_idx, col_idx[0]:col_idx[1] + 1])
            f_ax2 = fig.add_subplot(gs[row_idx + 1, col_idx[0]])
            f_ax3 = fig.add_subplot(gs[row_idx + 1, col_idx[1]])

            # modify topo axes position
            # -------------------------
            pos1 = list(f_ax2.get_position().bounds)
            pos2 = list(f_ax3.get_position().bounds)
            w, h = pos1[2], pos1[3]
            pos1[1] -= h * 0.334
            pos1[2] *= 1.334
            pos1[3] *= 1.334
            f_ax2.set_position(pos1)

            pos2[0] -= w * 0.334
            pos2[1] -= h * 0.334
            pos2[2] *= 1.334
            pos2[3] *= 1.334
            f_ax3.set_position(pos2)

            # read cluster results
            clst = ds.analysis.load_stat(study=study, contrast=contrast,
                                         space=space, stat_dir='stats add1',
                                         avg_freq=False, selection='frontal',
                                         transform=['log', 'zscore'])
            # sort channels left to right
            # (does not affect results, only channel order)
            clst = ds.analysis.sort_clst_channels(clst)

            # plot heatmap + two topographies
            ax_dict = dict(heatmap=f_ax1, topo1=f_ax2, topo2=f_ax3)
            fig, obj_dct = ds.viz.plot_heatmap_add1(clst, ax_dict=ax_dict,
                                                    scale=scale)

            # remove requency xlabel when topo titles are two lines long
            # (otherwise there is ugly text overlap)
            n_lines = [len(obj_dct[key].axes.get_title().split('\n'))
                       for key in ['topo1', 'topo2']]
            longer_line = (np.array(n_lines) > 1).any()
            if longer_line:
                obj_dct['heatmap'].set_xlabel('')

            # label rows, columns
            if study_idx == 0:
                # label row
                pos_y = pos1[1] + 1.5 * pos1[3]
                text = space.upper()
                plt.text(0.075, pos_y, text, va='center', ha='center',
                         transform=fig.transFigure, fontsize=21,
                         rotation=90)

            if row_idx == 0:
                # label column
                pos_x = pos1[0] + 1.15 * pos1[2]
                text = 'STUDY {}'.format(ds.utils.translate_study[study])
                plt.text(pos_x, 0.885, text, va='center', ha='center',
                         transform=fig.transFigure, fontsize=21)

    contrast_name = ds.utils.translate_contrast[contrast]
    fig.suptitle('{} contrast'.format(contrast_name), fontsize=24,
                 fontweight='bold')
    return fig


# TODO: move to borsar!
def zoom_topo(topo, xlim, ylim, linewidth=1.5):
    '''Zoom topography to a certain range of x and y values.
    Useful when presenting frontal asymmetry results.

    topo : borsar.viz.Topo
        Topomap object.
    xlim : (low, high)
        Limits used to crop the x axis.
    ylim : (low, high)
        Limits used to crop the y axis.
    linewidth : float
        Optional: change the head outline linewidth to counteract the zooming.
        Defaults to ``1.5``.
    '''
    # currntly works only for one topo (len(topo) == 1)
    for tp in topo:
        ax = tp.axes
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        [line.set_clip_on(True) for line in tp.head]
        [line.set_linewidth(linewidth) for line in tp.head]
