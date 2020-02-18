import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import seaborn as sns
import pandas as pd

from borsar.stats import format_pvalue
from borsar.viz import Topo

from sarna.viz import prepare_equal_axes

from . import freq
from .analysis import load_stat
from .utils import colors


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


def plot_grid_cluster(stats_clst, contrast, vlim=3):
    '''
    Plot cluster-corrected contrast results in a reference by study grid.

    Parameters
    ----------
    stats_clst : pandas.DataFrame
        DataFrame with information about all cluster-based analyses.
    contrast : str
        Statistical contrast represented as string. For example ``'cvsd'``
        means controls vs diagnosed.
    vlim : float
        Value limits for the topomap colormap (in values of the t statistic).

    Returns
    -------
    fig : matplotlib.Figure
        Matplotlib figure with visualized results.
    '''
    fig = plt.figure(figsize=(9, 6))
    ax = prepare_equal_axes(fig, [2, 3], space=[0.12, 0.85, 0.02, 0.8])

    example_stat = load_stat(study='C', contrast='cvsd', space='avg')
    vmin, vmax = -vlim, vlim
    axis_limit = 2.25

    for row_idx, space in enumerate(['avg', 'csd']):
        query_str = 'contrast=="{}" & space == "{}"'.format(contrast, space)
        this_stat = stats_clst.query(query_str)

        for col_idx, study in enumerate(list('ABC')):
            this_ax = ax[row_idx, col_idx]
            if study in this_stat.study.values:
                # read analysis result from disc
                stat = load_stat(study=study, contrast=contrast, space=space)

                # plot results
                topo = stat.plot(cluster_idx=0, outlines='skirt',
                                 extrapolate='head', axes=this_ax, vmin=vmin,
                                 vmax=vmax)

                # modify channel and cluster mark sizes and add text
                topo.chan.set_sizes([2.5])
                if len(topo.marks) > 0:
                    topo.marks[0].set_markersize(6.)
                    this_ax.set_title(format_pvalue(stat.pvals[0]),
                                      fontsize=14, pad=0)
                else:
                    this_ax.set_title('no clusters', fontsize=14, pad=0)
            else:
                plot_unavailable(example_stat, axis=this_ax)

            # set figure limits
            this_ax.set_ylim((-axis_limit, axis_limit))
            this_ax.set_xlim((-axis_limit, axis_limit))

    # add colorbar
    cbar_ax = fig.add_axes([0.87, 0.08, 0.03, 0.67])
    cbar = plt.colorbar(mappable=topo.img, cax=cbar_ax)
    cbar.set_label('t values', fontsize=12)

    # add study labels
    # ----------------
    for idx, letter in enumerate(list('ABC')):
        this_ax = ax[0, idx]
        box = this_ax.get_position()
        mid_x = box.corners()[:, 0].mean()

        plt.text(mid_x, 0.87, letter, va='center', ha='center',
                 transform=fig.transFigure, fontsize=21)

        if idx == 1:
            plt.text(mid_x, 0.935, 'STUDY', va='center', ha='center',
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

    return fig


def plot_multi_topo(psds_avg, info_frontal, info_asy):
    '''Creating combined Topo object for multiple psds'''
    gridspec = dict(height_ratios=[0.5, 0.5],
                    width_ratios=[0.47, 0.47, 0.06],
                    hspace=0.05, wspace=0.4,
                    bottom=0.05, top=0.95, left=0.07, right=0.85)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 7.5),
                            gridspec_kw=gridspec)

    topos = list()
    topomap_args = dict(extrapolate='head', outlines='skirt', border='mean')

    for val, ax in zip(psds_avg[:2], axs[0, :]):
        topos.append(Topo(val, info_frontal, cmap='Reds', vmin=-28,
                          vmax=-26, axes=ax, **topomap_args))

    for val, ax in zip(psds_avg[2:], axs[1, :]):
        topos.append(Topo(val, info_asy, cmap='RdBu_r', vmin=-0.13,
                          vmax=0.13, axes=ax, show=False,
                          **topomap_args))

    for tp in topos:
        tp.solid_lines()

    cbar1 = plt.colorbar(axs[0, 0].images[0], cax=axs[0, 2])
    plt.colorbar(axs[1, 0].images[0], cax=axs[1, 2])
    tck = cbar1.get_ticks()
    cbar1.set_ticks(tck[::2])
    # tck_lab = cbar1.get_ticklabels()
    # cbar1.set_ticklabels(tck_lab, fontsize=15)

    axs[0, 0].set_title('diagnosed', fontsize=22).set_position([.5, 1.1])
    axs[0, 1].set_title('healthy\nControls',
                        fontsize=22).set_position([.5, 1.1])
    axs[0, 0].set_ylabel('alpha asymmetry', fontsize=22, labelpad=20)
    axs[1, 0].set_ylabel('alpha asymmetry', fontsize=22, labelpad=20)
    axs[0, 2].set_ylabel('log(alpha power)', fontsize=22)
    axs[1, 2].set_ylabel('alpha power', fontsize=22)

    fig.canvas.draw()

    for row in range(2):
        cbar_bounds = axs[row, 2].get_position().bounds
        topo_bounds = axs[row, 0].get_position().bounds
        tcklb = axs[row, 2].get_yticklabels()
        axs[row, 2].set_yticklabels(tcklb, fontsize=15)
        axs[row, 2].set_position([cbar_bounds[0], topo_bounds[1],
                                  cbar_bounds[2], topo_bounds[3]])

    return fig


def plot_swarm(df, ax=None, ygrid=True):
    '''
    Swarmplot for single channel pairs asymmetry. Used for group contrast
    visualization.
    '''
    if ax is None:
        gridspec = dict(bottom=0.12, top=0.95, left=0.14, right=0.99)
        fig, ax = plt.subplots(figsize=(8, 6), gridspec_kw=gridspec)

    # swarmplot
    # ---------
    ax = sns.swarmplot("group", "asym", data=df, size=10,
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
    ax.set_ylabel('alpha asymmetry', fontsize=20)
    ax.set_xticklabels(['diagnosed', 'healthy\ncontrols'],
                       fontsize=20)
    ax.set_xlabel('')

    for tck in ax.yaxis.get_majorticklabels():
        tck.set_fontsize(14)

    # change width of spine lines
    sns.despine(ax=ax, trim=True)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.xaxis.set_tick_params(width=3, length=6)
    ax.yaxis.set_tick_params(width=3, length=6)

    if ygrid:
        ax.yaxis.grid(color=[0.88, 0.88, 0.88], linewidth=2,
                      zorder=0, linestyle='--')

    # t test value
    # ------------
    # ...
    # 1. t, p = ttest_ind(df.query(...), df.query(...))
    # 2. format text: 't = {:.2f}, p = {:.2f}'.format(t, p)
    # 3. plot text with matplotlib
    return ax


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


def create_swarm_df(psd_high, psd_low):
    df_list = list()
    groups = ['diag'] * psd_high.shape[0] + ['hc'] * psd_low.shape[0]
    for ar in [0, 1]:
        data = np.concatenate([psd_high[:, ar], psd_low[:, ar]])
        df_list.append(pd.DataFrame(data={'asym': data, 'group': groups}))
    return df_list


def plot_heatmap_add1(clst):
    '''Plot results of Standardized Analyses (ADD1) with heatmap and topo.'''
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.5, top=0.95, bottom=0.05,
                          left=0.07)

    f_ax1 = fig.add_subplot(gs[:2, :])
    f_ax2 = fig.add_subplot(gs[2, 0])
    f_ax3 = fig.add_subplot(gs[2, 1])

    clst_idx = [0, 1] if len(clst) > 1 else None
    clst.plot(dims=['chan', 'freq'], cluster_idx=clst_idx, axis=f_ax1,
              vmin=-4, vmax=4)
    f_ax1.set_xlabel('Frequency (Hz)', fontsize=18)
    f_ax1.set_ylabel('frontal channels', fontsize=18)

    contrast = clst.description['contrast']
    cbar_label = ('Regression t value' if 'reg' in contrast
                  else 't value')
    cbar_ax = fig.axes[-1]
    cbar_ax.set_ylabel(cbar_label, fontsize=16)

    # change ticklabels fontsize
    for tck in f_ax1.get_xticklabels():
        tck.set_fontsize(16)

    for tck in cbar_ax.get_yticklabels():
        tck.set_fontsize(15)

    # FIXME: change axis position (doesn't work...)
    # fig.canvas.draw()
    # bounds = cbar_ax.get_position().bounds
    # bounds = (0.86, *bounds[1:])
    # cbar_ax.set_position(bounds)

    freqs1, freqs2 = (9, 10), (11.5, 12.50)
    freqlabel1, freqlabel2 = '9 - 10 Hz', '11.5 - 12.5 Hz'
    idx1, idx2 = None, None

    if contrast == 'dreg' and clst.description['study'] == 'C':
        freqlabel1 += '\np = {:.3f}'.format(clst.pvals[1])
        freqlabel2 += '\np = {:.3f}'.format(clst.pvals[0])
        idx1, idx2 = 1, 0

    # topo 1
    tp1 = clst.plot(cluster_idx=idx1, freq=freqs1, axes=f_ax2,
                    vmin=-4, vmax=4, mark_clst_prop=0.3,
                    border='mean')
    tp1.axes.set_title(freqlabel1, fontsize=16)

    # topo 2
    tp2 = clst.plot(cluster_idx=idx2, freq=freqs2, axes=f_ax3,
                    vmin=-4, vmax=4, mark_clst_prop=0.3,
                    border='mean')
    tp2.axes.set_title(freqlabel2, fontsize=16)

    obj_dict = {'heatmap': f_ax1, 'colorbar': cbar_ax, 'topo1': tp1,
                'topo2': tp2}
    return fig, obj_dict


def bdi_bdi_histogram(bdi):
    '''Plot BDI histogram from ``bdi`` dataframe of given study.'''
    msk = bdi.DIAGNOZA
    bdi_col = [c for c in bdi.columns if 'BDI' in c][0]
    hc = bdi.loc[~msk, bdi_col].values
    diag = bdi.loc[msk, bdi_col].values
    hc1 = hc[hc <= 5]
    hc2 = hc[(hc > 5) & (hc <= 10)]
    hc3 = hc[hc > 10]

    # gridspec_kw=dict(height_ratios=[0.8, 0.2]))
    fig, ax = plt.subplots(figsize=(5, 6))
    if not isinstance(ax, (list, tuple, np.ndarray)):
        ax = [ax]

    plt.sca(ax[0])
    bins = np.arange(0, 51, step=5)
    plt.hist([hc1, hc2, hc3, diag], bins, stacked=True,
             color=[colors['hc'], colors['mid'], colors['subdiag'],
                    colors['diag']])
    plt.yticks([0, 5, 10, 15, 20, 25], fontsize=14)
    plt.ylabel('Number of participants', fontsize=16)
    plt.xticks([0, 10, 20, 30, 40, 50], fontsize=14)
    plt.xlabel(bdi_col, fontsize=16)
    ax[0].set_xlim((0, 50))
    ax[0].set_ylim((0, 28))
    return fig, ax
