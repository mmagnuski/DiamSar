import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import seaborn as sns
import pandas as pd

from borsar.stats import format_pvalue
from borsar.viz import Topo

from sarna.viz import prepare_equal_axes
from DiamSar.analysis import load_stat
from DiamSar.utils import colors
from DiamSar import freq

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


def plot_ds_swarm(df, axes=None):
    '''Plotting swarmplot for asymmetry in pairs comparison for single pair'''
    ax = sns.swarmplot("group", "asym", data=df, size=10,
                       palette=[colors['diag'], colors['hc']], ax=axes)
    means = df.groupby('group').mean()
    x_pos = ax.get_xticks()
    x_lab = [x.get_text() for x in ax.get_xticklabels()]
    width = np.diff(x_pos)[0] * 0.2

    for this_label, this_xpos in zip(x_lab, x_pos):
        this_mean = means.loc[this_label, 'asym']
        ax.plot([this_xpos - width, this_xpos + width], [this_mean, this_mean],
                color=colors[this_label], lw=2.)
        # add CI
        this_sem = sem(df.query('group == "{}"'.format(this_label)).asym.values)
        rct = plt.Rectangle((this_xpos - width, this_mean - this_sem),
                            width * 2, this_sem * 2,
                            facecolor=colors[this_label], alpha=0.3)
        ax.add_artist(rct)

    ax.set_ylabel('alpha asymmetry', fontsize=20)
    ax.set_xticklabels(['diagnosed', 'healthy\ncontrols'],
                       fontsize=20)
    ax.set_xlabel('')
    return ax


def plot_swarm_asy(study, space, contrast):
    '''Plotting swarmplot for asymmetry in pairs comparison
    for pairs F3 - F4 and F7 - F8'''
    psd_params = dict(study=study, space=space, contrast=contrast)
    psd_high, psd_low, ch_names = freq.get_psds(selection='asy_pairs',
                                                **psd_params)

    data_concat = []
    for ar in [0, 1]:
        data = np.concatenate([psd_high[:, ar], psd_low[:, ar]])
        data_concat.append(data)

    groups = ['diag'] * psd_high.shape[0] + ['hc'] * psd_low.shape[0]
    df_f1f3 = pd.DataFrame(data={'asym': data_concat[0], 'group': groups})
    df_f7f8 = pd.DataFrame(data={'asym': data_concat[1], 'group': groups})

    df_list = [df_f1f3, df_f7f8]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(17, 5),
                            gridspec_kw=dict(width_ratios=[1, 1],
                                             hspace=0.05, wspace=0.25,
                                             bottom=0.05, top=0.9, left=0.07,
                                             right=0.85))
    for idx, df in enumerate(df_list):
        plot_ds_swarm(df, axes=axs[idx])
        ch_name = ch_names[idx].replace('-', ' - ')
        axs[idx].set_title(ch_name, fontsize=22)

    return fig
