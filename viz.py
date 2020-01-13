from sarna.init import *
from borsar.stats import format_pvalue

from sarna.viz import prepare_equal_axes
from DiamSar.analysis import load_stat


# - [ ] change to use Info
# - [ ] make sure Info in .get_data() are cached
def plot_unavailable(stat, axis=None):
    '''Plot "empty" topography with 'NA' in the middle.'''
    topo = stat.plot(cluster_idx=0, outlines='skirt', extrapolate='head', axes=axis)
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


def plot_multi_topo(psd_avg_high_fr, psd_avg_low_fr, psd_avg_high_asy, psd_avg_low_asy,
                    info_sel_fr, info_sel_asy, vmax_fr, vmin_fr, vmax_asy):
    '''Creating figure for multiple psds'''
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                       gridspec_kw=dict(height_ratios=[0.5, 0.5], hspace=0.05, wspace=0.05,
                                        bottom=0.05, top=0.9, left=0.05, right=0.7))
    cax1 = fig.add_axes([0.75, 0.565, 0.05, 0.3])
    cax2 = fig.add_axes([0.75, 0.1, 0.05, 0.3])

    tp_low_fr = Topo(psd_avg_low_fr, info=info_sel_fr, cmap='Reds',
          vmin=vmin_fr, vmax=vmax_fr, extrapolate='head',
         outlines = 'skirt', axes = axs[0,0], border = 'mean')
    tp_high_fr = Topo(psd_avg_high_fr, info=info_sel_fr, cmap='Reds',
          vmin=vmin_fr, vmax=vmax_fr, extrapolate='head',
         outlines = 'skirt', axes = axs[0,1], border = 'mean')
    tp_low_asy = Topo(psd_avg_low_asy, info=info_sel_asy, cmap='RdBu_r',
          vmin=-vmax_asy, vmax=vmax_asy, extrapolate='head',
         outlines = 'skirt', axes = axs[1,0], border = 'mean')
    tp_high_asy = Topo(psd_avg_high_asy, info=info_sel_asy, cmap='RdBu_r',
          vmin=-vmax_asy, vmax=vmax_asy, extrapolate='head',
         outlines = 'skirt', axes = axs[1,1], border = 'mean')
    for tp in [tp_low_fr, tp_high_fr, tp_low_asy, tp_high_asy]:
        tp.solid_lines()

    cmap1 = mpl.cm.get_cmap('Reds')
    norm1 = mpl.colors.Normalize(vmin=-28, vmax=-26)
    cb1 = mpl.colorbar.ColorbarBase(cax1, cmap=cmap1,
                                norm=norm1,
                                orientation='vertical')
    cmap2 = mpl.cm.get_cmap('RdBu_r')
    norm2 = mpl.colors.Normalize(vmin=-0.13, vmax=0.13)
    cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap2,
                                norm=norm2,
                                orientation='vertical')

    cax1.set_ylabel('log(alpha power)')
    cax2.set_ylabel('alpha power')
    cax1.set_yticks([0., 0.25,  0.5 ,  0.75 , 1.   ])
    cax1.set_yticklabels([-28, -27.5, -27, -26.5, -26])
    axs[0, 0].set_title('group')
    axs[0, 0].set_ylabel('frontal')
    axs[1, 0].set_ylabel('asy')
    axs[0, 1].set_title('group')
    plt.locator_params(nbins=5)
    plt.rc('axes', labelsize=22)
    plt.rc('axes', titlesize=22)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)

    return fig
