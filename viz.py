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

    axs[0, 0].set_title('diagnosed', fontsize=20).set_position([.5, 1.1])
    axs[0, 1].set_title('healthy\ncontrols',
                        fontsize=20).set_position([.5, 1.1])
    axs[0, 0].set_ylabel('alpha\npower', fontsize=20, labelpad=18)
    axs[1, 0].set_ylabel('alpha\nasymmetry', fontsize=20, labelpad=18)
    cbar_ax[0].set_ylabel('log(alpha power)', fontsize=16)
    cbar_ax[1].set_ylabel('alpha power', fontsize=16)

    # correct cbar position with respect to the topo axes
    fig.canvas.draw()
    for row in range(2):
        cbar_bounds = cbar_ax[row].get_position().bounds
        topo_bounds = axs[row, 0].get_position().bounds
        cbar_ax[row].set_position([cbar_bounds[0], topo_bounds[1] + 0.05,
                                   cbar_bounds[2], topo_bounds[3] - 0.07])

    return fig, axs


def plot_swarm(df, ax=None, ygrid=True):
    '''
    Swarmplot for single channel pairs asymmetry. Used for group contrast
    visualization.
    '''
    if ax is None:
        gridspec = dict(bottom=0.2, top=0.92, left=0.2, right=0.99)
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
    axis_frame_aes(ax, ygrid=ygrid)

    # t test value
    # ------------
    # ...
    # 1. t, p = ttest_ind(df.query(...), df.query(...))
    # 2. format text: 't = {:.2f}, p = {:.2f}'.format(t, p)
    # 3. plot text with matplotlib
    return ax


def axis_frame_aes(ax, ygrid=True):
    '''Nice axis spines.'''
    sns.despine(ax=ax, trim=False, offset=25)
    _trim_y(ax)

    for tck in ax.yaxis.get_majorticklabels():
        tck.set_fontsize(14)

    # change width of spine lines
    axline_width = 2
    ax.spines['bottom'].set_linewidth(axline_width)
    ax.spines['left'].set_linewidth(axline_width)
    ax.xaxis.set_tick_params(width=axline_width, length=8)
    ax.yaxis.set_tick_params(width=axline_width, length=8)

    if ygrid:
        ax.yaxis.grid(color=[0.88, 0.88, 0.88], linewidth=2,
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


def create_swarm_df(psd_high, psd_low):
    df_list = list()
    groups = ['diag'] * psd_high.shape[0] + ['hc'] * psd_low.shape[0]
    for ar in [0, 1]:
        data = np.concatenate([psd_high[:, ar], psd_low[:, ar]])
        df_list.append(pd.DataFrame(data={'asym': data, 'group': groups}))
    return df_list


def plot_heatmap_add1(clst):
    '''Plot results of Standardized Analyses (ADD1) with heatmap and topo.'''
    fig = plt.figure(figsize=(7, 9))
    gs = fig.add_gridspec(2, 2, top=0.95, bottom=0.05, left=0.08, right=0.88,
                          height_ratios=[0.6, 0.4], hspace=0.4, wspace=0.25)
    f_ax1 = fig.add_subplot(gs[0, :])
    f_ax2 = fig.add_subplot(gs[1, 0])
    f_ax3 = fig.add_subplot(gs[1, 1])

    clst_idx = [0, 1] if len(clst) > 1 else None
    clst.plot(dims=['chan', 'freq'], cluster_idx=clst_idx, axis=f_ax1,
              vmin=-4, vmax=4, alpha=0.65)
    f_ax1.set_xlabel('Frequency (Hz)', fontsize=22)
    f_ax1.set_ylabel('frontal channels', fontsize=22)

    contrast = clst.description['contrast']
    cbar_label = ('Regression t value' if 'reg' in contrast
                  else 't value')
    cbar_ax = fig.axes[-1]
    cbar_ax.set_ylabel(cbar_label, fontsize=20)

    # change ticklabels fontsize
    for tck in f_ax1.get_xticklabels():
        tck.set_fontsize(18)

    for tck in cbar_ax.get_yticklabels():
        tck.set_fontsize(16)

    freqs1, freqs2 = (9, 10), (11.5, 12.50)
    freqlabel1, freqlabel2 = '9 - 10 Hz', '11.5 - 12.5 Hz'
    idx1, idx2 = None, None

    if contrast == 'dreg' and clst.description['study'] == 'C':
        freqlabel1 += '\np = {:.3f}'.format(clst.pvals[1])
        freqlabel2 += '\np = {:.3f}'.format(clst.pvals[0])
        idx1, idx2 = 1, 0

    # topo 1
    mark_kwargs = {'markersize': 8}
    topo_args = dict(vmin=-4, vmax=4, mark_clst_prop=0.3,
                     mark_kwargs=mark_kwargs, border='mean')
    tp1 = clst.plot(cluster_idx=idx1, freq=freqs1, axes=f_ax2, **topo_args)
    tp1.axes.set_title(freqlabel1, fontsize=18)

    # topo 2
    tp2 = clst.plot(cluster_idx=idx2, freq=freqs2, axes=f_ax3, **topo_args)
    tp2.axes.set_title(freqlabel2, fontsize=18)

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
    y = bdi['BDI-II'].values * 0.1 + noise
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
    cntr1, cntr2, cntr3 = -1, -2, 0
    bar1y, bar2y, bar3y = [c - bar_h / 2 for c in [cntr1, cntr2, cntr3]]
    length_to_end = 50 - second_min_diag_bdi

    # group contrast legends
    # ----------------------
    rectanges = _create_group_rectangles(
        bar1y, bar2y, bar_h, second_min_diag_bdi, length_to_end)
    for rct in rectanges:
        axs[1, 0].add_artist(rct)

    # regression contrast legends
    # ---------------------------
    rectanges = _create_regression_rectanges(
        bar1y, bar2y, bar3y, bar_h, length_to_end, second_min_diag_bdi)
    for rct in rectanges:
        axs[1, 1].add_artist(rct)

    # aesthetics
    # ----------
    axs[0, 1].set_ylabel("")
    axs[0, 1].set_xticklabels([])
    axs[0, 1].set_xlabel('BDI')

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
        ax.set_xlabel('BDI')
        ax.set_ylabel('')
        ax.set_xticks([0, 10, 20, 30, 40, 50])

        # tick labels fontsize
        for axside in [ax.xaxis, ax.yaxis]:
            for tck in axside.get_majorticklabels():
                tck.set_fontsize(17)

        # add grid
        ax.xaxis.grid(color=[0.88, 0.88, 0.88], linewidth=2,
                      zorder=0, linestyle='--')

    axs[1, 0].set_ylim((-2.5, -0.5))
    axs[1, 0].set_yticks([cntr2, cntr1])
    axs[1, 0].set_yticklabels(['SvsHC', 'DvsHC'])

    axs[1, 1].set_ylim((-2.75, 0.75))
    axs[1, 1].set_yticks([cntr2, cntr1, cntr3])
    axs[1, 1].set_yticklabels(['allReg', 'nonDReg', 'DReg'])

    return fig


def src_plot(clst, cluster_idx=0, azimuth_pos=[35, 125], colorbar=True,
             cluster_p=True, vmin=-3, vmax=3):
    '''Plot source-level clusters as multi-axis images.

    Parameters
    ----------
    clst : borsar.cluster.Clusters
        Cluster results to plot.
    cluster_idx : int
        Cluster to plot.
    azimuth_pos : list of int
        List of two azimuth position of the brain images.
    colorbar : bool
        Whether to show colorbar. True by default.
    cluster_p : bool
        Whether to show cluster p value text. True by default.
    vmin : value
        Minimum value of the colormap.
    vmax : value
        Maximum value of the colormap.

    Returns
    -------
    fig : matplotlib figure
        Matplotlib figure with images.'''

    from mayavi import mlab
    brain = clst.plot(cluster_idx=cluster_idx, vmin=vmin, vmax=vmax)

    if not colorbar:
        brain.hide_colorbar()

    if not cluster_p:
        # fing and hide cluster p text
        clst_txt = brain.texts_dict['time_label']['text']
        clst_txt.remove()

    imgs = list()
    for azi in azimuth_pos:
        mlab.view(azimuth=azi)
        img = mlab.screenshot(antialiased=True)
        imgs.append(img)
    mlab.close()

    gridspec = {'hspace': 0.1, 'wspace': 0.1, 'left': 0.025, 'right': 0.975,
                'top': 0.95, 'bottom': 0.05}
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6), gridspec_kw=gridspec)

    for this_ax, this_img in zip(ax, imgs):
        this_ax.imshow(this_img)
        this_ax.set_axis_off()

    return fig


def _create_regression_rectanges(bar1y, bar2y, bar3y, bar_h, length_to_end,
                                 second_min_diag_bdi):
    rct1 = plt.Rectangle((0, bar1y), 5, bar_h, color=colors['hc'], zorder=5)
    rct2 = plt.Rectangle((5, bar1y), 5, bar_h, color=colors['mid'], zorder=6)
    rct4 = plt.Rectangle((10, bar1y), 40, bar_h, color=colors['subdiag'],
                         zorder=5)
    rct3 = plt.Rectangle((0, bar2y), 5, bar_h, color=colors['hc'], zorder=5)
    rct5 = plt.Rectangle((10, bar2y), 40, bar_h, color=colors['subdiag'],
                         zorder=5)
    rct6 = plt.Rectangle((5, bar2y), 5, bar_h, color=colors['mid'], zorder=5)
    rct7 = plt.Rectangle((second_min_diag_bdi, bar2y), length_to_end,
                         bar_h / 2, color=colors['diag'], zorder=5)
    rct8 = plt.Rectangle((second_min_diag_bdi, bar3y), length_to_end, bar_h,
                         color=colors['diag'], zorder=5)
    return [rct1, rct2, rct3, rct4, rct5, rct6, rct7, rct8]


def _create_group_rectangles(bar1y, bar2y, bar_h, second_min_diag_bdi,
                             length_to_end):
    rct1 = plt.Rectangle((0, bar1y), 5, bar_h, color=colors['hc'], zorder=5)
    rct2 = plt.Rectangle((second_min_diag_bdi, bar1y), length_to_end, bar_h,
                         color=colors['diag'], zorder=5)
    rct3 = plt.Rectangle((0, bar2y), 5, bar_h, color=colors['hc'], zorder=5)
    rct4 = plt.Rectangle((10, bar2y), 40, bar_h,
                         color=colors['subdiag'], zorder=5)
    return [rct1, rct2, rct3, rct4]
