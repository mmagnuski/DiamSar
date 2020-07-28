import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import seaborn as sns
import pandas as pd

from borsar.stats import format_pvalue
from borsar.viz import Topo

from sarna.viz import prepare_equal_axes

from . import freq, analysis, utils
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
    fig = plt.figure(figsize=(9, 7.5))
    ax = prepare_equal_axes(fig, [2, 3], space=[0.12, 0.85, 0.02, 0.65])

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
    cbar_ax = fig.add_axes([0.87, 0.08, 0.03, 0.55])
    cbar = plt.colorbar(mappable=topo.img, cax=cbar_ax)
    cbar.set_label('t values', fontsize=12)

    # add study labels
    # ----------------
    for idx, letter in enumerate(['I', 'II', 'III']):
        this_ax = ax[0, idx]
        box = this_ax.get_position()
        mid_x = box.corners()[:, 0].mean()

        plt.text(mid_x, 0.72, letter, va='center', ha='center',
                 transform=fig.transFigure, fontsize=21)

        if idx == 1:
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

    cntrst = utils.translate_contrast[contrast]
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


# - [ ] consider adding CI per swarm
# - [ ] consider adding effsize and bootstrapped CIs
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
    # added to figures by hand in the end
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


# - [ ] compare and merge with the function in utils
def create_swarm_df(psd_high, psd_low):
    df_list = list()
    groups = ['diag'] * psd_high.shape[0] + ['hc'] * psd_low.shape[0]
    for ar in [0, 1]:
        data = np.concatenate([psd_high[:, ar], psd_low[:, ar]])
        df_list.append(pd.DataFrame(data={'asym': data, 'group': groups}))
    return df_list


def plot_heatmap_add1(clst, ax_dict=None, scale=None):
    '''Plot results of Analyses on Standardized data (ADD1) with heatmap
    and topomap.

    Parameters
    ----------
    clst: borsar.Clusters
        Cluster-based permutation test result.

    Returns
    -------
    fig: matplotlib Figure
        Matplotlib figure object.
    obj_dict: dict
        Dictionary with axes of the figure. The dictionary contains:
        'heatmap': heatmap axis
        'colorbar': colorbar axis
        'topo1': lower frequency topography
        'topo2': higher frequency topography
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

    show_p = ((contrast == 'dreg' and clst.description['study'] == 'C')
              or (contrast == 'cvsd' and clst.description['study'] == 'C')
              or (contrast == 'cdreg' and clst.description['study'] == 'C'))
    if show_p:
        freqlabel1 += '\np = {:.3f}'.format(clst.pvals[1])
        freqlabel2 += '\np = {:.3f}'.format(clst.pvals[0])
        idx1, idx2 = 1, 0

    # topo 1
    mark_kwargs = {'markersize': scale['markersize']}
    topo_args = dict(vmin=-4, vmax=4, mark_clst_prop=0.3,
                     mark_kwargs=mark_kwargs, border='mean')
    tp1 = clst.plot(cluster_idx=idx1, freq=freqs1, axes=f_ax2, **topo_args)
    tp1.axes.set_title(freqlabel1, fontsize=scale['topo_title'])

    # topo 2
    tp2 = clst.plot(cluster_idx=idx2, freq=freqs2, axes=f_ax3, **topo_args)
    tp2.axes.set_title(freqlabel2, fontsize=scale['topo_title'])

    obj_dict = {'heatmap': f_ax1, 'colorbar': cbar_ax, 'topo1': tp1,
                'topo2': tp2}
    return fig, obj_dict


def bdi_histogram(bdi):
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
    axs[0, 1].set_xlabel('BDI', fontsize=17)

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
        ax.set_xlabel('BDI', fontsize=17)
        ax.set_ylabel('')
        ax.set_xticks([0, 10, 20, 30, 40, 50])

        # tick labels fontsize
        for axside in [ax.xaxis, ax.yaxis]:
            for tck in axside.get_majorticklabels():
                tck.set_fontsize(17)

        # add grid
        ax.xaxis.grid(color=[0.88, 0.88, 0.88], linewidth=2,
                      zorder=0, linestyle='--')

    axs[0, 0].set_xticklabels(['diagnosed', 'healthy\ncontrols'], fontsize=17)

    axs[1, 0].set_ylim((-2.5, -0.5))
    axs[1, 0].set_yticks([cntr2, cntr1])
    axs[1, 0].set_yticklabels(['SvsHC', 'DvsHC'])

    axs[1, 1].set_ylim((-2.75, 0.75))
    axs[1, 1].set_yticks([cntr2, cntr1, cntr3])
    axs[1, 1].set_yticklabels(['allReg', 'nonDReg', 'DReg'])

    return fig


def src_plot(clst, cluster_idx=None, colorbar='mayavi', azimuth=[35, 125],
             elevation=[None, None], cluster_p=True, vmin=-3, vmax=3,
             figure_size=None):
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


def _pairs_aggregated_studies(space, contrast):
    '''
    Read all studies that include given contrast and aggregate their data.

    Used when plotting aggregated channel pairs figures (``plot_aggregated``).
    '''
    if contrast in ['cvsd', 'dreg']:
        studies = ['A', 'C']
    elif contrast in ['cvsc', 'creg']:
        studies = ['B', 'C']
    elif contrast ==  'cdreg':
        studies = ['C']

    psds = {'high': list(), 'low': list()}
    for study in studies:
        psd_high, psd_low, ch_names = freq.get_psds(
                            selection='asy_pairs', study=study,
                            space=space, contrast=contrast)
        psds['low'].append(psd_low)
        psds['high'].append(psd_high)

    low = np.concatenate(psds['low'], axis=0)
    high = np.concatenate(psds['high'], axis=0)

    studies = [utils.utils.translate_study[std] for std in studies]
    return high, low, studies, ch_names


def _compute_stats_group(high, low, ch_idx=0):
    '''Used when plotting aggregated channel pairs figures
    (``plot_aggregated``).'''
    from scipy.stats import ttest_ind
    stats = analysis.esci_indep_cohens_d(
        high[:, ch_idx], low[:, ch_idx])

    nx, ny = high.shape[0], low.shape[0]
    t, p = ttest_ind(high[:, ch_idx], low[:, ch_idx])
    out = pg.bayesfactor_ttest(t, nx, ny, paired=False)
    bf01 = 1 / float(out)
    stats.update({'bf01': bf01})

    return stats


def _compute_stats_regression(data1, data2, ch_idx=0):
    '''Used when plotting aggregated channel pairs figures
    (``plot_aggregated``).'''
    stats = analysis.esci_regression_r(data1[:, ch_idx], data2)

    nx = data1.shape[0]
    out = pg.bayesfactor_pearson(stats['es'], nx)
    bf01 = 1 / float(out)
    stats.update({'bf01': bf01})

    return stats


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


def plot_aggregated(ax=None, eff='d'):
    '''Plot aggregated effect sizes for channel pairs analyses.

    ax: matplotlib axis
        Axis to plot to.
        Should look good in the following setup:
        ```python
        ```
    eff: str
        Effect size to plot. ``'r'`` shows effects for linear relationship
        analyses with Pearson's r as the effect size. ``'d'`` shows effects for
        group contrasts with Cohen's d as the effect size.

    Returns
    -------
    ax: matplotlib axis
        Axis used to plot to.
    '''

    if ax is None:
        fig_size = (11, 13.5) if eff == 'r' else (11, 12)
        gridspec = ({'left': 0.25, 'bottom': 0.1, 'right': 0.85} if eff == 'r'
                    else {'left': 0.18, 'bottom': 0.1, 'right': 0.8})
        fig, ax = plt.subplots(figsize=fig_size, gridspec_kw=gridspec)

    ypos = 5
    labels_pos = list()
    labels = list()

    # 'd' vs 'r' effect size (group contrasts vs linear relationships)
    addpos = 0.09 if eff == 'd' else 0.06
    stat_fun = (_compute_stats_group if eff == 'd'
                else _compute_stats_regression)
    distr_color = (ds.utils.colors['hc'] if eff == 'd'
                   else ds.utils.colors['subdiag'])
    contrasts = ['cvsd', 'cvsc'] if eff == 'd' else ['dreg', 'creg', 'cdreg']

    for contrast in contrasts:
        for space in ['avg', 'csd']:
            # get relevant data
            data1, data2, studies, ch_names = _pairs_aggregated_studies(
                space, contrast)

            # channel pair loop
            for ch_idx in range(2):
                # compute es, bootstrap esci and bf01
                stats = stat_fun(data1, data2, ch_idx=ch_idx)

                # plot distribution, ES and CI
                v = _plot_dist_esci(ax, ypos, stats, color=distr_color)
                v['bodies'][0].set_zorder(4)

                # slight y tick labeling differences
                if len(studies) == 2:
                    label_schematic = ('{}, {}\n{}\nstudies {} & {}'
                                       if eff == 'd'
                                       else '{}, {}\n{}, studies {} & {}')
                    label = label_schematic.format(
                        utils.translate_contrast[contrast], space.upper(),
                        ch_names[ch_idx], studies[0], studies[1])
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


def full_fig5_supplement_plot(contrast, studies):
    '''Plot whole panel plot for figure 5 supplements, for given contrast
    and studies.'''
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
