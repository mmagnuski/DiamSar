import numpy as np
import mne

from . import pth


def find_label_vertices(src, labels, label_names=None, hemi='both'):
    '''
    Find vertex indices of a list of labels with respect to forward model.

    Parameters
    ----------
    src : mne.SourceSpaces | mne.SourceEstimate
        SourceSpaces with respect to which vertex indices of labels are found.
    labels : list of mne Labels
        Labels to find vertex indices for.
    label_names : list of str | None
        List to select labels by their names. Defaults to None: no label
        selection.
    hemi : 'both' | 'lh' | 'rh'
        Whether to select labels that belong to a specific hemisphere.

    Returns
    -------
    vert_idx : dict of hemi -> numpy array
        Dictionary mapping from hemisphere ('lh', 'rh') to numpy array
        of vertex indices to use gainst respective hemispheres.
    '''
    # check hemi
    if hemi == 'both':
        hemi = ['lh', 'rh']
    elif hemi in ['lh', 'rh']:
        hemi = list(hemi)
    else:
        raise ValueError('`hemi` must be either "both", "lh" or "rh"')

    vertices = _get_vertices(src)

    # select labels according to label_names
    if label_names is not None:
        labels = [label for label in labels if label.name in label_names]

    vert_idx = {k: list() for k in hemi}
    # check each label
    for lab in labels:
        if lab.hemi in hemi:
            hemi_idx = ['lh', 'rh'].index(lab.hemi)
            this_idx = np.where(np.in1d(vertices[hemi_idx],
                                        lab.vertices))[0]

            vert_idx[lab.hemi].append(this_idx)

    good_keys = [k for k in hemi if len(vert_idx[k]) > 0]
    vert_idx_all = {k: np.unique(np.concatenate(vert_idx[k]))
                    for k in good_keys}

    return vert_idx_all


def _to_data_vert(src, vert_dct, return_array=True):
    '''Format dictionary of vertex indices to indices for ``stc.data``.

    Parameters
    ----------
    src : mne.SourceSpaces
        Source space to use.
    vert_dct : dict
        Dictionary mapping hemisphere ('lh', 'rh') to vertex indices that could
        be applied to `src` but not to 'rh' of `stc.data`.
    return_array : bool
        Return one array that could be used to index into `stc.data` if True.
        If False: returns dictionary of hemisphere -> vertex indices, where the
        vertices for the right hemisphere are corrected to index into
        `stc.data`.

    Returns
    -------
    vert : array or dict
        If `return_array` is True: `vert` is one array that can be used to index
        into `stc.data` (`stc.data[vert, 0]` for example). If `return_array` is
        False `vert` is a dictionary with 'lh' and 'rh' where `vert['rh']`
        gives data vertex indices to `stc.data` for right hemisphere and
        `vert['lh']` gives data vertex indices to `stc.data` for left
        hemisphere.
    '''
    vertices = _get_vertices(src)
    num_lh_vert = len(vertices[0])
    if 'rh' in vert_dct and len(vert_dct['rh']) > 0:
        num_rh_vert = len(vertices[1])
        should_change = np.all(vert_dct['rh'] < num_rh_vert)
    else:
        should_change = False

    if should_change:
        vert_dct = vert_dct.copy()
        vert_dct['rh'] = vert_dct['rh'] + num_lh_vert

    if return_array:
        # make sure 'lh', 'rh' order is maintained even on older python
        keys = [key for key in vert_dct.keys() if len(vert_dct[key]) > 0]
        keys.sort()
        return np.concatenate([vert_dct[k] for k in keys])
    else:
        return vert_dct


# - [ ] add tests and move to borsar
def morph_hemi(data, src, morph='rh2lh', subjects_dir=None, has_subjects=True,
               method='sym', src_sym=None, return_stc=False):
    '''
    Morph one hemisphere to another. Useful in cross-hemisphere comparisons.

    Parameters
    ----------
    data : numpy array
        Data array of shape (subjects x time/freq x vertices) or
        (time/freq x vertices) or (subjects x vertices) or (vertices).
        To distinguish between (time/freq x vertices) and
        (subjects x vertices) `has_subjects` argument is used.
    src : mne.SourceSpaces
        SourceSpaces matching the subject (or all subjects if
        `has_subjects` is True).
    morph : str
        'rh2lh' (morph right hemisphere to left) or 'lh2rh' (morph
        left hemisphere to right). Defaults to 'rh2lh'.
    subjects_dir : str | None
        Freesurfer subjects directory. Defaults to None (which causes
        the function to look for a default subjects_dir).
    has_subjects : bool
        Whether data has subjects as the first dimension.
    method : str
        Method to use when morphing. FIXME
    return_stc : bool
        FIXME

    Returns
    -------
    morphed_data : numpy array
        Data of the same shape as input `data` with relevant hemispheres
        morphed.
    '''
    # check subjects_dir
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)

    # check morph
    if morph not in ['rh2lh', 'lh2rh']:
        raise ValueError('`morph` must be either "lh2rh" or "rh2lh", '
                         'got {}.'.format(morph))

    # check agreement between psd and src
    n_vert = [len(x['vertno']) for x in src]
    if not data.shape[-1] == sum(n_vert):
        raise ValueError('The last dimension of psd has to be of legth equal '
                         'to the number of vertices in the source space.')

    def get_single_subj(data, idx, has_subjects):
        single_subj = (data[idx].T if data.ndim == 3
                       else data.T if data.ndim == 2 and not has_subjects
                       else data[idx, :, np.newaxis] if data.ndim == 2
                       else data[:, np.newaxis])
        return single_subj

    # create stc
    subject = src[0]['subject_his_id']
    single_subj = get_single_subj(data, 0, has_subjects)
    stc = mne.SourceEstimate(single_subj, subject=subject, tmin=0., tstep=1.,
                             vertices=[src[0]['vertno'], src[1]['vertno']])

    morph_from, morph_to = morph.split('2')

    if method == 'direct':
        # this method is most likely inaccurate or even wrong, left for
        # comparison only!

        morph_dct = {'lh': [stc.vertices[0], []], 'rh': [[], stc.vertices[1]]}
        morph_dct_slices = {'lh': slice(0, n_vert[0]),
                            'rh': slice(n_vert[0], n_vert[0] + n_vert[1])}

        # create morph matrix
        mm = mne.compute_morph_matrix(
            subject, subject, xhemi=True, subjects_dir=subjects_dir,
            vertices_from=morph_dct[morph_from],
            vertices_to=morph_dct[morph_to])

        data_morphed = data.copy()
        slc_from = morph_dct_slices[morph_from]
        slc_to = morph_dct_slices[morph_to]

        if has_subjects:
            # morph subject by subject
            for subj_idx in range(data.shape[0]):
                data_morphed[subj_idx, ..., slc_from] = (
                    mm @ data[subj_idx, ..., slc_to].T).T
        else:
            # morph whole data at once
            data_morphed[..., slc_from] = (mm @ data[..., slc_to].T).T

    elif method == 'sym':
        if src_sym is None:
            src_sym = pth.paths.load_data('src_sym')
        vertices = _get_vertices(src_sym)

        fsavg2sym = mne.compute_source_morph(
            stc, subject_from=subject, subject_to='fsaverage_sym',
            smooth=5, spacing=vertices, warn=False,
            subjects_dir=subjects_dir)
        stc_sym = fsavg2sym.apply(stc)
        xhemi = mne.compute_source_morph(
            stc_sym, subject_from='fsaverage_sym', subject_to='fsaverage_sym',
            spacing=stc_sym.vertices, warn=False, subjects_dir=subjects_dir,
            xhemi=True, verbose='error')
        stc_xhemi = xhemi.apply(stc_sym)

        n_vert = [len(stc_sym.vertices[idx]) for idx in [0, 1]]
        morph_dct_slices = {'lh': slice(0, n_vert[0]),
                            'rh': slice(n_vert[0], n_vert[0] + n_vert[1])}

        data_morphed = data.copy()
        slc_from = morph_dct_slices[morph_from]
        slc_to = morph_dct_slices[morph_to]

        this_subj = stc_sym.data.T.copy()
        this_subj[..., slc_from] = stc_xhemi.data[slc_to].T

        if not has_subjects:
            if return_stc:
                return this_subj, stc_sym
            else:
                return this_subj
        else:
            n_subj = data.shape[0]
            data_morphed[0] = this_subj
            for idx in range(1, n_subj):
                this_subj = get_single_subj(data, idx, has_subjects)
                stc.data = this_subj
                stc_sym = fsavg2sym.apply(stc)
                stc_xhemi = xhemi.apply(stc_sym)
                data_morphed[idx, ..., slc_from] = stc_xhemi.data[slc_to].T

            if return_stc:
                return data_morphed, stc_sym
            else:
                return data_morphed

    return data_morphed


def select_vertices(src, hemi='both', selection='frontal', subjects_dir=None,
                   subject=None, parc='aparc'):
    '''Return vertex indices in source space based on ``selection`` string.'''
    if parc not in ['aparc', 'HCPMMP1']:
        msg = 'Parcellation has to be either "aparc" or "HCPMMP1", got {}.'
        raise ValueError(msg.format(parc))

    vertices = _get_vertices(src)
    n_vert_lh = len(vertices[0])
    n_vert_rh = len(vertices[1])

    if 'asy_all' == selection:
        labels_asy = dict()
        labels_asy['lh'] = np.arange(n_vert_lh)
        labels_asy['rh'] = np.arange(n_vert_rh)
        return labels_asy

    if 'all' == selection:
        return np.arange(n_vert_lh + n_vert_rh)

    if 'frontal' in selection:
        # FIXME: this currently works only for a specific parcellation...
        labels_all = mne.read_labels_from_annot(
            subject, parc=parc, subjects_dir=subjects_dir, verbose=False)

        labels = dict()
        if parc == 'HCPMMP1':
            labels['lh'] = ['Anterior Cingulate and Medial Prefrontal Cortex-lh',
                            'Insular and Frontal Opercular Cortex-lh',
                            'Orbital and Polar Frontal Cortex-lh',
                            'DorsoLateral Prefrontal Cortex-lh',
                            'Inferior Frontal Cortex-lh']
        elif parc == 'aparc':
            temp_labels = ['superiorfrontal', 'caudalmiddlefrontal',
                           'rostralmiddlefrontal', 'parsopercularis',
                           'parstriangularis', 'parsorbitalis',
                           'lateralorbitofrontal', 'medialorbitofrontal',
                           'frontalpole']
            labels['lh'] = [label + '-lh' for label in temp_labels]

        labels['rh'] = [label.replace('lh', 'rh')
                        for label in labels['lh']]

        labels = (labels['lh'] + labels['rh'] if hemi == 'both'
                  else labels[hemi])

        labels_vert = find_label_vertices(src, labels_all, label_names=labels)

    if 'asy' in selection:
        # the rest is done in format_psds
        return labels_vert
    else:
        return _to_data_vert(src, labels_vert)


def _get_vertices(src):
    '''Find used vertices indices irrespective of whether ``mne.SourceSpaces``
    or ``mne.SourceEstimate`` is passed.'''
    if isinstance(src, mne.SourceSpaces):
        vertices = [src[hemi_idx]['vertno'] for hemi_idx in [0, 1]]
    elif isinstance(src, mne.SourceEstimate):
        vertices = src.vertices
    return vertices
