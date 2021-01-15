import os
import os.path as op
import re
from pathlib import Path

import numpy as np
from borsar.project import Paths, get_valid_path
from sarna.proj import find_dropbox

global paths
global files
global files_id_mode


files = dict()
paths = Paths()

# - [?] create a diagram with study paths


# study C
# -------
def set_paths(base_dir=None):
    '''Setup study paths.'''
    global paths
    global files
    global files_id_mode

    paths = Paths()
    paths.register_study('C', tasks=['rest', 'sternberg', 'linord'])

    if base_dir is not None:
        paths.add_path('base', base_dir, relative_to=False)

    # study C
    # -------
    has_C = (paths.get_path('base', as_str=False) / 'DiamSar').exists()
    if has_C:
        paths.add_path('main', 'DiamSar', relative_to='base')
        paths.add_path('fig', 'fig', validate=False)
        paths.add_path('eeg', 'eeg')
        paths.add_path('subjects_dir', 'src', relative_to='eeg')
        paths.add_path('src', 'src', relative_to='eeg')
        paths.add_path('beh_base', 'beh', relative_to='main')
        paths.add_path('beh', 'stern', task='sternberg',
                       relative_to='beh_base', validate=False)

        # task-specific data
        translate = dict(rest='baseline', linord='linord',
                         sternberg='sternberg')
        for task in ['rest', 'sternberg']:
            task_eeg_dir = op.join('resampled set', translate[task] + '_clean_exported')
            paths.add_path('eeg', task_eeg_dir, study='C', task=task,
                           relative_to='eeg', validate=False)

    # study B
    # -------
    base_dir = paths.get_path('base')
    paths.register_study('B', tasks=['rest', 'linord'])
    study_B_path = Path(base_dir, 'Wronski')
    has_B = study_B_path.exists()

    if has_B:
        paths.add_path('main', study_B_path, study='B')
        paths.add_path('eeg', 'eeg', study='B')
        paths.add_path('src', 'src', study='B', relative_to='eeg')

        # task-specific
        for task in ['rest']:
            paths.add_path('eeg', translate[task] + '_clean_exported',
                           study='B', task=task, relative_to='eeg',
                           validate=False)

    # study A
    # -------
    paths.register_study('A', tasks=['rest'])
    study_A_path = Path(base_dir, 'Nowowiejska')
    has_A = study_A_path.exists()

    if has_A:
        paths.add_path('main', study_A_path, study='A')
        paths.add_path('eeg', 'eeg', study='A')
        paths.add_path('src', 'src', study='A', relative_to='eeg')

        paths.add_path('eeg', translate['rest'] + '_clean_exported', study='A',
                       task='rest', relative_to='eeg', validate=False)

    for study, has_study in zip(['A', 'B', 'C'], [has_A, has_B, has_C]):
        if has_study:
            paths.add_path('chanpos', 'chanpos', study=study,
                           relative_to='eeg', validate=False)

    # getting files
    # -------------
    # getting files is not yet supported in borsar.Paths so we write a few
    # functions for that
    files_id_mode = dict(A='anon', B='num', C='num')

    for study in paths.studies:
        files[study] = dict()
        for task in paths.tasks[study]:
            files[study][task] = list()

    # check files
    for study, has_study in zip(list('ABC'), [has_A, has_B, has_C]):
        if has_study:
            scan_files(study=study)

    # register data
    # -------------
    from .io import (load_GH, load_chanord, load_neighbours, load_info,
                     load_forward, load_src_sym, read_bdi, load_psd)

    for study, has_study in zip(list('ABC'), [has_A, has_B, has_C]):
        if has_study:
            paths.register_data('GH', load_GH, study=study, cache=True)
            paths.register_data('chanord', load_chanord, study=study,
                                cache=True)
            paths.register_data('neighbours', load_neighbours, study=study)
            paths.register_data('info', load_info, study=study, cache=True)
            paths.register_data('bdi', read_bdi, study=study, cache=True)
            paths.register_data('psd', load_psd, study=study, cache=False)
            paths.register_data('fwd', load_forward, study=study)

    paths.register_data('src_sym', load_src_sym, study='C')

    return paths


def scan_files(study='C', task='rest'):
    global files
    task_dir = paths.get_path('eeg', study=study, task=task)

    if op.isdir(task_dir):
        set_files = [f for f in os.listdir(task_dir) if f.endswith('.set')]
        rej_files = [f for f in os.listdir(task_dir) if f.endswith('.rej')]
        subj_id = [int(re.findall('[0-9]+', f)[0]) for f in set_files]
        files[study][task] = set_files, rej_files, subj_id
    else:
        files[study][task] = None, None, None


def get_file(file, study='C', task='rest'):
    '''
    Get `.set` and `.rej` filenames for given subject ID from specific
    study and task.
    '''
    global files
    global files_id_mode

    scan_files(study=study, task=task)
    set_files, rej_files, subj_id = files[study][task]

    # deal with subject identifiers, first in str format
    # (for example '2', '01', '042' or '003')
    if isinstance(file, str) and len(file) < 4:
        if file.isdigit():
            file = int(file)

    # deal with integers - subject identifiers
    if isinstance(file, int):
        if files_id_mode[study] == 'num':
            if file not in subj_id:
                msg = 'Given id ({}) was not found for study {}, task {}.'
                raise FileNotFoundError(msg.format(file, study, task))
            else:
                idx = subj_id.index(file)
                set_file = set_files[idx]
        elif files_id_mode[study] == 'anon':
            if file > len(set_files):
                msg = ('Requested file number {} while there are only '
                       '{} files for study {}, task {}.')
                raise FileNotFoundError(msg.format(file, len(set_files),
                                                   study, task))
            set_file = set_files[file - 1]
    else:
        set_file = file
        if set_file in set_files:
            msg = 'Given files ({}) was not found for study {}, task {}.'
            raise FileNotFoundError(msg.format(set_file, study, task))

    rej_file = set_file.replace('.set', '.rej')
    if rej_file not in rej_files:
        rej_file = None

    return set_file, rej_file


def get_subject_ids(study='C', task='rest', full_names=False):
    '''
    Get subject id's for specific study and task.
    '''
    global files
    global files_id_mode

    if files_id_mode[study] == 'num':
        set_files, _, subj_ids = files[study][task]
    elif files_id_mode[study] == 'anon':
        set_files, _, _ = files[study][task]
        subj_ids = list(range(1, len(set_files) + 1))
    if full_names:
        names = np.array([f.replace('.set', '') for f in set_files])
        return subj_ids, names
    else:
        return subj_ids


try:
    dropbox_dir = find_dropbox()
    if len(dropbox_dir) > 0:
        candidate_paths = [op.join(dropbox_dir, p)
                           for p in ['DANE/SarenkaData', 'DATA']]
        full_path = get_valid_path(candidate_paths)
        paths = set_paths(base_dir=full_path)
except:
    pass
