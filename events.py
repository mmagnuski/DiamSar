import numpy as np


sternberg_ids_shifted = [26, 30, 34, 40, 52, 55, 56, 64, 67, 69]


def get_task_event_id(raw, event_id, study='C', task='rest'):
    '''Returns trigger mapping specific to given study and task.

    Parameters
    ----------
    raw : mne Raw
        Data file. Used only in study C, sternberg task, when annotations
        are checked.
    event_id : dict
        Starting event_id dictionary that is filled with additional entries.
    study : str
        Study name. Defaults to ``'C'``.
    task : str
        Task name. Defaults to ``'rest'``.
    '''
    if task == 'rest':
        if study == 'C':
            event_id.update({'S 10': 10, 'S 11': 11})
        elif study == 'D':
            def event_id(name):
                if 'keyboard' in name:
                    return 99
                elif name == 'boundary':
                    return 999
                else:
                    return int(name)
        return event_id
    elif task == 'sternberg':
        if raw.annotations is not None:
            add_dct = dict()
            desc = np.unique(raw.annotations.description)
            for dsc in desc.tolist():
                if dsc[0] == 'S':
                    ints = ''.join([c for c in dsc if c.isdigit()])
                    add_dct[dsc] = int(ints)
            event_id.update(add_dct)
            return event_id


def translate_events_D(events):
    '''Translate PREDiCT triggers to a sane format where they signal event
    onset (by default events are sent periodically as long as given event,
    for example eyes closed, lasts).'''
    onset_events = list()
    types = [[1, 3, 5], [2, 4, 6]]
    translate_to = [11, 10]
    for event_types, translate in zip(types, translate_to):
        for event_type in event_types:
            all_types = np.where(events[:, -1] == event_type)[0]
            if len(all_types) > 0:
                this_event = events[all_types[0], :]
                this_event[-1] = translate
                onset_events.append(this_event)

    # join the new annotations with boundary events
    events_new = np.stack(onset_events, axis=0)
    where_boundaries = events[:, -1] == 999
    if where_boundaries.any():
        events_old = events[where_boundaries, :]
        events = np.concatenate([events_new, events_old], axis=0)
    else:
        events = events_new

    # sort the events array by onset sample
    ord_idx = np.argsort(events[:, 0])
    events = events[ord_idx, :]

    return events


# TODO - change name, fix suggests repairing, not fixation
def fix_epochs(raw, events, tmin=-0.2, tmax=0.5):
    '''Create fixation-centered epochs.

    Parameters
    ----------
    raw : instance of mne.io.BaseRaw
        Raw data to epoch.
    events : numpy 2d array
        Events array in mne-python convention.
    tmin : float
        Epoch start time with respect to fixation event (in seconds).
        Default : -0.2
    tmax : float
        Epoch end time with respect to fixation event (in seconds).
        Default: 0.5

    Returns
    -------
    epochs : instance of mne.Epochs
        Raw data epoched with respect to fixation events.

    This function is not used in "Three times NO" paper.
    '''
    import mne
    event_types = np.unique(events[:, -1])
    fix_types = event_types[(event_types > 99) & (event_types < 111)]
    event_id = {'load{}'.format(tp - 100): tp for tp in fix_types}
    return mne.Epochs(raw, tmin=tmin, tmax=tmax, events=events,
                      event_id=event_id, preload=True)


# - [ ] CONSIDER merging with construct_metadata_from_events?
def translate_events_sternberg(events):
    '''Translate sternberg events to include information about load.

    Information about load can be now obtained from fixation event value by
    subtracting 100 and dividing by 10. For example fixation event `150`
    denotes fixation that starts maintenance period with 5 elements held in
    memory. Information about load can also be obtained from probe event value
    using the following formula: `int(probe_event / 10)`.

    Parameters
    ----------
    events : numpy 2d array
        Events array in mne-python convention.

    Returns
    -------
    events : numpy 2d array
        Translated events array.

    Notes
    -----
    This function is not used in "Three times NO" paper.
    '''
    is_probe = (events[:, -1] > 10) & (events[:, -1] < 21)
    where_probe = np.where(is_probe)[0]
    where_fix = np.where(events[:, -1] == 100)[0]
    is_maintenance_fix = np.in1d(where_fix, where_probe - 1)
    trial_start_fix = where_fix[~is_maintenance_fix]
    new_events = events.copy()

    # change event 10 to 0 (digit zero was presented then)
    where_10 = np.where(events[:, -1] == 10)[0]
    new_events[where_10, -1] = 0

    # change event 20 to 10 (probe events should now be 10-19)
    where_20 = np.where(events[:, -1] == 20)[0]
    new_events[where_20, -1] = 10

    num_digits = list()
    for start_idx in trial_start_fix:
        idx = start_idx + 1
        n_dig = 0
        while idx < events.shape[0] - 1 and events[idx, -1] < 11:
            n_dig += 1
            idx += 1
        num_digits.append(n_dig)

    # fix missing maintenance fixation and probe event for last trial for
    # some of the subjects - these events are likely missing due to how some
    # data files were cropped
    n_fix_maint = np.sum(is_maintenance_fix)
    n_fix_start = np.sum(~is_maintenance_fix)
    equal_len = (n_fix_maint == n_fix_start == len(num_digits))
    if not equal_len:
        min_ind = min(n_fix_maint, n_fix_start, len(num_digits))
        num_digits = num_digits[:min_ind]

    num_digits = np.array(num_digits, dtype=new_events.dtype)

    # add load info to maintenance fix
    new_events[where_fix[is_maintenance_fix], -1] += num_digits * 10

    # add load info to probe
    new_events[where_probe, -1] = (num_digits * 10 +
                                   new_events[where_probe, -1] - 10)

    return new_events


def construct_metadata_from_events(events, subj_id=None):
    # go through events step by step and create rich dataframe representation
    # of what happens in the experiment
    import pandas as pd


    n_events = events.shape[0]
    digit_events = events[events[:, -1] < 10, :]
    n_digits = digit_events.shape[0]

    maint_events_mask = (events[:, -1] >= 120) & (events[:, -1] < 180)
    n_maint = maint_events_mask.sum()

    common_columns = ['trigger', 'sample']
    columns_digits = (['trial', 'digit', 'current_load', 'total_load']
                      + common_columns)
    columns_maint = ['trial', 'digits', 'load'] + common_columns
    columns_probe = (['trial', 'digits', 'load', 'probe', 'probe_in']
                     + common_columns)
    df_digits = pd.DataFrame(index=np.arange(n_digits), columns=columns_digits)
    df_maint = pd.DataFrame(index=np.arange(n_maint), columns=columns_maint)
    df_probe = pd.DataFrame(index=np.arange(n_maint), columns=columns_probe)

    # df_probes
    in_sequence = False
    maintenance = False
    fixation = False
    probe = True
    row_idx = -1
    trial = -1
    all_digits = []

    loads = list(range(2, 8))
    maint_events = np.array([100 + load * 10 for load in loads])

    for event_idx, event in enumerate(events[:, -1]):
        if event in [999, 111]:
            # ignore these events
            continue

        if fixation:
            if event < 10:
                in_sequence = True
                fixation = False
            else:
                raise ValueError(f'Unexpected event at position {event_idx}.')

        if probe:
            if event == 100:
                probe = False
                fixation = True
                trial += 1
                current_load = 0
                start_idx = row_idx + 1
                all_digits = []
            else:
                raise ValueError(f'Unexpected event at position {event_idx}.')

        if maintenance:
            event_str = str(event)
            if len(event_str) == 2 and int(event_str[0]) in loads:
                maintenance = False
                probe = True

                # fill probe df
                # -------------
                assert len(event_str) == 2
                current_load_probe = int(event_str[0])
                probe_value = int(event_str[1])
                assert current_load_probe == current_load

                df_probe.loc[trial, 'trial'] = trial
                df_probe.loc[trial, 'load'] = current_load
                df_probe.loc[trial, 'digits'] = digits
                df_probe.loc[trial, 'probe'] = probe_value
                df_probe.loc[trial, 'probe_in'] = probe_value in all_digits

                df_probe.loc[trial, 'trigger'] = event
                df_probe.loc[trial, 'sample'] = events[event_idx, 0]
            else:
                raise ValueError(f'Unexpected event at position {event_idx}.')

        if in_sequence:
            if event < 10:
                current_load += 1
                row_idx += 1

                # fill digits df
                # --------------
                all_digits.append(event)
                df_digits.loc[row_idx, 'trial'] = trial
                df_digits.loc[row_idx, 'digit'] = event
                df_digits.loc[row_idx, 'current_load'] = current_load
                df_digits.loc[row_idx, 'total_load'] = 0  # temporary

                df_digits.loc[row_idx, 'trigger'] = event
                df_digits.loc[row_idx, 'sample'] = events[event_idx, 0]
            else:
                if not event in maint_events:
                    if event == 100 and event_idx + 1 == n_events:
                        if subj_id is not None:
                            assert subj_id in sternberg_ids_shifted
                        else:
                            print('File does not have last maintenance '
                                  'event.')
                            print('This is likely due to how some of the '
                                  'files were cropped.')
                    else:
                        raise ValueError(
                            f'Unexpected event at position {event_idx}.')
                else:
                    # maintenance
                    in_sequence = False
                    maintenance = True

                    # fill maintenance df
                    # -------------------
                    digits = ' '.join(str(x) for x in all_digits)
                    df_maint.loc[trial, 'trial'] = trial
                    df_maint.loc[trial, 'load'] = current_load
                    df_maint.loc[trial, 'digits'] = digits

                    df_maint.loc[trial, 'trigger'] = event
                    df_maint.loc[trial, 'sample'] = events[event_idx, 0]
                    df_digits.loc[
                        start_idx:row_idx, 'total_load'] = current_load

    assert(df_digits.shape[0] == digit_events.shape[0])
    return df_digits, df_maint, df_probe


def get_probe_events(events):
    '''Returns probe events coded by load (irrespective of probe value).

    This function is not used in "Three times NO" paper.
    '''
    is_probe = (events[:, -1] > 10) & (events[:, -1] < 100)
    probe_events = events[is_probe, :]
    loads = ((probe_events[:, -1] - 1) / 10).astype('int') + 1
    probe_events[:, -1] = loads
    return probe_events
