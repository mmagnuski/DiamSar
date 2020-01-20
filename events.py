import numpy as np


def get_task_event_id(raw, event_id, study='C', task='rest'):
    if task == 'rest':
        if study == 'C':
            event_id.update({'S 10': 10, 'S 11': 11})
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
    '''
    event_types = np.unique(events[:, -1])
    fix_types = event_types[(event_types > 99) & (event_types < 111)]
    event_id = {'load{}'.format(tp - 100): tp for tp in fix_types}
    return mne.Epochs(raw, tmin=tmin, tmax=tmax, events=events,
                      event_id=event_id, preload=True)


# FIXME
# - [ ] change/clean up and test the FIX/CHECK part
# - [ ] add metadata instead of complex event values
def change_events_sternberg(events):
    '''Add load info to maintenance onset fixations and probe events.

    Information about load can be now obtained from fixation event value by
    subtracting 100. For example fixation event `150` denotes fixation that
    starts maintenance period with 5 elements held in memory.
    Iformation about load can also be obtained from probe event value using
    the following formula: `int((probe_event - 1) / 10) + 1`.
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

    # FIX/CHECK
    # min_ind i linijki poniżej są do upewnienia się, że wszystko jest ok
    # czasami po prostu brakuje kilku eventów na końcu - będę to musiał jeszcze
    # sprawdzić - a tutaj dodam pewnie Warning
    n_fix_maint = np.sum(is_maintenance_fix)
    n_fix_start = np.sum(~is_maintenance_fix)
    equal_len = (n_fix_maint == n_fix_start == len(num_digits))
    if not equal_len:
        from warnings import warn
        warn('Something may be wrong with this file - sternberg event'
            ' structure is perturbed. Number of maintenance fixes: '
            '{}, number of start fixes: {}, number of digit streams: '
            '{}.'.format(n_fix_maint, n_fix_start, len(num_digits)))
        min_ind = min(n_fix_maint, n_fix_start, len(num_digits))
        where_fix = where_fix[:min_ind * 2]
        is_maintenance_fix = is_maintenance_fix[:min_ind  * 2]
        num_digits = num_digits[:min_ind]
    num_digits = np.array(num_digits)

    # add load info to maintenance fix
    new_events[where_fix[is_maintenance_fix], -1] += num_digits * 10

    # add load info to probe
    new_events[where_probe, -1] = (num_digits * 10 +
        new_events[where_probe, -1] - 10)

    return new_events


def get_probe_events(events):
    '''Returns probe events coded by load (irrespective of probe value)'''
    is_probe = (events[:, -1] > 10) & (events[:, -1] < 100)
    probe_events = events[is_probe, :]
    loads = ((probe_events[:, -1] - 1) / 10).astype('int') + 1
    probe_events[:, -1] = loads
    return probe_events