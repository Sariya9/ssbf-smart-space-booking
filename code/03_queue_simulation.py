"""
Simulation of the release layer (Tables 4, 13, 14, 15).
20 rooms, 8 one-hour slots a day, 60 days, 150 users. Demand and no-shows are set
independently of the ML models. Every condition runs on the same random events per seed.
Run: python3 03_queue_simulation.py   (--quick for a short test)  -> ../results/simulation/
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results', 'simulation')
os.makedirs(OUT, exist_ok=True)

SLOT_MIN = 60
TIER_SEATS = np.array([6, 12, 30])

DEFAULTS = dict(
    days=60, slots=8, n_users=150,
    rooms_per_tier=(8, 8, 4), vc_room_share=0.4,
    rho=1.0,                                    # demand: requests per room-slot
    slot_profile=(0.6, 0.9, 1.2, 1.3, 1.1, 1.2, 1.0, 0.7),
    tier_pref=(0.6, 0.3, 0.1), vc_need=0.3,
    noshow_mean=0.20, noshow_conc=4.0,          # medical data no-show rate is 20.2%
    late_prob=0.15, late_max=25,                # 15% arrive 5-25 min late
    grace=10, ml_grace=5,                       # minutes; 5 if ML-flagged
    checkin_release=15, checkin_compliance=0.90,
    sensor_outage=0.02, sensor_false_vacant=0.005,
    ml_fpr=0.10, ml_tpr=0.24,                   # from the medical model (script 02)
    accept_prob=0.6, timeout=5, travel=5, min_useful=15,
    drf_window=30,
    flex_share=0.3,                             # waiting users happy to move by one slot
    walkin_rate=0.5,                            # walk-ins per freed room
    late_notify=0.5,                            # late users who press "running late"
    noshow_notify=0.05,                         # no-shows who press it anyway
    max_hold=30,                                # max hold, minutes
)

CONDITIONS = {
    'No release':                  dict(trigger=None,      policy=None),
    'Check-in release + FCFS':     dict(trigger='checkin', policy='fcfs'),
    'Sensing + FCFS':              dict(trigger='sensing', policy='fcfs'),
    'Sensing + Random':            dict(trigger='sensing', policy='random'),
    'Sensing + WFQ':               dict(trigger='sensing', policy='wfq'),
    'Sensing + DRF':               dict(trigger='sensing', policy='drf'),
    'ML + Sensing + DRF (SSBF)':   dict(trigger='ml+sensing', policy='drf'),
    'ML only + DRF':               dict(trigger='ml',      policy='drf'),
    'Oracle ML + Sensing + DRF (upper bound)': dict(trigger='oracle+sensing', policy='drf'),
    'SSBF + extended redistribution':          dict(trigger='ml+sensing', policy='drf', extended=True),
    'SSBF + late-arrival safeguards':          dict(trigger='ml+sensing', policy='drf', safeguards=True),
    'SSBF + extended + safeguards':            dict(trigger='ml+sensing', policy='drf', extended=True,
                                                    safeguards=True),
}


def jfi(x):
    x = np.asarray(x, dtype=float)
    return 1.0 if (x ** 2).sum() == 0 else x.sum() ** 2 / (len(x) * (x ** 2).sum())


def generate(p, seed):
    # all random events for one run, shared by every condition
    rng = np.random.default_rng(seed)
    n_rooms = sum(p['rooms_per_tier'])
    room_tier = np.repeat([0, 1, 2], p['rooms_per_tier'])
    room_vc = rng.random(n_rooms) < p['vc_room_share']
    room_vc[[0, 8, 16]] = True                       # at least one VC room per size

    U = p['n_users']
    weight = rng.gamma(1.5, 1.0, U)                  # some users book a lot, some rarely
    weight /= weight.sum()
    user_tier = rng.choice(3, U, p=p['tier_pref'])
    a = p['noshow_mean'] * p['noshow_conc']
    noshow_prop = rng.beta(a, p['noshow_conc'] - a, U)

    prof = np.array(p['slot_profile']) / np.mean(p['slot_profile'])
    bookings, waitlists = [], {}
    for d in range(p['days']):
        for s in range(p['slots']):
            n_req = rng.poisson(p['rho'] * n_rooms * prof[s])
            users = rng.choice(U, n_req, p=weight)
            free = np.ones(n_rooms, bool)
            wl = []
            for u in users:
                vc = rng.random() < p['vc_need']
                ok = np.where(free & (room_tier >= user_tier[u]) & (room_vc | ~vc))[0]
                if len(ok):
                    r = ok[np.argmin(room_tier[ok] * 2 + room_vc[ok])]   # smallest suitable room
                    free[r] = False
                    bookings.append((d, s, r, u, int(vc)))
                else:
                    wl.append((u, vc, rng.random() < p['flex_share']))
            waitlists[(d, s)] = wl

    B = len(bookings)
    bk = np.array(bookings)
    noshow = rng.random(B) < noshow_prop[bk[:, 3]]
    late = rng.random(B) < p['late_prob']
    arrival = np.where(late, rng.uniform(5, p['late_max'], B), rng.uniform(0, 5, B))
    # walk-ins, on their own random stream so they don't change other events
    wrng = np.random.default_rng(seed + 10_000_000)
    n_walk = wrng.poisson(p['walkin_rate'] * p['rho'] * prof[bk[:, 1]])
    walkins = [(np.sort(wrng.uniform(0, SLOT_MIN, k)), wrng.choice(U, k, p=weight), wrng.random(k) < p['vc_need'])
               for k in n_walk]
    # "running late" presses, also on their own stream
    press_u = np.random.default_rng(seed + 20_000_000).random(B)
    ev = dict(
        press_u=press_u,
        room_tier=room_tier, room_vc=room_vc, user_tier=user_tier, bk=bk, noshow=noshow, arrival=arrival,
        checks_in=rng.random(B) < p['checkin_compliance'],
        outage=rng.random(B) < p['sensor_outage'],
        false_vacant=rng.random(B) < p['sensor_false_vacant'],
        ml_flag=np.where(noshow, rng.random(B) < p['ml_tpr'], rng.random(B) < p['ml_fpr']),
        accept_u=rng.random((p['days'], p['slots'], U)),     # does the user accept
        resp_u=rng.random((p['days'], p['slots'], U)),       # how fast they answer
        rand_order=rng.random((p['days'], p['slots'], U)),   # tie-breaks
        waitlists=waitlists, walkins=walkins,
        requested=np.bincount(np.concatenate([bk[:, 3], [e[0] for wl in waitlists.values() for e in wl]]),
                              minlength=U),
    )
    return ev


def flagged(trig, ev, i):
    if trig == 'ml+sensing':
        return bool(ev['ml_flag'][i])
    if trig == 'oracle+sensing':
        return bool(ev['noshow'][i])
    return False


def release_time(c, ev, i, p):
    # minute the room is freed, or None if it is kept
    ns, arr = ev['noshow'][i], ev['arrival'][i]
    trig = c['trigger']
    if trig is None:
        return None
    if trig == 'checkin':
        present_by = (not ns) and arr <= p['checkin_release'] and ev['checks_in'][i]
        return None if present_by else p['checkin_release']
    if trig == 'ml':
        return 0.0 if ev['ml_flag'][i] else None
    if ev['outage'][i]:
        return None
    g = p['ml_grace'] if flagged(trig, ev, i) else p['grace']
    present = (not ns) and arr <= g and not ev['false_vacant'][i]
    if present:
        return None
    if c.get('safeguards'):
        if not ns and arr > g and arr <= p['max_hold'] and ev['press_u'][i] < p['late_notify']:
            return None                      # user said they're on the way
        if ns and ev['press_u'][i] < p['noshow_notify']:
            return float(p['max_hold'])      # misuse: freed when the hold runs out
    return float(g)


def simulate(cond_name, ev, p):
    c = CONDITIONS[cond_name]
    U, days, slots = p['n_users'], p['days'], p['slots']
    room_tier, room_vc, user_tier = ev['room_tier'], ev['room_vc'], ev['user_tier']
    seats = TIER_SEATS[room_tier]
    n_rooms = len(room_tier)

    # daily use per user, for the 30-day DRF/WFQ window
    day_slots = np.zeros((days, U))
    day_seat_h = np.zeros((days, U))
    received = np.zeros(U)            # slots each user actually got
    occupied_min = 0.0
    released = reallocated = wrongful = displaced = futile = rehoused = 0
    safeguards = c.get('safeguards', False)
    cand_count = 0

    walkin_min = 0.0
    walkin_claims = 0
    extended = c.get('extended', False)
    waitlists = {k: list(v) for k, v in ev['waitlists'].items()}   # served users get removed

    bk = ev['bk']
    order = np.lexsort((bk[:, 1], bk[:, 0]))
    idx_by_slot = {}
    for i in order:
        idx_by_slot.setdefault((bk[i, 0], bk[i, 1]), []).append(i)

    for d in range(days):
        lo = max(0, d - p['drf_window'])
        win_slots = day_slots[lo:d].sum(0)
        win_seat = day_seat_h[lo:d].sum(0)
        cap_slots = n_rooms * slots * max(1, d - lo)
        cap_seat = seats.sum() * slots * max(1, d - lo)
        for s in range(slots):
            wl = waitlists.get((d, s), [])
            slot_idx = idx_by_slot.get((d, s), [])
            if safeguards:
                # handle releases in time order
                rt = {i: release_time(c, ev, i, p) for i in slot_idx}
                slot_idx = sorted(slot_idx, key=lambda i: -1.0 if rt[i] is None else rt[i])
            stranded = []        # users who lost their room: [arrival, user, needs_vc]
            empty = {}           # room -> minute it became free and stayed empty
            for i in slot_idx:
                _, _, r, u, bvc = bk[i]
                ns, arr = ev['noshow'][i], ev['arrival'][i]
                t_rel = release_time(c, ev, i, p)
                if t_rel is None:
                    if not ns:
                        occupied_min += SLOT_MIN - arr
                        received[u] += 1
                        day_slots[d, u] += 1
                        day_seat_h[d, u] += seats[r]
                    continue
                released += 1
                attendee_coming = (not ns)
                if attendee_coming:
                    wrongful += 1
                # who can take it
                pools = [(s, wl)]
                if extended:
                    pools += [(s2, waitlists.get((d, s2), [])) for s2 in (s - 1, s + 1) if 0 <= s2 < slots]
                fits = [((s2, k), wu) for s2, pool in pools for k, (wu, vc, flex) in enumerate(pool)
                        if (s2 == s or flex) and room_tier[r] >= user_tier[wu] and (room_vc[r] or not vc)]
                if c['policy'] == 'fcfs':
                    ranked = fits
                elif c['policy'] == 'random':
                    ranked = sorted(fits, key=lambda kw: ev['rand_order'][d, s, kw[1]])
                elif c['policy'] == 'wfq':
                    ranked = sorted(fits, key=lambda kw: (win_slots[kw[1]] + day_slots[d, kw[1]],
                                                          ev['rand_order'][d, s, kw[1]]))
                else:  # drf: lowest dominant share first
                    ranked = sorted(fits, key=lambda kw: (max((win_seat[kw[1]] + day_seat_h[d, kw[1]]) / cap_seat,
                                                              (win_slots[kw[1]] + day_slots[d, kw[1]]) / cap_slots),
                                                          ev['rand_order'][d, s, kw[1]]))
                t = t_rel
                taker = None
                if safeguards:
                    # someone who lost their room goes first
                    k_s = next((k for k, (sa, su, sv) in enumerate(stranded) if sa <= t and
                                room_tier[r] >= user_tier[su] and (room_vc[r] or not sv)), None)
                    if k_s is not None and t <= SLOT_MIN - p['min_useful']:
                        su = stranded[k_s][1]
                        start = t + p['travel']
                        if not (attendee_coming and arr < start):
                            stranded.pop(k_s)
                            rehoused += 1
                            if attendee_coming:
                                displaced += 1
                                stranded.append([arr, u, bvc])
                            occupied_min += SLOT_MIN - start
                            received[su] += 1
                            day_slots[d, su] += 1
                            day_seat_h[d, su] += seats[r]
                            continue
                pre_notified = c['trigger'] == 'ml' and ev['ml_flag'][i] or flagged(c['trigger'], ev, i)
                for n_off, (k, wu) in enumerate(ranked):
                    if t > SLOT_MIN - p['min_useful']:
                        break
                    cand_count += 1
                    if ev['accept_u'][d, s, wu] < p['accept_prob']:
                        resp = 0.0 if (pre_notified and n_off == 0) else ev['resp_u'][d, s, wu] * p['timeout']
                        taker = (k, wu, t + resp)
                        break
                    t += p['timeout']
                if taker is None and extended:
                    # nobody in the queue took it, so it goes on the live map
                    claim = next(((wt, wu) for wt, wu, wvc in zip(*ev['walkins'][i])
                                  if t <= wt <= SLOT_MIN - p['min_useful'] and room_tier[r] >= user_tier[wu]
                                  and (room_vc[r] or not wvc)), None)
                    if claim is not None and not (attendee_coming and arr < claim[0]):
                        claim, cu = claim
                        walkin_claims += 1
                        walkin_min += SLOT_MIN - claim
                        occupied_min += SLOT_MIN - claim
                        # counts for DRF priority, not for satisfaction
                        day_slots[d, cu] += 1
                        day_seat_h[d, cu] += seats[r]
                        if attendee_coming:
                            displaced += 1
                            stranded.append([arr, u, bvc])
                        continue
                if taker is None:
                    # nobody took it; a late user can still walk in
                    if attendee_coming:
                        occupied_min += SLOT_MIN - arr
                        received[u] += 1
                        day_slots[d, u] += 1
                        day_seat_h[d, u] += seats[r]
                    else:
                        empty[r] = t_rel
                    continue
                k, wu, t_acc = taker
                start = t_acc + p['travel']
                if attendee_coming and arr < start:
                    # original user got there first and keeps it; wasted trip for the other user
                    futile += 1
                    occupied_min += SLOT_MIN - arr
                    received[u] += 1
                    day_slots[d, u] += 1
                    day_seat_h[d, u] += seats[r]
                    continue
                if attendee_coming:
                    displaced += 1
                    stranded.append([arr, u, bvc])
                reallocated += 1
                occupied_min += SLOT_MIN - start
                received[wu] += 1
                day_slots[d, wu] += 1
                day_seat_h[d, wu] += seats[r]
                s2, k2 = k
                waitlists[(d, s2)].pop(k2)
            if safeguards and stranded:
                # give anyone still without a room a free one
                booked_rooms = {bk[i, 2] for i in idx_by_slot.get((d, s), [])}
                for rr in range(n_rooms):
                    if rr not in booked_rooms:
                        empty[rr] = 0.0
                for sa, su, sv in sorted(stranded):
                    ok = [rr for rr, f in empty.items() if room_tier[rr] >= user_tier[su] and (room_vc[rr] or not sv)
                          and max(sa, f) + p['travel'] <= SLOT_MIN - p['min_useful']]
                    if not ok:
                        continue
                    rr = min(ok, key=lambda x: (room_tier[x] * 2 + room_vc[x], x))
                    start = max(sa, empty.pop(rr)) + p['travel']
                    rehoused += 1
                    occupied_min += SLOT_MIN - start
                    received[su] += 1
                    day_slots[d, su] += 1
                    day_seat_h[d, su] += seats[rr]

    req = np.maximum(ev['requested'], 1)
    sat = received / req
    active = ev['requested'] > 0
    tiers = ev['user_tier']
    n_book = len(bk)
    attended = int((~ev['noshow']).sum())
    res = dict(
        condition=cond_name,
        utilisation=occupied_min / (n_rooms * slots * days * SLOT_MIN) * 100,
        booked_share=n_book / (n_rooms * slots * days) * 100,
        noshow_rate=float(ev['noshow'].mean() * 100),
        released=released, reallocated=reallocated,
        realloc_rate=reallocated / released * 100 if released else 0.0,
        wrongful_per100=wrongful / attended * 100, displaced_per100=displaced / attended * 100,
        futile_per100=futile / attended * 100,
        rehoused_per100=rehoused / attended * 100, lost_per100=(displaced - rehoused) / attended * 100,
        walkin_claims=walkin_claims, walkin_share=walkin_min / max(1.0, occupied_min) * 100,
        offers_per_release=cand_count / released if released else 0.0,
        jfi_alloc=jfi(received[active]), jfi_satisfaction=jfi(sat[active]),
        sat_mean=float(sat[active].mean()),
        sat_small=float(sat[active & (tiers == 0)].mean()),
        sat_medium=float(sat[active & (tiers == 1)].mean()),
        sat_large=float(sat[active & (tiers == 2)].mean()),
    )
    heavy = ev['requested'] >= np.quantile(ev['requested'][active], 0.75)
    light = active & (ev['requested'] <= np.quantile(ev['requested'][active], 0.25))
    res['sat_heavy'] = float(sat[heavy].mean())
    res['sat_light'] = float(sat[light].mean())
    res['sat_tier_gap'] = max(res['sat_small'], res['sat_medium'], res['sat_large']) - \
        min(res['sat_small'], res['sat_medium'], res['sat_large'])
    return res


def run_rep(args):
    params, seed, conds, tag = args
    ev = generate(params, seed)
    out = []
    for cname in conds:
        r = simulate(cname, ev, params)
        r.update(seed=seed, **tag)
        out.append(r)
    return out


def paired(df, a, b, metric):
    x = df[df.condition == a].set_index('seed')[metric]
    y = df[df.condition == b].set_index('seed')[metric]
    diff = (y - x).dropna()
    n = len(diff)
    m, sd = diff.mean(), diff.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    t, pval = stats.ttest_rel(y.loc[diff.index], x.loc[diff.index]) if n > 1 and sd > 0 else (np.nan, np.nan)
    return dict(metric=metric, comparison=f'{b} vs {a}', n_reps=n, mean_diff=m, ci_low=m - half, ci_high=m + half,
                t=t, p=pval, cohens_dz=m / sd if sd > 0 else np.nan)


def summarise(df, metrics):
    g = df.groupby('condition', sort=False)[metrics]
    mean, sd, n = g.mean(), g.std(ddof=1), g.count()
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    out = mean.round(4).astype(str) + ' ± ' + half.round(4).astype(str)
    return mean, sd, out


def add_recovery(df, keys=('seed',)):
    # % of lost room time won back, compared with No release on the same seed
    keys = list(keys)
    base = df[df.condition == 'No release'].set_index(keys)[['utilisation', 'booked_share']]
    b = df.join(base, on=keys, rsuffix='_none')
    df['recovered_pct'] = ((b.utilisation - b.utilisation_none) /
                           (b.booked_share_none - b.utilisation_none) * 100).values
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--reps', type=int, default=30)
    args = ap.parse_args()
    reps = 5 if args.quick else args.reps
    sens_reps = 5 if args.quick else args.reps   # same seeds as the main run
    seeds = list(range(1000, 1000 + reps))
    t0 = time.time()

    with Pool() as pool:
        # main experiment, all 12 conditions
        jobs = [(DEFAULTS, s, list(CONDITIONS), {}) for s in seeds]
        main_df = add_recovery(pd.DataFrame([r for rs in pool.map(run_rep, jobs) for r in rs]))
        main_df.to_csv(os.path.join(OUT, 'main_runs.csv'), index=False)
        metrics = ['utilisation', 'recovered_pct', 'walkin_share', 'realloc_rate', 'wrongful_per100', 'displaced_per100',
                   'futile_per100', 'rehoused_per100', 'lost_per100', 'offers_per_release',
                   'jfi_alloc', 'jfi_satisfaction', 'sat_mean', 'sat_light', 'sat_heavy',
                   'sat_small', 'sat_medium', 'sat_large', 'sat_tier_gap', 'noshow_rate', 'booked_share']
        mean, sd, ci = summarise(main_df, metrics)
        mean.round(4).to_csv(os.path.join(OUT, 'main_mean.csv'))
        sd.round(4).to_csv(os.path.join(OUT, 'main_sd.csv'))
        ci.to_csv(os.path.join(OUT, 'main_mean_ci95.csv'))
        print(f'Main experiment ({reps} replications), mean ± 95% CI half-width')
        print(ci[['utilisation', 'recovered_pct', 'realloc_rate', 'wrongful_per100', 'displaced_per100', 'jfi_alloc',
                  'jfi_satisfaction', 'sat_tier_gap']].to_string())

        comps = []
        for a, b in [('No release', 'Sensing + DRF'), ('Check-in release + FCFS', 'Sensing + DRF'),
                     ('Sensing + FCFS', 'Sensing + DRF'), ('Sensing + WFQ', 'Sensing + DRF'),
                     ('Sensing + DRF', 'ML + Sensing + DRF (SSBF)'), ('No release', 'ML + Sensing + DRF (SSBF)'),
                     ('ML + Sensing + DRF (SSBF)', 'ML only + DRF'),
                     ('ML + Sensing + DRF (SSBF)', 'Oracle ML + Sensing + DRF (upper bound)'),
                     ('ML + Sensing + DRF (SSBF)', 'SSBF + extended redistribution'),
                     ('No release', 'SSBF + extended redistribution'),
                     ('ML + Sensing + DRF (SSBF)', 'SSBF + late-arrival safeguards'),
                     ('SSBF + extended redistribution', 'SSBF + extended + safeguards')]:
            for m in ['utilisation', 'recovered_pct', 'jfi_alloc', 'jfi_satisfaction', 'sat_tier_gap', 'wrongful_per100',
                      'displaced_per100', 'lost_per100']:
                comps.append(paired(main_df, a, b, m))
        comp_df = pd.DataFrame(comps)
        comp_df.round(5).to_csv(os.path.join(OUT, 'paired_comparisons.csv'), index=False)
        print('\nPaired comparisons across replications (utilisation, JFI)')
        print(comp_df[comp_df.metric.isin(['utilisation', 'jfi_alloc', 'jfi_satisfaction'])]
              .round(4).to_string(index=False))

        # sensitivity analysis, one setting at a time
        grid = {
            'rho': [0.5, 0.75, 1.0, 1.25, 1.5],
            'noshow_mean': [0.10, 0.20, 0.30, 0.40],
            'accept_prob': [0.3, 0.5, 0.6, 0.7, 0.9],
            'timeout': [2, 5, 10],
            'grace': [5, 10, 15],
            'late_prob': [0.0, 0.15, 0.30],
            'sensor_outage': [0.0, 0.02, 0.10, 0.20],
            'ml_fpr': [0.05, 0.10, 0.20],
            'vc_need': [0.0, 0.3, 0.6],
            'flex_share': [0.0, 0.15, 0.3, 0.5],
            'walkin_rate': [0.0, 0.25, 0.5, 1.0],
            # we measured ~2% in a trained room and 17-20% in other rooms (script 07)
            'sensor_false_vacant': [0.005, 0.02, 0.05, 0.10, 0.20, 0.40],
            'late_notify': [0.0, 0.25, 0.5, 0.75, 1.0],
            'noshow_notify': [0.0, 0.05, 0.10, 0.20],
        }
        # false-alarm rate -> recall, from script 02
        ml_points = {0.05: 0.13, 0.10: 0.24, 0.20: 0.43}
        sens_conds = ['No release', 'Check-in release + FCFS', 'Sensing + DRF', 'ML + Sensing + DRF (SSBF)',
                      'SSBF + extended redistribution', 'SSBF + late-arrival safeguards']
        jobs = []
        for param, values in grid.items():
            for v in values:
                pp = dict(DEFAULTS, **{param: v})
                if param == 'ml_fpr':
                    pp['ml_tpr'] = ml_points[v]
                for s in seeds[:sens_reps]:
                    jobs.append((pp, s, sens_conds, dict(param=param, value=v)))
        sens_df = add_recovery(pd.DataFrame([r for rs in pool.map(run_rep, jobs) for r in rs]),
                               keys=['param', 'value', 'seed'])
    sens_df.to_csv(os.path.join(OUT, 'sensitivity_runs.csv'), index=False)
    sens = sens_df.groupby(['param', 'value', 'condition'], sort=False)[
        ['utilisation', 'recovered_pct', 'wrongful_per100', 'displaced_per100', 'futile_per100', 'lost_per100',
         'jfi_alloc', 'jfi_satisfaction', 'realloc_rate']
    ].mean().round(4)
    sens.to_csv(os.path.join(OUT, 'sensitivity_summary.csv'))

    # gains for each sensitivity setting
    gains = []
    for (param, v), g in sens_df.groupby(['param', 'value'], sort=False):
        r = paired(g, 'No release', 'ML + Sensing + DRF (SSBF)', 'utilisation')
        r2 = paired(g, 'Sensing + DRF', 'ML + Sensing + DRF (SSBF)', 'utilisation')
        r3 = paired(g, 'No release', 'SSBF + extended redistribution', 'utilisation')
        rec = g.groupby('condition').recovered_pct.mean()
        gains.append(dict(param=param, value=v, gain_vs_none=r['mean_diff'], ci_low=r['ci_low'],
                          ci_high=r['ci_high'], recovered_pct=rec['ML + Sensing + DRF (SSBF)'],
                          ml_added=r2['mean_diff'], ml_ci_low=r2['ci_low'], ml_ci_high=r2['ci_high'],
                          ext_gain=r3['mean_diff'], ext_ci_low=r3['ci_low'], ext_ci_high=r3['ci_high'],
                          ext_recovered_pct=rec['SSBF + extended redistribution'],
                          safe_recovered_pct=rec['SSBF + late-arrival safeguards']))
    gains_df = pd.DataFrame(gains).round(3)
    gains_df.to_csv(os.path.join(OUT, 'sensitivity_gains.csv'), index=False)
    print('\nSensitivity: SSBF utilisation gain over No release (pp) and ML contribution over Sensing + DRF')
    print(gains_df.to_string(index=False))

    with open(os.path.join(OUT, 'config.json'), 'w') as f:
        json.dump(dict(defaults=DEFAULTS, conditions=CONDITIONS, seeds=seeds, sens_reps=sens_reps,
                       ml_operating_points=ml_points, sensitivity_grid=grid,
                       runtime_sec=round(time.time() - t0, 1)), f, indent=2, default=str)
    print(f'\nDone in {time.time() - t0:.0f}s. Results in {OUT}')


if __name__ == '__main__':
    main()
