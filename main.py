#!/usr/bin/env python3
# Roblox Top Trending forecaster, entry for roblox-trending-forecast-01
"""Forecast Roblox's Top Trending chart (device=computer, country=US).

Reads $MAKEFUN_INPUT_DIR (default /input): history.ndjson, task.json.
Writes $MAKEFUN_OUTPUT_DIR (default /output)/forecast.json:
  {"probabilities": [{"universe_id", "place", "p"}, ...], "top10": [...]}

The contest scores ONE forecast against every hourly read of the window, so
p[game, place] should be the expected share of the window's reads in which the
game holds that place: the average over the window's hours of
P(game at place | chart at the last read, h hours later).

Method (standard library only, deterministic):
  1. Every read -> rank table.  A game's state = its rank (blended with its mean
     rank over the previous 24 h, weight w), bucketed; "absent" for games seen
     in the last 72 h but not on the chart now.
  2. Empirical transition counts C[h][state][place 1..10 or other] from every
     pair of reads, a pair with gap g counting toward h with weight
     exp(-|g-h|/tau).  Smoothed toward a Gaussian rank-diffusion prior whose
     width grows with sqrt(h).
  3. Average over the window's hourly gaps, blend with each game's recent
     occupancy of each place, then iterative proportional fitting so each
     place's probabilities sum to 1 - (new-entrant share) and each game's sum
     to at most 1.
  4. Settings (prior strength, prior width, blend weight, occupancy weight) are
     chosen by a rolling-origin backtest on the history, scored exactly as the
     contest scores.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

IN_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

TOP = 10
OTHER = 10          # target class: not in the top 10
NS = 24             # states 0..19 = rank 1..20, 20: 21-30, 21: 31-45, 22: 46+, 23: absent
ABS = 23
REP = [i + 1 for i in range(20)] + [25, 38, 65, 90]
HMAX = 30
FLOOR = 1e-4
NEW_LO, NEW_HI = 0.002, 0.30


def to_hours(s):
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp() / 3600.0


# ---------------------------------------------------------------- history
def load_reads(path):
    reads = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if not o.get("ok", True):
                continue
            tt = o.get("top_trending")
            if not tt:
                continue
            ranks = {}
            for pos, row in enumerate(tt):
                uid = row.get("universe_id")
                if uid is None:
                    continue
                r = row.get("rank") or pos + 1
                if int(uid) not in ranks:
                    ranks[int(uid)] = int(r)
            if len(ranks) < TOP:
                continue
            try:
                reads.append((to_hours(o["at"]), ranks))
            except (KeyError, ValueError):
                continue
    reads.sort(key=lambda x: x[0])
    return reads


class Hist:
    def __init__(self, reads):
        self.t = [r[0] for r in reads]
        self.ranks = [r[1] for r in reads]
        self.N = len(reads)
        self.ent = []     # per read: list of (uid, rank|None, mean24|None), uid-sorted
        self.known = []
        lo24 = 0
        lo72 = 0
        for i in range(self.N):
            while self.t[i] - self.t[lo24] > 24:
                lo24 += 1
            while self.t[i] - self.t[lo72] > 72:
                lo72 += 1
            win = range(lo24, i + 1)
            n = len(win)
            ents = []
            ri = self.ranks[i]
            for g in sorted(ri):
                s = 0.0
                for j in win:
                    s += self.ranks[j].get(g, 70)
                ents.append((g, ri[g], s / n))
            seen = set()
            for j in range(lo72, i):
                seen.update(self.ranks[j].keys())
            for g in sorted(seen):
                if g not in ri:
                    ents.append((g, None, None))
            self.ent.append(ents)
            self.known.append(set(e[0] for e in ents))
        self.lo72 = None


def state_of(r, mean, w):
    if r is None:
        return ABS
    b = (1 - w) * r + w * mean
    if b <= 20:
        return max(1, int(b + 0.5)) - 1
    if b <= 30:
        return 20
    if b <= 45:
        return 21
    return 22


# ---------------------------------------------------------------- counts
def build_counts(H, Ws, origins):
    N = H.N
    C = {w: [[[0.0] * 11 for _ in range(NS)] for _ in range(HMAX + 1)] for w in Ws}
    newnum = [[0.0] * TOP for _ in range(HMAX + 1)]
    newden = [0.0] * (HMAX + 1)
    st = {w: [[state_of(r, m, w) for (_, r, m) in H.ent[i]] for i in range(N)] for w in Ws}
    snaps = {}

    def snap():
        return {
            "C": {w: [[row[:] for row in hh] for hh in C[w]] for w in Ws},
            "nn": [r[:] for r in newnum],
            "nd": newden[:],
        }

    for j in range(N):
        rj = H.ranks[j]
        inv = [None] * TOP
        for g, r in rj.items():
            if r <= TOP:
                inv[r - 1] = g
        for i in range(j):
            gap = H.t[j] - H.t[i]
            if gap < 0.4 or gap > HMAX + 3:
                continue
            tau = 1.0 + 0.1 * gap
            hs = []
            for h in range(max(1, int(gap - 3 * tau)), min(HMAX, int(gap + 3 * tau) + 1) + 1):
                wt = math.exp(-abs(gap - h) / tau)
                if wt >= 0.05:
                    hs.append((h, wt))
            if not hs:
                continue
            known = H.known[i]
            isnew = [1 if (inv[k] is not None and inv[k] not in known) else 0 for k in range(TOP)]
            for (h, wt) in hs:
                newden[h] += wt
                nh = newnum[h]
                for k in range(TOP):
                    if isnew[k]:
                        nh[k] += wt
            ents = H.ent[i]
            cls = []
            for (g, _, _) in ents:
                rr = rj.get(g)
                cls.append(rr - 1 if (rr is not None and rr <= TOP) else OTHER)
            for w in Ws:
                sti = st[w][i]
                Cw = C[w]
                for idx in range(len(ents)):
                    s = sti[idx]
                    c = cls[idx]
                    for (h, wt) in hs:
                        Cw[h][s][c] += wt
        if j in origins:
            snaps[j] = snap()
    if (N - 1) not in snaps and N > 0:
        snaps[N - 1] = snap()
    return snaps


# ---------------------------------------------------------------- prior
def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def make_prior(sd_scale):
    P0 = [None] * (HMAX + 1)
    for h in range(1, HMAX + 1):
        rows = []
        for s in range(NS):
            rho = REP[s]
            sd = sd_scale * (0.6 + 0.45 * math.sqrt(h)) * (1 + 0.06 * rho)
            p = [0.0] * 11
            p[0] = phi((1.5 - rho) / sd)
            for k in range(2, TOP + 1):
                p[k - 1] = phi((k + 0.5 - rho) / sd) - phi((k - 0.5 - rho) / sd)
            p[OTHER] = 1.0 - phi((TOP + 0.5 - rho) / sd)
            tot = sum(p)
            rows.append([x / tot for x in p])
        P0[h] = rows
    return P0


# ---------------------------------------------------------------- forecast
def forecast(H, snap, o, params, hs_list, prior):
    w = params["w"]
    m = params["m"]
    lam = params["occ"]
    ents = H.ent[o]
    cand = [e[0] for e in ents]
    states = [state_of(e[1], e[2], w) for e in ents]
    C = snap["C"][w]
    nh = float(len(hs_list))
    cache = {}
    f = [[0.0] * TOP for _ in cand]
    for h in hs_list:
        for ci, s in enumerate(states):
            key = (h, s)
            M = cache.get(key)
            if M is None:
                row = C[h][s]
                n = sum(row)
                P0 = prior[h][s]
                M = [(row[k] + m * P0[k]) / (n + m) for k in range(11)]
                cache[key] = M
            fc = f[ci]
            for k in range(TOP):
                fc[k] += M[k] / nh
    # new-entrant share per place
    nf = [0.0] * TOP
    for h in hs_list:
        for k in range(TOP):
            d = snap["nd"][h]
            v = snap["nn"][h][k] / d if d > 1.0 else 0.01
            nf[k] += min(NEW_HI, max(NEW_LO, v)) / nh
    # recent occupancy blend
    if lam > 0:
        idx = {g: ci for ci, g in enumerate(cand)}
        occ = [[0.0] * TOP for _ in cand]
        cnt = 0
        for j in range(o, -1, -1):
            if H.t[o] - H.t[j] > 72:
                break
            cnt += 1
            for g, r in H.ranks[j].items():
                if r <= TOP and g in idx:
                    occ[idx[g]][r - 1] += 1.0
        for ci in range(len(cand)):
            for k in range(TOP):
                f[ci][k] = (1 - lam) * f[ci][k] + lam * occ[ci][k] / max(cnt, 1)
    # iterative proportional fitting
    T = [1.0 - x for x in nf]
    G = len(cand)
    for _ in range(40):
        for k in range(TOP):
            cs = 0.0
            for ci in range(G):
                cs += f[ci][k]
            if cs > 0:
                sc = T[k] / cs
                for ci in range(G):
                    f[ci][k] *= sc
        for ci in range(G):
            rs = sum(f[ci])
            if rs > 1.0:
                for k in range(TOP):
                    f[ci][k] /= rs
    # floor, then cap each place's sum
    for k in range(TOP):
        for ci in range(G):
            if f[ci][k] < 2 * FLOOR:
                f[ci][k] = 2 * FLOOR
        cs = sum(f[ci][k] for ci in range(G))
        if cs > T[k]:
            sc = T[k] / cs
            for ci in range(G):
                f[ci][k] *= sc
    return cand, f


def to_rows(cand, f):
    rows = []
    for k in range(TOP):
        order = sorted(range(len(cand)), key=lambda ci: (-f[ci][k], cand[ci]))[:100]
        for ci in order:
            p = round(f[ci][k], 8)
            if p > 0:
                rows.append({"universe_id": int(cand[ci]), "place": k + 1, "p": p})
    return rows


def pick_top10(cand, f):
    used = set()
    out = []
    for k in range(TOP):
        best = None
        for ci in range(len(cand)):
            if ci in used:
                continue
            if best is None or f[ci][k] > f[best][k] or (f[ci][k] == f[best][k] and cand[ci] < cand[best]):
                best = ci
        used.add(best)
        out.append(int(cand[best]))
    return out


# ---------------------------------------------------------------- scoring (as the contest does)
def score_read(rows_by_place, colsum, ranks):
    inv = {}
    for g, r in ranks.items():
        if r <= TOP:
            inv[r] = g
    n_chart = len(ranks)
    tot = 0.0
    for k in range(1, TOP + 1):
        g = inv.get(k)
        if g is None:
            continue
        named = rows_by_place[k]
        if g in named:
            p = named[g]
        else:
            n_named_on_chart = 0
            for gg in named:
                if gg in ranks:
                    n_named_on_chart += 1
            p = (1.0 - colsum[k]) / max(1, n_chart - n_named_on_chart)
        tot += math.log(max(p, FLOOR))
    return tot


def score_forecast(cand, f, reads_ranks):
    rows = to_rows(cand, f)
    by = {k: {} for k in range(1, TOP + 1)}
    cs = {k: 0.0 for k in range(1, TOP + 1)}
    for r in rows:
        by[r["place"]][r["universe_id"]] = r["p"]
        cs[r["place"]] += r["p"]
    return sum(score_read(by, cs, rk) for rk in reads_ranks) / len(reads_ranks)


# ---------------------------------------------------------------- main
# Parameter grid for the rolling-origin backtest: the setting with the best contest score is used for the forecast.
GRID = [
    {"m": m, "sd": sd, "w": w, "occ": occ}
    for m in (6.0, 14.0)
    for sd in (1.1, 1.6)
    for w in (0.0, 0.4)
    for occ in (0.05, 0.2)
]
DEFAULT = {"m": 8.0, "sd": 1.2, "w": 0.0, "occ": 0.1}


def validate_forecast(out):
    try:
        counts = [0] * TOP
        sums = [0.0] * TOP
        for row in out["probabilities"]:
            uid = row["universe_id"]
            place = row["place"]
            p = row["p"]
            if type(uid) is not int or type(place) is not int or isinstance(p, bool) or not isinstance(p, (int, float)):
                raise ValueError("bad probability row")
            if place < 1 or place > TOP or p < 0 or p > 1:
                raise ValueError("bad probability row")
            counts[place - 1] += 1
            sums[place - 1] += p
        for k in range(TOP):
            if counts[k] > 100 or sums[k] > 1:
                raise ValueError("bad place totals")
        top10 = out["top10"]
        if len(top10) != TOP or len(set(top10)) != TOP:
            raise ValueError("bad top10")
    except (KeyError, TypeError, AttributeError):
        raise ValueError("bad forecast")


def log(msg):
    sys.stderr.write(msg + "\n")


def main():
    task = {}
    try:
        with open(os.path.join(IN_DIR, "task.json"), encoding="utf-8") as fh:
            task = json.load(fh)
    except (OSError, ValueError):
        pass
    reads = load_reads(os.path.join(IN_DIR, "history.ndjson"))
    if not reads:
        sys.exit("no readable history")
    H = Hist(reads)
    N = H.N
    t0 = H.t[-1]

    win = task.get("window") or {}
    try:
        wfrom = to_hours(win.get("from") or task.get("history_until"))
    except Exception:
        wfrom = t0 + 1.0
    slots = int(win.get("slots") or 24)
    smin = float(win.get("slot_minutes") or 60)
    gaps = []
    for i in range(slots):
        g = int(round(wfrom + i * smin / 60.0 - t0))
        gaps.append(min(HMAX, max(1, g)))

    # backtest origins
    origins = []
    cands = [o for o in range(N)
             if o >= 8 and H.t[o] - H.t[0] >= 36
             and sum(1 for r in range(o + 1, N) if 1 <= H.t[r] - H.t[o] <= 25) >= 2]
    if cands:
        step = max(1, len(cands) // 10)
        origins = cands[::step][:10]
    Ws = sorted(set(p["w"] for p in GRID) | {DEFAULT["w"]})
    snaps = build_counts(H, Ws, set(origins))
    priors = {}

    def prior_for(sd):
        if sd not in priors:
            priors[sd] = make_prior(sd)
        return priors[sd]

    best = dict(DEFAULT)
    if origins:
        bs = -1e18
        bt = list(range(1, 26))[:slots] if slots <= 25 else list(range(1, 26))
        for params in GRID:
            tot = 0.0
            for o in origins:
                tr = [H.ranks[r] for r in range(o + 1, N) if 1 <= H.t[r] - H.t[o] <= 25]
                hs_list = [min(HMAX, max(1, int(round(H.t[r] - H.t[o])))) for r in range(o + 1, N)
                           if 1 <= H.t[r] - H.t[o] <= 25]
                hs_use = sorted(set(bt)) if len(bt) else hs_list
                cand, f = forecast(H, snaps[o], o, params, hs_use, prior_for(params["sd"]))
                tot += score_forecast(cand, f, tr)
            tot /= len(origins)
            if tot > bs:
                bs, best = tot, params
        log("best params %s backtest %.4f" % (best, bs))

    final_snap = snaps[N - 1]
    cand, f = forecast(H, final_snap, N - 1, best, gaps, prior_for(best["sd"]))
    out = {"probabilities": to_rows(cand, f), "top10": pick_top10(cand, f)}
    os.makedirs(OUT_DIR, exist_ok=True)
    validate_forecast(out)
    with open(os.path.join(OUT_DIR, "forecast.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
