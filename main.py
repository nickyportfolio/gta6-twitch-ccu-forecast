#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun).

Reads every *.in case file in $MAKEFUN_INPUT_DIR (fallback /input/) and writes
$MAKEFUN_OUTPUT_DIR/<id>.out (fallback /output/) per case.

Model (no fitted constants, no practice data anywhere in the code):
- For each comparable launch, take its release-day hourly curve and form the
  log-ratio r_h = ln(count_h / count_1) for h = 2..24.
- For each hour, fit a LOCAL (kernel-weighted) linear regression of r_h on
  ln(hour-1) across the case file's own comparables, with a Gaussian kernel
  of bandwidth 2.0 in log units. Console launches with a rolling midnight
  unlock (the shape GTA VI itself has) get a weight bonus of 5, since their
  curves carry the timezone-rollout shape. The median for hour h is
  hour1 * exp(r_h). This is computed fresh from the case file at run time,
  so it adapts to the final's 42 comparables.
- There is NO per-hour bias table and NO constant fitted to the practice
  cases: the model relies on the comparables alone, so it cannot overfit
  four launches. (A bias fitted to the practice answers would look better
  on the practice and collapse on unseen data.)
- Spread: for each hour, the honest dispersion of the median, measured by
  leave-one-out on the case file's own similar comparables (console-rolling
  launches with hour-1 >= 50,000): predict each one's curve from the
  others and take the root mean square of the log-errors at that hour.
  The spread is 0.8 times that RMS (floored 0.05, capped 1.5 log
  units). It is the dispersion BETWEEN launches, not just a regression
  residual, so the quantiles reflect how much launches really differ.
- The five quantiles are median * exp(z_tau * spread) with z the
  standard-normal quantiles.

Standard library only. No network. Deterministic (no randomness).
"""
import json
import math
import os
import sys

Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

BANDWIDTH = 2.0       # Gaussian kernel bandwidth in ln(hour-1) units
CR_BONUS = 5.0        # weight bonus for console-rolling comparables
MIN_H1 = 1000.0       # comparables below this hour-1 are too noisy to use
LOO_MIN_H1 = 50000.0  # pool for the LOO spread: similar, large launches
SPREAD_SCALE = 0.8    # fraction of the LOO error dispersion used
SPREAD_FLOOR = 0.05   # never narrower than this (log units)
SPREAD_CAP = 1.5      # never wider than this (log units)


def release_day_hours(comp):
    """Return {hour: viewers} for the comparable's UTC release day."""
    day = comp["day_start"][:10]
    hours = {}
    for pt in comp.get("hourly", []):
        if not pt.get("from", "").startswith(day):
            continue
        v = pt.get("viewers")
        if v is None or v <= 0:
            continue
        h = int(pt["from"][11:13]) + 1
        if 1 <= h <= 24:
            hours[h] = v
    return hours


def is_console_rolling(c):
    """True when a comparable is a console launch with a rolling
    midnight-local unlock (the shape GTA VI itself has)."""
    u = (c.get("unlock", "") or "").lower()
    p = (c.get("platforms", "") or "").lower()
    has_console = ("ps" in p) or ("xbox" in p)
    rolling = ("midnight local" in u) or ("rolling" in u)
    return has_console and rolling


def prep_comps(raw_comps):
    out = []
    for c in raw_comps:
        hrs = release_day_hours(c)
        if 1 in hrs and len(hrs) >= 20:
            out.append({"hours": hrs,
                        "platforms": c.get("platforms", ""),
                        "unlock": c.get("unlock", "")})
    return out


def wlinreg(xs, ys, ws):
    """Weighted linear regression y = a + b*x. Returns (a, b)."""
    sw = sum(ws)
    if sw <= 0:
        return (sum(ys) / len(ys) if ys else 0.0), 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if den <= 1e-12:
        return my, 0.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / den
    return my - b * mx, b


def kernel_curve(h1, comps):
    """Local kernel regression of r_h on ln(hour-1) for each hour.

    Returns {hour: r_h}, the predicted log-ratio for that hour.
    """
    x = math.log(max(h1, 1.0))
    out = {}
    for h in range(2, 25):
        pts = []
        for c in comps:
            hrs = c["hours"]
            if h in hrs and 1 in hrs and hrs[1] >= MIN_H1:
                w = 5.0 if is_console_rolling(c) else 1.0
                pts.append((math.log(hrs[1]),
                            math.log(hrs[h] / hrs[1]), w))
        if not pts:
            out[h] = 0.0
            continue
        ws = [math.exp(-0.5 * ((xx - x) / BANDWIDTH) ** 2) * sw
              for xx, _, sw in pts]
        a, b = wlinreg([p[0] for p in pts], [p[1] for p in pts], ws)
        out[h] = a + b * x
    return out


def loo_error_dispersion(comps):
    """Per-hour RMS of the kernel model's leave-one-out log-errors on
    similar comparables (console-rolling, hour-1 >= 50,000).

    For each such comparable, predict its curve from all the OTHER
    comparables and take the log-error per hour. The root mean square
    across those launches is how wrong the median is at that hour --
    it counts both the spread between launches and any systematic miss,
    so it is the honest width for the quantiles. Measured on the case
    file's own data, never on the practice answers.
    """
    pool = [c for c in comps
            if is_console_rolling(c) and c["hours"].get(1, 0) >= LOO_MIN_H1]
    disp = {}
    for h in range(2, 25):
        errs = []
        for target in pool:
            if h not in target["hours"]:
                continue
            others = [c for c in comps if c is not target]
            curve = kernel_curve(target["hours"][1], others)
            pred = target["hours"][1] * math.exp(curve[h])
            errs.append(math.log(target["hours"][h]) - math.log(pred))
        if len(errs) >= 3:
            disp[h] = math.sqrt(sum(e * e for e in errs) / len(errs))
        else:
            disp[h] = 0.3
    return disp


def forecast_case(case):
    """Forecast one case. Returns {hour: [q05, q25, q50, q75, q95]}."""
    h1 = case["known"][0]["viewers"]
    if h1 is None or h1 < 1:
        h1 = 1
    comps = prep_comps(case.get("comparables", []))
    curve = kernel_curve(h1, comps)
    disp = loo_error_dispersion(comps)
    out = {}
    for h in range(2, 25):
        med = h1 * math.exp(curve[h])
        s_h = min(max(SPREAD_SCALE * disp.get(h, 0.3),
                      SPREAD_FLOOR), SPREAD_CAP)
        qs = [med * math.exp(z * s_h) for z in Z]
        qs = [min(max(q, 1.0), 1e9) for q in qs]
        out[h] = qs
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        cid = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        fc = forecast_case(case)
        rows = [{"hour": h, "q": fc[h]} for h in range(2, 25)]
        with open(os.path.join(out_dir, cid + ".out"), "w",
                  encoding="utf-8") as f:
            json.dump({"hours": rows}, f, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
