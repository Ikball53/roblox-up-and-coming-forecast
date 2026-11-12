# Roblox Up-and-Coming top-10 forecast

Entry for [make.fun](https://make.fun) competition
`roblox-up-and-coming-forecast-01`: forecast Roblox's Up-and-Coming chart
top 10, place by place, for the 72 hours after entries close
(2026-11-13 06:00 UTC to 2026-11-16 06:00 UTC), for the highest log score.

## Program

`main.py` (Python 3.13, standard library only) reads `/input/history.ndjson`
and `/input/task.json` and writes `/output/forecast.json` with exactly two
keys: `probabilities` (a place-by-game probability matrix) and `top10`
(the point guess at the top 10 in order).

Model: each game ever seen gets a latent rank that diffuses forward in
time — `R(h) = R0 + drift*h + sigma*sqrt(h)*Z`, with `R0` its rank in the
last read (off-chart games start past the chart's edge), drift its OLS rank
slope over its whole series (shrunk, capped at 0.5 places/hour), and sigma
`1.0 + 0.015*R0` places per sqrt(hour). One set of standard normals per
game is shared across the 72 hourly slots (Brownian scaling), so 4000
scenarios give a stable Monte Carlo; the forecast is the mean over the
slots of the per-slot place marginals, which is what the log score (a mean
over reads) rewards. Each place's named probabilities are scaled to 0.85,
leaving 0.15 unnamed: brand-new games debut straight into the top 10, and
the unnamed mass is the only way to score on them.

Rank velocity alone is mostly noise (corr 0.09 with the window's ranks)
and playing-growth tilts hurt in backtest, so neither is used; the last
rank is the anchor (corr 0.73 with the window's mean rank).

## Backtest (practice instance)

History = the 34 archive reads to 2026-11-07T20:53Z; answer = the 10
archive reads in the 72h window (reconstructed from the public
[roblox-snapshots](https://github.com/Kenny-Dhub/roblox-snapshots) archive
and verified against the rules' fingerprints: first read holds 9 of the
close's top 10, last read 5; South Hell leads 4).

| Forecast | Log score | Top-10 hits | Agreement |
| --- | --- | --- | --- |
| This program | **-34.514** | 5.4 | 39.5 |
| House baseline (persistence) | -43.512039 | 6.4 | 53.5 |

Beats the house baseline by +8.998 nats. The scorer replicates
`grader/forecast/ranked.ts` exactly; it reproduces the baseline's
-43.512039 on the same answer to 6 decimals.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: one fixed seed, sorted iteration, no network, no wall
clock. Two runs on the same history write byte-identical output. Runs in
about 40 seconds on one CPU, well inside the 300s limit.

## License

MIT — see `LICENSE`.
