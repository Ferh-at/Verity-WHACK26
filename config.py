"""Shared constants for data splits, statistics, the monitor and model calls."""

# --- general
SEED = 42                                  # base seed; seed for replicate b = SEED + b
DATA_START, DATA_END = "2000-01-31", "2026-10-31"  # fetch range (2000 gives warm-up for 36m windows)
START = "2003-01-31"                       # first month used in tests (DFII10 starts Jan 2003)
FREQ = "ME"                                # pandas month-end; all dates tz-naive month-end Timestamps
SPLITS = {"discovery": ("2003-01-31", "2014-12-31"),
          "validation": ("2015-01-31", "2019-12-31"),
          "holdout":    ("2020-01-31", "2026-10-31")}
SHORT_HISTORY_RULE = (0.6, 0.2, 0.2)       # assets with <60 months in discovery (e.g. BTC):
                                           # split their own history 60/20/20 chronologically
EMBARGO_MONTHS = 1                         # drop this many months at each split boundary
HORIZON_MONTHS = 1                         # position from data <= t earns return of t+1
LLM_CUTOFF = "2024-06-30"                  # set to the chosen model's stated knowledge cutoff
POST_CUTOFF_START = "2024-07-31"           # post-cutoff panel = months after LLM_CUTOFF
# --- trading
COST_BPS = 2.0                             # cost per unit of one-way turnover |p_t - p_{t-1}|
ZWIN = 36; ZMIN = 24                       # trailing window / min obs for clip_z transform
ZCLIP = 2.0                                # clip z at +-2, then divide by 2 -> position in [-1, 1]
# --- statistics
ALPHA = 0.05                               # one-sided per-test level
FDR_Q = 0.10                               # Benjamini-Yekutieli q across the ledger
N_BOOT = 1000                              # stationary-bootstrap replicates per test
N_HURDLE = 200                             # null replicates of the whole search (luck hurdle)
HURDLE_PCT = 95                            # hurdle = 95th percentile of null best-of statistic
MIN_SHIFT = 12                             # circular shift k ~ Uniform{MIN_SHIFT, ..., T-MIN_SHIFT}
BLOCK_LEN_RULE = "politis_white"           # arch.bootstrap.optimal_block_length, fallback ceil(T**(1/3))
NW_LAG_RULE = "nw1994"                     # L = floor(4*(T/100)**(2/9))
DUP_CORR = 0.90                            # strategy-return correlation above which = duplicate
CANT_TELL_MDE = 0.75                       # annual Sharpe; not significant and MDE80 > this -> Can't tell yet
POWER_TARGET = 0.80
VALID_T_MIN = 1.0                          # validation pass: same sign and NW t > 1.0
FACTOR_T_MIN = 1.645                       # factor check: alpha NW t must exceed this (one-sided 5%)
FACTOR_ENABLED = False                     # factor test becomes a kill test once enabled
OUTLIER_DROP_FRAC = 0.01                   # drop top 1% |net return| months (at least 1)
HINDSIGHT_MIN_N = 24                       # min post-cutoff months for the hindsight test to be able to kill
VENDOR_GAP = 0.01                          # monthly return gap > 1 pp counts as a disagreement
VENDOR_MAX_BAD = 2                         # more than 2 disagreeing months in the window -> flag
# --- leak audit
AUDIT_N_DATES = 24; AUDIT_TOL = 1e-9; AUDIT_EDIT_SCALE = 0.05
# --- monitor
ROLL_WIN = 36; ROLL_WIN_BTC = 24           # rolling window (months) for rule strength
STRENGTH_BOOT = 500; STRENGTH_BLOCK = 6    # bootstrap range for strength (5th-95th pct)
CUSUM_K = 0.5
FALSE_ALARM_YEARS = 20                     # target: about 1 false alarm per 20 years per rule
CALIB_WINDOW = ("2003-01-31", "2019-12-31")
N_CALIB_SIMS = 1000; SIM_MONTHS = 240; CALIB_BLOCK = 12
CALIB_QUANTILE = 1 - 2.718281828 ** -1     # = 0.632; threshold quantile for the monitor
BREAK_MONTH = 120; DETECT_WITHIN = 24; N_BREAK_SIMS = 200
REPLAY_START = "2015-01-31"
# --- benchmark (planted traps)
BENCH = dict(M=200, T=300, disc=180, val=60, hold=60, mu=0.005, sigma=0.04, ar=0.5,
             real=10, real_sr=[0.5, 1.0, 1.5, 2.0], leaky=10, outlier=10, regime=10,
             beta=10, dup=20, noise=130, outlier_sigma=8.0, regime_ic=0.5, regime_months=60,
             beta_mu=0.010, dup_noise=0.05)
LOOKBACKS = [1, 3, 6, 12, 24, 36]
# --- model calls
TEMPERATURE = {"generator": 0.7, "skeptic": 0.2, "explainer": 0.2, "probe": 0.0, "reviewer": 0.2}
LLM_TIMEOUT_S = 60; LLM_MAX_RETRIES = 2
N_HYPOTHESES_DEFAULT = 20
# --- research loop
RESEARCH_ROUNDS = 3                        # explore rounds per research run (fixed before the run)
REFINE_BUDGET = 5                          # max new hypotheses the reviewer may add per round
NOMINATE_MAX = 3                           # max ideas the reviewer may send to the one-shot holdout
COST_STRESS = 5.0                          # "costs" challenge re-tests at this multiple of COST_BPS
