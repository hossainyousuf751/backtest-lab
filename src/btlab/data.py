"""Synthetic markets with controllable signal, listings and delistings.

Because the data-generating process is known, every claim a backtest makes
can be checked: on a pure-noise market, no strategy should be significant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def synthetic_market(n_assets: int = 100, n_years: int = 10, momentum_strength: float = 0.0,
                     momentum_half_life: int = 126, delist_rate: float = 0.0,
                     delisting_return: float = -0.5, late_listing_frac: float = 0.0,
                     seed: int | None = 0, start: str = "2014-01-01") -> pd.DataFrame:
    """Daily prices (dates x assets) with NaN before listing and after delisting.

    Returns = beta * market + mu_t + idiosyncratic noise, where mu_t is a
    persistent AR(1) expected return with annualised cross-sectional standard
    deviation `momentum_strength`. Past returns are informative about mu_t,
    so momentum works exactly when momentum_strength > 0.

    Firms that delist (annual hazard `delist_rate`) drift down for six months,
    then suffer `delisting_return` on their final day.
    """
    rng = np.random.default_rng(seed)
    n_days = TRADING_DAYS * n_years
    dates = pd.bdate_range(start, periods=n_days)

    vol = rng.uniform(0.20, 0.50, n_assets) / np.sqrt(TRADING_DAYS)
    beta = rng.uniform(0.6, 1.4, n_assets)
    market = rng.normal(0.07 / TRADING_DAYS, 0.15 / np.sqrt(TRADING_DAYS), n_days)

    phi = 0.5 ** (1 / momentum_half_life)
    mu_sd = momentum_strength / TRADING_DAYS
    mu = np.zeros((n_days, n_assets))
    mu[0] = rng.normal(0, mu_sd, n_assets)
    shocks = rng.normal(0, mu_sd * np.sqrt(1 - phi**2), (n_days, n_assets))
    for t in range(1, n_days):
        mu[t] = phi * mu[t - 1] + shocks[t]

    rets = beta * market[:, None] + mu + rng.standard_normal((n_days, n_assets)) * vol

    alive = np.ones((n_days, n_assets), dtype=bool)
    daily_hazard = 1 - (1 - delist_rate) ** (1 / TRADING_DAYS)
    for j in range(n_assets):
        if late_listing_frac and rng.random() < late_listing_frac:
            alive[: rng.integers(1, int(0.6 * n_days)), j] = False
        if delist_rate:
            hits = np.flatnonzero(rng.random(n_days) < daily_hazard)
            hits = hits[hits > 130]
            if len(hits):
                d = hits[0]
                rets[d - 126:d, j] -= 0.4 / 126          # slow decline before failure
                rets[d, j] = delisting_return
                alive[d + 1:, j] = False

    rets = np.clip(rets, -0.95, None)
    rets[0] = 0.0
    prices = 100 * np.cumprod(1 + rets, axis=0)
    prices[~alive] = np.nan
    return pd.DataFrame(prices, index=dates, columns=[f"A{j:03d}" for j in range(n_assets)])


def synthetic_dollar_volume(prices: pd.DataFrame, median_adv: float = 2e7, size_dispersion: float = 1.0,
                            seed: int | None = 0) -> pd.DataFrame:
    """Daily traded value per asset: a lognormal firm size times lognormal daily noise.

    Firm sizes span roughly two orders of magnitude around `median_adv`
    (default $20m/day, a small/mid-cap stock), which is what makes capacity
    depend on which names a strategy trades.
    """
    rng = np.random.default_rng(seed)
    size = median_adv * np.exp(rng.normal(0, size_dispersion, prices.shape[1]))
    noise = np.exp(rng.normal(-0.045, 0.3, prices.shape))     # mean-one daily variation
    return pd.DataFrame(size * noise, index=prices.index, columns=prices.columns).where(prices.notna())


FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def load_french_factors(start: str = "1990-01-01") -> pd.DataFrame:
    """Daily US factors from Kenneth French's data library, as decimals.

    Columns: Mkt-RF, SMB (size), HML (value), RF (T-bill), Mom (momentum).
    """
    import io
    import urllib.request
    import zipfile

    frames = []
    for name in ("F-F_Research_Data_Factors_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip"):
        with urllib.request.urlopen(FRENCH_URL + name, timeout=60) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        lines = zf.read(zf.namelist()[0]).decode("latin1").splitlines()
        header = next(i for i, ln in enumerate(lines) if ln.startswith(","))
        body = [ln for ln in lines[header + 1:] if ln[:8].isdigit()]
        df = pd.read_csv(io.StringIO("\n".join(body)), header=None, index_col=0,
                         names=["date"] + [c.strip() for c in lines[header].split(",")[1:]])
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
        frames.append(df)
    out = pd.concat(frames, axis=1, join="inner").rename(columns={"MOM": "Mom"}) / 100
    return out.loc[start:]


def load_yfinance(tickers: list[str], start: str = "2005-01-01", end: str | None = None,
                  field: str = "Close") -> pd.DataFrame:
    """Adjusted prices (or field="Volume"). Today's ticker list is survivorship-biased for stocks."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("install the data extra: pip install -e '.[data]'") from exc
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    return data[field].dropna(how="all")
