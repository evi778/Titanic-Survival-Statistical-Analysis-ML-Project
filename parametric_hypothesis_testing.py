import numpy as np
from scipy.stats import norm, t, chi2, f


# =====================================
# Formatting
# =====================================
def _format_result(name, statistic, pvalue, ci=None):
    print(name)
    print(f"Test statistic: {statistic:.4f}")
    print(f"p-value:        {pvalue:.4g}")
    if ci is not None:
        print(f"Confidence interval: ({ci[0]:.4f}, {ci[1]:.4f})")


# =====================================
# One-sample Z test
# =====================================
def z_test_stats(xbar, n, mu0, sigma,
                 alternative="two-sided", conf_level=0.95):
    se = sigma / np.sqrt(n)
    z = (xbar - mu0) / se

    if alternative == "two-sided":
        p = 2 * (1 - norm.cdf(abs(z)))
    elif alternative == "greater":
        p = 1 - norm.cdf(z)
    else:
        p = norm.cdf(z)

    alpha = 1 - conf_level
    zcrit = norm.ppf(1 - alpha / 2)
    ci = (xbar - zcrit * se, xbar + zcrit * se)

    return {"statistic": z, "pvalue": p, "ci": ci}


def z_test(sample, mu0, sigma, verbose=True, **kwargs):
    sample = np.asarray(sample)
    res = z_test_stats(sample.mean(), len(sample), mu0, sigma, **kwargs)
    if verbose:
        _format_result("One-sample Z test (σ known)",
                       res["statistic"], res["pvalue"], res["ci"])
    return res


# =====================================
# One-sample t test
# =====================================
def t_test_stats(xbar, s, n, mu0,
                 alternative="two-sided", conf_level=0.95):
    se = s / np.sqrt(n)
    tstat = (xbar - mu0) / se
    df = n - 1

    if alternative == "two-sided":
        p = 2 * (1 - t.cdf(abs(tstat), df))
    elif alternative == "greater":
        p = 1 - t.cdf(tstat, df)
    else:
        p = t.cdf(tstat, df)

    alpha = 1 - conf_level
    tcrit = t.ppf(1 - alpha / 2, df)
    ci = (xbar - tcrit * se, xbar + tcrit * se)

    return {"statistic": tstat, "pvalue": p, "ci": ci, "df": df}


def t_test(sample, mu0, verbose=True, **kwargs):
    sample = np.asarray(sample)
    res = t_test_stats(sample.mean(),
                       sample.std(ddof=1),
                       len(sample),
                       mu0,
                       **kwargs)
    if verbose:
        _format_result("One-sample t test",
                       res["statistic"], res["pvalue"], res["ci"])
    return res


# =====================================
# One-sample proportion Z test
# =====================================
def z_test_proportion_stats(phat, n, p0,
                            alternative="two-sided", conf_level=0.95):
    se0 = np.sqrt(p0 * (1 - p0) / n)
    z = (phat - p0) / se0

    if alternative == "two-sided":
        p = 2 * (1 - norm.cdf(abs(z)))
    elif alternative == "greater":
        p = 1 - norm.cdf(z)
    else:
        p = norm.cdf(z)

    alpha = 1 - conf_level
    zcrit = norm.ppf(1 - alpha / 2)
    se = np.sqrt(phat * (1 - phat) / n)
    ci = (phat - zcrit * se, phat + zcrit * se)

    return {"statistic": z, "pvalue": p, "ci": ci}


def z_test_proportion(count, n, p0, alternative='two-sided', conf_level=0.95, verbose=True, **kwargs):
    phat = count / n
    res = z_test_proportion_stats(phat, n, p0, alternative=alternative, conf_level=conf_level)
    if verbose:
        _format_result("One-sample Z test for proportion",
                       res["statistic"], res["pvalue"], res["ci"])
    return res


# =====================================
# Chi-square test for variance
# =====================================
def chisq_test_stats(s2, n, var0,
                     alternative="two-sided", conf_level=0.95):
    df = n - 1
    chi_stat = df * s2 / var0

    if alternative == "two-sided":
        p = 2 * min(chi2.cdf(chi_stat, df),
                    1 - chi2.cdf(chi_stat, df))
    elif alternative == "greater":
        p = 1 - chi2.cdf(chi_stat, df)
    else:
        p = chi2.cdf(chi_stat, df)

    alpha = 1 - conf_level
    chi_low = chi2.ppf(alpha / 2, df)
    chi_high = chi2.ppf(1 - alpha / 2, df)
    ci = (df * s2 / chi_high, df * s2 / chi_low)

    return {"statistic": chi_stat, "pvalue": p, "ci": ci, "df": df}


def chisq_test(sample, var0, verbose=True, **kwargs):
    sample = np.asarray(sample)
    res = chisq_test_stats(sample.var(ddof=1), len(sample), var0, **kwargs)
    if verbose:
        _format_result("Chi-square test for variance",
                       res["statistic"], res["pvalue"], res["ci"])
    return res


# =====================================
# F test for equality of variances
# =====================================
def f_test_stats(s1_sq, n1, s2_sq, n2,
                 alternative="two-sided", conf_level=0.95):
    df1, df2 = n1 - 1, n2 - 1

    if s1_sq >= s2_sq:
        f_stat = s1_sq / s2_sq
        df_num, df_den = df1, df2
    else:
        f_stat = s2_sq / s1_sq
        df_num, df_den = df2, df1

    if alternative == "two-sided":
        p = 2 * min(f.cdf(f_stat, df_num, df_den),
                    1 - f.cdf(f_stat, df_num, df_den))
    elif alternative == "greater":
        p = 1 - f.cdf(f_stat, df_num, df_den)
    else:
        p = f.cdf(f_stat, df_num, df_den)

    alpha = 1 - conf_level
    f_low = f.ppf(alpha / 2, df_num, df_den)
    f_high = f.ppf(1 - alpha / 2, df_num, df_den)
    ci = (f_stat / f_high, f_stat / f_low)

    return {"statistic": f_stat, "pvalue": p, "ci": ci,
            "df_num": df_num, "df_den": df_den}


def f_test(x, y, verbose=True, **kwargs):
    x, y = np.asarray(x), np.asarray(y)
    res = f_test_stats(x.var(ddof=1), len(x),
                       y.var(ddof=1), len(y), **kwargs)
    if verbose:
        _format_result("F test for equality of variances",
                       res["statistic"], res["pvalue"], res["ci"])
    return res


# =====================================
# Two-sample Z test
# =====================================
def z_test_2samples_stats(xbar, n1, ybar, n2,
                          sigma1, sigma2,
                          alternative="two-sided", conf_level=0.95):
    se = np.sqrt(sigma1 ** 2 / n1 + sigma2 ** 2 / n2)
    z = (xbar - ybar) / se

    if alternative == "two-sided":
        p = 2 * (1 - norm.cdf(abs(z)))
    elif alternative == "greater":
        p = 1 - norm.cdf(z)
    else:
        p = norm.cdf(z)

    alpha = 1 - conf_level
    zcrit = norm.ppf(1 - alpha / 2)
    ci = ((xbar - ybar) - zcrit * se,
          (xbar - ybar) + zcrit * se)

    return {"statistic": z, "pvalue": p, "ci": ci}


def z_test_2samples(x, y, sigma1, sigma2, verbose=True, **kwargs):
    x, y = np.asarray(x), np.asarray(y)
    res = z_test_2samples_stats(x.mean(), len(x),
                                y.mean(), len(y),
                                sigma1, sigma2, **kwargs)
    if verbose:
        _format_result("Two-sample Z test (σ known)",
                       res["statistic"], res["pvalue"], res["ci"])
    return res


# =====================================
# Two-sample t test (pooled or Welch)
# =====================================
def t_test_2samples_stats(xbar, s1, n1,
                          ybar, s2, n2,
                          equal_var=True,
                          alternative="two-sided", conf_level=0.95):
    if equal_var:
        sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        se = sp * np.sqrt(1 / n1 + 1 / n2)
        df = n1 + n2 - 2
    else:
        se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
        df = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2 / (
            (s1 ** 4) / (n1 ** 2 * (n1 - 1)) +
            (s2 ** 4) / (n2 ** 2 * (n2 - 1))
        )

    tstat = (xbar - ybar) / se

    if alternative == "two-sided":
        p = 2 * (1 - t.cdf(abs(tstat), df))
    elif alternative == "greater":
        p = 1 - t.cdf(tstat, df)
    else:
        p = t.cdf(tstat, df)

    alpha = 1 - conf_level
    tcrit = t.ppf(1 - alpha / 2, df)
    ci = ((xbar - ybar) - tcrit * se,
          (xbar - ybar) + tcrit * se)

    return {"statistic": tstat, "pvalue": p, "ci": ci, "df": df}


def t_test_2samples(x, y, verbose=True, **kwargs):
    x, y = np.asarray(x), np.asarray(y)
    res = t_test_2samples_stats(x.mean(), x.std(ddof=1), len(x),
                                y.mean(), y.std(ddof=1), len(y),
                                **kwargs)
    if verbose:
        _format_result("Two-sample t test",
                       res["statistic"], res["pvalue"], res["ci"])
    return res
