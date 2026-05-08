#!/usr/bin/env python3
"""
Numerical verifier for the Erdős 690 non-unimodality proof.

This script verifies:
  * the finite prime sums appearing in the proof, using exact rational arithmetic;
  * the small-case R_r table, using exact rational elementary symmetric sums;
  * the interval for C = sum_p 1/(p(p-1)), using directed Decimal summation
    for the finite partial sum and the exact telescoping tail bound;
  * the displayed logarithmic numerical inequalities, using rational interval
    arithmetic for logarithms based on the atanh series with a rigorous tail bound.

This script treats the following as external mathematical or certified inputs:
  * the explicit prime estimates cited in the proof;
  * the stated interval for the Meissel-Mertens constant B;
  * the published large prime gap and twin-prime certificates.

Thus, conditional on these cited inputs, this script checks the
numerical comparisons used in the proof.
"""

from __future__ import annotations

import sys
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Iterable, List, Tuple
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING

# ---------------------------- exact decimals -----------------------------

def Q(s: str | int) -> Fraction:
    """Exact rational represented by a decimal/integer string."""
    if isinstance(s, int):
        return Fraction(s, 1)
    s = s.strip().replace(",", "")
    if "e" in s.lower():
        # minimal scientific notation support, e.g. 1e-5
        base, exp = s.lower().split("e")
        return Q(base) * (Fraction(10, 1) ** int(exp))
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    if "." not in s:
        ans = Fraction(int(s), 1)
    else:
        a, b = s.split(".")
        ans = Fraction(int((a or "0") + b), 10 ** len(b))
    return -ans if neg else ans


def dec(x: Fraction, digits: int = 30) -> str:
    """Decimal display; not used for proof comparisons."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    n, d = x.numerator, x.denominator
    whole = n // d
    rem = n % d
    if digits <= 0:
        return sign + str(whole)
    out = []
    for _ in range(digits):
        rem *= 10
        out.append(str(rem // d))
        rem %= d
    return sign + str(whole) + "." + "".join(out)

# -------------------------- rigorous ln intervals -------------------------

LN_TERMS = 30

@dataclass(frozen=True)
class I:
    lo: Fraction
    hi: Fraction

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError("bad interval")

    def __add__(self, other: "I") -> "I":
        return I(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: "I") -> "I":
        return I(self.lo - other.hi, self.hi - other.lo)

    def mul_pos(self, c: Fraction | int) -> "I":
        c = Fraction(c, 1)
        if c >= 0:
            return I(c * self.lo, c * self.hi)
        return I(c * self.hi, c * self.lo)

    def div_pos(self, other: "I") -> "I":
        assert other.lo > 0
        return I(self.lo / other.hi, self.hi / other.lo)

    def recip_pos(self) -> "I":
        assert self.lo > 0
        return I(Fraction(1, 1) / self.hi, Fraction(1, 1) / self.lo)


def add(*xs: I) -> I:
    ans = I(Fraction(0), Fraction(0))
    for x in xs:
        ans = ans + x
    return ans


def const(x: Fraction | str | int) -> I:
    if not isinstance(x, Fraction):
        x = Q(x)
    return I(x, x)


def floor_log2_q(x: Fraction) -> int:
    assert x > 0
    # Initial estimate from bit lengths.
    m = x.numerator.bit_length() - x.denominator.bit_length()
    def pow2(k: int) -> Fraction:
        if k >= 0:
            return Fraction(1 << k, 1)
        return Fraction(1, 1 << (-k))
    while pow2(m + 1) <= x:
        m += 1
    while pow2(m) > x:
        m -= 1
    return m


def ln_1_to_2_interval(y: Fraction, terms: int = LN_TERMS) -> I:
    """Rigorous interval for ln(y), where 1 <= y < 2."""
    assert Fraction(1) <= y < Fraction(2)
    if y == 1:
        return I(Fraction(0), Fraction(0))
    z = (y - 1) / (y + 1)  # 0 <= z < 1/3
    z2 = z * z
    term = z
    s = Fraction(0)
    for n in range(terms):
        s += term / (2 * n + 1)
        term *= z2
    # tail <= z^(2N+1)/(2N+1)/(1-z^2)
    tail = term / (2 * terms + 1) / (1 - z2)
    return I(2 * s, 2 * (s + tail))


def ln2_interval(terms: int = LN_TERMS) -> I:
    z = Fraction(1, 3)
    z2 = z * z
    term = z
    s = Fraction(0)
    for n in range(terms):
        s += term / (2 * n + 1)
        term *= z2
    tail = term / (2 * terms + 1) / (1 - z2)
    return I(2 * s, 2 * (s + tail))

_LN2 = ln2_interval()

def ln_q(x: Fraction, terms: int = LN_TERMS) -> I:
    """Rigorous interval for ln(x), x>0."""
    assert x > 0
    if x == 1:
        return I(Fraction(0), Fraction(0))
    if x < 1:
        y = ln_q(1 / x, terms)
        return I(-y.hi, -y.lo)
    m = floor_log2_q(x)
    if m >= 0:
        y = x / Fraction(1 << m, 1)
    else:
        y = x * Fraction(1 << (-m), 1)
    # Now 1 <= y < 2.
    ly = ln_1_to_2_interval(y, terms)
    return _LN2.mul_pos(m) + ly


def ln_interval(x: I) -> I:
    assert x.lo > 0
    lo = ln_q(x.lo).lo
    hi = ln_q(x.hi).hi
    return I(lo, hi)


def sqr_interval_pos(x: I) -> I:
    assert x.lo >= 0
    return I(x.lo * x.lo, x.hi * x.hi)


def cube_interval_pos(x: I) -> I:
    assert x.lo >= 0
    return I(x.lo ** 3, x.hi ** 3)


def eps_from_log_interval(L: I) -> I:
    """epsilon(y) as an interval, supplied L=ln(y)>0."""
    assert L.lo > 0
    # eps(L)=1/(10 L^2)+4/(15 L^3), decreasing for L>0.
    lo = Fraction(1, 10 * L.hi * L.hi) + Fraction(4, 15) / (L.hi ** 3)
    hi = Fraction(1, 10 * L.lo * L.lo) + Fraction(4, 15) / (L.lo ** 3)
    return I(lo, hi)


def ln10_interval() -> I:
    return ln_q(Fraction(10, 1))

_LN10 = ln10_interval()

def log_of_10_power(N: int) -> I:
    return _LN10.mul_pos(N)


def loglog_10_power(N: int) -> I:
    # log(log(10^N)) = log(N) + log(log(10)); this is much faster
    # and avoids taking log of a very large rational interval.
    return ln_q(Fraction(N, 1)) + ln_interval(_LN10)

# ----------------------------- prime utilities ----------------------------

def sieve(n: int) -> List[int]:
    if n < 2:
        return []
    bs = bytearray(b"\x01") * (n + 1)
    bs[0:2] = b"\x00\x00"
    for p in range(2, isqrt(n) + 1):
        if bs[p]:
            start = p * p
            bs[start:n + 1:p] = b"\x00" * (((n - start) // p) + 1)
    return [i for i in range(n + 1) if bs[i]]

PRIMES_2M = sieve(2_000_000)

def sum_w_le(y: int) -> Fraction:
    return sum((Fraction(1, p - 1) for p in PRIMES_2M if p <= y), Fraction(0))

def sum_w_lt(y: int) -> Fraction:
    return sum((Fraction(1, p - 1) for p in PRIMES_2M if p < y), Fraction(0))

def W(N: int) -> Fraction:
    return sum((Fraction(1, PRIMES_2M[j] - 1) for j in range(N)), Fraction(0))

# ----------------------------- assertion tools ----------------------------

pass_count = 0

def check(cond: bool, msg: str) -> None:
    global pass_count
    if not cond:
        raise AssertionError("FAILED: " + msg)
    pass_count += 1
    print("PASS:", msg)


def show_interval(name: str, x: I, digits: int = 25) -> None:
    print(f"  {name} in [{dec(x.lo, digits)}, {dec(x.hi, digits)}]")

# ----------------------------- exact R values -----------------------------

def R_exact_before_prime(r: int, a: int) -> Fraction:
    weights = [Fraction(1, p - 1) for p in PRIMES_2M if p < a]
    E = [Fraction(0) for _ in range(r + 1)]
    E[0] = Fraction(1)
    for w in weights:
        for m in range(min(r, len(weights)), 0, -1):
            E[m] += w * E[m - 1]
    assert E[r] > 0
    return E[r - 1] / E[r]

# ------------------------------- verifiers --------------------------------

def verify_small_table() -> None:
    print("\n[1] Small table 4 <= k <= 20: exact rational R_r checks")
    rows = [
        (3, 13, 17, Q("3.506"), 17, 19, Q("3.048")),
        (4, 23, 29, Q("4.759"), 29, 31, Q("4.371")),
        (5, 31, 37, Q("6.748"), 37, 41, Q("6.263")),
        (6, 73, 79, Q("6.437"), 79, 83, Q("6.282")),
        (7, 89, 97, Q("8.085"), 97, 101, Q("7.911")),
        (8, 113, 127, Q("9.303"), 127, 131, Q("9.145")),
        (9, 113, 127, Q("11.677"), 127, 131, Q("11.452")),
        (10, 113, 127, Q("14.414"), 127, 131, Q("14.101")),
        (11, 293, 307, Q("12.085"), 307, 311, Q("12.011")),
        (12, 293, 307, Q("14.050"), 307, 311, Q("13.959")),
        (13, 523, 541, Q("13.651"), 541, 547, Q("13.607")),
        (14, 523, 541, Q("15.415"), 541, 547, Q("15.364")),
        (15, 523, 541, Q("17.279"), 541, 547, Q("17.218")),
        (16, 887, 907, Q("16.752"), 907, 911, Q("16.721")),
        (17, 887, 907, Q("18.438"), 907, 911, Q("18.402")),
        (18, 887, 907, Q("20.191"), 907, 911, Q("20.151")),
        (19, 1129, 1151, Q("20.742"), 1151, 1153, Q("20.711")),
    ]
    for r, a, b, up, c, d, low in rows:
        R1 = R_exact_before_prime(r, a)
        R2 = R_exact_before_prime(r, c)
        g1, g2 = b - a, d - c
        check(R1 < up < Q(str(g1 + 1)), f"r={r}: R_r({a}^-)={dec(R1,12)} < displayed upper < g+1")
        check(R2 > low > Q(str(g2 + 1)), f"r={r}: R_r({c}^-)={dec(R2,12)} > displayed lower > g+1")


def verify_C_interval() -> None:
    print("\n[2] C interval by directed Decimal partial sum plus exact tail")
    # Exact Fraction summation over ~150k primes is unnecessarily slow.
    # Here every division and addition is done with directed rounding at 80 digits.
    # Since all terms are positive, this gives a rigorous lower/upper enclosure.
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.rounding = ROUND_FLOOR
        lower = Decimal(0)
        for p in PRIMES_2M:
            if p > 1_999_993:
                break
            lower = ctx.add(lower, ctx.divide(Decimal(1), Decimal(p * (p - 1))))
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.rounding = ROUND_CEILING
        upper_partial = Decimal(0)
        for p in PRIMES_2M:
            if p > 1_999_993:
                break
            upper_partial = ctx.add(upper_partial, ctx.divide(Decimal(1), Decimal(p * (p - 1))))
        upper = ctx.add(upper_partial, ctx.divide(Decimal(1), Decimal(1_999_993)))
    check(lower > Decimal("0.773156636699192"), "C lower: partial sum > 0.773156636699192")
    check(upper < Decimal("0.773157136700943"), "C upper: partial + 1/1999993 < 0.773157136700943")
    print("  partial lower =", lower)
    print("  upper         =", upper)


def verify_medium_ranges() -> None:
    print("\n[3] Medium finite ranges 21 <= k <= 48: exact finite sums")
    A_lt_15683 = sum_w_lt(15683)
    A_le_15683 = sum_w_le(15683)
    W29 = W(29)
    print("  sum_{p<15683}  =", dec(A_lt_15683, 30))
    print("  sum_{p<=15683} =", dec(A_le_15683, 30))
    print("  W_29           =", dec(W29, 30))
    A1_low = Q("3.303755162423773")
    A1_up = Q("3.303818929800384")
    W29_up = Q("2.612642166507777")
    check(A_lt_15683 > A1_low, "sum_{p<15683} > 3.303755162423773")
    check(A_le_15683 < A1_up, "sum_{p<=15683} < 3.303818929800384")
    check(W29 < W29_up, "W_29 < 2.612642166507777")
    check(Fraction(30, 1) / (A1_low - W29_up) < Q("43.409"), "30/(lower sum - upper W29) < 43.409 < 45")
    check(Fraction(20, 1) / A1_up > Q("6.053"), "20/(upper sum<=15683) > 6.053 > 5")

    A_lt_31397 = sum_w_lt(31397)
    A_le_31397 = sum_w_le(31397)
    W46 = W(46)
    print("  sum_{p<31397}  =", dec(A_lt_31397, 30))
    print("  sum_{p<=31397} =", dec(A_le_31397, 30))
    print("  W_46           =", dec(W46, 30))
    A2_low = Q("3.372584257226677")
    A2_up = Q("3.372616108417913")
    W46_up = Q("2.721441010945543")
    check(A_lt_31397 > A2_low, "sum_{p<31397} > 3.372584257226677")
    check(A_le_31397 < A2_up, "sum_{p<=31397} < 3.372616108417913")
    check(W46 < W46_up, "W_46 < 2.721441010945543")
    check(Fraction(47, 1) / (A2_low - W46_up) < Q("72.181"), "47/(lower sum - upper W46) < 72.181 < 73")
    check(Fraction(31, 1) / A2_up > Q("9.191"), "31/(upper sum<=31397) > 9.191 > 9")


def verify_finite_large() -> None:
    print("\n[4] Record gap + twin prime finite-large numerical checks")
    Bminus = Q("0.261497212847642")
    Bplus = Q("0.261497212847643")
    Cminus = Q("0.773156636699192")
    Cplus = Q("0.773157136700943")

    # s_L digit checks from the formula, not primality/gap proof.
    primes_43103 = [p for p in PRIMES_2M if p <= 43103]
    primorial_43103 = 1
    for p in primes_43103:
        primorial_43103 *= p
    sL = 587 * (primorial_43103 // 2310) - 455_704
    gL = 1_113_106
    check(len(str(sL)) == 18_662, "s_L formula has 18662 decimal digits")
    check(len(str(sL + gL)) == 18_662, "s_L+g_L has 18662 decimal digits")

    # s_T digit check from the formula, not primality proof.
    sT = pow(504_983_334, 8192) - pow(504_983_334, 4096) - 1
    check(len(str(sT)) == 71_298, "s_T formula has 71298 decimal digits")
    check(sT > sL + gL, "s_T occurs after the large-gap endpoints by digit/formula comparison")

    # A(s_T) upper bound. Since s_T has 71298 digits,
    # 10^71297 <= s_T < 10^71298.  Use 10^71298 for loglog,
    # but 10^71297 for epsilon because epsilon is decreasing.
    N_T = 71_298
    LT_eps = log_of_10_power(N_T - 1)
    epsT = eps_from_log_interval(LT_eps)
    A_sT_up = loglog_10_power(N_T).hi + Bplus + epsT.hi + Cplus
    print("  A(s_T) analytic upper <=", dec(A_sT_up, 30))
    check(A_sT_up < Q("13.04331036"), "A(s_T) < 13.04331036")
    check(Fraction(40, 1) / Q("13.04331036") > Q("3.066"), "40/13.04331036 > 3.066 > 3")

    # A(s_L^-) lower bound; tail term -1/(10^N_L - 1).
    N_L = 18_661
    LL = log_of_10_power(N_L)
    epsL = eps_from_log_interval(LL)
    A_sL_lower = loglog_10_power(N_L).lo + Bminus - epsL.hi + Cminus - Fraction(1, 10 ** N_L - 1)
    print("  A(s_L^-) lower >=", dec(A_sL_lower, 30))
    check(A_sL_lower > Q("11.70287735"), "A(s_L^-) > 11.70287735")

    # Dusart p_n upper for the s_T definability statement, at n=8,600,000.
    n_twin_def = 8_600_000
    ln_n = ln_q(Fraction(n_twin_def, 1))
    lnln_n = ln_interval(ln_n)
    numerator = lnln_n - const(2)
    frac_term = numerator.div_pos(ln_n)
    U_interval = (ln_n + lnln_n - const(1) + frac_term).mul_pos(n_twin_def)
    print("  U(8,600,000) upper <=", dec(U_interval.hi, 30))
    check(U_interval.hi < Q("152960215"), "U(8,600,000) < 152,960,215")

    # Dusart p_n upper for W_{8,599,999}, used in the descent estimate.
    n = 8_599_999
    ln_n = ln_q(Fraction(n, 1))
    lnln_n = ln_interval(ln_n)
    numerator = lnln_n - const(2)
    frac_term = numerator.div_pos(ln_n)
    U_interval = (ln_n + lnln_n - const(1) + frac_term).mul_pos(n)
    print("  U(8,599,999) upper <=", dec(U_interval.hi, 30))
    check(U_interval.hi < Q("152960196"), "U(8,599,999) < 152,960,196")

    Y = 152_960_196
    LY = ln_q(Fraction(Y, 1))
    epsY = eps_from_log_interval(LY)
    A_U_up = ln_interval(LY).hi + Bplus + epsY.hi + Cplus
    print("  A(152960196) upper <=", dec(A_U_up, 30))
    check(A_U_up < Q("3.9713"), "A(U) <= A(152960196) < 3.9713")
    check(Fraction(8_600_000, 1) / (Q("11.70287735") - Q("3.9713")) < Q("1112322"), "8600000/(11.70287735-3.9713) < 1,112,322")
    check(Q("1112322") < Q("1113107"), "1,112,322 < 1,113,107 = g_L+1")


def verify_tail_numerics() -> None:
    print("\n[5] Uniform tail k >= 8,600,002: numerical inequalities")
    R0 = 8_600_001
    v0 = ln_q(Fraction(R0, 1))
    check(v0.lo > Q("15.96"), "log(8,600,001) > 15.96")
    x0_lower = Fraction(99, 100) * Fraction(R0, 1) / v0.hi
    print("  x0 lower >=", dec(x0_lower, 30))
    check(x0_lower > Q("533000"), "x >= 0.99*8600001/log(8600001) > 533000")
    check(x0_lower / 2 > Q("266500"), "x/2 > 266500 > 3275")

    # Worst cases use the lower endpoints x=533000 and x/2=266500.
    L533 = ln_q(Q("533000"))
    L266 = ln_q(Q("266500"))
    inv2log2_533_hi = Fraction(1, 2) / (L533.lo ** 2)
    inv2log2_266_hi = Fraction(1, 2) / (L266.lo ** 2)
    check(inv2log2_533_hi < Q("0.002876"), "1/(2 log^2 x) < 0.002876 for x>533000")
    check(inv2log2_266_hi < Q("0.00321"), "1/(2 log^2(x/2)) < 0.00321 for x/2>266500")
    check(Q("1.2323") / L533.lo < Q("0.1"), "1.2323/log x < 0.1 for x>533000")

    # (1+...)*(1+1/36260)+log8/x < 1.003 at x=533000.
    ln8 = ln_q(Fraction(8, 1))
    expr7_hi = (1 + inv2log2_533_hi) * (1 + Fraction(1, 36260)) + ln8.hi / Q("533000")
    check(expr7_hi < Q("1.003"), "((1+1/(2log^2x))*(1+1/36260)+log8/x) < 1.003")

    # eps if log y > 18.9.
    eps_18_9_hi = Fraction(1, 10) / (Q("18.9") ** 2) + Fraction(4, 15) / (Q("18.9") ** 3)
    check(eps_18_9_hi < Q("0.001"), "if log y > 18.9 then epsilon(y)<0.001")

    # Elementary inequalities (9), (10).
    ln_15_96 = ln_q(Q("15.96"))
    f9_lo = Q("0.44") * Q("15.96") - 2 * ln_15_96.hi - Q("1.300")
    check(Q("0.44") - Fraction(2, 1) / Q("15.96") > 0, "derivative 0.44-2/v is positive for v>=15.96")
    check(f9_lo > 0, "0.44v-2logv-1.300 > 0 at v=15.96, hence for all v>=15.96")
    check(Fraction(1, 1) / Q("0.99") > Q("1.010"), "v/(0.99(v-1.50)) > 1/0.99 > 1.010")

    # q^- and gap constants.
    check(Fraction(1, 2) * (1 + Q("0.00321")) < 1, "D0 at x/2 gives a prime below x")
    check(Fraction(2, 1) / Q("1.00321") > Q("1.993"), "2/1.00321 > 1.993")

    # Small gap D(M) constants.
    ln2 = _LN2
    check(Q("9") - 10 * ln2.hi > Q("2"), "9 - 10 log 2 > 2")
    check(10 * ln2.hi + Q("11") < Q("18"), "10 log 2 + 11 < 18")
    check(Q("20") > 3 * ln_q(Q("20")).hi, "e^20/20^3 > 1, certified as 20 > 3 log 20")

    # Inequality for primes in (P,2P].
    a = ln2.hi
    # numerator of 2/(L+ln2-1)-1/(L-1.1)-1/(2L) with positive denom is
    # L^2 -(3 ln2 + 0.3)L + 1.1(ln2-1). It is increasing for L>=20.
    num20_lo = Q("20") ** 2 - (3 * a + Q("0.3")) * Q("20") + Q("1.1") * (ln2.lo - 1)
    deriv20_lo = 2 * Q("20") - (3 * a + Q("0.3"))
    check(deriv20_lo > 0 and num20_lo > 0, "for L>20: 2/(L+log2-1)-1/(L-1.1) > 1/(2L)")
    check(Q("20") > ln_q(Q("80")).hi, "e^20/40 > 2, certified as 20 > log 80")

    # h(t) checks.
    ln_15_96 = ln_q(Q("15.96"))
    h15_lo = Q("0.2") * Q("15.96") - ln_15_96.hi + 1 - (ln_15_96.hi - 2) / Q("15.96")
    hprime_lower = Q("0.2") - Fraction(1, 1) / Q("15.96") - Q("0.230") / (Q("15.96") ** 2)
    check(ln_15_96.lo - 3 > -Q("0.230"), "log(15.96)-3 > -0.230")
    check(hprime_lower > 0, "h'(t)>0 for t>=15.96 using the proof's lower bound")
    check(h15_lo > Q("1.37"), "h(15.96)>1.37")

    # log(1.2r log r)>18.9 at r0 and log P>0.9x>18.9.
    log_1p2_r_logr_lo = ln_q(Q("1.2")).lo + v0.lo + ln_interval(v0).lo
    check(log_1p2_r_logr_lo > Q("18.9"), "log(1.2 r log r)>18.9 at r=8,600,001, hence after")
    check(Q("0.9") * Q("533000") > Q("18.9"), "0.9x>18.9 from x>533000")

    # log(1.2v)<0.2v at v=15.96 and increasing gap.
    gap_1p2v_lo = Q("0.2") * Q("15.96") - ln_q(Q("1.2") * Q("15.96")).hi
    deriv_gap_lo = Q("0.2") - Fraction(1, 1) / Q("15.96")
    check(deriv_gap_lo > 0 and gap_1p2v_lo > Q("0.23"), "log(1.2v)<0.2v for v>=15.96")

    const_descent = ln_q(Q("0.9") * Q("0.99")).lo - ln_q(Q("1.2")).hi - Q("1.002")
    check(const_descent > -Q("1.300"), "log(0.9*0.99)-log1.2-1.002 > -1.300")
    check(Fraction(1, 1) / (Q("0.99") * Q("0.56")) < Q("1.805"), "1/(0.99*0.56)<1.805")
    check(Q("1.805") < Q("1.993"), "descent constants: 1.805 < 1.993")

    # Ascent constants.
    const_A8P = ln_q(Q("1.003")).hi + Q("0.262") + Q("1.001")
    check(const_A8P < Q("1.266"), "log(1.003)+0.262+1.001 < 1.266")
    ascent_v15_hi = -ln_15_96.lo + ln_q(Q("0.99")).hi + Q("1.266")
    check(ascent_v15_hi < -Q("1.50"), "-log v+log0.99+1.266 < -1.50 for v>=15.96")
    check(Q("0.001") * Q("533000") > 1, "x>533000 implies 1 < 0.001x")
    check(Q("1.004") < Q("1.010"), "ascent constants: 1.004 < 1.010")


def main() -> None:
    print("Erdos 690 numerical verifier")
    print("All assertions use exact rational arithmetic or directed-rounded interval arithmetic; no binary floating point is used.")
    verify_small_table()
    verify_C_interval()
    verify_medium_ranges()
    verify_finite_large()
    verify_tail_numerics()
    print(f"\nALL CHECKS PASSED ({pass_count} assertions).")

if __name__ == "__main__":
    main()
