# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

from __future__ import annotations

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange, threadid
from libc.math cimport exp, log, log1p, isfinite, sqrt, fabs
from libc.stdlib cimport malloc, free
from scipy.special.cython_special cimport gammaln, psi

cdef double NEG_INF = float("-inf")
cdef double EPSILON = 1.0e-4
cdef double NAN_VAL = float("nan")

cdef extern from *:
    """
    #ifdef _OPENMP
    #include <omp.h>
    static int pydeseq2_openmp_enabled(void) { return 1; }
    static int pydeseq2_omp_max_threads(void) { return omp_get_max_threads(); }
    #else
    static int pydeseq2_openmp_enabled(void) { return 0; }
    static int pydeseq2_omp_max_threads(void) { return 1; }
    #endif
    """
    int pydeseq2_openmp_enabled()
    int pydeseq2_omp_max_threads()


cdef inline double _log_posterior_no_cr(
    double log_alpha,
    const double[:, ::1] y,
    const double[:, ::1] mu,
    const double[:, ::1] weights,
    Py_ssize_t row,
    double log_alpha_prior_mean,
    double log_alpha_prior_sigmasq,
    bint use_prior,
    bint use_weights,
) noexcept nogil:
    cdef Py_ssize_t j
    cdef Py_ssize_t n_samples = y.shape[1]
    cdef double alpha = exp(log_alpha)
    cdef double alpha_neg1
    cdef double ll_part = 0.0
    cdef double yv
    cdef double muv
    cdef double ll
    cdef double prior_part = 0.0
    if alpha <= 0.0 or not isfinite(alpha):
        return NEG_INF

    alpha_neg1 = 1.0 / alpha
    for j in range(n_samples):
        yv = y[row, j]
        muv = mu[row, j]
        ll = (
            gammaln(yv + alpha_neg1)
            - gammaln(alpha_neg1)
            - yv * log(muv + alpha_neg1)
            - alpha_neg1 * log1p(muv * alpha)
        )
        if use_weights:
            ll_part += weights[row, j] * ll
        else:
            ll_part += ll

    if use_prior:
        prior_part = -0.5 * ((log_alpha - log_alpha_prior_mean) ** 2) / log_alpha_prior_sigmasq

    return ll_part + prior_part


cdef inline void _log_and_dlog_posterior_no_cr(
    double log_alpha,
    const double[:, ::1] y,
    const double[:, ::1] mu,
    const double[:, ::1] weights,
    Py_ssize_t row,
    double log_alpha_prior_mean,
    double log_alpha_prior_sigmasq,
    bint use_prior,
    bint use_weights,
    double* lp_out,
    double* dlp_out,
) noexcept nogil:
    cdef Py_ssize_t j
    cdef Py_ssize_t n_samples = y.shape[1]
    cdef double alpha = exp(log_alpha)
    cdef double alpha_neg1
    cdef double alpha_neg2
    cdef double yv
    cdef double muv
    cdef double ll
    cdef double inner
    cdef double ll_part = 0.0
    cdef double dlp_part = 0.0
    cdef double prior_lp = 0.0
    cdef double prior_dlp = 0.0
    cdef double wv
    if alpha <= 0.0 or not isfinite(alpha):
        lp_out[0] = NEG_INF
        dlp_out[0] = 0.0
        return

    alpha_neg1 = 1.0 / alpha
    alpha_neg2 = alpha_neg1 * alpha_neg1
    for j in range(n_samples):
        yv = y[row, j]
        muv = mu[row, j]
        ll = (
            gammaln(yv + alpha_neg1)
            - gammaln(alpha_neg1)
            - yv * log(muv + alpha_neg1)
            - alpha_neg1 * log1p(muv * alpha)
        )
        inner = (
            psi(alpha_neg1)
            + log1p(muv * alpha)
            - (muv * alpha) / (1.0 + muv * alpha)
            - psi(yv + alpha_neg1)
            + yv / (muv + alpha_neg1)
        )
        if use_weights:
            wv = weights[row, j]
            ll_part += wv * ll
            dlp_part += wv * inner
        else:
            ll_part += ll
            dlp_part += inner

    if use_prior:
        prior_lp = -0.5 * ((log_alpha - log_alpha_prior_mean) ** 2) / log_alpha_prior_sigmasq
        prior_dlp = -(log_alpha - log_alpha_prior_mean) / log_alpha_prior_sigmasq

    dlp_part = dlp_part * alpha_neg2
    lp_out[0] = ll_part + prior_lp
    dlp_out[0] = dlp_part * alpha + prior_dlp


cdef inline void _fit_row_no_cr(
    Py_ssize_t row,
    const double[:, ::1] y,
    const double[:, ::1] mu_hat,
    const double[:, ::1] weights,
    const double[::1] log_alpha_prior_mean,
    double log_alpha_prior_sigmasq,
    double min_log_alpha,
    double kappa_0,
    double tol,
    int maxit,
    bint use_prior,
    bint use_weights,
    double[::1] log_alpha_out,
    int[::1] iterations,
    int[::1] iter_accept,
    double[::1] last_change,
    double[::1] initial_lp,
    double[::1] last_lp,
) noexcept nogil:
    cdef double a = log_alpha_out[row]
    cdef double lp
    cdef double dlp
    _log_and_dlog_posterior_no_cr(
        a,
        y,
        mu_hat,
        weights,
        row,
        log_alpha_prior_mean[row],
        log_alpha_prior_sigmasq,
        use_prior,
        use_weights,
        &lp,
        &dlp,
    )
    cdef double kappa = kappa_0
    cdef double change = -1.0
    cdef double a_propose
    cdef double theta_kappa
    cdef double theta_hat_kappa
    cdef double lp_new
    cdef int local_iterations = 0
    cdef int local_accept = 0
    cdef int it

    initial_lp[row] = lp

    for it in range(maxit):
        local_iterations += 1
        if dlp == 0.0 or not isfinite(dlp):
            break

        a_propose = a + kappa * dlp
        if a_propose < -30.0:
            kappa = (-30.0 - a) / dlp
        elif a_propose > 10.0:
            kappa = (10.0 - a) / dlp

        theta_kappa = -_log_posterior_no_cr(
            a + kappa * dlp,
            y,
            mu_hat,
            weights,
            row,
            log_alpha_prior_mean[row],
            log_alpha_prior_sigmasq,
            use_prior,
            use_weights,
        )
        theta_hat_kappa = -lp - kappa * EPSILON * (dlp * dlp)
        if theta_kappa <= theta_hat_kappa:
            local_accept += 1
            a = a + kappa * dlp
            lp_new = _log_posterior_no_cr(
                a,
                y,
                mu_hat,
                weights,
                row,
                log_alpha_prior_mean[row],
                log_alpha_prior_sigmasq,
                use_prior,
                use_weights,
            )
            change = lp_new - lp
            if change < tol:
                lp = lp_new
                break
            if a < min_log_alpha:
                break
            lp = lp_new
            _log_and_dlog_posterior_no_cr(
                a,
                y,
                mu_hat,
                weights,
                row,
                log_alpha_prior_mean[row],
                log_alpha_prior_sigmasq,
                use_prior,
                use_weights,
                &lp,
                &dlp,
            )
            kappa = kappa * 1.1
            if kappa > kappa_0:
                kappa = kappa_0
            if local_accept % 5 == 0:
                kappa = kappa / 2.0
        else:
            kappa = kappa / 2.0
            if kappa < 1e-12:
                break

    last_lp[row] = lp
    last_change[row] = change
    log_alpha_out[row] = a
    iterations[row] = local_iterations
    iter_accept[row] = local_accept


cdef inline int _cholesky_logdet(double* a, double* l, int p, double* logdet_out) noexcept nogil:
    cdef int i
    cdef int j
    cdef int k
    cdef double sumv
    cdef double diag
    cdef double logdet = 0.0

    for i in range(p):
        for j in range(p):
            l[i * p + j] = 0.0

    for i in range(p):
        for j in range(i + 1):
            sumv = a[i * p + j]
            for k in range(j):
                sumv -= l[i * p + k] * l[j * p + k]

            if i == j:
                if (not isfinite(sumv)) or (sumv <= 1e-12):
                    return 0
                diag = sqrt(sumv)
                if (not isfinite(diag)) or (diag <= 0.0):
                    return 0
                l[i * p + j] = diag
                logdet += 2.0 * log(diag)
            else:
                diag = l[j * p + j]
                if (not isfinite(diag)) or (diag <= 0.0):
                    return 0
                l[i * p + j] = sumv / diag
                if not isfinite(l[i * p + j]):
                    return 0

    if not isfinite(logdet):
        return 0
    logdet_out[0] = logdet
    return 1


cdef inline void _cox_reid_terms(
    double alpha,
    const double[:, ::1] mu,
    const double[:, ::1] x,
    const double[:, ::1] weights,
    Py_ssize_t row,
    bint use_weights,
    double weight_threshold,
    int* active,
    int* active_idx,
    double* b_mat,
    double* db_mat,
    double* chol_mat,
    double* inv_mat,
    double* work_mat,
    double* cr_lp_out,
    double* cr_dlp_out,
) noexcept nogil:
    cdef int n_samples = x.shape[0]
    cdef int p = x.shape[1]
    cdef int j
    cdef int a
    cdef int b
    cdef int ia
    cdef int ib
    cdef int k = 0
    cdef int a_idx
    cdef int b_idx
    cdef bint keep_any = False
    cdef double muv
    cdef double denom
    cdef double w
    cdef double dw
    cdef double xa
    cdef double xb
    cdef double val
    cdef double tr = 0.0
    cdef double logdet = 0.0
    cdef int ok

    if use_weights:
        for a in range(p):
            active[a] = 0
        for j in range(n_samples):
            if weights[row, j] > weight_threshold:
                keep_any = True
                for a in range(p):
                    if fabs(x[j, a]) > 0.0:
                        active[a] = 1
        if not keep_any:
            cr_lp_out[0] = NEG_INF
            cr_dlp_out[0] = 0.0
            return
    else:
        for a in range(p):
            active[a] = 1

    for a in range(p):
        if active[a] != 0:
            active_idx[k] = a
            k += 1

    if k == 0:
        cr_lp_out[0] = 0.0
        cr_dlp_out[0] = 0.0
        return

    for a in range(k):
        for b in range(k):
            b_mat[a * k + b] = 0.0
            db_mat[a * k + b] = 0.0

    for j in range(n_samples):
        if use_weights and (weights[row, j] <= weight_threshold):
            continue
        muv = mu[row, j]
        denom = (1.0 / muv) + alpha
        w = 1.0 / denom
        dw = -1.0 / (denom * denom)
        for ia in range(k):
            a_idx = active_idx[ia]
            xa = x[j, a_idx]
            for ib in range(k):
                b_idx = active_idx[ib]
                xb = x[j, b_idx]
                val = xa * xb
                b_mat[ia * k + ib] += val * w
                db_mat[ia * k + ib] += val * dw

    ok = _cholesky_logdet(b_mat, chol_mat, k, &logdet)
    if ok == 0:
        cr_lp_out[0] = NEG_INF
    else:
        cr_lp_out[0] = -0.5 * logdet

    ok = _invert_matrix(b_mat, inv_mat, work_mat, k)
    if ok == 0:
        cr_dlp_out[0] = 0.0
    else:
        tr = 0.0
        for ia in range(k):
            for ib in range(k):
                tr += inv_mat[ia * k + ib] * db_mat[ib * k + ia]
        cr_dlp_out[0] = -0.5 * tr


cdef inline void _log_and_dlog_posterior_with_cr(
    double log_alpha,
    const double[:, ::1] y,
    const double[:, ::1] mu,
    const double[:, ::1] x,
    const double[:, ::1] weights,
    Py_ssize_t row,
    double log_alpha_prior_mean,
    double log_alpha_prior_sigmasq,
    bint use_prior,
    bint use_weights,
    double weight_threshold,
    int* active,
    int* active_idx,
    double* b_mat,
    double* db_mat,
    double* chol_mat,
    double* inv_mat,
    double* work_mat,
    double* lp_out,
    double* dlp_out,
) noexcept nogil:
    cdef Py_ssize_t j
    cdef Py_ssize_t n_samples = y.shape[1]
    cdef double alpha = exp(log_alpha)
    cdef double alpha_neg1
    cdef double alpha_neg2
    cdef double yv
    cdef double muv
    cdef double ll
    cdef double inner
    cdef double ll_part = 0.0
    cdef double dlp_part = 0.0
    cdef double prior_lp = 0.0
    cdef double prior_dlp = 0.0
    cdef double wv
    cdef double cr_lp = 0.0
    cdef double cr_dlp = 0.0
    cdef bint keep_any = False

    if alpha <= 0.0 or not isfinite(alpha):
        lp_out[0] = NEG_INF
        dlp_out[0] = 0.0
        return

    if use_weights:
        for j in range(n_samples):
            if weights[row, j] > weight_threshold:
                keep_any = True
                break
        if not keep_any:
            lp_out[0] = NEG_INF
            dlp_out[0] = 0.0
            return

    alpha_neg1 = 1.0 / alpha
    alpha_neg2 = alpha_neg1 * alpha_neg1
    for j in range(n_samples):
        yv = y[row, j]
        muv = mu[row, j]
        ll = (
            gammaln(yv + alpha_neg1)
            - gammaln(alpha_neg1)
            - yv * log(muv + alpha_neg1)
            - alpha_neg1 * log1p(muv * alpha)
        )
        inner = (
            psi(alpha_neg1)
            + log1p(muv * alpha)
            - (muv * alpha) / (1.0 + muv * alpha)
            - psi(yv + alpha_neg1)
            + yv / (muv + alpha_neg1)
        )
        if use_weights:
            wv = weights[row, j]
            ll_part += wv * ll
            dlp_part += wv * inner
        else:
            ll_part += ll
            dlp_part += inner

    if use_prior:
        prior_lp = -0.5 * ((log_alpha - log_alpha_prior_mean) ** 2) / log_alpha_prior_sigmasq
        prior_dlp = -(log_alpha - log_alpha_prior_mean) / log_alpha_prior_sigmasq

    _cox_reid_terms(
        alpha,
        mu,
        x,
        weights,
        row,
        use_weights,
        weight_threshold,
        active,
        active_idx,
        b_mat,
        db_mat,
        chol_mat,
        inv_mat,
        work_mat,
        &cr_lp,
        &cr_dlp,
    )

    dlp_part = dlp_part * alpha_neg2
    lp_out[0] = ll_part + prior_lp + cr_lp
    dlp_out[0] = (dlp_part + cr_dlp) * alpha + prior_dlp


cdef inline void _fit_row_with_cr(
    Py_ssize_t row,
    const double[:, ::1] y,
    const double[:, ::1] x,
    const double[:, ::1] mu_hat,
    const double[:, ::1] weights,
    const double[::1] log_alpha_prior_mean,
    double log_alpha_prior_sigmasq,
    double min_log_alpha,
    double kappa_0,
    double tol,
    int maxit,
    bint use_prior,
    bint use_weights,
    double weight_threshold,
    double[::1] log_alpha_out,
    int[::1] iterations,
    int[::1] iter_accept,
    double[::1] last_change,
    double[::1] initial_lp,
    double[::1] last_lp,
) noexcept nogil:
    cdef int p = x.shape[1]
    cdef size_t vec_bytes = <size_t>p * sizeof(int)
    cdef size_t mat_bytes = <size_t>p * <size_t>p * sizeof(double)
    cdef int* active = <int*>malloc(vec_bytes)
    cdef int* active_idx = <int*>malloc(vec_bytes)
    cdef double* b_mat = <double*>malloc(mat_bytes)
    cdef double* db_mat = <double*>malloc(mat_bytes)
    cdef double* chol_mat = <double*>malloc(mat_bytes)
    cdef double* inv_mat = <double*>malloc(mat_bytes)
    cdef double* work_mat = <double*>malloc(mat_bytes)
    cdef double a = log_alpha_out[row]
    cdef double lp
    cdef double dlp
    cdef double kappa
    cdef double change = -1.0
    cdef double a_propose
    cdef double theta_kappa
    cdef double theta_hat_kappa
    cdef double lp_probe
    cdef double dlp_probe
    cdef double lp_new
    cdef double dlp_new
    cdef int local_iterations = 0
    cdef int local_accept = 0
    cdef int it

    if (
        (active == NULL)
        or (active_idx == NULL)
        or (b_mat == NULL)
        or (db_mat == NULL)
        or (chol_mat == NULL)
        or (inv_mat == NULL)
        or (work_mat == NULL)
    ):
        if active != NULL:
            free(active)
        if active_idx != NULL:
            free(active_idx)
        if b_mat != NULL:
            free(b_mat)
        if db_mat != NULL:
            free(db_mat)
        if chol_mat != NULL:
            free(chol_mat)
        if inv_mat != NULL:
            free(inv_mat)
        if work_mat != NULL:
            free(work_mat)
        initial_lp[row] = NEG_INF
        last_lp[row] = NEG_INF
        last_change[row] = -1.0
        iterations[row] = 0
        iter_accept[row] = 0
        return

    _log_and_dlog_posterior_with_cr(
        a,
        y,
        mu_hat,
        x,
        weights,
        row,
        log_alpha_prior_mean[row],
        log_alpha_prior_sigmasq,
        use_prior,
        use_weights,
        weight_threshold,
        active,
        active_idx,
        b_mat,
        db_mat,
        chol_mat,
        inv_mat,
        work_mat,
        &lp,
        &dlp,
    )
    kappa = kappa_0
    initial_lp[row] = lp

    for it in range(maxit):
        local_iterations += 1
        if dlp == 0.0 or not isfinite(dlp):
            break

        a_propose = a + kappa * dlp
        if a_propose < -30.0:
            kappa = (-30.0 - a) / dlp
        elif a_propose > 10.0:
            kappa = (10.0 - a) / dlp

        _log_and_dlog_posterior_with_cr(
            a + kappa * dlp,
            y,
            mu_hat,
            x,
            weights,
            row,
            log_alpha_prior_mean[row],
            log_alpha_prior_sigmasq,
            use_prior,
            use_weights,
            weight_threshold,
            active,
            active_idx,
            b_mat,
            db_mat,
            chol_mat,
            inv_mat,
            work_mat,
            &lp_probe,
            &dlp_probe,
        )
        theta_kappa = -lp_probe
        theta_hat_kappa = -lp - kappa * EPSILON * (dlp * dlp)
        if theta_kappa <= theta_hat_kappa:
            local_accept += 1
            a = a + kappa * dlp
            _log_and_dlog_posterior_with_cr(
                a,
                y,
                mu_hat,
                x,
                weights,
                row,
                log_alpha_prior_mean[row],
                log_alpha_prior_sigmasq,
                use_prior,
                use_weights,
                weight_threshold,
                active,
                active_idx,
                b_mat,
                db_mat,
                chol_mat,
                inv_mat,
                work_mat,
                &lp_new,
                &dlp_new,
            )
            change = lp_new - lp
            if change < tol:
                lp = lp_new
                break
            if a < min_log_alpha:
                break
            lp = lp_new
            dlp = dlp_new
            kappa = kappa * 1.1
            if kappa > kappa_0:
                kappa = kappa_0
            if local_accept % 5 == 0:
                kappa = kappa / 2.0
        else:
            kappa = kappa / 2.0
            if kappa < 1e-12:
                break

    last_lp[row] = lp
    last_change[row] = change
    log_alpha_out[row] = a
    iterations[row] = local_iterations
    iter_accept[row] = local_accept

    free(active)
    free(active_idx)
    free(b_mat)
    free(db_mat)
    free(chol_mat)
    free(inv_mat)
    free(work_mat)


cdef inline int _solve_linear_inplace(double* a, double* b, int p) noexcept nogil:
    cdef int i
    cdef int j
    cdef int k
    cdef int pivot_row
    cdef double pivot_abs
    cdef double cand
    cdef double pivot
    cdef double factor
    cdef double tmp
    cdef double val

    for k in range(p):
        pivot_row = k
        pivot_abs = fabs(a[k * p + k])
        for i in range(k + 1, p):
            cand = fabs(a[i * p + k])
            if cand > pivot_abs:
                pivot_abs = cand
                pivot_row = i
        if (not isfinite(pivot_abs)) or (pivot_abs < 1e-12):
            return 0

        if pivot_row != k:
            for j in range(p):
                tmp = a[k * p + j]
                a[k * p + j] = a[pivot_row * p + j]
                a[pivot_row * p + j] = tmp
            tmp = b[k]
            b[k] = b[pivot_row]
            b[pivot_row] = tmp

        pivot = a[k * p + k]
        for i in range(k + 1, p):
            factor = a[i * p + k] / pivot
            a[i * p + k] = 0.0
            if factor != 0.0:
                for j in range(k + 1, p):
                    a[i * p + j] -= factor * a[k * p + j]
                b[i] -= factor * b[k]

    i = p - 1
    while i >= 0:
        val = b[i]
        for j in range(i + 1, p):
            val -= a[i * p + j] * b[j]
        pivot = a[i * p + i]
        if (not isfinite(pivot)) or (fabs(pivot) < 1e-12):
            return 0
        b[i] = val / pivot
        i -= 1

    return 1


cdef inline int _invert_matrix(double* a, double* inv, double* work, int p) noexcept nogil:
    cdef int i
    cdef int j
    cdef int k
    cdef int r
    cdef int pivot_row
    cdef double pivot_abs
    cdef double cand
    cdef double pivot
    cdef double factor
    cdef double tmp

    for i in range(p):
        for j in range(p):
            work[i * p + j] = a[i * p + j]
            inv[i * p + j] = 1.0 if i == j else 0.0

    for i in range(p):
        pivot_row = i
        pivot_abs = fabs(work[i * p + i])
        for r in range(i + 1, p):
            cand = fabs(work[r * p + i])
            if cand > pivot_abs:
                pivot_abs = cand
                pivot_row = r
        if (not isfinite(pivot_abs)) or (pivot_abs < 1e-12):
            return 0

        if pivot_row != i:
            for j in range(p):
                tmp = work[i * p + j]
                work[i * p + j] = work[pivot_row * p + j]
                work[pivot_row * p + j] = tmp
                tmp = inv[i * p + j]
                inv[i * p + j] = inv[pivot_row * p + j]
                inv[pivot_row * p + j] = tmp

        pivot = work[i * p + i]
        for j in range(p):
            work[i * p + j] /= pivot
            inv[i * p + j] /= pivot

        for r in range(p):
            if r == i:
                continue
            factor = work[r * p + i]
            if factor != 0.0:
                for j in range(p):
                    work[r * p + j] -= factor * work[i * p + j]
                    inv[r * p + j] -= factor * inv[i * p + j]

    return 1


cdef inline void _fit_glm_row(
    Py_ssize_t row,
    const double[:, ::1] counts,
    const double[:, ::1] x,
    const double[:, ::1] nf,
    const double[:, ::1] wmat,
    const double[::1] alpha,
    const double[:, ::1] beta_nat_init,
    const double[::1] lambda_nat,
    int n_samples,
    int p,
    int maxit,
    double beta_tol,
    double minmu,
    double large,
    bint use_weights,
    bint mu_only,
    double[:, ::1] beta_nat,
    double[:, ::1] beta_var_nat,
    double[:, ::1] hat_diagonals,
    double[:, ::1] mu_out,
    int[::1] beta_iter,
    double* beta_hat,
    double* mu_hat,
    double* w_vec,
    double* a_mat,
    double* b_vec,
    double* xtwx,
    double* inv_a,
    double* work,
    double* temp,
    double* xw,
) noexcept nogil:
    cdef int t
    cdef int j
    cdef int a
    cdef int b
    cdef int k
    cdef double eta
    cdef double mu
    cdef double alpha_i
    cdef double size
    cdef double yv
    cdef double ll
    cdef double ll_sum
    cdef double dev_old = 0.0
    cdef double dev = 0.0
    cdef double conv_test
    cdef double w
    cdef double ww
    cdef double z
    cdef double ws
    cdef double tmp
    cdef double h
    cdef int ok

    alpha_i = alpha[row]
    if alpha_i <= 0.0 or (not isfinite(alpha_i)):
        beta_iter[row] = maxit
        if mu_only:
            for j in range(n_samples):
                # Match Python-side fallback behavior for invalid rows.
                mu_out[row, j] = nf[row, j]
        else:
            for a in range(p):
                beta_nat[row, a] = NAN_VAL
                beta_var_nat[row, a] = NAN_VAL
            for j in range(n_samples):
                hat_diagonals[row, j] = NAN_VAL
        return

    for a in range(p):
        beta_hat[a] = beta_nat_init[row, a]

    for j in range(n_samples):
        eta = 0.0
        for a in range(p):
            eta += x[j, a] * beta_hat[a]
        if not isfinite(eta):
            eta = 0.0
        if eta > 30.0:
            eta = 30.0
        elif eta < -30.0:
            eta = -30.0
        mu = nf[row, j] * exp(eta)
        if mu < minmu:
            mu = minmu
        mu_hat[j] = mu

    beta_iter[row] = 0
    for t in range(maxit):
        beta_iter[row] += 1

        for a in range(p):
            b_vec[a] = 0.0
            for b in range(p):
                a_mat[a * p + b] = lambda_nat[a] if a == b else 0.0

        for j in range(n_samples):
            mu = mu_hat[j]
            ww = wmat[row, j] if use_weights else 1.0
            w = ww * mu / (1.0 + alpha_i * mu)
            w_vec[j] = w
            z = log(mu / nf[row, j]) + (counts[row, j] - mu) / mu
            for a in range(p):
                b_vec[a] += x[j, a] * z * w
                for b in range(p):
                    a_mat[a * p + b] += x[j, a] * x[j, b] * w

        ok = _solve_linear_inplace(a_mat, b_vec, p)
        if ok == 0:
            beta_iter[row] = maxit
            break

        for a in range(p):
            beta_hat[a] = b_vec[a]
            if fabs(beta_hat[a]) > large:
                beta_iter[row] = maxit
                break
        if beta_iter[row] == maxit:
            break

        for j in range(n_samples):
            eta = 0.0
            for a in range(p):
                eta += x[j, a] * beta_hat[a]
            if not isfinite(eta):
                eta = 0.0
            if eta > 30.0:
                eta = 30.0
            elif eta < -30.0:
                eta = -30.0
            mu = nf[row, j] * exp(eta)
            if mu < minmu:
                mu = minmu
            mu_hat[j] = mu

        size = 1.0 / alpha_i
        ll_sum = 0.0
        for j in range(n_samples):
            yv = counts[row, j]
            mu = mu_hat[j]
            ll = (
                gammaln(yv + size)
                - gammaln(size)
                - gammaln(yv + 1.0)
                + size * (log(size) - log(size + mu))
                + yv * (log(mu) - log(size + mu))
            )
            if use_weights:
                ll_sum += wmat[row, j] * ll
            else:
                ll_sum += ll
        dev = -2.0 * ll_sum
        conv_test = fabs(dev - dev_old) / (fabs(dev) + 0.1)
        if not isfinite(conv_test):
            beta_iter[row] = maxit
            break
        if (t > 0) and (conv_test < beta_tol):
            break
        dev_old = dev

    if mu_only:
        for j in range(n_samples):
            mu_out[row, j] = mu_hat[j]
        return

    for a in range(p):
        beta_nat[row, a] = beta_hat[a]

    for a in range(p):
        for b in range(p):
            xtwx[a * p + b] = 0.0

    for j in range(n_samples):
        mu = mu_hat[j]
        ww = wmat[row, j] if use_weights else 1.0
        w = ww * mu / (1.0 + alpha_i * mu)
        ws = sqrt(w) if w > 0.0 else 0.0
        w_vec[j] = w
        for a in range(p):
            xw[j * p + a] = x[j, a] * ws
            for b in range(p):
                xtwx[a * p + b] += x[j, a] * x[j, b] * w

    for a in range(p):
        for b in range(p):
            a_mat[a * p + b] = xtwx[a * p + b]
        a_mat[a * p + a] += lambda_nat[a]

    ok = _invert_matrix(a_mat, inv_a, work, p)
    if ok == 0:
        for a in range(p):
            beta_var_nat[row, a] = NAN_VAL
        for j in range(n_samples):
            hat_diagonals[row, j] = NAN_VAL
    else:
        for j in range(n_samples):
            h = 0.0
            for a in range(p):
                tmp = 0.0
                for b in range(p):
                    tmp += inv_a[a * p + b] * xw[j * p + b]
                h += xw[j * p + a] * tmp
            hat_diagonals[row, j] = h

        for a in range(p):
            for b in range(p):
                tmp = 0.0
                for k in range(p):
                    tmp += inv_a[a * p + k] * xtwx[k * p + b]
                temp[a * p + b] = tmp

        for a in range(p):
            tmp = 0.0
            for k in range(p):
                tmp += temp[a * p + k] * inv_a[k * p + a]
            beta_var_nat[row, a] = tmp


def openmp_enabled() -> bool:
    return bool(pydeseq2_openmp_enabled())


def fit_disp_core(
    cnp.ndarray[cnp.float64_t, ndim=2] y,
    cnp.ndarray[cnp.float64_t, ndim=2] mu_hat,
    cnp.ndarray[cnp.float64_t, ndim=1] log_alpha,
    cnp.ndarray[cnp.float64_t, ndim=1] log_alpha_prior_mean,
    double log_alpha_prior_sigmasq,
    double min_log_alpha,
    double kappa_0,
    double tol,
    int maxit,
    bint use_prior,
    cnp.ndarray[cnp.float64_t, ndim=2] weights,
    bint use_weights,
    x=None,
    bint use_cr=False,
    double weight_threshold=1e-2,
    int n_threads=0,
):
    if y.ndim != 2:
        raise ValueError("y must be a 2D matrix.")
    if mu_hat.shape[0] != y.shape[0] or mu_hat.shape[1] != y.shape[1]:
        raise ValueError("mu_hat must have same shape as y.")
    if weights.shape[0] != y.shape[0] or weights.shape[1] != y.shape[1]:
        raise ValueError("weights must have same shape as y.")
    if log_alpha.shape[0] != y.shape[0]:
        raise ValueError("log_alpha must have one value per row in y.")
    if log_alpha_prior_mean.shape[0] != y.shape[0]:
        raise ValueError("log_alpha_prior_mean must have one value per row in y.")
    if use_cr and x is None:
        raise ValueError("x is required when use_cr=True.")

    cdef cnp.ndarray[cnp.float64_t, ndim=1] log_alpha_out = np.asarray(log_alpha, dtype=np.float64).copy()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] initial_lp = np.empty(y.shape[0], dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] last_lp = np.empty(y.shape[0], dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] last_change = np.empty(y.shape[0], dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] iterations = np.zeros(y.shape[0], dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] iter_accept = np.zeros(y.shape[0], dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] x_c

    cdef const double[:, ::1] y_mv = np.asarray(y, dtype=np.float64, order="C")
    cdef const double[:, ::1] mu_mv = np.asarray(mu_hat, dtype=np.float64, order="C")
    cdef const double[:, ::1] w_mv = np.asarray(weights, dtype=np.float64, order="C")
    if x is None:
        x_c = np.empty((y.shape[1], 0), dtype=np.float64)
    else:
        x_c = np.asarray(x, dtype=np.float64, order="C")
    if x_c.shape[0] != y.shape[1]:
        raise ValueError("x rows must equal the number of columns in y.")
    cdef const double[:, ::1] x_mv = x_c
    cdef double[::1] log_alpha_out_mv = log_alpha_out
    cdef const double[::1] log_alpha_prior_mean_mv = np.asarray(log_alpha_prior_mean, dtype=np.float64, order="C")
    cdef int[::1] iterations_mv = iterations
    cdef int[::1] iter_accept_mv = iter_accept
    cdef double[::1] last_change_mv = last_change
    cdef double[::1] initial_lp_mv = initial_lp
    cdef double[::1] last_lp_mv = last_lp
    cdef Py_ssize_t i
    cdef Py_ssize_t n_rows = y.shape[0]

    if use_cr:
        if n_threads > 0:
            for i in prange(n_rows, nogil=True, schedule="static", num_threads=n_threads):
                _fit_row_with_cr(
                    i,
                    y_mv,
                    x_mv,
                    mu_mv,
                    w_mv,
                    log_alpha_prior_mean_mv,
                    log_alpha_prior_sigmasq,
                    min_log_alpha,
                    kappa_0,
                    tol,
                    maxit,
                    use_prior,
                    use_weights,
                    weight_threshold,
                    log_alpha_out_mv,
                    iterations_mv,
                    iter_accept_mv,
                    last_change_mv,
                    initial_lp_mv,
                    last_lp_mv,
                )
        else:
            for i in prange(n_rows, nogil=True, schedule="static"):
                _fit_row_with_cr(
                    i,
                    y_mv,
                    x_mv,
                    mu_mv,
                    w_mv,
                    log_alpha_prior_mean_mv,
                    log_alpha_prior_sigmasq,
                    min_log_alpha,
                    kappa_0,
                    tol,
                    maxit,
                    use_prior,
                    use_weights,
                    weight_threshold,
                    log_alpha_out_mv,
                    iterations_mv,
                    iter_accept_mv,
                    last_change_mv,
                    initial_lp_mv,
                    last_lp_mv,
                )
    else:
        if n_threads > 0:
            for i in prange(n_rows, nogil=True, schedule="static", num_threads=n_threads):
                _fit_row_no_cr(
                    i,
                    y_mv,
                    mu_mv,
                    w_mv,
                    log_alpha_prior_mean_mv,
                    log_alpha_prior_sigmasq,
                    min_log_alpha,
                    kappa_0,
                    tol,
                    maxit,
                    use_prior,
                    use_weights,
                    log_alpha_out_mv,
                    iterations_mv,
                    iter_accept_mv,
                    last_change_mv,
                    initial_lp_mv,
                    last_lp_mv,
                )
        else:
            for i in prange(n_rows, nogil=True, schedule="static"):
                _fit_row_no_cr(
                    i,
                    y_mv,
                    mu_mv,
                    w_mv,
                    log_alpha_prior_mean_mv,
                    log_alpha_prior_sigmasq,
                    min_log_alpha,
                    kappa_0,
                    tol,
                    maxit,
                    use_prior,
                    use_weights,
                    log_alpha_out_mv,
                    iterations_mv,
                    iter_accept_mv,
                    last_change_mv,
                    initial_lp_mv,
                    last_lp_mv,
                )

    return {
        "log_alpha": np.asarray(log_alpha_out),
        "iter": np.asarray(iterations, dtype=np.int64),
        "iter_accept": np.asarray(iter_accept, dtype=np.int64),
        "last_change": np.asarray(last_change),
        "initial_lp": np.asarray(initial_lp),
        "last_lp": np.asarray(last_lp),
    }


def fit_glm_core(
    cnp.ndarray[cnp.float64_t, ndim=2] counts,
    cnp.ndarray[cnp.float64_t, ndim=2] model_matrix,
    cnp.ndarray[cnp.float64_t, ndim=2] normalization_factors,
    cnp.ndarray[cnp.float64_t, ndim=1] alpha_hat,
    cnp.ndarray[cnp.float64_t, ndim=2] beta_nat_init,
    cnp.ndarray[cnp.float64_t, ndim=1] lambda_nat,
    double beta_tol,
    int maxit,
    double minmu,
    bint use_weights,
    cnp.ndarray[cnp.float64_t, ndim=2] weights,
    bint use_qr,
    bint mu_only=False,
    int n_threads=0,
):
    if counts.ndim != 2:
        raise ValueError("counts must be a 2D matrix.")
    if normalization_factors.shape[0] != counts.shape[0] or normalization_factors.shape[1] != counts.shape[1]:
        raise ValueError("normalization_factors must have the same shape as counts.")
    if weights.shape[0] != counts.shape[0] or weights.shape[1] != counts.shape[1]:
        raise ValueError("weights must have the same shape as counts.")
    if alpha_hat.shape[0] != counts.shape[0]:
        raise ValueError("alpha_hat must have one value per gene.")
    if model_matrix.shape[0] != counts.shape[1]:
        raise ValueError("model_matrix rows must equal the number of samples.")
    if beta_nat_init.shape[0] != counts.shape[0] or beta_nat_init.shape[1] != model_matrix.shape[1]:
        raise ValueError("beta_nat_init must have one row per gene and one column per coefficient.")
    if lambda_nat.shape[0] != model_matrix.shape[1]:
        raise ValueError("lambda_nat must have one value per coefficient.")

    cdef cnp.ndarray[cnp.float64_t, ndim=2] counts_c = np.asarray(counts, dtype=np.float64, order="C")
    cdef cnp.ndarray[cnp.float64_t, ndim=2] x_c = np.asarray(model_matrix, dtype=np.float64, order="C")
    cdef cnp.ndarray[cnp.float64_t, ndim=2] nf_c = np.asarray(normalization_factors, dtype=np.float64, order="C")
    cdef cnp.ndarray[cnp.float64_t, ndim=2] w_c = np.asarray(weights, dtype=np.float64, order="C")
    cdef cnp.ndarray[cnp.float64_t, ndim=2] beta_init_c = np.asarray(beta_nat_init, dtype=np.float64, order="C")
    cdef cnp.ndarray[cnp.float64_t, ndim=1] alpha_c = np.asarray(alpha_hat, dtype=np.float64, order="C")
    cdef cnp.ndarray[cnp.float64_t, ndim=1] lambda_c = np.asarray(lambda_nat, dtype=np.float64, order="C")

    cdef int n_genes = counts_c.shape[0]
    cdef int n_samples = counts_c.shape[1]
    cdef int p = x_c.shape[1]
    cdef int threads_eff = n_threads if n_threads > 0 else pydeseq2_omp_max_threads()
    if threads_eff < 1:
        threads_eff = 1
    cdef double large = 30.0
    cdef bint _ = use_qr  # Currently solved via normal equations in core kernel.

    cdef cnp.ndarray[cnp.float64_t, ndim=2] beta_nat
    cdef cnp.ndarray[cnp.float64_t, ndim=2] beta_var_nat
    cdef cnp.ndarray[cnp.float64_t, ndim=2] hat_diagonals
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mu_out
    cdef cnp.ndarray[cnp.int32_t, ndim=1] beta_iter = np.zeros(n_genes, dtype=np.int32)

    if mu_only:
        beta_nat = np.empty((1, 1), dtype=np.float64)
        beta_var_nat = np.empty((1, 1), dtype=np.float64)
        hat_diagonals = np.empty((1, 1), dtype=np.float64)
        mu_out = np.full((n_genes, n_samples), np.nan, dtype=np.float64)
    else:
        beta_nat = np.full((n_genes, p), np.nan, dtype=np.float64)
        beta_var_nat = np.full((n_genes, p), np.nan, dtype=np.float64)
        hat_diagonals = np.full((n_genes, n_samples), np.nan, dtype=np.float64)
        mu_out = np.empty((1, 1), dtype=np.float64)

    cdef const double[:, ::1] counts_mv = counts_c
    cdef const double[:, ::1] x_mv = x_c
    cdef const double[:, ::1] nf_mv = nf_c
    cdef const double[:, ::1] w_mv = w_c
    cdef const double[::1] alpha_mv = alpha_c
    cdef const double[:, ::1] beta_init_mv = beta_init_c
    cdef const double[::1] lambda_mv = lambda_c
    cdef double[:, ::1] beta_nat_mv = beta_nat
    cdef double[:, ::1] beta_var_mv = beta_var_nat
    cdef double[:, ::1] hat_mv = hat_diagonals
    cdef double[:, ::1] mu_mv = mu_out
    cdef int[::1] beta_iter_mv = beta_iter
    cdef Py_ssize_t i
    cdef int tid
    cdef Py_ssize_t off_p
    cdef Py_ssize_t off_s
    cdef Py_ssize_t off_pp
    cdef Py_ssize_t off_sp
    cdef Py_ssize_t pp = <Py_ssize_t>p * p
    cdef Py_ssize_t sp = <Py_ssize_t>n_samples * p

    cdef double* beta_hat_buf = <double*>malloc((<Py_ssize_t>threads_eff * p) * sizeof(double))
    cdef double* mu_hat_buf = <double*>malloc((<Py_ssize_t>threads_eff * n_samples) * sizeof(double))
    cdef double* w_vec_buf = <double*>malloc((<Py_ssize_t>threads_eff * n_samples) * sizeof(double))
    cdef double* a_mat_buf = <double*>malloc((<Py_ssize_t>threads_eff * pp) * sizeof(double))
    cdef double* b_vec_buf = <double*>malloc((<Py_ssize_t>threads_eff * p) * sizeof(double))
    cdef double* xtwx_buf = <double*>malloc((<Py_ssize_t>threads_eff * pp) * sizeof(double))
    cdef double* inv_a_buf = <double*>malloc((<Py_ssize_t>threads_eff * pp) * sizeof(double))
    cdef double* work_buf = <double*>malloc((<Py_ssize_t>threads_eff * pp) * sizeof(double))
    cdef double* temp_buf = <double*>malloc((<Py_ssize_t>threads_eff * pp) * sizeof(double))
    cdef double* xw_buf = <double*>malloc((<Py_ssize_t>threads_eff * sp) * sizeof(double))

    if (
        beta_hat_buf == NULL
        or mu_hat_buf == NULL
        or w_vec_buf == NULL
        or a_mat_buf == NULL
        or b_vec_buf == NULL
        or xtwx_buf == NULL
        or inv_a_buf == NULL
        or work_buf == NULL
        or temp_buf == NULL
        or xw_buf == NULL
    ):
        if beta_hat_buf != NULL:
            free(beta_hat_buf)
        if mu_hat_buf != NULL:
            free(mu_hat_buf)
        if w_vec_buf != NULL:
            free(w_vec_buf)
        if a_mat_buf != NULL:
            free(a_mat_buf)
        if b_vec_buf != NULL:
            free(b_vec_buf)
        if xtwx_buf != NULL:
            free(xtwx_buf)
        if inv_a_buf != NULL:
            free(inv_a_buf)
        if work_buf != NULL:
            free(work_buf)
        if temp_buf != NULL:
            free(temp_buf)
        if xw_buf != NULL:
            free(xw_buf)
        raise MemoryError("Failed to allocate GLM workspace buffers.")

    if threads_eff > 1:
        for i in prange(n_genes, nogil=True, schedule="static", num_threads=threads_eff):
            tid = threadid()
            off_p = <Py_ssize_t>tid * p
            off_s = <Py_ssize_t>tid * n_samples
            off_pp = <Py_ssize_t>tid * pp
            off_sp = <Py_ssize_t>tid * sp
            _fit_glm_row(
                i,
                counts_mv,
                x_mv,
                nf_mv,
                w_mv,
                alpha_mv,
                beta_init_mv,
                lambda_mv,
                n_samples,
                p,
                maxit,
                beta_tol,
                minmu,
                large,
                use_weights,
                mu_only,
                beta_nat_mv,
                beta_var_mv,
                hat_mv,
                mu_mv,
                beta_iter_mv,
                beta_hat_buf + off_p,
                mu_hat_buf + off_s,
                w_vec_buf + off_s,
                a_mat_buf + off_pp,
                b_vec_buf + off_p,
                xtwx_buf + off_pp,
                inv_a_buf + off_pp,
                work_buf + off_pp,
                temp_buf + off_pp,
                xw_buf + off_sp,
            )
    else:
        for i in range(n_genes):
            _fit_glm_row(
                i,
                counts_mv,
                x_mv,
                nf_mv,
                w_mv,
                alpha_mv,
                beta_init_mv,
                lambda_mv,
                n_samples,
                p,
                maxit,
                beta_tol,
                minmu,
                large,
                use_weights,
                mu_only,
                beta_nat_mv,
                beta_var_mv,
                hat_mv,
                mu_mv,
                beta_iter_mv,
                beta_hat_buf,
                mu_hat_buf,
                w_vec_buf,
                a_mat_buf,
                b_vec_buf,
                xtwx_buf,
                inv_a_buf,
                work_buf,
                temp_buf,
                xw_buf,
            )

    free(beta_hat_buf)
    free(mu_hat_buf)
    free(w_vec_buf)
    free(a_mat_buf)
    free(b_vec_buf)
    free(xtwx_buf)
    free(inv_a_buf)
    free(work_buf)
    free(temp_buf)
    free(xw_buf)

    if mu_only:
        return {
            "mu": np.asarray(mu_out),
            "beta_iter": np.asarray(beta_iter, dtype=np.int64),
        }
    return {
        "beta_nat": np.asarray(beta_nat),
        "beta_var_nat": np.asarray(beta_var_nat),
        "hat_diagonals": np.asarray(hat_diagonals),
        "beta_iter": np.asarray(beta_iter, dtype=np.int64),
    }
