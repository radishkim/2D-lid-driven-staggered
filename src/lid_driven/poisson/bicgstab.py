import numpy as np


def _project_zero_mean(field):
    field -= field.mean()
    return field


def _apply_laplacian(phi, dx, dy):
    padded = np.pad(phi, 1, mode="edge")
    result = (
        (padded[2:, 1:-1] - 2.0 * phi + padded[:-2, 1:-1]) / dx**2
        + (padded[1:-1, 2:] - 2.0 * phi + padded[1:-1, :-2]) / dy**2
    )
    return _project_zero_mean(result)


def _inner_product_breakdown(value, left, right):
    scale = np.linalg.norm(left.ravel()) * np.linalg.norm(right.ravel())
    threshold = 100.0 * np.finfo(float).eps * max(scale, 1.0)
    return abs(value) <= threshold


def _scalar_breakdown(value):
    return abs(value) <= 100.0 * np.finfo(float).eps


def solve_poisson_bicgstab(
    rhs,
    dx,
    dy,
    tolerance,
    max_iter,
    initial=None,
):
    """Solve a homogeneous-Neumann Poisson problem with BiCGSTAB."""
    b = _project_zero_mean(rhs[1:-1, 1:-1].copy())

    if initial is None:
        x = np.zeros_like(b)
    else:
        x = _project_zero_mean(initial[1:-1, 1:-1].copy())

    r = _project_zero_mean(b - _apply_laplacian(x, dx, dy))
    r_hat = r.copy()
    rho_old = alpha = omega = 1.0
    v = np.zeros_like(b)
    p = np.zeros_like(b)
    restart = True
    restart_count = 0
    max_restarts = 20

    for _ in range(max_iter):
        if np.max(np.abs(r)) < tolerance:
            break

        rho_new = np.vdot(r_hat, r)
        if _inner_product_breakdown(rho_new, r_hat, r):
            restart_count += 1
            if restart_count > max_restarts:
                raise RuntimeError(
                    "BiCGSTAB failed after repeated rho breakdowns."
                )
            r_hat = r.copy()
            rho_new = np.vdot(r_hat, r)
            restart = True

        if restart:
            p = r.copy()
            restart = False
        else:
            beta = (rho_new / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)

        p = _project_zero_mean(p)
        v = _apply_laplacian(p, dx, dy)

        denominator = np.vdot(r_hat, v)
        if _inner_product_breakdown(denominator, r_hat, v):
            restart_count += 1
            if restart_count > max_restarts:
                raise RuntimeError(
                    "BiCGSTAB failed after repeated alpha breakdowns."
                )
            r_hat = r.copy()
            p = r.copy()
            v = _apply_laplacian(p, dx, dy)
            rho_new = np.vdot(r_hat, r)
            denominator = np.vdot(r_hat, v)
            if _inner_product_breakdown(denominator, r_hat, v):
                raise RuntimeError(
                    "BiCGSTAB alpha breakdown persisted after restart."
                )

        alpha = rho_new / denominator
        s = _project_zero_mean(r - alpha * v)

        if np.max(np.abs(s)) < tolerance:
            x += alpha * p
            break

        t = _apply_laplacian(s, dx, dy)
        t_norm = np.vdot(t, t)
        if _inner_product_breakdown(t_norm, t, t):
            if np.max(np.abs(s)) < tolerance:
                x += alpha * p
                break
            raise RuntimeError(
                "BiCGSTAB omega denominator is zero before convergence."
            )

        omega = np.vdot(t, s) / t_norm
        x += alpha * p

        if _scalar_breakdown(omega):
            r = s
            r_hat = r.copy()
            rho_old = alpha = omega = 1.0
            v.fill(0.0)
            p.fill(0.0)
            restart = True
            restart_count += 1
            if restart_count > max_restarts:
                raise RuntimeError(
                    "BiCGSTAB failed after repeated omega breakdowns."
                )
            continue

        x += omega * s
        x = _project_zero_mean(x)
        r = _project_zero_mean(s - omega * t)
        rho_old = rho_new
    else:
        raise RuntimeError(
            f"BiCGSTAB did not converge within {max_iter} iterations."
        )

    x = _project_zero_mean(x)
    phi = np.zeros_like(rhs)
    phi[1:-1, 1:-1] = x
    phi[0, :] = phi[1, :]
    phi[-1, :] = phi[-2, :]
    phi[:, 0] = phi[:, 1]
    phi[:, -1] = phi[:, -2]
    return phi
