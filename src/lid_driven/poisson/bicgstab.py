import numpy as np


def _apply_laplacian(phi, dx, dy):
    padded = np.pad(phi, 1, mode="edge")
    result = (
        (padded[2:, 1:-1] - 2.0 * phi + padded[:-2, 1:-1]) / dx**2
        + (padded[1:-1, 2:] - 2.0 * phi + padded[1:-1, :-2]) / dy**2
    )
    result -= result.mean()
    return result


def solve_poisson_bicgstab(
    rhs,
    dx,
    dy,
    tolerance,
    max_iter,
    initial=None,
):
    """Solve a homogeneous-Neumann Poisson problem with BiCGSTAB."""
    b = rhs[1:-1, 1:-1].copy()
    b -= b.mean()

    if initial is None:
        x = np.zeros_like(b)
    else:
        x = initial[1:-1, 1:-1].copy()
        x -= x.mean()

    r = b - _apply_laplacian(x, dx, dy)
    r_hat = r.copy()
    rho_old = alpha = omega = 1.0
    v = np.zeros_like(b)
    p = np.zeros_like(b)
    tiny = np.finfo(float).eps

    for _ in range(max_iter):
        if np.max(np.abs(r)) < tolerance:
            break

        rho_new = np.vdot(r_hat, r)
        if abs(rho_new) <= tiny:
            raise RuntimeError("BiCGSTAB breakdown: rho is zero.")

        beta = (rho_new / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = _apply_laplacian(p, dx, dy)

        denominator = np.vdot(r_hat, v)
        if abs(denominator) <= tiny:
            raise RuntimeError("BiCGSTAB breakdown: alpha denominator is zero.")

        alpha = rho_new / denominator
        s = r - alpha * v

        if np.max(np.abs(s)) < tolerance:
            x += alpha * p
            break

        t = _apply_laplacian(s, dx, dy)
        t_norm = np.vdot(t, t)
        if abs(t_norm) <= tiny:
            raise RuntimeError("BiCGSTAB breakdown: omega denominator is zero.")

        omega = np.vdot(t, s) / t_norm
        if abs(omega) <= tiny:
            raise RuntimeError("BiCGSTAB breakdown: omega is zero.")

        x += alpha * p + omega * s
        x -= x.mean()
        r = s - omega * t
        rho_old = rho_new
    else:
        raise RuntimeError(
            f"BiCGSTAB did not converge within {max_iter} iterations."
        )

    x -= x.mean()
    phi = np.zeros_like(rhs)
    phi[1:-1, 1:-1] = x
    phi[0, :] = phi[1, :]
    phi[-1, :] = phi[-2, :]
    phi[:, 0] = phi[:, 1]
    phi[:, -1] = phi[:, -2]
    return phi
