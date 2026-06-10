import numpy as np

def solve_poisson_sor(
    rhs,
    dx,
    dy,
    omega,
    tolerance,
    max_iter,
    initial=None,
):
    
    """
    Solve the 2D Poisson equation using SOR.
    
    Equation:
        ∇²φ = f
        d²(phi)/dx² + d²(phi)/dy² = rhs
        
    Boundary conditions:
        Homogeneous Neumann BCs:
            d(phi)/dn = 0 on all boundaries
            
    Parameters
    ----------
    rhs : np.ndarray
        Right-hand side of the Poisson equation.
        Shape is expected to include ghost cells.

    dx, dy : float
        Grid spacing in x and y directions.

    omega : float
        SOR relaxation factor.
        omega = 1.0 corresponds to Gauss-Seidel.
        Usually 1.0 < omega < 2.0.

    tolerance : float
        Iteration stops when the maximum update becomes smaller than this value.

    max_iter : int
        Maximum number of SOR iterations.

    Returns
    -------
    phi : np.ndarray
        Solution of the Poisson equation.
    """
    
    rhs = rhs.copy()
    rhs[1:-1, 1:-1] -= rhs[1:-1, 1:-1].mean()

    phi = np.zeros_like(rhs) if initial is None else initial.copy()
    phi -= np.mean(phi[1:-1, 1:-1])
    
    dx2 = dx * dx
    dy2 = dy * dy
    
    coef = 1.0 / (2.0 / dx2 + 2.0 / dy2)
    
    for iteration in range(max_iter):
        max_error = 0.0
        
        # Interior points only
        for i in range(1, rhs.shape[0] - 1):
            for j in range(1, rhs.shape[1] - 1):
                
                phi_old = phi[i, j]
                
                phi_gs = coef * (
                    (phi[i + 1, j] + phi[i - 1, j]) / dx2
                    + (phi[i, j + 1] + phi[i, j - 1]) / dy2
                    - rhs[i, j]
                )
                
                phi[i, j] = (1.0 - omega) * phi_old + omega * phi_gs
                
                error = abs(phi[i, j] - phi_old)
                if error > max_error:
                    max_error = error
                    
                
        # Homogeneous Neumann boundary condition
        phi[0, :] = phi[1, :]
        phi[-1, :] = phi[-2, :]
        phi[:, 0] = phi[:, 1]
        phi[:, -1] = phi[:, -2]
                
        # Remove arbitary constant offset
        phi -= np.mean(phi[1:-1, 1:-1]) 
        
        if max_error < tolerance:
            break
        
    return phi
