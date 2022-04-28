import numpy as np
import scipy.optimize as opt

def modelo(x, c_alpha=6, c_beta=6, c_gamma=6):
    """
    asdfasçdlfkjaçdfl
    """
    dof = c_alpha+c_beta+c_gamma-3+c_alpha*c_beta+c_beta*c_gamma+c_alpha*c_gamma
    p_alpha = x[0:c_alpha-1]
    p_alpha = np.concatenate((p_alpha,[1-np.sum(p_alpha)]))
    p_beta = x[c_alpha-1:c_alpha+c_beta-2]
    p_beta = np.concatenate((p_beta,[1]-np.sum(p_beta)))
    p_gamma = x[c_alpha+c_beta-2:c_alpha+c_beta+c_gamma-3]
    p_gamma = np.concatenate((p_gamma,[1-np.sum(p_gamma)]))
    p_a = x[c_alpha+c_beta+c_gamma-3:c_alpha+c_beta+c_gamma-3+c_beta*c_gamma]
    p_a = np.reshape(p_a, (c_beta,c_gamma))
    p_a = np.array([p_a, 1-p_a])
    p_b = x[c_alpha+c_beta+c_gamma-3+c_beta*c_gamma:c_alpha+c_beta+c_gamma-3+c_beta*c_gamma+c_alpha*c_gamma]
    p_b = np.reshape(p_b, (c_gamma,c_alpha))
    p_b = np.array([p_b, 1-p_b])
    p_c = x[c_alpha+c_beta+c_gamma-3+c_beta*c_gamma+c_alpha*c_gamma:dof]
    p_c = np.reshape(p_c, (c_alpha,c_beta))
    p_c = np.array([p_c, 1-p_c])
    return p_alpha, p_beta, p_gamma, p_a, p_b, p_c

def comportamento(x, c_alpha=6, c_beta=6, c_gamma=6):
    """
    \dfsdafsdfadfg
    """
    p_alpha, p_beta, p_gamma, p_a, p_b, p_c = modelo(x, c_alpha, c_beta, c_gamma)
    #Array indices for np.einsum:
    #   p_alpha: alpha -> i
    #   p_beta: beta -> j
    #   p_gamma: gamma -> k
    #   p_a: a, beta, gamma -> ljk
    #   p_b: b, gamma, alpha -> mki
    #   p_c: c, alpha, beta -> nij
    #   px: a, b, c -> lmn
    px = np.einsum('i,j,k,ljk,mki,nij->lmn',p_alpha, p_beta, p_gamma, p_a, p_b, p_c)
    return px

def cost(x, p, c_alpha=6, c_beta=6, c_gamma=6):
    """
    adfasdfasdf
    """
    px = comportamento(x, c_alpha, c_beta, c_gamma)
    return np.sum((px-p)**2)

def triangle(p=None, ma=2, mb=2, mc=2, c_alpha=6, c_beta=6, c_gamma=6):
    """
    aasdfasdfasdf
    """
    # -------------------------------------------- calculates p if not provided
    if p is None:
        v = 0.33
        A = np.arange(0,ma)
        B = np.arange(0,mb)
        C = np.arange(0,mc)
        Am, Bm, Cm = np.meshgrid(A,B,C,indexing='ij')
        pGHZ = 1/2*(Am==Bm)*(Bm==Cm)
        p0 = 1/8
        p = v*pGHZ + (1-v)*p0
    # -------------------------------------------------------------------------
    dof = c_alpha+c_beta+c_gamma-3+c_alpha*c_beta+c_beta*c_gamma+c_alpha*c_gamma
    rng = np.random.default_rng()
    bounds = opt.Bounds(np.zeros(dof), np.ones(dof))
    coeffs = np.zeros(shape=(3,dof))
    coeffs[0,0:c_alpha-1] = 1
    coeffs[1,c_alpha-1:c_alpha+c_beta-2] = 1
    coeffs[2,c_alpha+c_beta-2:c_alpha+c_beta+c_gamma-3] = 1
    linear_constraints = opt.LinearConstraint(coeffs, -np.inf*np.ones(3), np.ones(3))
    x0 = rng.random(size=dof)
    x0[0:c_alpha-1] = 1/c_alpha
    x0[c_alpha-1:c_alpha+c_beta-2] = 1/c_beta
    x0[c_alpha+c_beta-2:c_alpha+c_beta+c_gamma-3] = 1/c_gamma
    solution = opt.minimize(cost, x0, args=(p, c_alpha, c_beta, c_gamma), 
                            method='trust-constr', 
                            constraints=linear_constraints, 
                            options={'verbose': 1}, bounds=bounds)
    return solution