import numpy as np
import scipy.optimize as opt

def modelo(x, c_lambda=4, c_mu=4):
    """
    ssssssss
    """
    dof = 2*c_lambda*c_mu + 3*c_lambda + 3*c_mu - 2
    p_lambda = x[0:c_lambda-1]
    p_lambda = np.concatenate((p_lambda,[1-np.sum(p_lambda)]))
    p_mu = x[c_lambda-1:c_lambda+c_mu-2]
    p_mu = np.concatenate((p_mu,[1-np.sum(p_mu)]))
    p_a = x[c_lambda+c_mu-2:3*c_lambda+c_mu-2]
    p_a = np.reshape(p_a, (2,c_lambda))
    p_a = np.array([p_a, 1-p_a])
    p_b = x[3*c_lambda+c_mu-2:3*c_lambda+c_mu-2+2*c_lambda*c_mu]
    p_b = np.reshape(p_b, (2,c_lambda,c_mu))
    p_b = np.array([p_b, 1-p_b])
    p_c = x[3*c_lambda+c_mu-2+2*c_lambda*c_mu:dof]
    p_c = np.reshape(p_c, (2,c_mu))
    p_c = np.array([p_c, 1-p_c])
    return p_lambda, p_mu, p_a, p_b, p_c

def comportamento(x, c_lambda=4, c_mu=4):
    """
    aaaaaaaaaaaa
    """
    p_lambda, p_mu, p_a, p_b, p_c = modelo(x, c_lambda, c_mu)
    #Array indices for np.einsum:
    #   p_lambda: lambda -> i
    #   p_mu: mu -> j
    #   p_a: a, x, lambda -> kli
    #   p_b: b, y, lambda, mu -> mnij
    #   p_c: c, z, mu -> pqj
    #   px: a, b, c, x, y, z -> kmplnq
    px = np.einsum('i,j,kli,mnij,pqj->kmplnq',p_lambda,p_mu,p_a,p_b,p_c)
    return px

def cost(x, p, c_lambda, c_mu):
    """
    fgsfsfg
    """
    px = comportamento(x, c_lambda=4, c_mu=4)
    return np.sum((px-p)**2)


def bilocal(p=None, Ma=2, Mb=2, Mc=2, ma=2, mb=2, mc=2, c_lambda=4, c_mu=4):
    """
    Tries to find a local model for behaviour p in the bilocal scenario.
    
    
    
    :param p: the behaviour to be optimized against (default=None)
    :type p: numpy.ndarray of floats and shape=(ma,mb,mc,Ma,Mb,Mc)
    
    :param Ma: number of Alice's inputs (default=2)
    :type Ma: integer
    
    :param Mb: number of Bob's inputs (default=2)
    :type Mb: integer
    
    :param Mc: number of Charles's inputs (default=2)
    :type Mc: integer
    
    :param ma: number of Alice's outputs (default=2)
    :type ma: integer
    
    :param mb: number of Bob's outputs (default=2)
    :type mb: integer
    
    :param mc: number of Charles's outputs (default=2)
    :type mc: integer
    
    :param c_lambda: cardinality of lambda (default=4)
    :type c_lambda: integer
    
    :param c_mu: cardinality of mu (default=4)
    :type c_mu: integer
    
    :return: solution to the optimization problem
    :rtype: scipy.optimize.optimize.OptimizeResult
    """
    # -------------------------------------------- calculates p if not provided
    if p is None:
        I = 0
        J = 1
        A = np.arange(0,ma)
        B = np.arange(0,mb)
        C = np.arange(0,mc)
        X = np.arange(0,Ma)
        Y = np.arange(0,Mb)
        Z = np.arange(0,Mc)
        Am, Bm, Cm, Xm, Ym, Zm = np.meshgrid(A,B,C,X,Y,Z,indexing='ij')
        pI = 1/8*(1+(Ym==0)*(-1)**(Am+Bm+Cm))
        pJ = 1/8*(1+(Ym==1)*(-1)**(Xm+Zm+Am+Bm+Cm))
        p0 = 1/8
        p = I*pI + J*pJ + (1-I-J)*p0
    # -------------------------------------------------------------------------
    dof = 2*c_lambda*c_mu + 3*c_lambda + 3*c_mu - 2
    rng = np.random.default_rng()
    bounds = opt.Bounds(np.zeros(dof), np.ones(dof))
    coeffs = np.zeros(shape=(2,dof))
    coeffs[0,0:c_lambda-1] = 1
    coeffs[1,c_lambda-1:c_lambda+c_mu-2] = 1
    linear_constraints = opt.LinearConstraint(coeffs, -np.inf*np.ones(2), np.ones(2))
    x0 = rng.random(size=dof)
    x0[0:c_lambda-1] = 1/c_lambda
    x0[c_lambda-1:c_lambda+c_mu-2] = 1/c_mu
    solution = opt.minimize(cost, x0, args=(p, c_lambda, c_mu), method='trust-constr',
                        constraints=linear_constraints,
                        options={'verbose': 1}, bounds=bounds)
    return solution