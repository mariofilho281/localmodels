import numpy as np
import scipy.optimize as opt

def model(x, ma=2, mb=2, mc=2, c_alpha=6, c_beta=6, c_gamma=6):
    """
    Assembles hidden variables probability distributions and response functions
    
    This function takes the ``x`` attribute of the solution for the 
    optimization problem solved in the function ``triangle`` and extract the 
    model to a more readble format.
    
    :param x: solution of optimization problem solved by ``triangle``
    :type x: numpy.ndarray of floats
    
    :param c_alpha: cardinality of alpha (default=6)
    :type c_alpha: integer
    
    :param c_beta: cardinality of beta (default=6)
    :type c_beta: integer
    
    :param c_gamma: cardinality of gamma (default=6)
    :type c_gamma: integer
    
    :return: 
        the hidden variables probability distributions ``p_alpha``, ``p_beta``,
        ``p_gamma`` and the response functions ``p_a``, ``p_b``, ``p_c`` of 
        Alice, Bob and Charles
    :rtype: tuple
    """
    dof = c_alpha+c_beta+c_gamma-3+c_alpha*c_beta*(mc-1)+c_beta*c_gamma*(ma-1)+c_alpha*c_gamma*(mb-1)
    p_alpha = x[0:c_alpha-1]
    p_alpha = np.concatenate((p_alpha,[1-np.sum(p_alpha)]))
    p_beta = x[c_alpha-1:c_alpha+c_beta-2]
    p_beta = np.concatenate((p_beta,[1]-np.sum(p_beta)))
    p_gamma = x[c_alpha+c_beta-2:c_alpha+c_beta+c_gamma-3]
    p_gamma = np.concatenate((p_gamma,[1-np.sum(p_gamma)]))
    
    p_a = np.zeros(shape=(ma,c_beta,c_gamma))
    p_a[0:ma-1,:,:] = np.reshape(x[c_alpha+c_beta+c_gamma-3:c_alpha+c_beta+c_gamma-3+(ma-1)*c_beta*c_gamma],(ma-1,c_beta,c_gamma))
    p_a[ma-1,:,:] = 1-np.sum(p_a[0:ma-1,:,:], axis=0)

    p_b = np.zeros(shape=(mb,c_gamma,c_alpha))
    p_b[0:mb-1,:,:] = np.reshape(x[c_alpha+c_beta+c_gamma-3+(ma-1)*c_beta*c_gamma:c_alpha+c_beta+c_gamma-3+(ma-1)*c_beta*c_gamma+(mb-1)*c_gamma*c_alpha],(mb-1,c_gamma,c_alpha))
    p_b[mb-1,:,:] = 1-np.sum(p_b[0:mb-1,:,:], axis=0)
    
    p_c = np.zeros(shape=(mc,c_alpha,c_beta))
    p_c[0:mc-1,:,:] = np.reshape(x[c_alpha+c_beta+c_gamma-3+(ma-1)*c_beta*c_gamma+(mb-1)*c_gamma*c_alpha:dof],(mc-1,c_alpha,c_beta))
    p_c[mc-1,:,:] = 1-np.sum(p_c[0:mc-1,:,:])
    return p_alpha, p_beta, p_gamma, p_a, p_b, p_c

def behaviour(x, ma=2, mb=2, mc=2, c_alpha=6, c_beta=6, c_gamma=6):
    """
    Calculates the behaviour of a local model in the triangle scenario
    
    This function takes the ``x`` attribute of the solution for the 
    optimization problem solved in the function ``triangle`` and calculates the
    probability distribution p(a,b,c) that it reproduces.
    
    :param x: solution of optimization problem solved by ``triangle``
    :type x: numpy.ndarray of floats
    
    :param c_alpha: cardinality of alpha (default=6)
    :type c_alpha: integer
    
    :param c_beta: cardinality of beta (default=6)
    :type c_beta: integer
    
    :param c_gamma: cardinality of gamma (default=6)
    :type c_gamma: integer
    
    :return: 
        the probability distribution p(a,b,c) indexed in the usual order,
        i.e. p[a,b,c]
    :rtype: numpy.ndarray of floats
    """
    p_alpha, p_beta, p_gamma, p_a, p_b, p_c = model(x, ma, mb, mc, c_alpha, c_beta, c_gamma)
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

def cost(x, p, ma=2, mb=2, mc=2, c_alpha=6, c_beta=6, c_gamma=6):
    """
    Calculates the sum of squared errors between behaviour p and model x
    
    This function calculates the cost that is optimized in function 
    ``triangle``. Its minimum value of zero is attained when the model
    represented by the solution ``x`` represents exactly all the probabilities 
    of behaviour ``p``.
    
    :param x: array that represents an explicit trilocal model
    :type x: numpy.ndarray of floats
    
    :param p: the behaviour to be optimized against
    :type p: numpy.ndarray of floats
    
    :param c_alpha: cardinality of alpha (default=6)
    :type c_alpha: integer
    
    :param c_beta: cardinality of beta (default=6)
    :type c_beta: integer
    
    :param c_gamma: cardinality of gamma (default=6)
    :type c_gamma: integer
    
    :return: 
        sum of squared errors between probabilities ``p`` and those generated
        by ``x``
    :rtype: numpy.ndarray of floats
    """
    px = behaviour(x, ma, mb, mc, c_alpha, c_beta, c_gamma)
    return np.sum((px-p)**2)

def triangle(p=None, ma=2, mb=2, mc=2, c_alpha=6, c_beta=6, c_gamma=6):
    """
    Tries to find a local model for behaviour p in the triangle scenario with
    no inputs.
    
    This function takes a user supplied probability distribution p(a,b,c) in
    in the triangle scenario with no inputs [1] and tries to find an explicit 
    local model with given cardinalities for the hidden variables alpha, beta
    and gamma. If a probability distribution is not supplied, the function 
    considers the GHZ distribution mixed with the uniform distribution with
    visibility v = 0.33.
    
    References:
    
    [1]: RENOU, M.-O.; BÄUMER, E.; BOREIRI, S.; BRUNNER, N.; GISIN, N.;
    BEIGI, S. Genuine quantum nonlocality in the triangle network. Physical 
    review letters, APS, v. 123, n. 14, p. 140401, 2019.
    
    :param p: the behaviour to be optimized against (default=None)
    :type p: numpy.ndarray of floats and shape=(ma,mb,mc,Ma,Mb,Mc)
    
    :param ma: number of Alice's outputs (default=2)
    :type ma: integer
    
    :param mb: number of Bob's outputs (default=2)
    :type mb: integer
    
    :param mc: number of Charles's outputs (default=2)
    :type mc: integer
    
    :param c_alpha: cardinality of alpha (default=6)
    :type c_alpha: integer
    
    :param c_beta: cardinality of beta (default=6)
    :type c_beta: integer
    
    :param c_gamma: cardinality of gamma (default=6)
    :type c_gamma: integer
    
    :return: solution to the optimization problem
    :rtype: scipy.optimize.optimize.OptimizeResult
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
    dof = c_alpha+c_beta+c_gamma-3+c_alpha*c_beta*(mc-1)+c_beta*c_gamma*(ma-1)+c_alpha*c_gamma*(mb-1)
    rng = np.random.default_rng()
    bounds = opt.Bounds(np.zeros(dof), np.ones(dof))
    coeffs = np.zeros(shape=(3+c_beta*c_gamma+c_gamma*c_alpha+c_alpha*c_beta,dof))
    coeffs[0,0:c_alpha-1] = 1
    coeffs[1,c_alpha-1:c_alpha+c_beta-2] = 1
    coeffs[2,c_alpha+c_beta-2:c_alpha+c_beta+c_gamma-3] = 1
    coeffs[3:3+c_beta*c_gamma,c_alpha+c_beta+c_gamma-3:c_alpha+c_beta+c_gamma-3+(ma-1)*c_beta*c_gamma] = np.tile(np.eye(c_beta*c_gamma),ma-1)
    coeffs[3+c_beta*c_gamma:3+c_beta*c_gamma+c_gamma*c_alpha,c_alpha+c_beta+c_gamma-3+(ma-1)*c_beta*c_gamma:c_alpha+c_beta+c_gamma-3+(ma-1)*c_beta*c_gamma+(mb-1)*c_gamma*c_alpha] = np.tile(np.eye(c_gamma*c_alpha),mb-1)
    coeffs[3+c_beta*c_gamma+c_gamma*c_alpha:3+c_beta*c_gamma+c_gamma*c_alpha+c_alpha*c_beta,c_alpha+c_beta+c_gamma-3+(ma-1)*c_beta*c_gamma+(mb-1)*c_gamma*c_alpha:c_alpha+c_beta+c_gamma-3+(ma-1)*c_beta*c_gamma+(mb-1)*c_gamma*c_alpha+(mc-1)*c_alpha*c_beta] = np.tile(np.eye(c_alpha*c_beta),mc-1)
    linear_constraints = opt.LinearConstraint(coeffs, -np.inf*np.ones(3+c_beta*c_gamma+c_gamma*c_alpha+c_alpha*c_beta), np.ones(3+c_beta*c_gamma+c_gamma*c_alpha+c_alpha*c_beta))
    x0 = rng.random(size=dof)
    x0[0:c_alpha-1] = 1/c_alpha
    x0[c_alpha-1:c_alpha+c_beta-2] = 1/c_beta
    x0[c_alpha+c_beta-2:c_alpha+c_beta+c_gamma-3] = 1/c_gamma
    solution = opt.minimize(cost, x0, args=(p, ma, mb, mc, c_alpha, c_beta, c_gamma), 
                            method='trust-constr', 
                            constraints=linear_constraints, 
                            options={'verbose': 1}, bounds=bounds)
    return solution