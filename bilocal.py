import numpy as np
import scipy.optimize as opt

def model(x, c_lambda=4, c_mu=4):
    """
    Assembles hidden variables probability distributions and response functions
    
    This function takes the ``x`` attribute of the solution for the 
    optimization problem solved in the function ``bilocal`` and extract the 
    model to a more readble format.
    
    :param x: solution of optimization problem solved by ``bilocal``
    :type x: numpy.ndarray of floats
    
    :param c_lambda: cardinality of lambda (default=4)
    :type c_lambda: integer
    
    :param c_mu: cardinality of mu (default=4)
    :type c_mu: integer
    
    :return: 
        the hidden variables probability distributions ``p_lambda``, ``p_mu`` 
        and the response functions ``p_a``, ``p_b``, ``p_c`` of Alice, Bob and
        Charles
    :rtype: tuple
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

def behaviour(x, c_lambda=4, c_mu=4):
    """
    Calculates the behaviour of a local model in the bilocal scenario
    
    This function takes the ``x`` attribute of the solution for the 
    optimization problem solved in the function ``bilocal`` and calculates the 
    probability distribution p(a,b,c|x,y,z) that it reproduces.
    
    :param x: solution of optimization problem solved by ``bilocal``
    :type x: numpy.ndarray of floats
    
    :param c_lambda: cardinality of lambda (default=4)
    :type c_lambda: integer
    
    :param c_mu: cardinality of mu (default=4)
    :type c_mu: integer
    
    :return: 
        the probability distribution p(a,b,c|x,y,z) indexed in the usual order,
        i.e. p[a,b,c,x,y,z]
    :rtype: numpy.ndarray of floats
    """
    p_lambda, p_mu, p_a, p_b, p_c = model(x, c_lambda, c_mu)
    #Array indices for np.einsum:
    #   p_lambda: lambda -> i
    #   p_mu: mu -> j
    #   p_a: a, x, lambda -> kli
    #   p_b: b, y, lambda, mu -> mnij
    #   p_c: c, z, mu -> pqj
    #   px: a, b, c, x, y, z -> kmplnq
    px = np.einsum('i,j,kli,mnij,pqj->kmplnq',p_lambda,p_mu,p_a,p_b,p_c)
    return px

def cost(x, p, c_lambda=4, c_mu=4):
    """
    Calculates the sum of squared errors between behaviour p and model x
    
    This function calculates the cost that is optimized in function 
    ``bilocal``. Its minimum value of zero is attained when the model
    represented by the solution ``x`` represents exactly all the probabilities 
    of behaviour ``p``.
    
    :param x: array that represents an explicit bilocal model
    :type x: numpy.ndarray of floats
    
    :param p: the behaviour to be optimized against
    :type p: numpy.ndarray of floats
    
    :param c_lambda: cardinality of lambda (default=4)
    :type c_lambda: integer
    
    :param c_mu: cardinality of mu (default=4)
    :type c_mu: integer
    
    :return: 
        sum of squared errors between probabilities ``p`` and those generated
        by ``x``
    :rtype: numpy.ndarray of floats
    """
    px = behaviour(x, c_lambda, c_mu)
    return np.sum((px-p)**2)


def bilocal(p=None, Ma=2, Mb=2, Mc=2, ma=2, mb=2, mc=2, c_lambda=4, c_mu=4):
    """
    Tries to find a local model for behaviour p in the bilocal scenario.
    
    This function takes a user supplied probability distribution p(a,b,c|x,y,z)
    in the bilocal scenario [1] and tries to find an explicit local model with
    given cardinalities for the hidden variables lambda and mu. If a
    probability distribution is not supplied, the function considers the
    distribution pJ in footnote 17 of [1].
    
    References:
    
    [1]: BRANCIARD, C.; ROSSET, D.; GISIN, N.; PIRONIO, S. Bilocal versus 
    nonbilocal correlations in entanglement-swapping experiments. Physical 
    Review A, APS, v. 85, n. 3, p. 032119, 2012.
    
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