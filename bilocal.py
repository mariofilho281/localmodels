import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#number of inputs
Ma = 2
Mb = 2
Mc = 2

#number of outputs
ma=2
mb=2
mc=2

I = 0.25
J = 0.75

c_lambda = 2
c_mu = 4

A = np.arange(0,ma)
B = np.arange(0,mb)
C = np.arange(0,mc)
X = np.arange(0,Ma)
Y = np.arange(0,Mb)
Z = np.arange(0,Mc)
Am, Bm, Cm, Xm, Ym, Zm = np.meshgrid(A,B,C,X,Y,Z,indexing='ij')
pX = 1/8*(1+(Ym==0)*(-1)**(Am+Bm+Cm))
pY = 1/8*(1+(Ym==1)*(-1)**(Xm+Zm+Am+Bm+Cm))
p0 = 1/8
pX = 1/8*(1/2+(Am==0))*(1+(Ym==0)*(-1)**(Am+Bm+Cm))
pY = 1/8*(1/2+(Am==0))*(1+(Ym==1)*(-1)**(Zm+Am+Bm+Cm))
p0 = 1/8*(1/2+(Am==0))

p = I*pX + J*pY + (1-I-J)*p0

dof = 2*c_lambda*c_mu + 3*c_lambda + 3*c_mu - 2

def cost(x):
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
    #Array indices
    #p_lambda: lambda = i
    #p_mu: mu = j
    #p_a: a, x, lambda = kli
    #p_b: b, y, lambda, mu = mnij
    #p_c: c, z, mu = pqj
    #p_x: a, b, c, x, y, z = kmplnq
    px = np.einsum('i,j,kli,mnij,pqj->kmplnq',p_lambda,p_mu,p_a,p_b,p_c)
    d = px - p
    return np.sum(d**2)

rng = np.random.default_rng()
x0 = rng.random(size=dof)
#x0 = 1/2*np.ones(dof)
x0[0:c_lambda-1] = 1/c_lambda
x0[c_lambda-1:c_lambda+c_mu-2] = 1/c_mu
bounds = opt.Bounds(np.zeros(dof), np.ones(dof))
coeffs = np.zeros(shape=(2,dof))
coeffs[0,0:c_lambda-1] = 1
coeffs[1,c_lambda-1:c_lambda+c_mu-2] = 1
linear_constraints = opt.LinearConstraint(coeffs, -np.inf*np.ones(2), np.ones(2))
solution = opt.minimize(cost, x0, method='trust-constr',
                        constraints=linear_constraints,
                        options={'verbose': 1}, bounds=bounds)

print('I = '+str(I))
print('J = '+str(J))
print('c_lambda = '+str(c_lambda))
print('c_mu = '+str(c_mu))

def modelo(x):
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

p_lambda, p_mu, p_a, p_b, p_c = modelo(solution.x)

print('p_lambda:')
print(p_lambda)
print('p_mu:')
print(p_mu)
print('p_a:')
print(p_a[0,:,:])
print('p_b:')
print(p_b[0,:,:,:])
print('p_c:')
print(p_c[0,:,:])
print('erro quadrático médio nas probabilidades = '+str(np.sqrt(cost(solution.x)/64)))

def comportamento(x):
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
    #Array indices
    #p_lambda: lambda = i
    #p_mu: mu = j
    #p_a: a, x, lambda = kli
    #p_b: b, y, lambda, mu = mnij
    #p_c: c, z, mu = pqj
    #p_x: a, b, c, x, y, z = kmplnq
    px = np.einsum('i,j,kli,mnij,pqj->kmplnq',p_lambda,p_mu,p_a,p_b,p_c)
    return px

#print('comportamento final:')
#print(comportamento(solution.x))
#print('comportamento objetivo:')
#print(p)

#fig, ax = plt.subplots(2,2)
#ax[0,0].matshow(p_a,cmap='plasma')
#ax[0,0].set_xlabel(r'$\gamma$')
#ax[0,0].set_ylabel(r'$\beta$')
#ax[0,1].matshow(p_c.transpose(),cmap='plasma')
#ax[1,0].matshow(p_b.transpose(),cmap='plasma')
#ax[1,0].set_ylabel(r'$\alpha$')
#ax[1,1].matshow(p_b,cmap='plasma')
#fig.colorbar(cm.ScalarMappable(cmap='plasma'))
#plt.show()