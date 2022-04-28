# localmodels

The modules bilocal.py and triangle.py provide tools to find explicit local
models in two network topologies: the bilocal scenario [1] and the triangle
scenario with no inputs [2].

See the two examples below to learn how use the modules.

## Bilocal scenario 

This example tries to find a local model for the (I,J) distribution found in 
figure 3 of reference [1] in the bilocal scenario with binary inputs and 
outputs.

```python
import numpy as np
from bilocal import *

a = np.arange(0,2)
b = np.arange(0,2)
c = np.arange(0,2)
x = np.arange(0,2)
y = np.arange(0,2)
z = np.arange(0,2)
a, b, c, x, y, z = np.meshgrid(a,b,c,x,y,z,indexing='ij')
I = 0
J = 1
pI = 1/8*(1+(y==0)*(-1)**(a+b+c))
pJ = 1/8*(1+(y==1)*(-1)**(x+z+a+b+c))
p0 = 1/8
p = I*pI + J*pJ + (1-I-J)*p0
solution = bilocal(p, Ma=2, Mb=2, Mc=2, ma=2, mb=2, mc=2, c_lambda=2, c_mu=2)
p_lambda, p_mu, p_a, p_b, p_c = model(solution.x, c_lambda=2, c_mu=2)
```

After running the code above, the probability distributions of the hidden
variables should be in the variables ``p_lambda``, ``p_mu``, and the response
functions of Alice, Bob and Charles should be in the variables ``p_a``, 
``p_b``, ``p_c``. The indexing of these variables works as follows:

``p_lambda[i]`` is the probability that lambda assumes the value ``i``.

``p_mu[i]`` is the probability that mu assumes the value ``i``.

``p_a[i,j,k]`` is the probability that Alice outputs value ``i``, given that
she receives x=``j`` and lambda=``k``.

``p_b[i,j,k,l]`` is the probability that Bob outputs value ``i``, given that
he receives y=``j``, lambda=``k`` and mu=``l``.

``p_c[i,j,k]`` is the probability that Charles outputs value ``i``, given that
he receives z=``j`` and mu=``k``.

## Triangle scenario with no inputs

This example tries to find a local model for the GHZ distribution mixed with
a uniform distribution with visibility v = 0.33 in the triangle scenario with no
inputs and binary outputs.

```python
import numpy as np
from triangle import *

a = np.arange(0,2)
b = np.arange(0,2)
c = np.arange(0,2)
a, b, c = np.meshgrid(a,b,c,indexing='ij')
v = 0.33
pGHZ = 1/2*(a==b)*(b==c)
p0 = 1/8
p = v*pGHZ + (1-v)*p0
solution = triangle(p, ma=2, mb= 2, mc=2, c_alpha=3, c_beta=2, c_gamma=2)
p_alpha, p_beta, p_gamma, p_a, p_b, p_c = model(solution.x, c_alpha=3, c_beta=2, c_gamma=2)
```

After running the code above, the probability distributions of the hidden
variables should be in the variables ``p_alpha``, ``p_beta``, ``p_gamma``, and
the response functions of Alice, Bob and Charles should be in the variables
``p_a``, ``p_b``, ``p_c``. The indexing of these variables works as follows:

``p_alpha[i]`` is the probability that alpha assumes the value ``i``.

``p_beta[i]`` is the probability that beta assumes the value ``i``.

``p_gamma[i]`` is the probability that gamma assumes the value ``i``.

``p_a[i,j,k]`` is the probability that Alice outputs value ``i``, given that
she receives beta=``j`` and gamma=``k``.

``p_b[i,j,k]`` is the probability that Bob outputs value ``i``, given that
he receives gamma=``j``, alpha=``k``.

``p_c[i,j,k]`` is the probability that Charles outputs value ``i``, given that
he receives alpha=``j`` and beta=``k``.

## References:

[1]: BRANCIARD, C.; ROSSET, D.; GISIN, N.; PIRONIO, S. Bilocal versus 
nonbilocal correlations in entanglement-swapping experiments. Physical 
Review A, APS, v. 85, n. 3, p. 032119, 2012.

[2]: RENOU, M.-O.; BÄUMER, E.; BOREIRI, S.; BRUNNER, N.; GISIN, N.;
BEIGI, S. Genuine quantum nonlocality in the triangle network. Physical 
review letters, APS, v. 123, n. 14, p. 140401, 2019.
