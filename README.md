# localmodels

The modules bilocal.py and triangle.py provide tools to find explicit local
models in two network topologies: the bilocal scenario [1] and the triangle
scenario with no inputs [2].

See the two examples below to learn how use the modules.

## Bilocal scenario 



## Triangle scenario with no inputs

import numpy as np
from triangle import *

a = np.arange(0,2)
b = np.arange(0,2)
c = np.arange(0,2)
a, b, c = np.meshgrid(a,b,c,indexing='ij')
v = 0.2
pGHZ = 1/2*(a==b)*(b==c)
p0 = 1/8
p = v*pGHZ + (1-v)*p0
solution = triangle(p, c_alpha=3, c_beta=2, c_gamma=2)
solution = triangle(p, c_alpha=3, c_beta=2, c_gamma=2)
p_alpha, p_beta, p_gamma, p_a, p_b, p_c = modelo(solution.x, c_alpha=3, c_beta=2, c_gamma=2)
p_alpha, p_beta, p_gamma, p_a, p_b, p_c = model(solution.x, c_alpha=3, c_beta=2, c_gamma=2)
p_alpha
p_beta
p_gamma
print(p_alpha)

## References:

[1]: BRANCIARD, C.; ROSSET, D.; GISIN, N.; PIRONIO, S. Bilocal versus 
nonbilocal correlations in entanglement-swapping experiments. Physical 
Review A, APS, v. 85, n. 3, p. 032119, 2012.

[2]: RENOU, M.-O.; BÄUMER, E.; BOREIRI, S.; BRUNNER, N.; GISIN, N.;
BEIGI, S. Genuine quantum nonlocality in the triangle network. Physical 
review letters, APS, v. 123, n. 14, p. 140401, 2019.
