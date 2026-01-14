import numpy as np
import sympy as sp

q1, q2, l1, l2, m1, m2, J1, J2 = sp.symbols('q1 q2 l1 l2 m1 m2 J1 J2')
q = sp.Matrix([q1, q2])

c_1_1 = sp.Matrix([-l1/2, 0, 0, 1])
Dc1 = sp.diag(0, J1, J1)
c_2_2 = sp.Matrix([-l2/2, 0, 0, 1])
Dc2 = sp.diag(0, J2, J2)

c1=sp.cos(q1)
s1=sp.sin(q1)
c2=sp.cos(q2)
s2=sp.sin(q2)

T_1_0 = sp.Matrix([[c1, -s1, 0, l1*c1],[s1, c1, 0, l1*s1], [0, 0, 1, 0], [0, 0, 0, 1]])
T_2_1 = sp.Matrix([[c2, -s2, 0, l2*c2],[s2, c2, 0, l2*s2], [0, 0, 1, 0], [0, 0, 0, 1]])

c_1_0 = T_1_0 * c_1_1
D1 = sp.simplify(T_1_0[:3,:3] * Dc1 * T_1_0[:3,:3].T)
A1 = sp.zeros(3, 2)
A1[:,0] = sp.diff(c_1_0, q1)[:3]
A1=sp.simplify(A1)
B1 = sp.Matrix([[0, 0], [0, 0], [1, 0]])
D = sp.simplify(A1.T * A1 * m1 + B1.T * D1 * B1)

T_2_0 = T_1_0 * T_2_1
c_2_0 = T_2_0 * c_2_2
D2 = sp.simplify(T_2_0[:3,:3] * Dc2 * T_2_0[:3,:3].T)
A2 = sp.simplify(sp.diff(c_2_0, q1))
A2 = A2.row_join(sp.simplify(sp.diff(c_2_0, q2)))
B2 = B1
B2[:,1] = T_1_0[:3,2]

D = sp.simplify(D + sp.simplify(A2.T * A2 * m2 + B2.T * D2 * B2))

print('D=')
sp.pprint(D)

dq1, dq2 = sp.symbols('dq1 dq2')
dq = sp.Matrix([dq1, dq2])
N = sp.zeros(2,1)
for i in range(2):	
	C = sp.zeros(2,2)
	for j in range(2):
		for k in range(2):
			C[k,j] = sp.simplify(sp.diff(D[i,j], q[k]) - sp.diff(D[k,j], q[i]) / 2)
	N[i] = sp.simplify(dq.T * C[:,:] * dq)

print('N=')
sp.pprint(N)

g = sp.symbols('g')
g = sp.Matrix([0, -g])
h = sp.zeros(2,1)
for i in range(2):	
	for k in range(2):
		h[i] = h[i] - g[k] * (m1 * A1[k,i] + m2 * A2[k,i])
	h[i] = sp.simplify(h[i])

print('h=')
sp.pprint(h)

print('D=', D)
print('N=', N)
print('h=', h)




