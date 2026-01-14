import numpy as np
import sympy as sp

q1, q2, q3, l1, l2, l3, m1, m2, m3, J1, J2, J3 = sp.symbols('q1 q2 q3 l1 l2 l3 m1 m2 m3 J1 J2 J3')
q = sp.Matrix([q1, q2, q3])

c_1_1 = sp.Matrix([-l1/2, 0, 0, 1])
Dc1 = sp.diag(0, J1, J1)
c_2_2 = sp.Matrix([-l2/2, 0, 0, 1])
Dc2 = sp.diag(0, J2, J2)
c_3_3 = sp.Matrix([-l3/2, 0, 0, 1])
Dc3 = sp.diag(0, J3, J3)

c1=sp.cos(q1)
s1=sp.sin(q1)
c2=sp.cos(q2)
s2=sp.sin(q2)
c3=sp.cos(q3)
s3=sp.sin(q3)

T_1_0 = sp.Matrix([
	[c1, -s1, 0, l1*c1],
	[s1, c1, 0, l1*s1], 
	[0, 0, 1, 0], 
	[0, 0, 0, 1]])

T_2_1 = sp.Matrix([
	[c2, -s2, 0, l2*c2],
	[s2, c2, 0, l2*s2], 
	[0, 0, 1, 0], 
	[0, 0, 0, 1]])

T_3_2 = sp.Matrix([
	[c3, -s3, 0, l3*c3],
	[s3, c3, 0, l3*s3], 
	[0, 0, 1, 0], 
	[0, 0, 0, 1]])

c_1_0 = T_1_0 * c_1_1
D1 = sp.simplify(T_1_0[:3,:3] * Dc1 * T_1_0[:3,:3].T)
A1 = sp.zeros(3, 3)
A1[:,0] = sp.diff(c_1_0, q1)[:3]
A1=sp.simplify(A1)
B1 = sp.Matrix([
	[0, 0, 0], 
	[0, 0, 0], 
	[1, 0, 0]]) 
#dim(B) = 3xn, n - br. zglobova
D = sp.simplify(A1.T * A1 * m1 + B1.T * D1 * B1)

T_2_0 = T_1_0 * T_2_1
c_2_0 = T_2_0 * c_2_2
D2 = sp.simplify(T_2_0[:3,:3] * Dc2 * T_2_0[:3,:3].T)
A2 = sp.zeros(3,3)      
A2[:,0] = sp.diff(c_2_0, q1)[:3]
A2[:,1] = sp.diff(c_2_0, q2)[:3]
B2 = B1
B2[:,1] = T_1_0[:3,2]
D = sp.simplify(D + sp.simplify(A2.T * A2 * m2 + B2.T * D2 * B2))

T_3_0 = T_2_0 * T_3_2
c_3_0 = T_3_0 * c_3_3
D3 = sp.simplify(T_3_0[:3,:3] * Dc3 * T_3_0[:3,:3].T)
A3 = sp.simplify(sp.diff(c_3_0, q1))
A3 = A3.row_join(sp.simplify(sp.diff(c_3_0, q2)))
A3 = A3.row_join(sp.simplify(sp.diff(c_3_0, q3)))
B3 = sp.zeros(3,3)
B3[:,0] = sp.Matrix([0,0,1])       
B3[:,1] = T_1_0[:3,2]              
B3[:,2] = T_2_0[:3,2]  
D = sp.simplify(D + sp.simplify(A3.T * A3 * m3 + B3.T * D3 * B3))

print('D=')
sp.pprint(D)

dq1, dq2, dq3 = sp.symbols('dq1 dq2 dq3')
dq = sp.Matrix([dq1, dq2, dq3])
N = sp.zeros(3,1)
for i in range(3):	
	C = sp.zeros(3,3)
	for j in range(3):
		for k in range(3):
			C[k,j] = sp.simplify(sp.diff(D[i,j], q[k]) - sp.diff(D[k,j], q[i]) / 2)
	N[i] = sp.simplify(dq.T * C[:,:] * dq)

print('N=')
sp.pprint(N)

g = sp.symbols('g')
g = sp.Matrix([0, -g, 0])
h = sp.zeros(3,1)
for i in range(3):	
	for k in range(3):
		h[i] = h[i] - g[k] * (m1 * A1[k,i] + m2 * A2[k,i] + m3 * A3[k,i])
	h[i] = sp.simplify(h[i])

print('h=')
sp.pprint(h)

print('D=', D)
print('N=', N)
print('h=', h)




