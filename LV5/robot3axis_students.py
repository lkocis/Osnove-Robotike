import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from LV5.dynplanar_LE import J1
import vtk_visualizer as vis
import or_util as rob

class Robot():
	def __init__(self, m2, m3, l2, l3, b1, b2, b3, r_tool, scene, T0S, gravity=False):
		self.m2 = m2
		self.m3 = m3
		self.l2 = l2
		self.l3 = l3
		self.b1 = b1
		self.b2 = b2
		self.b3 = b3
		self.J2 = m2 * l2 / 12
		self.J3 = m3 * l3 / 12
		self.T0S = T0S
		self.gravity = gravity

		q = np.zeros(3)
		d = np.array([0, 0, 0])
		a = np.array([0, l2, l3])
		al = np.array([np.pi/2, 0, 0])
		self.DH = np.stack((q, d, a, al), 1)
	
		s = scene

		min_link_size = min(l2, l3)
		r_link = 0.05 * min_link_size
		r_joint = 1.5 * r_link
		h_joint = 3.0 * r_link
		self.c = 2.0 * r_tool

		# Joint 1.
		self.joint1 = vis.cylinder(r_joint, h_joint)
		s.add_actor(self.joint1)	
	
		# Joint 2.
		self.joint2 = vis.cylinder(r_joint, h_joint)
		s.add_actor(self.joint2)	

		# Link 2.
		self.link2 = vis.cylinder(r_link, l2)
		s.add_actor(self.link2)	

		# Joint 3.
		self.joint3 = vis.cylinder(r_joint, h_joint)
		s.add_actor(self.joint3)	

		# Link 3.
		self.link3 = vis.cylinder(r_link, l3 - self.c)
		s.add_actor(self.link3)		

		# Tool.
		self.tool = vis.sphere(r_tool)
		self.tool.GetProperty().SetColor(1.0, 0.0, 0.0)
		s.add_actor(self.tool)

	def set_configuration(self, q):
		d = self.DH[:,1]
		a = self.DH[:,2]
		al = self.DH[:,3]
		
		# Joint 1.
		TJ10 = np.identity(4)
		TJ10[:3,:3] = rob.rotx(np.pi/2)
		TJ1S = self.T0S @ TJ10		
		vis.set_pose(self.joint1, TJ1S)
		T10 = rob.dh(q[0], d[0], a[0], al[0])
		T1S = self.T0S @ T10

		# Joint 2.
		TJ21 = np.identity(4)
		TJ21[:3,:3] = rob.rotx(np.pi/2)
		TJ2S = T1S @ TJ21
		vis.set_pose(self.joint2, TJ2S)
		T21 = rob.dh(q[1], d[1], a[1], al[1])
		T2S = T1S @ T21	
		
		# Link 2.
		TL22 = np.identity(4)
		TL22[:3,:3] = rob.rotz(np.pi/2)
		TL22[0,3] = -0.5 * self.l2
		TL2S = T2S @ TL22
		vis.set_pose(self.link2, TL2S)

		# Joint 3.
		TJ32 = np.identity(4)
		TJ32[:3,:3] = rob.rotx(np.pi/2)
		TJ3S = T2S @ TJ32
		vis.set_pose(self.joint3, TJ3S)
		T32 = rob.dh(q[2], d[2], a[2], al[2])
		T3S = T2S @ T32	
		
		# Link 3.
		TL33 = np.identity(4)
		TL33[:3,:3] = rob.rotz(np.pi/2)
		TL33[0,3] = -0.5 * self.l3 - self.c
		TL3S = T3S @ TL33
		vis.set_pose(self.link3, TL3S)		

		# Tool.
		vis.set_pose(self.tool, T3S)
		
		return T3S	

	def fwdkin(self, q):
		if q.ndim == 1:
			self.DH[:,0] = q
			return rob.fwdkin(self.DH)
		elif q.ndim == 2:
			traj_W = np.zeros((q.shape[0], 3))
			for i in range(q.shape[0]):
				self.DH[:,0] = q[i,:]
				traj_W[i,:] = rob.fwdkin(self.DH)[:3,3]
			return traj_W
	
	def invkin(self, T30, solution):
		q = np.zeros(3)
		a = self.DH[1:,2]	
		p = np.sqrt(T30[0,3]**2 + T30[1,3]**2)
		z = T30[2,3]
		q[0] = np.atan2(T30[1,3], T30[0,3])
		q[2] = np.arccos((p**2 + z**2 - a[0]**2 - a[1]**2) / (2.0 * a[0] * a[1]))
		if(solution == 1):
			q[2] = -q[2]
		q[1] = np.arctan2(z, p) - np.arctan2(a[1] * np.sin(q[2]), a[0] + a[1] * np.cos(q[2]))

		return q
	
	def dynmodel(self, q, dq):
		m1 = self.m1
		m2 = self.m2
		m3 = self.m3
		l1 = self.l1
		l2 = self.l2
		l3 = self.l3
		b1 = self.b1
		b2 = self.b2
		b3 = self.b3
		J1 = self.J1
		J2 = self.J2
		J3 = self.J3

		q1 = q[0]
		q2 = q[1]
		q3 = q[2]
		dq1 = dq[0]
		dq2 = dq[1]
		dq3 = dq[2]

		#ADD YOUR CODE HERE:************
		D= np.array([[J1 + J2 + J3 + l1**2*m1/4 + l1**2*m2 + l1*l2*m2*np.cos(q(2)) + l2**2*m2/4, m2*(4*l1**2 + 4*l1*l2*np.cos(q(2)) + l2**2)/4, l2*m2*(2*l1*np.cos(q(2)) + l2)/4], [m2*(4*l1**2 + 4*l1*l2*cos(q2) + l2**2)/4, J2*l1**2*sin(q2)**2 + J3*(l1*sin(q1) + l2*sin(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*cos(q2 + q3)**2 - (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3))/2 - J3*(l1*cos(q1) + l2*cos(q1 + q2))*((l1*sin(q1) + l2*sin(q1 + q2))*sin(2*q2 + 2*q3) - 2*(l1*cos(q1) + l2*cos(q1 + q2))*sin(q2 + q3)**2)/2 + l1**2*m2 + l1*l2*m2*cos(q2) + l2**2*m2/4 + m3*(4*l2**2 + 4*l2*l3*cos(q3) + l3**2)/4, l2*m2*(2*l1*cos(q2) + l2)/4 + l3*m3*(2*l2*cos(q3) + l3)/4], [l2*m2*(2*l1*cos(q2) + l2)/4, l2*m2*(2*l1*cos(q2) + l2)/4 + l3*m3*(2*l2*cos(q3) + l3)/4, l2**2*m2/4 + l3**2*m3/4]])
		
		N= np.array([[dq2*(-4*dq1*l1*l2*m2*sin(q2) + dq2*(-J3*(l1*sin(q1) + l2*sin(q1 + q2))*((l1*sin(q1) + l2*sin(q1 + q2))*sin(2*q2 + 2*q3) - 2*(l1*cos(q1) + l2*cos(q1 + q2))*sin(q2 + q3)**2) - J3*(l1*sin(q1) + l2*sin(q1 + q2))*((l1*sin(q1) + l2*sin(q1 + q2))*sin(2*q2 + 2*q3) + 2*(l1*cos(q1) + l2*cos(q1 + q2))*cos(q2 + q3)**2) + J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*sin(q2 + q3)**2 + (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) - J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*cos(q2 + q3)**2 - (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) - 4*l1*l2*m2*sin(q2)) - 2*dq3*l1*l2*m2*sin(q2))/4], [dq1*l1*l2*m2*(2*dq1 - 2*dq2 + dq3)*sin(q2)/4 + dq2*(2*dq1*(J3*(l1*sin(q1) + l2*sin(q1 + q2))*((l1*sin(q1) + l2*sin(q1 + q2))*sin(2*q2 + 2*q3) - 2*(l1*cos(q1) + l2*cos(q1 + q2))*sin(q2 + q3)**2) + J3*(l1*sin(q1) + l2*sin(q1 + q2))*((l1*sin(q1) + l2*sin(q1 + q2))*sin(2*q2 + 2*q3) + 2*(l1*cos(q1) + l2*cos(q1 + q2))*cos(q2 + q3)**2) - J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*sin(q2 + q3)**2 + (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) + J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*cos(q2 + q3)**2 - (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) + l1*l2*m2*sin(q2)) + dq2*(2*J2*l1**2*sin(2*q2) + 2*J3*l1**2*sin(-2*q1 + 2*q2 + 2*q3) + J3*l1*l2*sin(q1)*sin(q1 + q2)*sin(2*q2 + 2*q3) + 2*J3*l1*l2*sin(q1)*cos(q1 + q2)*cos(q2 + q3)**2 - J3*l1*l2*sin(q2) - 2*J3*l1*l2*sin(q1 + q2)*sin(q2 + q3)*(l1*cos(q1) + l2*cos(q1 + q2))*sin(q2 + q3)**2) - J3*(l1*sin(q1) + l2*sin(q1 + q2))*((l1*sin(q1) + l2*sin(q1 + q2))*sin(2*q2 + 2*q3) + 2*(l1*cos(q1) + l2*cos(q1 + q2))*cos(q2 + q3)**2) + J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*sin(q2 + q3)**2 + (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) - J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*cos(q2 + q3)**2 - (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) - 4*l1*l2*m2*sin(q2)) - 2*dq3*l1*l2*m2*sin(q2))/4], [dq1*l1*l2*m2*(2*dq1 - 2*dq2 + dq3)*sin(q2)/4 + dq2*(2*dq1*(J3*(l1*sin(q1) + l2*sin(q1 + q2))*((l1*sin(q1) + l2*sin(q1 + q2))*sin(2*q2 + 2*q3) - 2*(l1*cos(q1) + l2*cos(q1 + q2))*sin(q2 + q3)**2) + J3*(l1*sin(q1) + l2*sin(q1 + q2))*((l1*sin(q1) + l2*sin(q1 + q2))*sin(2*q2 + 2*q3) + 2*(l1*cos(q1) + l2*cos(q1 + q2))*cos(q2 + q3)**2) - J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*sin(q2 + q3)**2 + (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) + J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*cos(q2 + q3)**2 - (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) + l1*l2*m2*sin(q2)) + dq2*(2*J2*l1**2*sin(2*q2) + 2*J3*l1**2*sin(-2*q1 + 2*q2 + 2*q3) + J3*l1*l2*sin(q1)*sin(q1 + q2)*sin(2*q2 + 2*q3) + 2*J3*l1*l2*sin(q1)*cos(q1 + q2)*cos(q2 + q3)**2 - J3*l1*l2*sin(q2) - 2*J3*l1*l2*sin(q1 + q2)*sin(q2 + q3)*- (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) - 4*l1*l2*m2*sin(q2)) - 2*dq3*l1*l2*m2*sin(q2))/4], [dq1*l1*l2*m2*(2*dq1 - 2*dq2 + dq3)*sin(q2)/4 + dq2*(2*dq1*(J3*(l1*sin(q1) + l2*sin(q1 + q2))*((l1*sin(q1) + l2*sin(q1 + q2))*sin(2*q2 + 2*q3) - 2*(l1*cos(q1) + l2*cos(q1 + q2))*sin(q2 + q3)**2) + J3*(l1*sin(q1) + l2*sin(q1 + q2))*((l1*sin(q1) + l2*sin(q1 + q2))*sin(2*q2 + 2*q3) + 2*(l1*cos(q1) + l2*cos(q1 + q2))*cos(q2 + q3)**2) - J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*sin(q2 + q3)**2 + (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) + J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*cos(q2 + q3)**2 - (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) + l1*l2*m2*sin(q2)) + dq2*(2*J2*l1**2*sin(2*q2) + 2*J3*l1**2*sin(-2*q1 + 2*q2 + 2*q3) + J3*l1*l2*sin(q1)*sin(q1 + q2)*sin(2*q2 + 2*q3) + 2*J3*l1*l2*sin(q1)*cos(q1 + q2)*cos(q2 + q3)**2 - J3*l1*l2*sin(q2) - 2*J3*l1*l2*sin(q1 + q2)*sin(q2 + q3)*1 + q2))*sin(q2 + q3)**2) + J3*(l1*sin(q1) + l2*sin(q1 + q2))*((l1*sin(q1) + l2*sin(q1 + q2))*sin(2*q2 + 2*q3) + 2*(l1*cos(q1) + l2*cos(q1 + q2))*cos(q2 + q3)**2) - J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*sin(q2 + q3)**2 + (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) + J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*cos(q2 + q3)**2 - (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) + l1*l2*m2*sin(q2)) + dq2*(2*J2*l1**2*sin(2*q2) + 2*J3*l1**2*sin(-2*q1 + 2*q2 + 2*q3) + J3*l1*l2*sin(q1)*sin(q1 + q2)*sin(2*q2 + 2*q3) + 2*J3*l1*l2*sin(q1)*cos(q1 + q2)*cos(q2 + q3)**2 - J3*l1*l2*sin(q2) - 2*J3*l1*l2*sin(q1 + q2)*sin(q2 + q3)*q1 + q2))*sin(2*q2 + 2*q3)) + J3*(l1*cos(q1) + l2*cos(q1 + q2))*(2*(l1*sin(q1) + l2*sin(q1 + q2))*cos(q2 + q3)**2 - (l1*cos(q1) + l2*cos(q1 + q2))*sin(2*q2 + 2*q3)) + l1*l2*m2*sin(q2)) + dq2*(2*J2*l1**2*sin(2*q2) + 2*J3*l1**2*sin(-2*q1 + 2*q2 + 2*q3) + J3*l1*l2*sin(q1)*sin(q1 + q2)*sin(2*q2 + 2*q3) + 2*J3*l1*l2*sin(q1)*cos(q1 + q2)*cos(q2 + q3)**2 - J3*l1*l2*sin(q2) - 2*J3*l1*l2*sin(q1 + q2)*sin(q2 + q3)**2*cos(q1) - J3*l1*l2*sin(2*q2 + 2*q3)*cos(q1)*cos(q1 + q2) + 3*J3*l1*l2*sin(-2*q1 + q2 + 2*q3) + J3*l2**2*sin(q1 + q2)**2*sin(2*q2 + 2*q3) - 2*J3*l2**2*sin(q1 + q2)*sin(q2 + q3)**2*cos(q1 + q2) + 2*J3*l2**2*sin(q1 + q2)*cos(q1 + q2)*cos(q2 + q3)**2 - J3*l2**2*sin(2*q1 - n(q1 + q2)*sin(2*q2 + 2*q3) + 2*J3*l1*l2*sin(q1)*cos(q1 + q2)*cos(q2 + q3)**2 - J3*l1*l2*sin(q2) - 2*J3*l1*l2*sin(q1 + q2)*sin(q2 + q3)**2*cos(q1) - J3*l1*l2*sin(2*q2 + 2*q3)*cos(q1)*cos(q1 + q2) + 3*J3*l1*l2*sin(-2*q1 + q2 + 2*q3) + J3*l2**2*sin(q1 + q2)**2*sin(2*q2 + 2*q3) - 2*J3*l2**2*sin(q1 + q2)*sin(q2 + q3)**2*cos(q1 + q2) + 2*J3*l2**2*sin(q1 + q2)*cos(q1 + q2)*cos(q2 + q3)**2 - J3*l2**2*sin(2*q1 - *2*cos(q1) - J3*l1*l2*sin(2*q2 + 2*q3)*cos(q1)*cos(q1 + q2) + 3*J3*l1*l2*sin(-2*q1 + q2 + 2*q3) + J3*l2**2*sin(q1 + q2)**2*sin(2*q2 + 2*q3) - 2*J3*l2**2*sin(q1 + q2)*sin(q2 + q3)**2*cos(q1 + q2) + 2*J3*l2**2*sin(q1 + q2)*cos(q1 + q2)*cos(q2 + q3)**2 - J3*l2**2*sin(2*q1 - 2*q3) - J3*l2**2*sin(2*q2 + 2*q3)*cos(q1 + q2)**2 - 2*l1*l2*m2*sin(q2)) + dq3*(4*J3*l1**2*sin(-2*q1 + 2*q2 + 2*q3) + 8*J3*l1*l2*sin(-2*qq3) - 2*J3*l2**2*sin(q1 + q2)*sin(q2 + q3)**2*cos(q1 + q2) + 2*J3*l2**2*sin(q1 + q2)*cos(q1 + q2)*cos(q2 + q3)**2 - J3*l2**2*sin(2*q1 - 2*q3) - J3*l2**2*sin(2*q2 + 2*q3)*cos(q1 + q2)**2 - 2*l1*l2*m2*sin(q2)) + dq3*(4*J3*l1**2*sin(-2*q1 + 2*q2 + 2*q3) + 8*J3*l1*l2*sin(-2*q2*q3) - J3*l2**2*sin(2*q2 + 2*q3)*cos(q1 + q2)**2 - 2*l1*l2*m2*sin(q2)) + dq3*(4*J3*l1**2*sin(-2*q1 + 2*q2 + 2*q3) + 8*J3*l1*l2*sin(-2*q1 + q2 + 2*q3) - 4*J3*l2**2*sin(2*q1 - 2*q3) + l1*l2*m2*sin(q2) - 4*l2*l3*m3*sin(q3)))/4 - dq3*l2*(-dq1*l1*m2*sin(q2) + dq2*l1*m2*sin(q2) + 2*dq3*l3*m3*sin(q3))/4], [dq2*(-dq1*l1*l2*m2*sin(q2) - dq2*(J3*l1**2*sin(-2*q1 + 2*q2 + 2*q3) + 2*J3*l1*l2*sin(-2*q1 + q2 + 2*q3) - J3*l2**2*sin(2*q1 - 2*q3) + l1*l2*m2*sin(q2) - l2*l3*m3*sin(q3)))/2]])J3*l2**2*sin(2*q1 - 2*q3) + l1*l2*m2*sin(q2) - l2*l3*m3*sin(q3)))/2]])
		# if self.gravity:
		# 	g = 9.81
		# else:
		# 	g = 0

		# h = ...
		# B = ...
		#END*****************************

		return D, N, h, B
	
	def acc(self, q, dq, tau):
		D, N, h, B = self.dynmodel(q, dq)
		return np.linalg.inv(D) @ (tau - N - h - B)
	
class simulator():
	def __init__(self, robot, Q):
		self.timer_count = 0
		self.robot = robot
		self.Q = Q
		self.trajW = []
		self.T0S = np.identity(4)
	
	def execute(self,iren,event):
		T2S = self.robot.set_configuration(self.Q[:,self.timer_count % self.Q.shape[1]], self.T0S)
		self.trajW.append(T2S)
		iren.GetRenderWindow().Render()
		self.timer_count += 1





