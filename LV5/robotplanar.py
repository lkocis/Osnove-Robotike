import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import vtk_visualizer as vis
import or_util as rob

class Robot():
	def __init__(self, m1, m2, l1, l2, b1, b2, r_tool, scene, T0S, gravity=False):
		self.m1 = m1
		self.m2 = m2
		self.l1 = l1
		self.l2 = l2
		self.b1 = b1
		self.b2 = b2
		self.J1 = m1 * l1 / 12
		self.J2 = m2 * l2 / 12
		self.T0S = T0S
		self.gravity = gravity

		q = np.zeros(2)
		d = np.array([0, 0])
		a = np.array([l1, l2])
		al = np.array([0, 0])
		self.DH = np.stack((q, d, a, al), 1)
	
		s = scene

		min_link_size = min(l1, l2)
		r_link = 0.05 * min_link_size
		r_joint = 0.1 * min_link_size
		h_joint = 0.1 * min_link_size
		self.c = 2.0 * r_tool
		
		# Joint 1.
		self.joint1 = vis.cylinder(r_joint, h_joint)
		s.add_actor(self.joint1)	

		# Link 1.
		self.link1 = vis.cylinder(r_link, l1)
		s.add_actor(self.link1)	

		# Joint 2.
		self.joint2 = vis.cylinder(r_joint, h_joint)
		s.add_actor(self.joint2)	

		# Link 2.
		self.link2 = vis.cylinder(r_link, l2 - self.c)
		s.add_actor(self.link2)		

		# Tool.
		self.tool = vis.sphere(r_tool)
		self.tool.GetProperty().SetColor(1.0, 0.0, 0.0)
		s.add_actor(self.tool)

	def set_configuration(self, q):
		d = self.DH[:,1]
		a = self.DH[:,2]
		al = self.DH[:,3]
		
		# Joint 1.
		T10 = rob.dh(q[0], d[0], a[0], al[0])
		T1S = self.T0S @ T10
		TJ11 = np.identity(4)
		TJ11[:3,:3] = rob.rotx(np.pi/2)
		TJ11[0,3] = -self.l1
		TJ1S = T1S @ TJ11
		vis.set_pose(self.joint1, TJ1S)

		# Link 1.
		TL11 = np.identity(4)
		TL11[:3,:3] = rob.rotz(np.pi/2)
		TL11[0,3] = -0.5 * self.l1
		TL1S = T1S @ TL11
		vis.set_pose(self.link1, TL1S)

		# Joint 2.
		T21 = rob.dh(q[1], d[1], a[1], al[1])
		T2S = T1S @ T21	
		TJ22 = np.identity(4)
		TJ22[:3,:3] = rob.rotx(np.pi/2)
		TJ22[0,3] = -self.l2
		TJ2S = T2S @ TJ22
		vis.set_pose(self.joint2, TJ2S)
		
		# Link 2.
		TL22 = np.identity(4)
		TL22[:3,:3] = rob.rotz(np.pi/2)
		TL22[0,3] = -0.5 * self.l2 - self.c
		TL2S = T2S @ TL22
		vis.set_pose(self.link2, TL2S)

		# Tool.
		vis.set_pose(self.tool, T2S)
		
		return T2S	

	def fwdkin(self, q):
		if q.ndim == 1:
			self.DH[:,0] = q
			return rob.fwdkin(self.DH)
		elif q.ndim == 2:
			traj_W = np.zeros((q.shape[0], 2))
			for i in range(q.shape[0]):
				self.DH[:,0] = q[i,:]
				traj_W[i,:] = rob.fwdkin(self.DH)[:2,3]
			return traj_W
	
	def invkin(self, T20, solution):
		q = np.zeros(2)
		a = self.DH[:,2]
		q[1] = np.arccos((T20[0,3]**2 + T20[1,3]**2 - a[0]**2 - a[1]**2) / (2.0 * a[0] * a[1]))
		if(solution == 1):
			q[1] = -q[1]
		q[0] = np.arctan2(T20[1,3], T20[0,3]) - np.arctan2(a[1] * np.sin(q[1]), a[0] + a[1] * np.cos(q[1]))

		return q
	
	def dynmodel(self, q, dq):
		m1 = self.m1
		m2 = self.m2
		l1 = self.l1
		l2 = self.l2
		b1 = self.b1
		b2 = self.b2
		J1 = self.J1
		J2 = self.J2

		D = np.array([[J1 + J2 + l1**2*m1/4 + l1**2*m2 + l1*l2*m2*np.cos(q[1]) + l2**2*m2/4, J2 + l2*m2*(2*l1*np.cos(q[1]) + l2)/4], [J2 + l2*m2*(2*l1*np.cos(q[1]) + l2)/4, J2 + l2**2*m2/4]])
		N = np.array([[dq[1]*l1*l2*m2*(-2*dq[0] - dq[1])*np.sin(q[1])/2], [dq[0]**2*l1*l2*m2*np.sin(q[1])/2]])
		if self.gravity:
			g = 9.81
		else:
			g = 0
		h = np.array([[g*(l1*m1*np.cos(q[0]) + m2*(2*l1*np.cos(q[0]) + l2*np.cos(q[0] + q[1])))/2], [g*l2*m2*np.cos(q[0] + q[1])/2]])
		B = np.array([[b1*dq[0]], [b2*dq[1]]])

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





