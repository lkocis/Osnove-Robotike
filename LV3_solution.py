import sys
import numpy as np
import vtk
import matplotlib.pyplot as plt
import vtk_visualizer as vis
#import br_lectures as br
from hocook import hocook
from planecontact import planecontact
from mobrobsim import mobrobsimanimate, set_goal, set_map
from scipy import ndimage
from PIL import Image
from camerasim import CameraSimulator
#from skimage import feature
#from skimage.transform import hough_line, hough_line_peaks

# TASK 0
class tool():
	def __init__(self, scene):
		s = scene	
		self.finger1 = vis.cube(0.02, 0.01, 0.05)
		s.add_actor(self.finger1)
		self.finger2 = vis.cube(0.02, 0.01, 0.05)
		s.add_actor(self.finger2)
		self.palm = vis.cube(0.03, 0.08, 0.03)
		s.add_actor(self.palm)
		self.wrist = vis.cylinder(0.015, 0.04)
		s.add_actor(self.wrist)
		
	def set_configuration(self, g, TGS):	
		TF1G = np.identity(4)
		TF1G[:3,3] = np.array([0, -0.5*g-0.005, -0.025])
		TF1S = TGS @ TF1G
		vis.set_pose(self.finger1, TF1S)
		TF2G = np.identity(4)
		TF2G[:3,3] = np.array([0, 0.5*g+0.005, -0.025])	
		TF2S = TGS @ TF2G
		vis.set_pose(self.finger2, TF2S)
		TPG = np.identity(4)
		TPG[:3,3] = np.array([0, 0, -0.065])
		TPS = TGS @ TPG
		vis.set_pose(self.palm, TPS)
		TWG = np.block([[rotx(np.pi/2), np.array([[0], [0], [-0.1]])], [np.zeros((1, 3)), 1]])
		TWS = TGS @ TWG
		vis.set_pose(self.wrist, TWS)

def rotx(q):
	c = np.cos(q)
	s = np.sin(q)
	return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
	
def roty(q):
	c = np.cos(q)
	s = np.sin(q)
	return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(q):
	c = np.cos(q)
	s = np.sin(q)
	return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def set_floor(s, size):
	floor = vis.cube(size[0], size[1], 0.01)
	TFS = np.identity(4)
	TFS[2,3] = -0.005
	vis.set_pose(floor, TFS)
	s.add_actor(floor)

def task0():
	# Scene
	s = vis.visualizer()

	# Floor.
	set_floor(s, [1, 1])

	# Cube.
	cube = vis.cube(0.03, 0.03, 0.03)
	TCS = np.identity(4)
	TCS[:3,3] = np.array([0, 0, 1.5])
	vis.set_pose(cube, TCS)
	s.add_actor(cube)
	
	# Tool.
	TGS = np.identity(4)
	TGS[:3,3] = np.array([0, 0, 1.0])
	tool_ = tool(s)
	tool_.set_configuration(0.015, TGS)
		
	# Render scene.
	s.run()


# TASK 1
class robot():
	def __init__(self, scene):
		q = np.zeros(6)
		d = np.array([0, 0, 0, -0.065, 0, 0.11])
		a = np.array([0, 0.25, 0.25, 0, 0, 0])
		al = np.array([np.pi/2, 0, 0, np.pi/2, np.pi/2, 0])
		self.DH = np.stack((q, d, a, al), 1)
	
		s = scene
		
		# Base.
		self.base = vis.cylinder(0.025, 0.05)
		s.add_actor(self.base)	

		# Link 1.
		self.link1 = vis.cylinder(0.025, 0.05)
		s.add_actor(self.link1)	
		
		# Link 2.
		self.link2 = vis.cube(0.3, 0.05, 0.05)
		s.add_actor(self.link2)

		# Link 3.
		self.link3 = vis.cube(0.3, 0.05, 0.05)
		s.add_actor(self.link3)
		
		# Link 4.
		self.link4 = vis.cylinder(0.015, 0.04)
		s.add_actor(self.link4)	
		
		# Link 5.
		self.link5 = vis.cylinder(0.02, 0.04)
		s.add_actor(self.link5)		
		
		# Tool.
		self.tool = tool(s)
		
	def set_configuration(self, q, g, T0S):
		d = self.DH[:,1]
		a = self.DH[:,2]
		al = self.DH[:,3]
		
		# Base.
		TB0 = np.identity(4)
		TB0[:3,:3] = rotx(np.pi/2)
		TB0[2,3] = -0.025
		TBS = T0S @ TB0
		vis.set_pose(self.base, TBS)

		# Link 1.
		T10 = dh(q[0], d[0], a[0], al[0])
		T1S = T0S @ T10
		TL11 = np.identity(4)
		TL11[:3,:3] = rotx(np.pi/2)	
		TL1S = T1S @ TL11
		vis.set_pose(self.link1, TL1S)
		
		# Link 2.
		T21 = dh(q[1], d[1], a[1], al[1])
		T2S = T1S @ T21	
		TL22 = np.identity(4)
		TL22[0,3] = -0.12
		TL22[2,3] = 0.05
		TL2S = T2S @ TL22
		vis.set_pose(self.link2, TL2S)

		# Link 3.
		T32 = dh(q[2], d[2], a[2], al[2])
		T3S = T2S @ T32	
		TL33 = np.identity(4)
		TL33[0,3] = -0.125
		TL3S = T3S @ TL33
		vis.set_pose(self.link3, TL3S)
		
		# Link 4.
		T43 = dh(q[3], d[3], a[3], al[3])
		T4S = T3S @ T43	
		TL44 = np.identity(4)
		TL44[1,3] = 0.02
		TL4S = T4S @ TL44
		vis.set_pose(self.link4, TL4S)
		
		# Link 5.
		T54 = dh(q[4], d[4], a[4], al[4])
		T5S = T4S @ T54	
		TL55 = np.identity(4)
		TL5S = T5S @ TL55
		vis.set_pose(self.link5, TL5S)
		
		# Link 6.
		T65 = dh(q[5], d[5], a[5], al[5])
		T6S = T5S @ T65	
		self.tool.set_configuration(g, T6S)
		
		return T6S	

def dh(q, d, a, al):
	cq = np.cos(q)
	sq = np.sin(q)
	ca = np.cos(al)
	sa = np.sin(al)
	T = np.array([[cq, -sq*ca, sq*sa, a*cq],
		[sq, cq*ca, -cq*sa, a*sq],
		[0, sa, ca, d],
		[0, 0, 0, 1]])
	return T

def task1(q):
	# Scene.
	s = vis.visualizer()

	# Axes.
	axes = vtk.vtkAxesActor()
	s.add_actor(axes)

	# Floor.
	set_floor(s, [1, 1])
	
	# Robot.
	T0S = np.identity(4)
	T0S[2,3] = 0.05
	rob = robot(s)
	rob.set_configuration(q, 0.03, T0S)
	
	# Render scene.
	s.run()	


# TASK 2

def invkin(DH, T60, solution):
	d = DH[:,1]
	a = DH[:,2]
	al = DH[:,3]
	
	p = T60 @ np.expand_dims(np.array([0, 0, -d[5], 1]), 1)
	x = p[0]
	y = p[1]
	z = p[2]
	r = p[:3].T @ p[:3]
	
	q = np.zeros(6)
	
	q[2] = np.arccos((r - a[2]**2 - d[3]**2 - a[1]**2) / (2*a[1]*a[2]))
	if solution[0] == 1:
		q[2] = -q[2]
		
	c3 = np.cos(q[2])
	s3 = np.sin(q[2])
	f1 = a[2]*c3 + a[1]
	
	f2 = a[2]*s3
	
	f3 = d[3]
	
	A = np.sqrt(f1**2+f2**2)
	phi = np.arctan2(f2, f1)
	if solution[1] == 0:
		q[1] = np.arcsin(z/A) - phi
	else:
		q[1] = np.pi - np.arcsin(z/A) - phi
	

	c2 = np.cos(q[1])
	s2 = np.sin(q[1])
	g1 = c2*f1 - s2*f2
		
	g2 = -f3
	
	c1 = g1*x + g2*y
	s1 = -g2*x + g1*y
	q[0] = np.arctan2(s1, c1)
	
	T10 = dh(q[0], d[0], a[0], al[0])
	T21 = dh(q[1], d[1], a[1], al[1])
	T32 = dh(q[2], d[2], a[2], al[2])
	T30 = T10 @ T21 @ T32
	R30 = T30[:3,:3]
	R60 = T60[:3,:3]
	R63 = R30.T @ R60
	
	c5 = -R63[2,2]
	q[4] = np.arccos(c5)
	if solution[2] == 1:
		q[4] = -q[4]
	s5 = np.sin(q[4])
	if np.abs(s5) > 1e-10:
		q[3] = np.arctan2(R63[1,2]/s5, R63[0,2]/s5)
		q[5] = np.arctan2(-R63[2,1]/s5, R63[2,0]/s5)
	else:
		c46 = R63[0,0]
		s46 = R63[0,1]
		q46 = np.arctan2(s46, c46)
		q[3] = q46
		q[5] = 0
	
	return q

def task2(solution):
	TTS = np.identity(4)
	TTS[0,3]=0.3
	TTS[2,3] = 0.1
	
	# Scene
	s = vis.visualizer()

	# Floor.
	set_floor(s, [1, 1])
	
	# Target object.
	target = vis.cube(0.03, 0.03, 0.03) #change size
	vis.set_pose(target, TTS)
	s.add_actor(target)	

	# Robot.
	T0S = np.identity(4)
	T0S[2,3] = 0.05
	T6T = np.identity(4)
	T6T[:3,:3] = roty(np.pi)
	T60 = np.linalg.inv(T0S) @ TTS @ T6T
	rob = robot(s)
	q = invkin(rob.DH, T60, solution)
	rob.set_configuration(q, 0.03, T0S)
	
	# Render scene.
	s.run()

# TASK 3

class simulator():
	def __init__(self, robot, Qc):
		self.timer_count = 0
		self.robot = robot
		self.Qc = Qc
		self.trajW = []
		self.T0S = np.identity(4)
		self.T0S[2,3] = 0.05

	def execute(self,iren,event):
		T6S = self.robot.set_configuration(self.Qc[:,self.timer_count % self.Qc.shape[1]], 0.03, self.T0S)
		self.trajW.append(T6S)
		iren.GetRenderWindow().Render()
		self.timer_count += 1

def task3():
	# Scene
	s = vis.visualizer()

	# Floor.
	set_floor(s, [1, 1])
	
	# Robot.
	rob = robot(s)
	
	# Robot velocity and acceleration limits.
	dqgr=np.pi*np.ones((1,6))
	ddqgr=10*np.pi*np.ones((1,6))
	
	# Trajectory.
	q_home = np.array([-np.pi/2, np.pi/2, 0, 0, 0, 0])
	T60_1 = np.identity(4)
	T60_1[:3,:3] = roty(np.pi)
	T60_1[:3,3] = np.array([0.17, 0.03, 0.2]) # y koordinata - lijevo/desno kada smo ispred robota
	q1 = invkin(rob.DH,T60_1,[1, 0, 0])

	# Zvijezda
	T60_2 = T60_1.copy()
	T60_2[:3,3] = np.array([0.17, 0.03, 0.02]) 
	q2 = invkin(rob.DH,T60_2,[1, 0, 0])

	T60_3 = T60_1.copy()
	T60_3[:3,3] = np.array([0.21, 0.05, 0.02]) 
	q3 = invkin(rob.DH,T60_3,[1, 0, 0])

	T60_4 = T60_1.copy()
	T60_4[:3,3] = np.array([0.17, 0.03, 0.2]) 
	q4 = invkin(rob.DH,T60_4,[1, 0, 0])

	T60_5 = T60_1.copy()
	T60_5[:3,3] = np.array([0.17, 0.03, 0.02]) 
	q5 = invkin(rob.DH,T60_5,[1, 0, 0])

	T60_6 = T60_1.copy()
	T60_6[:3,3] = np.array([0.17, 0.07, 0.02]) 
	q6 = invkin(rob.DH,T60_6,[1, 0, 0])

	T60_7 = T60_1.copy()
	T60_7[:3,3] = np.array([0.2, 0.07, 0.2]) 
	q7 = invkin(rob.DH,T60_7,[1, 0, 0])

	T60_8 = T60_1.copy()
	T60_8[:3,3] = np.array([0.2, 0.07, 0.02]) 
	q8 = invkin(rob.DH,T60_8,[1, 0, 0])

	T60_9 = T60_1.copy()
	T60_9[:3,3] = np.array([0.2, 0.03, 0.02]) 
	q9 = invkin(rob.DH,T60_9,[1, 0, 0])

	T60_10 = np.identity(4)
	T60_10[:3,:3] = roty(np.pi)
	T60_10[:3,3] = np.array([0.21, 0.05, 0.2]) 
	q10 = invkin(rob.DH,T60_10,[1, 0, 0])

	T60_11 = T60_10.copy()
	T60_11[:3,3] = np.array([0.21, 0.05, 0.02]) 
	q11 = invkin(rob.DH,T60_11,[1, 0, 0])

	T60_12 = T60_10.copy()
	T60_12[:3,3] = np.array([0.17, 0.07, 0.02]) 
	q12 = invkin(rob.DH,T60_12,[1, 0, 0])

	T60_13 = T60_10.copy()
	T60_13[:3,3] = np.array([0.2, 0.03, 0.2]) 
	q13 = invkin(rob.DH,T60_13,[1, 0, 0])

	T60_14 = T60_10.copy()
	T60_14[:3,3] = np.array([0.2, 0.03, 0.02]) 
	q14 = invkin(rob.DH,T60_14,[1, 0, 0])

	T60_15 = T60_10.copy()
	T60_15[:3,3] = np.array([0.16, 0.05, 0.02]) 
	q15 = invkin(rob.DH,T60_15,[1, 0, 0])

	T60_16 = T60_10.copy()
	T60_16[:3,3] = np.array([0.2, 0.07, 0.2]) 
	q16 = invkin(rob.DH,T60_16,[1, 0, 0])

	T60_17 = T60_10.copy()
	T60_17[:3,3] = np.array([0.2, 0.07, 0.02]) 
	q17 = invkin(rob.DH,T60_17,[1, 0, 0])

	T60_18 = T60_10.copy()
	T60_18[:3,3] = np.array([0.16, 0.05, 0.02]) 
	q18 = invkin(rob.DH,T60_18,[1, 0, 0])

	#Krakovi
	T60_19 = np.identity(4)
	T60_19[:3,:3] = rotx(np.pi)
	T60_19[:3,3] = np.array([0.17, 0.12, 0.2]) 
	q19 = invkin(rob.DH,T60_19,[1, 0, 0])

	T60_20 = T60_19.copy()
	T60_20[:3,3] = np.array([0.17, 0.12, 0.02]) 
	q20 = invkin(rob.DH,T60_20,[1, 0, 0])

	T60_21 = T60_19.copy()
	T60_21[:3,3] = np.array([0.17, 0.08, 0.02]) 
	q21 = invkin(rob.DH,T60_21,[1, 0, 0])

	T60_22 = T60_19.copy()
	T60_22[:3,3] = np.array([0.185, 0.11, 0.2]) 
	q22 = invkin(rob.DH,T60_22,[1, 0, 0])

	T60_23 = T60_19.copy()
	T60_23[:3,3] = np.array([0.185, 0.11, 0.02]) 
	q23 = invkin(rob.DH,T60_23,[1, 0, 0])

	T60_24 = T60_19.copy()
	T60_24[:3,3] = np.array([0.185, 0.07, 0.02]) 
	q24 = invkin(rob.DH,T60_24,[1, 0, 0])

	T60_25 = T60_19.copy()
	T60_25[:3,3] = np.array([0.2, 0.12, 0.2]) 
	q25 = invkin(rob.DH,T60_25,[1, 0, 0])

	T60_26 = T60_19.copy()
	T60_26[:3,3] = np.array([0.2, 0.12, 0.02]) 
	q26 = invkin(rob.DH,T60_26,[1, 0, 0])

	T60_27 = T60_19.copy()
	T60_27[:3,3] = np.array([0.2, 0.08, 0.02]) 
	q27 = invkin(rob.DH,T60_27,[1, 0, 0])

	Q = np.stack((q_home, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20, q21, q22, q23, q24, q25, q26, q27, q_home), 1)
	Ts = 0.01
	Qc, dQc, ddQc, tc = hocook(Q, dqgr, ddqgr, Ts)
	
	# Display trajectory.
	# Položaj zglobova u vremenu
	plt.figure()
	plt.plot(tc, Qc[0,:], label='q1')
	plt.plot(tc, Qc[1,:], label='q2')
	plt.plot(tc, Qc[2,:], label='q3')
	plt.plot(tc, Qc[3,:], label='q4')
	plt.plot(tc, Qc[4,:], label='q5')
	plt.plot(tc, Qc[5,:], label='q6')
	plt.xlabel('Vrijeme [s]')
	plt.ylabel('Položaj zglobova [rad]')
	plt.title('Položaj zglobova u vremenu')
	plt.legend()
	plt.grid()
	plt.show()

	# Brzina zglobova u vremenu
	plt.figure()
	plt.plot(tc, dQc[0,:], label='dq1')
	plt.plot(tc, dQc[1,:], label='dq2')
	plt.plot(tc, dQc[2,:], label='dq3')
	plt.plot(tc, dQc[3,:], label='dq4')
	plt.plot(tc, dQc[4,:], label='dq5')
	plt.plot(tc, dQc[5,:], label='dq6')
	plt.xlabel('Vrijeme [s]')
	plt.ylabel('Brzina zglobova [rad/s]')
	plt.title('Brzine zglobova u vremenu')
	plt.legend()
	plt.grid()
	plt.show()

	# Ubrzanje zglobova u vremenu
	plt.figure()
	plt.plot(tc, ddQc[0,:], label='ddq1')
	plt.plot(tc, ddQc[1,:], label='ddq2')
	plt.plot(tc, ddQc[2,:], label='ddq3')
	plt.plot(tc, ddQc[3,:], label='ddq4')
	plt.plot(tc, ddQc[4,:], label='ddq5')
	plt.plot(tc, ddQc[5,:], label='ddq6')
	plt.xlabel('Vrijeme [s]')
	plt.ylabel('Ubrzanje zglobova [rad/s²]')
	plt.title('Ubrzanja zglobova u vremenu')
	plt.legend()
	plt.grid()
	plt.show()

	
	# Create animation callback.
	sim = simulator(rob, Qc)
	
	# Start animation.
	s.run(animation_timer_callback=sim.execute)
	
	# Display tool trajectory in 3D.
	trajW = np.array(sim.trajW)
	tool_tip_W = trajW[:,:3,3]

	ax = plt.axes(projection='3d')
	ax.plot3D(tool_tip_W[:,0], tool_tip_W[:,1], tool_tip_W[:,2], 'b')
	ax.set_xlabel('X [m]')
	ax.set_ylabel('Y [m]')
	ax.set_zlabel('Z [m]')
	ax.set_title('3D putanja vrha alata')
	plt.show()

	
	# Display plane contact.
	print('Visina alata:', np.min(tool_tip_W[:,2]))
	d_board = np.min(tool_tip_W[:,2]) + 0.004
	n_board = np.array([0, 0, 1])
	board_draw = planecontact(tool_tip_W, n_board, d_board)

	fig, ax = plt.subplots(1, 1)
	ax.plot(board_draw[:,0], board_draw[:,1], 'b.')
	ax.set_xlabel('X [m]')
	ax.set_ylabel('Y [m]')
	ax.set_title('Kontakt vrha alata s ravninom')
	ax.axis('equal')
	ax.grid()
	plt.show()

		
	return tool_tip_W
	


def main():
	#task0()
	#task1([0, np.pi/2, -np.pi/2, 0, 0, 0])
	#task2([0, 1, 0])
	task3()


if __name__ == '__main__':
    main()