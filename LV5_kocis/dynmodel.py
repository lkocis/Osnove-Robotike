import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import vtk_visualizer as vis
import or_util as rob
import robotplanar
from hocook import hocook

m1 = 1
m2 = 1
l1 = 1
l2 = 1
b1 = 0.1
b2 = 0.1
r_tool = 0.02

q = np.array([0.0, 0.0])

# Simulation.
class Simulation():
	def __init__(self, scene, robot, ctrlalg, Ts, ref_traj):
		self.scene = scene
		self.robot = robot		
		self.ctrlalg = ctrlalg
		self.ref_traj = ref_traj
		self.Ts = Ts

	def step(self, t, state, u):
		q = state[:2]
		dq = state[2:4]
		D, N, h, B = self.robot.dynmodel(q, dq)
		ddq = np.linalg.inv(D) @ (u - N - h - B)
		dstate = np.concatenate((dq, ddq[:,0]), axis=0)
		return dstate
	
	def ctrlstep(self, state, k):
		q = state[:2]
		dq = state[2:4]
		qr = self.ref_traj[:2,k]
		dqr = self.ref_traj[2:4,k]
		ddqr = self.ref_traj[4:6,k]
		return self.ctrlalg.step(q, dq, qr, dqr, ddqr)
	
	def update_scene(self, state):
		q = state[:2]
		dq = state[2:4]
		T2S = self.robot.set_configuration(q)

# Control algorithm.
class RobCtrlAlg():
	def __init__(self, robot, Kp, Kv, friction_model=False, feedback=True, cc_model=True, g_model=True, full_D=True):
		self.robot = robot
		self.Kp = Kp
		self.Kv = Kv		
		self.friction_model = friction_model
		self.feedback = feedback
		self.cc_model = cc_model
		self.g_model = g_model
		self.full_D = full_D

	def step(self, q, dq, qr, dqr, ddqr):
		D, N, h, B = self.robot.dynmodel(q, dq)
		uFF = ddqr
		uFB = np.zeros(2)
		if self.feedback:
			ep = qr - q
			ev = dqr - dq
			uFB = self.Kp * ep + self.Kv * ev
		if not self.full_D:
			D[0,1] = 0.0
			D[1,0] = 0.0
		u = D @ (uFF + uFB)[:, np.newaxis]
		if self.cc_model:
			u = u + N
		if self.g_model:
			u = u + h
		if self.friction_model:
			u = u + B
		return u

# Scene
s = vis.visualizer(camera_position=[0, 0, 6])

# Robot.
T0S = np.identity(4)
robot = robotplanar.Robot(m1, m2, l1, l2, b1, b2, r_tool, s, T0S, gravity=True)
# robot.set_configuration(q)
dqgr=np.pi*np.ones((1,2))
ddqgr=10*np.pi*np.ones((1,2))
# dqgr=5*dqgr
# ddqgr=20*ddqgr

# Reference trajectory.
Ts = 0.001
r = 0.5
c = np.array([r, 2 * r])
traj_res = 12
Qpt = np.zeros((2, traj_res + 1))
T20 = np.identity(4)
w = 2 * np.pi / traj_res
for i in range(traj_res+1):
	T20[0,3] = c[0] + r * np.cos(w * i)
	T20[1,3] = c[1] + r * np.sin(w * i)
	Qpt[:,i] = robot.invkin(T20, 0)
Qr, dQr, ddQr, t = hocook(Qpt, dqgr, ddqgr, Ts)
ref_traj = np.concatenate((Qr, dQr, ddQr), axis=0)
ref_traj_W = robot.fwdkin(ref_traj[:2,:].T)

# Simulation selection loop.

next_simulation = True
while next_simulation:
	# Menu.
	print(' ')
	print('Select simulation:')
	print(' ')
	print('0. Visualization')
	print('1. Feed forward with friction model')
	print('2. Feed forward w/o friction model')
	print('3. Feed forward w/o friction model, independent joint ctrl + feedback')
	print('4. Feed forward w/o friction model, w/o cc model + feedback')
	print('5. Feed forward w/o friction model, w/o gravity model + feedback')
	print('6. Feed forward w/o friction model + feedback')
	print('7. Exit')
	key = input('Enter selection: ')	

	# Simulation selection.
	friction_model = False
	g_model = True
	feedback = True
	cc_model = True
	full_D = True
	animation = False
	if key == '0':
		animation = True
	elif key == '1':
		friction_model = True
		feedback = False
	elif key == '2':
		feedback = False
	elif key == '3':
		cc_model = False
		g_model = False
		full_D = False
	elif key == '4':
		cc_model = False
	elif key == '5':
		g_model = False
	elif key == '7':
		break

	# Control algorithm.
	w = 20
	Kp = w**2
	Kv = 2 * 0.7 * w
	ctrlalg = RobCtrlAlg(robot=robot, Kp=Kp, Kv=Kv, friction_model=friction_model, feedback=feedback, cc_model=cc_model, g_model=g_model, full_D=full_D)

	# Simulation.
	sim = Simulation(s, robot, ctrlalg, Ts, ref_traj)
	traj = rob.simulate(sim, t.shape[0], Ts, ref_traj[:4,0], animation=animation, subsample=30)
	traj_W = robot.fwdkin(traj[:,:2])

	# Display trajectory.
	if not animation:
		plt.plot(t,Qr[0,:],t,Qr[1,:],t,traj[:,0],t,traj[:,1])
		plt.show()
		# plt.plot(t,dQ[0,:],t,dQ[1,:])
		# plt.show()
		# plt.plot(t,ddQ[0,:],t,ddQ[1,:])
		# plt.show()
		plt.plot(ref_traj_W[:,0], ref_traj_W[:,1], traj_W[:,0], traj_W[:,1])
		plt.axis('equal')
		plt.show()

