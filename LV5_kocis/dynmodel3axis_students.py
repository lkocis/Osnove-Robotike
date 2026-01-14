import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import vtk_visualizer as vis
import or_util as rob
import robot3axis_students
from hocook import hocook

m1 = 1
m2 = 1
m3 = 1
l1 = 1
l2 = 1
l3 = 1
b1 = 0.1
b2 = 0.1
b3 = 0.1
r_tool = 0.02

q = np.array([0.0, 0.0, 0.0])

# Simulation.
class Simulation():
	def __init__(self, scene, robot, ctrlalg, Ts, ref_traj):
		self.scene = scene
		self.robot = robot		
		self.ctrlalg = ctrlalg
		self.ref_traj = ref_traj
		self.Ts = Ts
		self.D_history = []

	def step(self, t, state, u):
		q = state[:3]
		dq = state[3:6]
		D, N, h, B = self.robot.dynmodel(q, dq)
		self.D_history.append(D)
		ddq = np.linalg.inv(D) @ (u - N - h - B)
		dstate = np.concatenate((dq, ddq[:,0]), axis=0)
		return dstate
	
	def ctrlstep(self, state, k):
		q = state[:3]
		dq = state[3:6]
		qr = self.ref_traj[:3,k]
		dqr = self.ref_traj[3:6,k]
		ddqr = self.ref_traj[6:9,k]
		return self.ctrlalg.step(q, dq, qr, dqr, ddqr)
	
	def update_scene(self, state):
		q = state[:3]
		dq = state[3:6]
		T3S = self.robot.set_configuration(q)

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
		#ADD YOUR CODE HERE: *****************************
		D, N, h, B = self.robot.dynmodel(q, dq)
		uFF = ddqr
		uFB = np.zeros(3)
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
		#END ***********************************************
		return u
	
def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so spheres look like spheres."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])

# Colors
colors = vis.colors()
red = colors.GetColor3d("Tomato")
green = colors.GetColor3d("Green")
blue = colors.GetColor3d("CornflowerBlue")

# Scene
s = vis.visualizer(camera_position=[0, 0, 6])
rf = vis.Coordinate_System(s, 0.3, green)
rf.set_pose(np.eye(4))

# Robot.
T0S = np.identity(4)
robot = robot3axis_students.Robot(m1, m2, m3, l1, l2, l3, b1, b2, b3, r_tool, s, T0S, gravity=True)
dqgr=np.pi*np.ones((1,3))
ddqgr=10*np.pi*np.ones((1,3))
dqgr=2*dqgr
ddqgr=2*ddqgr

# T30 = np.identity(4)
# T30[:3,3] = np.array([0.4, 0.4, 0.4])
# q = robot.invkin(T30, 1)
# T30_ = robot.fwdkin(q)
# T3S = robot.set_configuration(q)
# s.run()

# Reference trajectory.
Ts = 0.001
r = 0.8
c = np.array([0.5, 0, 1.0])
traj_res = 12
Qpt = np.zeros((3, traj_res + 1))
T30 = np.identity(4)
T30[0,3] = c[0]
w = 2 * np.pi / traj_res
for i in range(traj_res+1):
	T30[1,3] = c[1] - r * np.sin(w * i)
	T30[2,3] = c[2] + r * np.cos(w * i)
	Qpt[:,i] = robot.invkin(T30, 1)
Qr, dQr, ddQr, t = hocook(Qpt, dqgr, ddqgr, Ts)
ref_traj = np.concatenate((Qr, dQr, ddQr), axis=0)
ref_traj_W = robot.fwdkin(ref_traj[:3,:].T)

# Test Simulation.

# robot.set_configuration(ref_traj[:,0])
# s.run()
# sim = Simulation(s, robot, None, Ts, ref_traj)
# anim = rob.Animation(sim, ref_traj[:,::20].T)
# print('animation started.')
# sim.scene.run(animation_timer_callback=anim.execute)
# traj = rob.simulate(sim, t.shape[0], Ts, ref_traj[:4,0], animation=animation, subsample=30) 

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
	print('6. Feed forward w/o friction model, w/o cc model, w/o gravity model + feedback')
	print('7. Feed forward w/o friction model + feedback')
	print('8. Exit')
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
	elif key == '6':
		cc_model = False
		g_model = False
	elif key == '8':
		break

	# Control algorithm.
	w = 10
	Kp = w**2
	Kv = 2 * 0.7 * w
	ctrlalg = RobCtrlAlg(robot=robot, Kp=Kp, Kv=Kv, friction_model=friction_model, feedback=feedback, cc_model=cc_model, g_model=g_model, full_D=full_D)

	# Simulation.
	sim = Simulation(s, robot, ctrlalg, Ts, ref_traj)
	traj = rob.simulate(sim, t.shape[0], Ts, ref_traj[:6,0], animation=animation, subsample=30)
	traj_W = robot.fwdkin(traj[:,:3])

	# Display trajectory.
	if not animation:
		# Zad 3, 4, 5, 6:*******************************
		D_array = np.array(sim.D_history)  
		print('Srednje vrijednosti momenta inercije D za svaki zglob: ')
		print('D1 srednje: ', np.mean(D_array[:,0,0]))
		print('D2 srednje: ', np.mean(D_array[:,1,1]))
		print('D3 srednje: ', np.mean(D_array[:,2,2]))
		#****Graf t/D****************************
		t_D = np.arange(D_array.shape[0]) * Ts
		plt.figure()
		plt.plot(t_D, D_array[:,0,0], label='1. zglob (D1)')
		plt.plot(t_D, D_array[:,1,1], label='2. zglob (D2)')
		plt.plot(t_D, D_array[:,2,2], label='3. zglob (D3)')
		plt.xlabel('Vrijeme [s]')
		plt.ylabel('D moment inercije [kgm²]')
		plt.legend()
		plt.title('Promjena momenta inercije D tijekom vremena')
		plt.grid(True)
		plt.show()
		#****???*****************************************
		plt.figure()
		plt.plot(t,Qr[0,:],t,Qr[1,:],t,Qr[1,:], label='Zglob')
		plt.plot(t,traj[:,0],t,traj[:,1],t,traj[:,2], '--', label='Ostvareno')
		plt.show()
		# plt.plot(t,dQ[0,:],t,dQ[1,:])
		# plt.show()
		# plt.plot(t,ddQ[0,:],t,ddQ[1,:])
		# plt.show()
		#****2D prikaz************************************
		plt.figure()
		plt.plot(ref_traj_W[:,1], ref_traj_W[:,2], label='Referentna')
		plt.plot(traj_W[:,1], traj_W[:,2], label='Ostvarena')
		plt.axis('equal')
		plt.xlabel('Y')
		plt.ylabel('Z')
		plt.legend()
		plt.grid(True)
		plt.title('Prikaz referentne i ostvarene putanje')
		plt.show()
		#****3D prikaz************************************
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(ref_traj_W[:,0], ref_traj_W[:,1], ref_traj_W[:,2], label='Referentna')
		ax.plot(traj_W[:,0], traj_W[:,1], traj_W[:,2], label='Ostvarena')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.legend()
		ax.set_title('3D prikaz referentne i ostvarene putanje')
		set_axes_equal(ax)
		plt.show()
		#****Gibanje vrha alata u 2D i 3D s konst. D*********


		