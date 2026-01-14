import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

def trot(R):
	T = np.identity(4)
	T[:3,:3] = R
	return T

def transl(t):
	T = np.identity(4)
	T[:3,3] = t
	return T

def skew(x):
    Y = np.zeros((x.shape[0], 3, 3))
    Y[:,0,1] = -x[:,2]
    Y[:,0,2] = x[:,1]
    Y[:,1,0] = x[:,2]
    Y[:,1,2] = -x[:,0]
    Y[:,2,0] = -x[:,1]
    Y[:,2,1] = x[:,0]
    return Y

def angle_axis_to_rotmx(k, q):
    cq = np.cos(q)
    sq = np.sin(q)
    cqcomp = 1.0 - cq
    kxy = k[0] * k[1] * cqcomp
    kyz = k[1] * k[2] * cqcomp
    kzx = k[2] * k[0] * cqcomp
    return np.array([[k[0] * k[0] * cqcomp + cq, kxy - k[2] * sq, kzx + k[1] * sq, 0.0],
    [kxy + k[2] * sq, k[1] * k[1] * cqcomp + cq, kyz - k[0] * sq, 0.0],
    [kzx - k[1] * sq, kyz + k[0] * sq, k[2] * k[2] * cqcomp + cq, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

def identity_array(m, n):
    I = np.zeros((n,m,m))
    for i in range(m):
        I[:,i,i] = 1.0
    return I

def rotvect_to_rotmx(r):
    q = np.linalg.norm(r, axis=1)
    cq = np.cos(q)
    sq = np.sin(q)
    cqcomp = 1.0 - cq
    idx = (np.abs(q) > 1e-6)
    R = identity_array(3,r.shape[0])
    k = np.zeros((r.shape[0], 3))
    k[idx,:] = r[idx,:] / q[idx,np.newaxis]
    kxy = k[:,0] * k[:,1] * cqcomp
    kyz = k[:,1] * k[:,2] * cqcomp
    kzx = k[:,2] * k[:,0] * cqcomp
    R[:,0,0] = k[:,0] * k[:,0] * cqcomp + cq
    R[:,0,1] = kxy - k[:,2] * sq
    R[:,0,2] = kzx + k[:,1] * sq
    R[:,1,0] = kxy + k[:,2] * sq
    R[:,1,1] = k[:,1] * k[:,1] * cqcomp + cq
    R[:,1,2] = kyz - k[:,0] * sq
    R[:,2,0] = kzx - k[:,1] * sq
    R[:,2,1] = kyz + k[:,0] * sq
    R[:,2,2] = k[:,2] * k[:,2] * cqcomp + cq
    return R

def rotmx_to_angle_axis(R):
    k = 0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1.0)
    if k > 1.0:
        theta = 0.0
        axis = np.zeros(3)
    elif k < -1.0:
        theta = np.pi
        axis = np.zeros(3)
    theta = np.arccos(k)
    k = 0.5 / np.sin(theta)	
    axis = np.array([k * (R[2, 1] - R[1, 2]), k * (R[0, 2] - R[2, 0]), k * (R[1, 0] - R[0, 1])])
    axis = axis / np.linalg.norm(axis)
    return axis, theta

def rotangle(R):
    cs = 0.5 * (R[:,0,0] + R[:,1,1] + R[:,2,2] - 1.0)
    cs[cs > 1.0] = 1.0
    cs[cs < -1.0] = -1.0
    return np.arccos(cs)

def rotmx_to_rotvect(R):
	th = rotangle(R)
	sn_th = np.sin(th)
	u = np.zeros((R.shape[0], 3))
	idx = (np.abs(sn_th) > 1e-6)
	u[idx,:] = np.expand_dims(0.5 / sn_th[idx], 1) * np.stack((R[idx,2,1] - R[idx,1,2], R[idx,0,2] - R[idx,2,0], R[idx,1,0] - R[idx,0,1]), 1)
	r = th[:,np.newaxis] * u
	return r

def inv_transf(T):
    invT = np.eye(4)
    invT[:3,:3] = T[:3,:3].T
    invT[:3,3] = -T[:3,:3].T @ T[:3,3]
    return invT

def homogeneous(points):
    return np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
	
def fwdkin(DH):
	q = DH[:,0]
	d = DH[:,1]
	a = DH[:,2]
	al = DH[:,3]
	
	n = DH.shape[0]
	T = np.eye(4)
	for i in range(n):
		T = T @ dh(q[i], d[i], a[i], al[i])
	
	return T
	
def plot(x, title, colors=None):
	x_ = x.copy()
	if not isinstance(x_, list):
		x_ = [x_]
	m = len(x_)
	for j in range(m):
		if x_[j].ndim < 2:
			x_[j] = np.expand_dims(x_[j], 1)
	n = x_[0].shape[1]
	N = x_[0].shape[0]
	k = np.arange(N)
	for i in range(n):
		plt.subplot(n,1,i+1)
		for j in range(m):
			if colors is None:
				plt.plot(k, x_[j][:,i])
			else:
				plt.plot(k, x_[j][:,i], color=colors[j])
		plt.grid()
		if i==0:
			plt.title(title)
	plt.show()	

class Animation():
	def __init__(self, sim, traj):
		self.timer_count = 0		
		self.sim = sim
		self.traj = traj		

	def execute(self,iren,event):
		if self.timer_count < self.traj.shape[0]:
			self.sim.update_scene(self.traj[self.timer_count,:])
			render_window = iren.GetRenderWindow()
			render_window.Render()		
		elif self.timer_count == self.traj.shape[0]:
			print('animation completed.')			
		self.timer_count += 1	

def simulate(sim, num_steps, Ts, initial_state, animation=False, subsample=1):
	traj = np.zeros((num_steps, initial_state.shape[0]))
	traj[0,:] = initial_state
	state = initial_state
	for k in range(num_steps - 1):		
		u = sim.ctrlstep(state, k)
		t0 = k * Ts
		tf = (k+1) * Ts
		sol = solve_ivp(sim.step, [t0, tf], state, args=(u,), t_eval=[tf])
		state = sol.y[:, -1]
		traj[k+1,:] = state
	#traj = integrate.odeint(sim.step, initial_state, t)

	# Animation.

	if animation:
		anim = Animation(sim, traj[::subsample,:])
		print('animation started.')
		sim.scene.run(animation_timer_callback=anim.execute)
	
	return traj
