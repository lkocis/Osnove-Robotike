import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import br_lectures as br
import vtk
import vtk_visualizer as vis

class MobileRobot():
	def __init__(self, scene):
		s = scene
		
		# Parameters.
		self.wheel_radius = 0.06
		self.wheel_distance = 0.35
		
		# Color.
		red = [1.0, 0.0, 0.0]
		black = [0.0, 0.0, 0.0]
		
		# Body.
		self.body = vis.cylinder(0.15, 0.16, resolution=20)
		self.body.GetProperty().SetColor(red)
		s.add_actor(self.body)
		
		# Deck.
		self.deck = vis.cylinder(0.2, 0.01, resolution=20)
		self.deck.GetProperty().SetColor(red)
		s.add_actor(self.deck)
		
		# Wheel 1.
		self.wheel1 = vis.cylinder(self.wheel_radius, 0.04, resolution=20)
		self.wheel1.GetProperty().SetColor(black)
		s.add_actor(self.wheel1)
		
		# Wheel 2.
		self.wheel2 = vis.cylinder(self.wheel_radius, 0.04, resolution=20)
		self.wheel2.GetProperty().SetColor(black)
		s.add_actor(self.wheel2)
		
	def set_pose(self, position, orientation):
		# Robot pose on the scene.
		TRS = np.identity(4)
		TRS[:3,:3] = br.rotz(orientation)
		TRS[0,3] = position[0]
		TRS[1,3] = position[1]
		TRS[2,3] = position[2]
	
		# Body.
		TBR = np.identity(4)
		TBR[:3,:3] = br.rotx(np.pi/2)
		TBR[2,3] = 0.12
		TBS = TRS @ TBR
		vis.set_pose(self.body, TBS)
		
		# Deck.
		TDR = np.identity(4)
		TDR[:3,:3] = br.rotx(np.pi/2)
		TDR[2,3] = 0.205
		TDS = TRS @ TDR
		vis.set_pose(self.deck, TDS)
		
		# Wheel 1.
		TW1R = np.identity(4)
		TW1R[1,3] = 0.5 * self.wheel_distance
		TW1R[2,3] = self.wheel_radius
		TW1S = TRS @ TW1R
		vis.set_pose(self.wheel1, TW1S)		
		
		# Wheel 2.
		TW2R = np.identity(4)
		TW2R[1,3] = -0.5 * self.wheel_distance
		TW2R[2,3] = self.wheel_radius
		TW2S = TRS @ TW2R
		vis.set_pose(self.wheel2, TW2S)	

class Animation():
	def __init__(self, scene, robot, traj):
		self.timer_count = 0
		self.robot = robot
		self.traj = traj

	def execute(self,iren,event):
		#print('timer_count=', self.timer_count)
		if self.timer_count < self.traj.shape[0]:
			position = [self.traj[self.timer_count,0], self.traj[self.timer_count,1], 0.0]
			orientation = self.traj[self.timer_count,2]
			self.robot.set_pose(position, orientation)
			render_window = iren.GetRenderWindow()
			render_window.Render()		
		elif self.timer_count == self.traj.shape[0]:
			print('simulation completed.')			
		self.timer_count += 1
		#else:
		#	render_window.Finalize()
		#	iren.TerminateApp()
		#	del render_window, iren

class Simulator():
	def __init__(self, robot, ctrlalg):
		self.robot_wheel_radius = robot.wheel_radius
		self.robot_wheel_distance = robot.wheel_distance
		self.ctrlalg = ctrlalg		
		
	def step(self, pose, t):
		# Control.
		u = self.ctrlalg.step(pose)
		
		# Robot motion.
		v = 0.5 * self.robot_wheel_radius * (u[0] + u[1])
		w = self.robot_wheel_radius * (u[1] - u[0]) / self.robot_wheel_distance
		
		return v * np.cos(pose[2]), v * np.sin(pose[2]), w 	
			
def mobrobsimanimate(scene, robot_initial_pose, ctrlalg, t_sim, dt):
	s = scene
	robot = MobileRobot(s)
	robot.set_pose([robot_initial_pose[0], robot_initial_pose[1], 0.0], robot_initial_pose[2])
	sim = Simulator(robot, ctrlalg)
	t = np.linspace(0, t_sim, int(t_sim/dt))
	traj = integrate.odeint(sim.step, robot_initial_pose, t)
	anim = Animation(s, robot, traj)
	print('simulation started.')
	s.run(animation_timer_callback=anim.execute)
	
	return traj

def set_goal(s, position, orientation):
	axes = vtk.vtkAxesActor()
	transform = vtk.vtkTransform()
	transform.Translate(position[0], position[1], 0.0)	
	transform.RotateZ(np.rad2deg(orientation))
	transform.Scale(0.5, 0.5, 0.5)
	axes.SetUserTransform(transform)
	for label in [
        axes.GetXAxisCaptionActor2D(),
        axes.GetYAxisCaptionActor2D(),
        axes.GetZAxisCaptionActor2D(),
        ]:
		label.SetWidth(label.GetWidth() * 0.2)
		label.SetHeight(label.GetHeight() * 0.2)
	s.add_actor(axes)	
	
def set_map(s, map_file_name, size):
	readerFactory = vtk.vtkImageReader2Factory()
	textureFile = readerFactory.CreateImageReader2(map_file_name)
	textureFile.SetFileName(map_file_name)
	textureFile.Update()
	
	atext = vtk.vtkTexture()
	atext.SetInputConnection(textureFile.GetOutputPort())
	atext.InterpolateOn()
	
	plane = vtk.vtkPlaneSource()
	plane.SetOrigin(0,0,0)
	plane.SetPoint1(size[0],0.0,0)
	plane.SetPoint2(0.0,size[1],0)
	planeMapper = vtk.vtkPolyDataMapper()
	planeMapper.SetInputConnection(plane.GetOutputPort())

	planeActor = vtk.vtkActor()
	planeActor.SetMapper(planeMapper)
	planeActor.SetTexture(atext)	
		
	s.add_actor(planeActor)
