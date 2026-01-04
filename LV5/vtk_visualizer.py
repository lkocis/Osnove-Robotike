import math
import vtk

class visualizer:
	def __init__(self, camera_position=None, camera_focal_point=None, camera_up=None, time_interval=100):
		# create a rendering window and renderer
		self.ren = vtk.vtkRenderer()
		self.ren.SetBackground(0, 1, 1)
		self.renWin = vtk.vtkRenderWindow()
		self.renWin.AddRenderer(self.ren)

		# create a renderwindowinteractor
		self.iren = vtk.vtkRenderWindowInteractor()
		self.iren.SetRenderWindow(self.renWin)
		self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

		# camera pose
		self.camera = self.ren.GetActiveCamera()
		if camera_position is not None:
			self.camera.SetPosition(camera_position[0], camera_position[1], camera_position[2])
		if camera_focal_point is not None:
			self.camera.SetFocalPoint(camera_focal_point[0], camera_focal_point[1], camera_focal_point[2])
		if camera_up is not None:
			self.camera.SetViewUp(camera_up[0], camera_up[1], camera_up[2])

		# time interval
		self.time_interval = time_interval
		
	def run(self, animation_timer_callback=None):
		# enable user interface interactor
		self.iren.Initialize()
		self.renWin.Render()
		
		if animation_timer_callback is not None:
			self.iren.AddObserver('TimerEvent', animation_timer_callback)
			timerId = self.iren.CreateRepeatingTimer(self.time_interval);		
					
		self.iren.Start()
					
	def add_actor(self, object):
		self.ren.AddActor(object)
			
def cube(a, b, c):
	# create cube
	cube = vtk.vtkCubeSource()
	cube.SetXLength(a)
	cube.SetYLength(b)
	cube.SetZLength(c)
	cube.Update()

	# mapper
	cubeMapper = vtk.vtkPolyDataMapper()
	cubeMapper.SetInputConnection(cube.GetOutputPort())

	# actor
	cubeActor = vtk.vtkActor()
	cubeActor.SetMapper(cubeMapper)
	
	return cubeActor
	
def cylinder(r, h, resolution=12):
	# create cylinder
	cylinder = vtk.vtkCylinderSource()
	cylinder.SetRadius(r)
	cylinder.SetHeight(h)
	cylinder.SetResolution(resolution)
	cylinder.Update()

	# mapper
	cylinderMapper = vtk.vtkPolyDataMapper()
	cylinderMapper.SetInputConnection(cylinder.GetOutputPort())

	# actor
	cylinderActor = vtk.vtkActor()
	cylinderActor.SetMapper(cylinderMapper)
	
	return cylinderActor

def sphere(r, resolution=12):
	# create sphere
	sphere = vtk.vtkSphereSource()
	sphere.SetRadius(r)
	sphere.SetThetaResolution(resolution)  # Longitude resolution
	sphere.SetPhiResolution(resolution)    		  # Latitude resolution
	sphere.Update()

	# mapper
	sphereMapper = vtk.vtkPolyDataMapper()
	sphereMapper.SetInputConnection(sphere.GetOutputPort())

	# actor
	sphereActor = vtk.vtkActor()
	sphereActor.SetMapper(sphereMapper)
	
	return sphereActor

def arrow(origin, endpoint, color, shaft_radius=0.02, tip_length=0.2, tip_radius=0.06):
	"""
	Create an arrow actor from origin to endpoint.
	Uses vtkArrowSource and a transform to place/orient/scale it.
	"""
	# Create an arrow (points along +X by default)
	arrow_src = vtk.vtkArrowSource()
	arrow_src.SetShaftRadius(shaft_radius)
	arrow_src.SetShaftResolution(16)
	arrow_src.SetTipLength(tip_length)
	arrow_src.SetTipRadius(tip_radius)
	arrow_src.SetTipResolution(16)
	arrow_src.Update()

	# Compute transform from (0,0,0)->(1,0,0) to origin->endpoint
	ox, oy, oz = origin
	ex, ey, ez = endpoint
	direction = [ex - ox, ey - oy, ez - oz]
	length = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
	if length == 0:
		length = 1.0
	# Normalize
	ndir = [d / length for d in direction]

	# Default arrow is from (0,0,0) to (1,0,0)
	# Calculate rotation that maps X axis to ndir
	transform = vtk.vtkTransform()
	transform.PostMultiply()
	# Move to origin
	transform.Translate(ox, oy, oz)

	# Find rotation axis and angle
	# cross product of (1,0,0) and ndir gives axis:
	vx, vy, vz = ndir
	# If ndir is nearly (1,0,0), no rotation needed
	dot = vx * 1.0 + vy * 0.0 + vz * 0.0
	if abs(dot - 1.0) > 1e-6:
		axis = [0.0, -vz, vy]  # cross((1,0,0), ndir) = (0, -vz, vy)
		axis_mag = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
		if axis_mag > 1e-6:
			axis = [a / axis_mag for a in axis]
			angle_deg = math.degrees(math.acos(max(min(dot, 1.0), -1.0)))
			transform.RotateWXYZ(angle_deg, axis)
		else:
			# opposite direction: rotate 180 deg around Y or Z
			transform.RotateWXYZ(180.0, 0, 1, 0)

	# Scale to the required length
	transform.Scale(length, length, length)

	# Apply transform to arrow geometry
	tf_filter = vtk.vtkTransformPolyDataFilter()
	tf_filter.SetTransform(transform)
	tf_filter.SetInputConnection(arrow_src.GetOutputPort())
	tf_filter.Update()

	mapper = vtk.vtkPolyDataMapper()
	mapper.SetInputConnection(tf_filter.GetOutputPort())

	actor = vtk.vtkActor()
	actor.SetMapper(mapper)
	actor.GetProperty().SetColor(color)
	return actor

def make_label(text, pos, camera, scale=0.08, color=(1, 1, 1)):
    """
    Create a 3D text label that follows the camera (vtkFollower).
    camera: renderer camera to attach follower to
    """
    source = vtk.vtkVectorText()
    source.SetText(text)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())

    actor = vtk.vtkFollower()
    actor.SetMapper(mapper)
    actor.SetScale(scale, scale, scale)
    actor.SetPosition(*pos)
    actor.GetProperty().SetColor(color)
    actor.SetCamera(camera)
    return actor

class Coordinate_System():	
	def __init__(self, scene, axis_length, color):
		s = scene
		  
		origin = (0.0, 0.0, 0.0)

		# Axes endpoints
		x_end = (axis_length, 0.0, 0.0)
		y_end = (0.0, axis_length, 0.0)
		z_end = (0.0, 0.0, axis_length)		  

		# Axes
		self.x_axis = arrow(origin, x_end, color)
		s.add_actor(self.x_axis)
		self.y_axis = arrow(origin, y_end, color)
		s.add_actor(self.y_axis)
		self.z_axis = arrow(origin, z_end, color)
		s.add_actor(self.z_axis)

		# Axis name labels (X, Y, Z) a bit beyond arrow tip
		self.label_x = make_label("X", (x_end[0] + 0.08 * axis_length, 0.0, 0.0), s.camera, scale=0.12, color = color)
		self.label_y = make_label("Y", (0.0, y_end[1] + 0.08 * axis_length, 0.0), s.camera, scale=0.12, color = color)
		self.label_z = make_label("Z", (0.0, 0.0, z_end[2] + 0.08 * axis_length), s.camera, scale=0.12, color = color)
		s.add_actor(self.label_x)
		s.add_actor(self.label_y)
		s.add_actor(self.label_z)

	def set_pose(self, T):
		set_pose(self.x_axis, T)
		set_pose(self.y_axis, T)
		set_pose(self.z_axis, T)
		set_pose(self.label_x, T)
		set_pose(self.label_y, T)
		set_pose(self.label_z, T)
		
def set_pose(actor, T):
    transfo_mat = vtk.vtkMatrix4x4()
    for i in range(0,4):
        for j in range(0,4):
            transfo_mat.SetElement(i,j, T[i,j])        
    actor.SetUserMatrix(transfo_mat) 

def colors():
	return vtk.vtkNamedColors()
		

