import vtk

class visualizer:
	def __init__(self):
		# create a rendering window and renderer
		self.ren = vtk.vtkRenderer()
		self.ren.SetBackground(0, 1, 1)
		self.renWin = vtk.vtkRenderWindow()
		self.renWin.AddRenderer(self.ren)

		# create a renderwindowinteractor
		self.iren = vtk.vtkRenderWindowInteractor()
		self.iren.SetRenderWindow(self.renWin)
		self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
		
	def run(self, animation_timer_callback=None):
		# enable user interface interactor
		self.iren.Initialize()
		self.renWin.Render()
		
		if animation_timer_callback is not None:
			self.iren.AddObserver('TimerEvent', animation_timer_callback)
			timerId = self.iren.CreateRepeatingTimer(100);		
					
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
		
def set_pose(actor, T):
    transfo_mat = vtk.vtkMatrix4x4()
    for i in range(0,4):
        for j in range(0,4):
            transfo_mat.SetElement(i,j, T[i,j])        
    actor.SetUserMatrix(transfo_mat) 
		

