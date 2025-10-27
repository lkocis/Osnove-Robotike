import vtk

def cube():
	# create a rendering window and renderer
	ren = vtk.vtkRenderer()
	ren.SetBackground(0, 1, 1)
	renWin = vtk.vtkRenderWindow()
	renWin.AddRenderer(ren)

	# create a renderwindowinteractor
	iren = vtk.vtkRenderWindowInteractor()
	iren.SetRenderWindow(renWin)
	iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

	# create cube
	cube = vtk.vtkCubeSource()

	# mapper
	cubeMapper = vtk.vtkPolyDataMapper()
	cubeMapper.SetInputConnection(cube.GetOutputPort())

	# actor
	cubeActor = vtk.vtkActor()
	cubeActor.SetMapper(cubeMapper)

	# assign actor to the renderer
	ren.AddActor(cubeActor)

	# enable user interface interactor
	iren.Initialize()
	renWin.Render()
	iren.Start()
	
	
	

 