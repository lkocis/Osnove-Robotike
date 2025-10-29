import sys
import numpy as np
import vtk
import matplotlib.pyplot as plt
import vtk_visualizer as vis
#import br_lectures as br
from hocook import hocook
from planecontact import planecontact
#from mobrobsim import mobrobsimanimate, set_goal, set_map
from scipy import ndimage
from PIL import Image
#from camerasim import CameraSimulator
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks

# TASK 0
class tool():
    def __init__(self, scene):
        s = scene   
        self.finger1 = vis.cube(0.04, 0.06, 0.08)
        s.add_actor(self.finger1)
        self.finger2 = vis.cube(0.04, 0.06, 0.08)
        s.add_actor(self.finger2)
        self.palm = vis.cube(0.04, 0.25, 0.02)
        s.add_actor(self.palm)
        self.wrist = vis.cylinder(0.015, 0.04)
        s.add_actor(self.wrist)
        
    def set_configuration(self, g, TGS):    
        TF1G = np.identity(4)
        TF1G[:3,3] = np.array([0, -0.095, -0.03])
        TF1S = TGS @ TF1G
        vis.set_pose(self.finger1, TF1S)
        TF2G = np.identity(4)
        TF2G[:3,3] = np.array([0, 0.095, -0.03])    
        TF2S = TGS @ TF2G
        vis.set_pose(self.finger2, TF2S)
        TPG = np.identity(4)
        TPG[:3,3] = np.array([0, 0, -0.065])
        TPS = TGS @ TPG
        vis.set_pose(self.palm, TPS)
        TWG = np.block([[rotx(np.pi/2), np.array([[0], [0], [-0.08]])], [np.zeros((1, 3)), 1]])
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
    # Scene.
    s = vis.visualizer()

    # Axes.
    #axes = vtk.vtkAxesActor()
    #s.add_actor(axes)
    
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
        #kinematic parameters:
        q = np.array([0, np.pi/2, -np.pi/2, np.pi/2, 0, 0])
        d = np.array([0, 0, 1, 0, 0, 0.95])
        a = np.array([0, -0.46, 0, 0.64, 0, 0])
        al = np.array([-np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])

        self.DH = np.stack((q, d, a, al), 1)
    
        s = scene

        
        # Base.
        #self.base = vis.cylinder(0.025, 0.05)
        self.base = vis.cube(0.3, 0.3, 0.3)
        s.add_actor(self.base)  

        # Link 1.
        self.link1 = vis.cylinder(0.13, 0.3)
        #self.link1 = vis.cube(0.26, 0.67, 0.26)
        s.add_actor(self.link1) 
        
        #Link 2
        self.link2 = vis.cylinder(0.1, 0.614)
        s.add_actor(self.link2) 

        #Link 3
        self.link3 = vis.cube(0.3, 0.3, 0.3)
        s.add_actor(self.link3)
        
        # Tool.

        
    def set_configuration(self, q, g, T0S):
        d = self.DH[:,1]
        a = self.DH[:,2]
        al = self.DH[:,3]
        
        # Base.
        TB0 = np.identity(4)
        #TB0[:3,:3] = rotx(np.pi/2)
        TB0[2,3] = -(0.45 - 0.3/2)
        TBS = T0S @ TB0
        vis.set_pose(self.base, TBS)

        # Link 1.
        T10 = dh(q[0], d[0], a[0], al[0]) #todo: create dh function
        T1S = T0S @ T10
        TL11 = np.identity(4)    
        TL1S = T1S @ TL11
        vis.set_pose(self.link1, TL1S)
        
        #Link 2
        T21 = dh(q[1], d[1], a[1], al[1])
        T20 = T10 @ T21
        T2S = T0S @ T20
        TL21 = np.identity(4)
        TL21[:3, :3] = rotz(np.pi/2)  
        TL2S = T2S @ TL21 
        vis.set_pose(self.link2, TL2S)

        #Link3
        T23 = dh(q[2], d[2], a[2], al[2])
        T30 = T21 @ T23
        T3S = T0S @ T30
        TL32 = np.identity(4)
        TL3S = T3S @ TL32
        vis.set_pose(self.link3, TL3S)

        return TBS  

    

def dh(q, d, a, al):

    #Define cq, sq, ca, sa and T
    cq = np.cos(q)
    sq = np.sin(q)
    ca = np.cos(al)
    sa = np.sin(al)
    T = np.array([[cq, -sq*ca, sq*sa, a*cq],
        [sq, cq*ca, -cq*sa, a*sq],
        [0, sa, ca, d],
        [0, 0, 0, 1]])
    return T
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
    T0S[2,3] = 0.45
    rob = robot(s)
    rob.set_configuration(q, 0.03, T0S)
    
    # Render scene.
    s.run() 

def main():
    #task0()
    task1([0, np.pi/2, -np.pi/2, 0, 0, 0])


if __name__ == '__main__':
    main()