import numpy as np
from scipy import ndimage
import br_lectures as br

class CameraSimulator():
	def __init__(self, camera, height, box_size, noise):
		self.camera = camera
		self.height = height
		self.box_size = box_size
		self.noise = noise

	def get_image(self):
		img_w = self.camera['img_size'][0]
		img_h = self.camera['img_size'][1]		
		box_half_w = 0.5 * self.box_size[0] / self.height * self.camera['focal_length']
		box_half_h = 0.5 * self.box_size[1] / self.height * self.camera['focal_length']
		img_region = np.array([[0, 0], [img_w-1, img_h-1]])
		box_size = np.array([box_half_w, box_half_h])
		self.alpha = np.pi/2 * (2 * np.random.rand() - 1)
		R = br.rotz(self.alpha)
		R = R[:2,:2]
		vertices_nrm = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
		vertices = vertices_nrm * box_size @ R.T
		bbox = np.stack((vertices.min(0), vertices.max(0)),0)
		box_center_region = img_region - bbox
		t_img = (box_center_region[1,:] - box_center_region[0,:]) * np.random.rand(2) + box_center_region[0,:]
		vertices += t_img
		u, v = np.meshgrid(np.arange(img_w), np.arange(img_h))
		p = np.expand_dims(np.stack((u, v), 2) - t_img,3)
		p = R.T @ p 
		p = np.squeeze(p,3)
		p = np.abs(p)
		B = np.logical_or(p[:,:,0] > box_half_w, p[:,:,1] > box_half_h).astype('float64')
		B = 0.5 * B + 0.25 + self.noise * (2 * np.random.rand(img_h, img_w) - 1)
		RGB = np.stack((B, B, B), 2)
		RGB = ndimage.filters.gaussian_filter(RGB,sigma=1.5)
		self.t = self.height * np.ones(3)
		self.t[:2] = (t_img - np.array(self.camera['principal_point'])) / self.camera['focal_length'] * self.height 
		
		return RGB
		
	def evaluate(self, alpha, t):
		e_alpha = np.abs(alpha - self.alpha)
		if e_alpha > np.pi/2:
			e_alpha = np.pi - e_alpha
		e_t = np.linalg.norm(t - self.t)
		
		return e_alpha, e_t