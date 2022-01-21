import numpy as np
import skimage
import skimage.filters
import time

from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb
from skimage import io


class DisjointSet: 

	def __init__(self, noc_):
		self.noc = noc_ #number of components
		self.forest = np.empty((self.noc,3), dtype=np.int32)
		for i in range(self.noc):
			self.forest[i][0] = i #parent
			self.forest[i][1] = 0 #rank
			self.forest[i][2] = 1 #component size
		self.is_modified = True
		self.root_indexes = dict() #dictionary/maps

	def find(self, x): 
		if self.forest[x][0] == x:
			return x
		else:
			self.forest[x][0] = self.find(self.forest[x][0])
			return self.forest[x][0]

	def union(self,x,y): 
		root1 = self.find(x)
		root2 = self.find(y)
		if root1 == root2:
			return root1
		self.noc -= 1
		if self.forest[root1][1] < self.forest[root2][1]:
			self.forest[root1][0] = root2
			self.forest[root2][2] += self.forest[root1][2]
			return root2
		elif self.forest[root1][1] > self.forest[root2][1]:
			self.forest[root2][0] = root1
			self.forest[root1][2] += self.forest[root2][2];
			return root1
		else:
			self.forest[root2][0] = root1;
			self.forest[root1][1] += 1;
			self.forest[root1][2] += self.forest[root2][2];
			return root1

	def getNumberOfComponents(self):
		return self.noc

	def getParent(self,x):
		root = self.find(x)
		return self.forest[root][0]

	def getRank(self,x):
		root = self.find(x)
		return self.forest[root][1]

	def getComponentSize(self,x):
		root = self.find(x)
		return self.forest[root][2]

	def isModified(self):
		return self.is_modified 

	def getRootIndexes(self):
		if self.is_modified == False:
			return self.root_indexes 
		self.root_indexes.clear()
		current_index = 0
		for i in range(len(self.forest)):
			root = self.find(i)
			key_index = (self.root_indexes).get(root)
			if key_index == None:
				self.root_indexes[root] = current_index
				current_index += 1
		self.is_modified = False
		return self.root_indexes

	def fuseSmallComponents(self,min_size,edges):
		for edge in edges:
			root_0 = self.find(edge[0])
			root_1 = self.find(edge[1])
			if root_0 == root_1:
				continue
			if self.forest[root_0][2]<min_size or self.forest[root_1][2]<min_size:
				new_root = self.union(root_0, root_1)
		for i in range(len(self.forest)):
			self.find(i)

	def felsenszwalbSegment(self,k,min_size,costs,edges,n):  
		if len(costs) != len(edges):
			raise Exception('length of costs array and egde array is not same')
		internal_differences = np.zeros(n,dtype=np.float32)
		index = 0
		for edge in edges:
			root_0 = self.find(edge[0])
			root_1 = self.find(edge[1])
			segment_size_0 = self.forest[root_0][2]
			segment_size_1 = self.forest[root_1][2]
			inner_cost_0 = internal_differences[root_0] + k/segment_size_0
			inner_cost_1 = internal_differences[root_1] + k/segment_size_1
			if root_0 != root_1 and costs[index] < min(inner_cost_0,inner_cost_1):
				new_root = self.union(root_0,root_1)
				internal_differences[new_root] = costs[index]
			index += 1
		# Post processing: merge small segments with nearby segments
		self.fuseSmallComponents(min_size,edges)
		# Replace random labels by fixed continuous labels {0,1,...,#segments}
		root_indexes_ = self.getRootIndexes()
		labels = np.ones(n, dtype = np.int32)
		for i in range(n):
			root = int(self.find(i))
			label = int(root_indexes_[root])
			labels[i] = label
		return labels
		
		
#########################################################
# \param[in] image : (N,M,C) ndarray. 
# Input image.
# \param[in] scale : float, optional (default=1).
# Sets the observation level. Higher means larger clusters.
# \param[in] sigma : float, optional (default=0.8)
# Width of Gaussian smoothing kernel used in preprocessing.
# \param[in] min_size : int,opional (default=20)
# Minimum coponent size. Enforced using postprocessing.
# \returns labels : (N,M) ndarray
#########################################################
def fh(image,scale=1,sigma=0.8,min_size=20):

	if image.shape[2] > 3:
		raise Exception('Images with more than 3 channels cannot be handled right now')

	image = skimage.img_as_float64(image)

	# rescale scale to behave like in reference explanation
	scale = float(scale)/255

	image = skimage.filters.gaussian(image,sigma,cval=0,multichannel=True)

	height,width,channels = image.shape
	n = height*width

	# Compute edge weights in 8 connectivity
	down_cost = np.sqrt(np.sum((image[1:, :, :]-image[:height-1, :, :])*(image[1:, :, :]-image[:height-1, :, :]),axis=-1))
	right_cost = np.sqrt(np.sum((image[:, 1:, :]-image[:, :width-1, :])*(image[:, 1:, :]-image[:, :width-1, :]),axis=-1))
	dright_cost = np.sqrt(np.sum((image[1:, 1:, :]-image[:height-1, :width-1, :])*(image[1:, 1:, :]-image[:height-1, :width-1, :]),axis=-1))
	uright_cost = np.sqrt(np.sum((image[1:, :width-1, :]-image[:height-1, 1:, :])*(image[1:, :width-1, :]-image[:height-1, 1:, :]),axis=-1))
	costs = np.hstack([right_cost.ravel(),down_cost.ravel(),dright_cost.ravel(),uright_cost.ravel()])

    # Compute edges between pixels
	segments = np.arange(height*width,dtype=np.int32).reshape(height,width)
	down_edges  = np.c_[segments[1:, :].ravel(), segments[:height-1, :].ravel()]
	right_edges = np.c_[segments[:, 1:].ravel(), segments[:, :width-1].ravel()]
	dright_edges = np.c_[segments[1:, 1:].ravel(), segments[:height-1, :width-1].ravel()]
	uright_edges = np.c_[segments[:height-1, 1:].ravel(), segments[1:, :width-1].ravel()]
	edges = np.vstack([right_edges, down_edges, dright_edges, uright_edges])

    # Sort edges according to the cost
	edge_queue = np.argsort(costs)
	edges = np.ascontiguousarray(edges[edge_queue])
	costs = np.ascontiguousarray(costs[edge_queue])

    # Apply greedy merging and returning final oversegmentation image
	ds = DisjointSet(n)
	labels = ds.felsenszwalbSegment(scale,min_size,costs,edges,n)
	labels = labels.reshape(height,width)

	return labels


image = skimage.io.imread('/home/sushmita/Documents/SUSHMITA/PROJECT1/lib_fh/self/version_2/input/113016.jpg')

# Self
a=time.time()
labels = fh(image,scale=100,sigma=0.0,min_size=20)
# (unique, counts) = np.unique(labels,return_counts=True)
# print(len(unique))
print(time.time()-a)

#Comparing with inbuilt
a=time.time()
labels = skimage.segmentation.felzenszwalb(image,scale=100,sigma=0.0,min_size=20)
# (unique, counts) = np.unique(labels,return_counts=True)
# print(len(unique))
print(time.time()-a)

# Saving boundary image
boundary_image = skimage.segmentation.mark_boundaries(image,labels)
boundary_image = 255 * boundary_image # Convert float64 image into uint8 before saving.
boundary_image = boundary_image.astype(np.uint8)
skimage.io.imsave('/home/sushmita/Documents/SUSHMITA/PROJECT1/lib_fh/self/version_2/output/boundary.jpg',boundary_image)                                                                                                                

