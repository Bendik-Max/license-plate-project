import numpy as np


MIN_HEIGHT_RATIO = 0.5
MAX_HEIGHT_RATIO = 0.95

MIN_WIDTH_RATIO = 0.03
MAX_WIDTH_RATIO = 0.3

MIN_SIZE_RATIO = 0.01
MAX_SIZE_RATIO = 0.1


class BoundingBox:
	def __init__(self, min_x, max_x, min_y, max_y):
		self.min_x = min_x
		self.max_x = max_x
		self.min_y = min_y
		self.max_y = max_y

	def __str__(self):
		return "min_x: " + str(self.min_x) + ", max_x: " + str(self.max_x) + ", min_y: " + str(self.min_y) + ", max_y: " + str(self.max_y)


	"""
	Get the size of a bounding box in pixels
	
	Inputs:(Zero)
	Outputs:(One)
		1. size: size in pixels of bounding box
		type: int
	"""
	def size(self):
		width = self.max_x - self.min_x
		height = self.max_y - self.min_y
		return width * height


class Params:
	def __init__(self, image_width, image_height):
		self.min_height = int(image_height * MIN_HEIGHT_RATIO)
		self.max_height = int(image_height * MAX_HEIGHT_RATIO)
		self.min_width = int(image_width * MIN_WIDTH_RATIO)
		self.max_width = int(image_width * MAX_WIDTH_RATIO)
		self.min_size = int(image_width * image_height * MIN_SIZE_RATIO)
		self.max_size = int(image_width * image_height * MAX_SIZE_RATIO)

	def __str__(self):
		return "min_height: " + str(self.min_height) + ", max_height: " + str(self.max_height) \
			   + ", min_width: " + str(self.min_width) + ", max_width: " + str(self.max_width) \
			   + ", min_size: " + str(self.min_size) + ", max_size: " + str(self.max_size)


class Plate:
	def __init__(self, frame_nr, content):
		self.frame_nr = frame_nr
		self.content = content

	def __str__(self):
		return "Plate(content: " + str(self.content) + ", frame_nr: " + str(self.frame_nr) + ")"

	def __repr__(self):
		return "Plate(content: " + str(self.content) + ", frame_nr: " + str(self.frame_nr) + ")"


class Group:
	def __init__(self, plates=None):
		if plates is None:
			plates = []
		self.plates = plates

	def __str__(self):
		return "Group(plates: " + str(self.plates) + ")"

	def __repr__(self):
		return "Group(plates: " + str(self.plates) + ")"

	"""
	Add plate to group's list of plates
	
	Inputs:(One)
		1. plate: plate to add
		type: Plate
	Outputs:(Zero)
	"""
	def add_plate(self, plate):
		self.plates.append(plate)


	"""
	Get the frame of this group as half of minimum and maximum frame of group
	
	Inputs:(Zero)
	Outputs:(One)
		1. frame: frame of this group
		type: int
	"""
	def get_frame(self):
		min_frame = np.Inf
		for plate in self.plates:
			curr_frame = plate.frame_nr
			if curr_frame < min_frame:
				min_frame = curr_frame
		return min_frame + 1
