import cv2
import numpy as np


MAX_RATIO = 0.35
MIN_RATIO = 0.2
MAX_SIZE = 120000
MIN_SIZE = 1500
MAX_WIDTH = 600
MIN_WIDTH = 50
MAX_HEIGHT = 200
MIN_HEIGHT = 30
MIN_FILL = 0.85


"""
Given an image and a bounding box, extract the image inside the bounding box

Inputs:(Two)
	1. bounding_box: bounding box defined by x_min, x_max, y_min and y_max
	type: BoundingBox
	2. image: image to extract the part out of
	type: array with dimensions > 1

Outputs:(One)
	1. cropped_image: the image cropped using the bounding box
	tye: array with dimension >1
"""
def crop_image(bounding_box, image):
	return image[bounding_box.min_x:bounding_box.max_x, bounding_box.min_y:bounding_box.max_y]


"""
Given an image and a color range in hsv, 
defines a mask which for each pixel has value:
	 0  - if the pixel color is not within the range
	255 - if the pixel color is within the range

Inputs:(Three)
	1. image: rgb image
	type: array (3D)
	2. minV: minimum color value of range in hsi
	type: array (1D, size=3)
	3. maxV: maximum color value of range in hsi
	type: array (1D, size=3)

Outputs:(One)
	1. mask: image with values either 0 or 255, same size as input image
	type: array (3D)
"""
def create_mask(image, minV, maxV):
	# convert the rgb image to hsv
	hsi = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	# define the mask
	mask = cv2.inRange(hsi, minV, maxV)
	return mask


"""
returns a tuple of colorMin and colorMax which
contains the range of yellow colors for a license plate

Inputs:(Zero)
Outputs:(One)
	1. color_range: tuple defining color range in hsi values
	type: tuple(size = 2) of 1D numpy arrays
"""
def yellow_range():
	color_min = np.array([80, 70, 70])
	color_max = np.array([110, 255, 250])
	color_range = (color_min, color_max)
	return color_range


"""
Given a width, a height and an area, determine if 
it is possible for an object of those dimensions to be
a license plate.

Inputs:(Three)
	1. width: width of the object
	type: int
	2. height: height of the object
	type: int
	3. area: amount of pixels that it fills
	type: int

Outputs:(One)
	1. boolean: true if it cannot be a license plate, false otherwise
"""
def is_not_license_plate_prelim(width, height, area):
	if size_is_off(width, height, area):
		return True
	return False


"""
Given a mask, determine if it could be a license plate.

Inputs:(One)
	1. mask: the cropped mask given by localization
	type: 2D array
Outputs:(One)
	1. boolean: true if it cannot be a license plate, false otherwise
"""
def is_not_license_plate(mask):
	return ratio_is_wrong(mask) or components_not_connected(mask) or fill_ratio_not_satisfied(mask)


"""
Given a mask, determine if the ratio is that of a license plate.

Inputs:(One)
	1. mask: the cropped mask given by localization
	type: 2D array
Outputs:(One)
	1. boolean: true if it cannot be a license plate, false otherwise
"""
def ratio_is_wrong(mask):
	ratio = len(mask)/len(mask[0])
	return not MIN_RATIO < ratio < MAX_RATIO


"""
Given a mask, determine if the amount of connected components exceeds the
possible amount from a mask corresponding to that of a license plate.

Inputs:(One)
	1. mask: the cropped mask given by localization
	type: 2D array
Outputs:(One)
	1. boolean: true if it cannot be a license plate, false otherwise
"""
def components_not_connected(mask):
	stats = cv2.connectedComponentsWithStats(mask, 4)[2]
	return len(stats) > 2


"""
Given a mask, determine if the mask fills up the space it occupies
enough to be a potential license plate.

Inputs:(One)
	1. mask: the cropped mask given by localization
	type: 2D array
Outputs:(One)
	1. boolean: true if it cannot be a license plate, false otherwise
"""
def fill_ratio_not_satisfied(mask):
	comparator = np.zeros(mask.shape)
	xor = np.logical_xor(mask, comparator)
	fill_ratio = np.sum(xor) / mask.size
	return fill_ratio < MIN_FILL


"""
Given a width, a height and an area, determine if 
it is possible for an object of those dimensions to be
a license plate.

Inputs:(Three)
	1. width: width of the object
	type: int
	2. height: height of the object
	type: int
	3. area: amount of pixels that it fills
	type: int

Outputs:(One)
	1. boolean: true if it cannot be a license plate, false otherwise
"""
def size_is_off(width, height, area):
	width_bool = MIN_WIDTH < width < MAX_WIDTH
	height_bool = MIN_HEIGHT < height < MAX_HEIGHT
	size_bool = MIN_SIZE < area < MAX_SIZE
	return not (width_bool and height_bool and size_bool)


"""
Applies another mask onto the cropped image, and slices the license plate
in such a way that most of the external parts are no longer included. 

Inputs:(Two)
	1. image: the actual cropped image
	type: 3D array
	2. mask: the mask of the cropped image
	type: 2D array

Outputs:(Two)
	1. refitted_image: the refitted image, cropped neatly
	type: 3D array
	2. refitted_mask: the refitted mask, cropped neatly
	type: 2D array
"""
def refit_image(image, mask):
	color_min, color_max = yellow_range()
	new_mask = create_mask(image, color_min, color_max)

	stats = cv2.connectedComponentsWithStats(new_mask, 4)[2]
	index_max_area = np.argmax(stats, axis=0)[4]
	plate_stats = stats[index_max_area]
	leftmost_coordinate, top_coordinate, width, height = plate_stats[0], plate_stats[1], plate_stats[2], plate_stats[3]

	offset_vertical = int(height / 20)
	offset_horizontal = int(width / 24)

	refitted_image = image[top_coordinate + offset_vertical:top_coordinate + height - offset_vertical, leftmost_coordinate + offset_horizontal:leftmost_coordinate + width - int(offset_horizontal*1.4)]
	refitted_mask = mask[top_coordinate + offset_vertical:top_coordinate + height - offset_vertical, leftmost_coordinate + offset_horizontal:leftmost_coordinate + width - int(offset_horizontal*1.4)]
	return refitted_image, refitted_mask
