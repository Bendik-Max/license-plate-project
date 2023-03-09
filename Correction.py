import cv2
import numpy as np


"""
Correct plate rotation using the hough transform

Inputs:(Two)
    1. img: image to be corrected
    type: 2D array
    2. mask: mask of the image
    type: 2D array
Outputs:(One)
    1. rotated: rotated and corrected image
    type: 2D array
"""
def correct_plate_hough(img, mask):
	canny = canny_edge_detection(mask)
	lines = cv2.HoughLines(canny, 1, np.pi / 180, 30)

	rotation_angle = 0
	if lines is not None and len(lines) > 1:
		angle = (lines[0][0][1] + lines[1][0][1])/2
		rotation_angle = (np.pi / 2 - angle) * 180 / np.pi

	if not (-1.5 < rotation_angle < 1.5):
		rotation_angle = min(16, rotation_angle)
		M = cv2.getRotationMatrix2D((len(img[0]) / 2, len(img) / 2), -rotation_angle, 1)

		rotated = cv2.warpAffine(img, M, (len(img[0]), len(img)))
		rotated_mask = cv2.warpAffine(mask, M, (len(mask[0]), len(mask)))
		return post_rotation_crop(rotated, rotated_mask)
	return img, mask


"""
Detect edges using Canny

Inputs:(One)
	1. img: the image to perform Canny on
	type: np array
Outputs:(One)
	1. canny: edge image
	type: np array
"""
def canny_edge_detection(img):
	return cv2.Canny(img, 100, 150, apertureSize=3);


"""
Crop the image and mask after rotation

Inputs:(Two)
	1. img: the image to crop
	type: np array
	2. mask: mask of the image
	type: 2D array
Outputs:(Two)
	1. cropped_image: image after cropping
	type: np array
	2. cropped_mask: mask after cropping
	type: 2D array
"""
def post_rotation_crop(img, mask):
	stats = cv2.connectedComponentsWithStats(mask, 4)[2]
	index_max_area = np.argmax(stats, axis=0)[4]
	plate_stats = stats[index_max_area]
	leftmost_coordinate, top_coordinate, width, height = plate_stats[0], plate_stats[1], plate_stats[2], plate_stats[3]

	cropped_image = img[top_coordinate:top_coordinate+height, leftmost_coordinate:leftmost_coordinate+width]
	cropped_mask = mask[top_coordinate:top_coordinate + height, leftmost_coordinate:leftmost_coordinate + width]

	return cropped_image, cropped_mask
