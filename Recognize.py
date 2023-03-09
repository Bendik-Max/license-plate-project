import cv2
import Enhance
from Morphology import denoise_plate
from LocalizationUtils import crop_image
from Classes import Params
from RecognizeUtils import resize_image, extract_characters, recognize_char,\
	good_distance_between_bbs, valid_plate, convertArrayToString, overwrite_mistakes


"""
Given (hopefully) a license plate as an input image, segments and
recognizes the characters of the license plate, and returns the 
recognized plate as a string.

Inputs:(Two)
	1. plate_imgs: cropped plate image by Localization.plate_detection function
	type: 3D numpy array
	2. binarize_technique: technique used to binarize the image,
	1 is adaptive thresholding, 2 is isoData
	type: int
Outputs:(One)
	1. final_plate: recognized plate characters
	type: string
"""
def segment_and_recognize(plate_img, binarize_technique = 1, is_cat3= False):
	## first, pre-process the image
	copy = pre_process_image(plate_img, is_cat3)

	# calculate parameters for the image
	params = Params(len(copy[0]), len(copy))

	## binarize and denoise based on image size
	copy = binarize_and_denoise(copy, binarize_technique)

	## find the connected components and extract the ones which
	## are likely to be characters
	stats = cv2.connectedComponentsWithStats(copy, 4)[2]
	listOfChars = extract_characters(stats, params)

	## recognize the characters
	recognized_plate = recognize_plate(listOfChars, copy)

	## convert to a string and check if the result is a valid plate
	final_plate = convertArrayToString(recognized_plate)
	final_plate = overwrite_mistakes(final_plate)
	is_valid = valid_plate(final_plate)

	## if not valid, we repeat with isoData and not adaptive
	if not is_valid and binarize_technique == 1:
		return segment_and_recognize(plate_img, 2, is_cat3)
	if not is_valid and binarize_technique == 2 and not is_cat3:
		return segment_and_recognize(plate_img, 1, True)
	return final_plate.upper() if is_valid else None


"""
Given an image, pre-processes it by doubling its size and converting
it to grayscale. The size is increased in order to make the morphology
more effective in the next step. Also copies it in order to not affect
the original image.

Inputs:(One)
	1. image: image to pre-process
	type: 3D numpy array
Outputs:(One)
	1. copy: a processed copy of the original image
	type: 2D numpy array
"""
def pre_process_image(image, is_cat3=False):
	width, height = len(image[0]) * 3, len(image) * 3
	if is_cat3:
		width, height = 250, 70
	copy = image
	copy = resize_image(copy, width, height)
	if is_cat3:
		copy = Enhance.unsharp_mask(copy)
	copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
	return copy


"""
Given a grayscale image, binarize it using the specified technique.
Technique 1 is adaptive thresholding, technique 2 is isodata 
thresholding. Also applies morphology on the image, based on the
size.

Inputs:(Two)
	1. image: image to binarize and denoise
	type: 2D numpy array
	2. binarize_technique: technique used for binarization
	type: int
Outputs:(One)
	1. image: a binarized and denoised version of the input image
	type: 2D numpy array
"""
def binarize_and_denoise(image, binarize_technique):
	if binarize_technique == 1:
		image = Enhance.binarize_adaptive(image, 1)
	if binarize_technique == 2:
		image = Enhance.binarize(image, 1)
	image = denoise_plate(image)
	return image


"""
Given a list of bounding boxes that most likely contain characters, 
recognizes the characters. Also adds dashes between characters that
have a separation larger than a given threshold.

Inputs:(Two)
	1. listOfChars: a list of bounding boxes likely to contain characters
	type: list containing instances of BoundingBox class, defined in Classes.py
	2. image: image to obtain the characters from
	type: 2D numpy array
Outputs:(One)
	1. recognized_plate: a list of all the characters that were recognized
	type: list of chars
"""
def recognize_plate(listOfChars, image):
	recognized_plate = []
	for i in range(len(listOfChars)):
		bounding_box = listOfChars[i]
		next_bounding_box = None

		if i + 1 < len(listOfChars):
			next_bounding_box = listOfChars[i + 1]

		cropped_image = crop_image(bounding_box, image)
		recognized_char = recognize_char(cropped_image)
		recognized_plate.append(recognized_char)
		if next_bounding_box is not None and good_distance_between_bbs(bounding_box, next_bounding_box, len(image[0])):
			recognized_plate.append('-')

	return recognized_plate
