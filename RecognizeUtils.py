import cv2
import re
import numpy as np
from Classes import BoundingBox


MIN_SIDE_RATIO = 0.2
MAX_SIDE_RATIO = 0.9

MAX_DIST_DASH_RATIO = 0.055


"""
Given a filepath and a filename, load the image.

Inputs:(Two)
	1. filepath: path to the file to be read
	type: string
	2. filename: name of the file to be read
	type: string
Outputs:(One)
	1. image: loaded image
	type: 2D Array
"""
def loadImage(filepath, filename, grayscale=True):
    return cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


## Initializes the characters and the numbers for the dataset
letter_set = set(['b','d','f','g','h','j','k','l','m','n','p','r','s','t','v','x','z'])
number_set = set(['0','1','2','3','4','5','6','7','8','9'])
reference_characters = {}
for char in letter_set:
	list = []
	list.append(loadImage("dataset/SameSizeLetters/", char +".bmp"))
	list.append(loadImage("dataset/SameSizeLetters/", char + "_left.bmp"))
	list.append(loadImage("dataset/SameSizeLetters/", char + "_right.bmp"))
	reference_characters[char] = list
for char in number_set:
	list = []
	list.append(loadImage("dataset/SameSizeNumbers/", char + ".bmp"))
	list.append(loadImage("dataset/SameSizeNumbers/", char + "_left.bmp"))
	list.append(loadImage("dataset/SameSizeNumbers/", char + "_right.bmp"))
	reference_characters[char] = list


"""
Given a character, find the character with the lowest
xor score to the ones in our dataset.

Inputs:(One)
	1. image: segmented character to be recognized
	type: 2D array
Outputs:(One)
	1. recognized_char: the recognized character
	type: char
"""
def recognize_char(image):
	recognized_char = give_label_lowest_score(image)
	return recognized_char


"""
Given the bounding boxes of the connected components of
an image, determine which of these could potentially be 
a character based on some checks, and add them to a list.

Inputs:(Two)
	1. stats: the stats of the connected components
	type: 2D array
	1. params: the calculated parameters to check for
	type: instance of Params class
Outputs:(One)
	1. result: the list of components assumed to be 
	characters
	type: list
"""
def extract_characters(stats, params):
	result = []
	for stat in stats:
		min_x = stat[0]
		min_y = stat[1]
		width = stat[2]
		height = stat[3]
		area = stat[4]

		bounding_box = BoundingBox(min_y, min_y + height, min_x, min_x + width)

		side_ratio = width / height

		if not_character(width, height, area, side_ratio, params):
			continue

		index = 0

		for j in range(len(result)):
			bb = result[j]
			if bb.min_y < min_x:
				index += 1

		result.insert(index, bounding_box)
	return result


"""
Given two images, calculate the difference between them 
by computing an xor of the images and summing up 
the resulting values.

Inputs:(Two)
	1. test_image: our segmented character from the plate
	type: 2D array
	2. reference_character: a character in the dataset
	type: 2D array
Outputs:(One)
	1. sum: the difference between the images
	type: int
"""
# !The test_image and reference_character must have the same shape
def difference_score(test_image, reference_character):
	xor = np.bitwise_xor(test_image, reference_character).astype(np.uint8)
	return np.sum(xor)


"""
Given a character, find the character with the lowest
xor score to the ones in our dataset.

Inputs:(One)
	1. test_image: segmented character to be recognized
	type: 2D array
Outputs:(One)
	1. min_char: the recognized character
	type: char
"""
def give_label_lowest_score(test_image):
	# Get the difference score with each of the reference characters
	# (or only keep track of the lowest score)
	min_score = 100000000

	for i in reference_characters:
		list_of_chars = reference_characters[i]

		for comparison_file in list_of_chars:
			resized_image = resize_image(comparison_file, len(test_image[0]), len(test_image))
			temp = difference_score(resized_image, test_image)

			if temp < min_score:
				min_score = temp
				min_char = i

	# Return a single character based on the lowest score
	return min_char


"""
Given an image, resize it to be of the dimensions specified.

Inputs:(Three)
	1. image: image to be resized
	type: 2D array
	2. width: new width of the image
	type: int
	3. height: new height of the image
	type: int
Outputs:(One)
	1. resized: the resized image
	type: 2D array
"""
def resize_image(image, width, height):
	resized = cv2.resize(image, (width, height))
	return resized


"""
Given a list of characters, convert it into
a string.

Inputs:(One)
	1. arr: char array
	type: list
Outputs:(One)
	1. res: the computed string
	type: string
"""
def convertArrayToString(arr):
	res = ""
	for i in arr:
		res += i
	return res


"""
Given two bounding boxes, determine if the distance
between them is wide enough to warrant the placement
of a dash in our recognized license plate.

Inputs:(Three)
	1. bb1: first bounding box
	type: instance of BoundingBox class
	2. bb2: 2nd bounding box
	type: list
	3. image_width: width of the image
	type: int
Outputs:(One)
	1. bool: whether or not the distance is good enough
	type: boolean
"""
def good_distance_between_bbs(bb1, bb2, image_width):
	distance = bb2.min_y - bb1.max_y
	return distance > image_width * MAX_DIST_DASH_RATIO


"""
Check whether the connected component is likely to be a 
character based on the pre-computed parameters of the image 
and the stats of the component.

Inputs:(Five)
	1. width: the width of the component
	type: int
	2. height: the height of the component
	type: int
	3. area: the area covered by the component
	type: int
	1. ratio: the ratio of width to height, of the component
	type: int
	1. params: the parameters to check against
	type: instance of Params class
Outputs:(One)
	1. bool: false if the component is likely to 
	contain a character
	type: boolean
"""
def not_character(width, height, area, ratio, params):
	width_bool = params.min_width < width < params.max_width
	height_bool = params.min_height < height < params.max_height
	size_bool = params.min_size < area < params.max_size
	ratio_bool = MIN_SIDE_RATIO < ratio < MAX_SIDE_RATIO
	return not (width_bool and height_bool and size_bool and ratio_bool)


"""
Check whether a plate as a string is a valid license plate

Inputs:(One)
	1. plate: the plate to check
	type: string
Outputs:(One)
	1. is_valid: True iff the plate is a valid license plate
"""
def valid_plate(plate):
	no_dashes = re.sub('-', '', plate)
	return plate is not None and len(no_dashes) == 6

"""
Checks for edge cases following the rules of a dutch license plate.
A letter cannot directly follow a number and must be separated by a
hyphen. In the case that this does happen, we set the letter to the 
number it is most often misclassified as, and vice versa.

Inputs:(One)
	1. plate: the plate to check
	type: string
Outputs:(One)
	1. fixed_plate: the plate with possible fixes
	type: string
"""
def overwrite_mistakes(plate):
	amount_of_letters, amount_of_numbers = count_plate(plate)
	fixed_plate = plate
	for i in range(len(plate)):
		current_letter = fixed_plate[i]
		corrected_letter = None

		next_letter = fixed_plate[i+1] if i + 1 < len(plate) else None
		prev_letter = fixed_plate[i-1] if i - 1 >= 0 else None

		if current_letter in letter_set and (next_letter in number_set or prev_letter in number_set) and amount_of_numbers < 3:
				corrected_letter = mistake_letter_to_number(corrected_letter, current_letter)

		if current_letter in number_set and (next_letter in letter_set or prev_letter in letter_set) and amount_of_numbers > 1:
			corrected_letter = mistake_number_to_letter(corrected_letter, current_letter)

		if corrected_letter != None:
			fixed_plate = fixed_plate[:i] + corrected_letter + fixed_plate[i + 1:]

	return fixed_plate

def mistake_letter_to_number(corrected_letter, current_letter):
	corrected_letter = '8' if current_letter == 'b' else corrected_letter
	corrected_letter = '5' if current_letter == 's' else corrected_letter
	corrected_letter = '0' if current_letter == 'd' else corrected_letter
	corrected_letter = '3' if current_letter == "j" else corrected_letter
	corrected_letter = '2' if current_letter == 'z' else corrected_letter
	corrected_letter = '6' if current_letter == 'g' else corrected_letter
	return corrected_letter

def mistake_number_to_letter(corrected_letter, current_letter):
	corrected_letter = 'd' if current_letter == '0' else corrected_letter
	corrected_letter = 'b' if current_letter == '8' else corrected_letter
	corrected_letter = 's' if current_letter == '5' else corrected_letter
	corrected_letter = 'j' if current_letter == '3' else corrected_letter
	corrected_letter = 'z' if current_letter == '2' else corrected_letter
	corrected_letter = 'g' if current_letter == '6' else corrected_letter
	return corrected_letter

def count_plate(plate):
	letters = 0
	numbers = 0
	for i in range(len(plate)):
		if plate[i] in letter_set:
			letters+=1
		if plate[i] in number_set:
			numbers+=1
	return letters, numbers




