import cv2
import LocalizationUtils
from Morphology import denoise
from Correction import correct_plate_hough
from Classes import BoundingBox


"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
	
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
"""
def plate_detection(image):
	# creating mask
	color_min, color_max = LocalizationUtils.yellow_range()
	mask = LocalizationUtils.create_mask(image, color_min, color_max)

	# denoise mask
	# defined in Morphology.py
	denoised_mask = denoise(mask)

	# filter out components in mask that are likely to be noise
	# append the bounding boxes of potential license plates to a list
	potential_plate_bbs = []
	stats = cv2.connectedComponentsWithStats(denoised_mask, 4)[2]
	for stat in stats:
		width = stat[2]
		height = stat[3]
		area = stat[4]

		if LocalizationUtils.is_not_license_plate_prelim(width, height, area):
			continue
		bounding_box = BoundingBox(stat[1], stat[1] + stat[3], stat[0], stat[0] + stat[2])
		potential_plate_bbs.append(bounding_box)

	# for each potential plate, run some more restricting checks
	# append a rotation corrected version of the plate to a list
	plates = []
	for potential_plate_bb in potential_plate_bbs:
		cropped_image = LocalizationUtils.crop_image(potential_plate_bb, image)
		cropped_mask = LocalizationUtils.crop_image(potential_plate_bb, denoised_mask)
		corrected_image, corrected_mask = correct_plate_hough(cropped_image, cropped_mask)
		refitted_image, refitted_mask = LocalizationUtils.refit_image(corrected_image, corrected_mask)
		if LocalizationUtils.is_not_license_plate(refitted_mask):
			continue
		plates.append((refitted_image, potential_plate_bb))

	return plates
