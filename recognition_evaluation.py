import os
import cv2
from json import load
import argparse
import Recognize


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--print', type=bool, default=False)
	args = parser.parse_args()
	return args


"""
Read json file of labels to map of file names to license plates as string.

Inputs:(One)
	1. file_path: path of file too read
	type: string
Outputs:(One)
	1. labels: map of file names to license plates as string.
	type: dictionary (string to string)
"""
def read_labels(file_path):
	with open(file_path) as json_file:
		return load(json_file)


"""
Calculate the recognition evaluation score of data in a folder.

Inputs:(One)
	1. file_path: path to folder containing images of plates and json file of labels
	type: string
Ouputs:(One)
	1. score: evaluation score of recognition (value between 0 and 100)
"""
def recognition_score(file_path, training_data, print_all = True):
	labels = None
	images = {}
	# reading images and labels into maps
	for filename in os.listdir(file_path):
		if "plate" in filename:
			file_nr = ''.join(c for c in filename if c.isdigit())
			images[file_nr] = cv2.imread(file_path + "/" + filename)
		else:
			labels = read_labels(file_path + "/" + filename)

	correctly_recognized = 0
	for filename, label in labels.items():
		actual = Recognize.segment_and_recognize(images[filename])
		if actual is None:
			continue
		if actual == label:
			correctly_recognized += 1
			if training_data and print_all:
				print(label + " --- CORRECT")
		elif training_data and print_all:
			print(actual + " --- should have been --- " + label)

	if labels is None or len(labels) == 0:
		print("No labels found")
		return 0
	return correctly_recognized/len(labels) * 100


"""
Evaluate recognition
"""
if __name__ == '__main__':
	print_all = get_args().print
	print("Evaluating recognition...")
	training_score_1 = recognition_score('dataset/RecognitionTrainingSet/category1', True, print_all)
	print("Training score category 1: " + str(training_score_1) + "%")
	training_score_2 = recognition_score('dataset/RecognitionTrainingSet/category2', True, print_all)
	print("Training score category 2: " + str(training_score_2) + "%")
	training_score_3 = recognition_score('dataset/RecognitionTrainingSet/category3', True, print_all)
	print("Training score category 3: " + str(training_score_3) + "%")
	training_score_4 = recognition_score('dataset/RecognitionTrainingSet/category4', True, print_all)
	print("Training score category 4: " + str(training_score_4) + "%")
	validation_score_1 = recognition_score('dataset/RecognitionValidationSet/category1', False, print_all)
	print("Validation score category 1: " + str(validation_score_1) + "%")
	validation_score_2 = recognition_score('dataset/RecognitionValidationSet/category2', False, print_all)
	print("Validation score category 2: " + str(validation_score_2) + "%")
	validation_score_3 = recognition_score('dataset/RecognitionValidationSet/category3', False, print_all)
	print("Validation score category 3: " + str(validation_score_3) + "%")
	validation_score_4 = recognition_score('dataset/RecognitionValidationSet/category4', False, print_all)
	print("Validation score category 4: " + str(training_score_4) + "%")
