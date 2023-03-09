import os.path
import numpy as np
import CaptureFrame_Process
import Localization
import json
from Localization import BoundingBox


image_size = None
ERROR_THRESHOLD = 100


"""
Read json file of dictionary of frame number mapped to lists of bounding boxes (see Localization.py)
example:
{
    1: [[0, 1, 0, 1]],
    2: [[1, 2, 1, 2]]
}

Inputs:(One)
    1. file_path: file path of json file to read
    type: string
Outputs:(One)
    1. actuals_bbs: dictionary of frame numbers mapped to a list of bounding boxes
    type: dictionary (int to list of BoundingBox)
"""
def read_bounding_box_labels(file_path):
    actual_bbs = {}
    with open(file_path, 'r') as file:
        pre_processed = json.load(file)
    for frame_nr, pre_processed_bbs in pre_processed.items():
        processed_bbs = []
        for pre_processed_bb in pre_processed_bbs:
            min_x = int(pre_processed_bb[0])
            max_x = int(pre_processed_bb[1])
            min_y = int(pre_processed_bb[2])
            max_y = int(pre_processed_bb[3])
            processed_bbs.append(BoundingBox(min_x, max_x, min_y, max_y))
        actual_bbs[int(frame_nr)] = processed_bbs

    return actual_bbs


"""
Given frame numbers mapped to lists of bounding boxes of training data and labeled data,
Evaluate localization on a scale of 0 - 100 by determining for each plate the endpoint error
If this error is below a certain threshold, it counts as correctly localized plate
The return value is a per
False positives count as -1 correctly localized plate

Inputs:(Two)
    1. localized_bbs: map of frame numbers and bounding box determined by localization algorithm
    type: dictionary (int to BoundingBox)
    2. actual_bbs: map of frame numbers and bounding box predefined manually
    type: dictionary (int to BoundingBox)
Outputs:(One)
    1. correctly_localized_percentage: percentage of correctly localized plates
    type: float
"""
def evaluate_localization(localized_bbs, actual_bbs, frame_map):
    # calculate how many should be localized
    target_localized = 0
    for frame_nr, bbs in actual_bbs.items():
        if frame_nr in frame_map.keys():
            target_localized += len(bbs)
    if target_localized == 0:
        print("No plates to be localised.")
        return 100

    # calculate correctly localized plates
    correctly_localized = 0
    for frame_nr, bbs in localized_bbs.items():
        if frame_nr not in frame_map.keys():
            continue
        for localized_bb in bbs:
            actual_bb_list = actual_bbs.get(frame_nr)
            if actual_bb_list is None or len(actual_bb_list) == 0:  # false positive
                # correctly_localized -= 1
                continue
            # for each bb in the actual frame, calculate error
            # if an error is below the threshold, count as correctly_localized
            # if none are below the threshold it's a false positive
            for actual_bb in actual_bb_list:
                error = endpoint_error(actual_bb, localized_bb)
                if error < ERROR_THRESHOLD:
                    correctly_localized += 1
                    break

    correctly_localized_percentage = (correctly_localized / target_localized) * 100
    return correctly_localized_percentage


"""
Calculate the euclidian distance between a bounding box calculated by
our algorithm and the actual bounding box label

Inputs:(Two)
    1. training_bounding_box: bounding box calculated by algorithm
    type: BoundingBox (see Localization.py)
    2. actual_bounding_box: bounding box predefined manually
    type: BoundingBox (see Localization.py)
Outputs:(One)
    1. error: sum of euclidian distance between corners of bounding boxes
    type: float
"""
def endpoint_error(training_bounding_box, actual_bounding_box):
    # Defining corners
    training_ld_corner = (training_bounding_box.min_x, training_bounding_box.min_y)
    training_rd_corner = (training_bounding_box.max_x, training_bounding_box.min_y)
    training_lu_corner = (training_bounding_box.min_x, training_bounding_box.max_y)
    training_ru_corner = (training_bounding_box.max_x, training_bounding_box.max_y)

    actual_ld_corner = (actual_bounding_box.min_x, actual_bounding_box.min_y)
    actual_rd_corner = (actual_bounding_box.max_x, actual_bounding_box.min_y)
    actual_lu_corner = (actual_bounding_box.min_x, actual_bounding_box.max_y)
    actual_ru_corner = (actual_bounding_box.max_x, actual_bounding_box.max_y)

    # Calculate euclidian distance between each corner and sum
    ld_distance = euclidian_distance(training_ld_corner, actual_ld_corner)
    rd_distance = euclidian_distance(training_rd_corner, actual_rd_corner)
    lu_distance = euclidian_distance(training_lu_corner, actual_lu_corner)
    ru_distance = euclidian_distance(training_ru_corner, actual_ru_corner)
    error = ld_distance + rd_distance + lu_distance + ru_distance
    return error


"""
Calculate euclidian distance between two 2D coordinates defined as a pair of x and y

Inputs:(Two)
    1. p1: first point
    type: pair 
    3. p2: second point
    type: pair
Ouputs:(One)
    1. distance: euclidian distance between the two points
    type: float
"""
def euclidian_distance(p1, p2):
    p1x, p1y = p1
    p2x, p2y = p2
    return np.sqrt(((p2x - p1x) ** 2) + ((p2y - p1y) ** 2))


"""
Evaluate localization for a specific category (1-4)
The evaluation score for the specific category will be printed for the training set
Make sure that the files for both exist and are properly named.

Inputs:(One)
    1. cat: category to report evaluation for
    type: int (1-4)
Outputs:(Zero)
"""
def evaluate_category_training(cat, sample_freq):
    global image_size
    # loading training video
    vid_path = "training/training_vid_cat" + str(cat) + ".mp4"
    if not os.path.exists(vid_path):
        print("Training video for category " + str(cat) + " not found, skipping this category")
        return
    frame_map, _ = CaptureFrame_Process.loadFrames(vid_path, sample_freq)
    image_size = frame_map[0].shape[0] * frame_map[0].shape[1]
    # loading training labels
    label_path = "training/training_labels_cat" + str(cat) + ".json"
    if not os.path.exists(label_path):
        print("Training labels for category " + str(cat) + " not found, skipping this category")
        return
    training_labels = read_bounding_box_labels(label_path)
    # get bounding boxes for training set
    training_bbs = {}
    for frame_nr, frame in frame_map.items():
        localized_list = Localization.plate_detection(frame)
        bbs = []
        for _, bb in localized_list:
            bbs.append(bb)
        training_bbs[frame_nr] = bbs
    # get evaluation score
    score = evaluate_localization(training_bbs, training_labels, frame_map)
    print("Training category " + str(cat) + ": " + str(score) + "%")


"""
Evaluate localization for a specific category (1-4)
The evaluation score for the specific category will be printed for the validation set
Make sure that the files for both exist and are properly named.

Inputs:(One)
    1. cat: category to report evaluation for
    type: int (1-4)
Outputs:(Zero)
"""
def evaluate_category_validation(cat, sample_freq):
    global image_size
    # loading validation video
    vid_path = "validation/tst_vid_cat" + str(cat) + ".mp4"
    if not os.path.exists(vid_path):
        print("Validation video for category " + str(cat) + " not found, skipping this category")
        return
    frame_map, _ = CaptureFrame_Process.loadFrames(vid_path, sample_freq)
    image_size = frame_map[0].shape[0] * frame_map[0].shape[1]
    # loading validation labels
    label_path = "validation/tst_labels_cat" + str(cat) + ".json"
    if not os.path.exists(label_path):
        print("Validation labels for category " + str(cat) + " not found, skipping this category")
        return
    testing_labels = read_bounding_box_labels(label_path)
    # get bounding boxes for training set
    testing_bbs = {}
    for frame_nr, frame in frame_map.items():
        localized_list = Localization.plate_detection(frame)
        bbs = []
        for _, bb in localized_list:
            bbs.append(bb)
        testing_bbs[frame_nr] = bbs
    # get evaluation score
    score = evaluate_localization(testing_bbs, testing_labels, frame_map)
    print("Validation category " + str(cat) + ": " + str(score) + "%")
