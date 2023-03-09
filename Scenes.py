import LocalizationEvaluation
from Classes import Plate, Group


BB_ERROR_FACTOR = 0.03
SIMILARITY_THRESHOLD = 0.5


"""
Check if two bounding boxes are similar, aka their endpoint error is small enough

Inputs:(Two)
    1. bb1: first bounding box
    type: BoundingBox
    2. bb2: second bounding box
    type: BoundingBox
Outputs:(One)
    1. similar: true iff two bounding boxes are similar
    type: boolean
"""
def similar_bounding_boxes(bb1, bb2):
    if bb1 is None and bb2 is None:
        return True
    if bb1 is None or bb2 is None:
        return False
    error = LocalizationEvaluation.endpoint_error(bb1, bb2)
    max_error = BB_ERROR_FACTOR * bb1.size()
    return error < max_error


"""
Divide frames into scenes based on location of bounding boxes

Inputs:(One)
    1. localized: map of frame numbers to pair of plate and bounding box
    type: dictionary int to pair of image and BoundingBox
Outputs:(One)
    1. scenes: list where the ith element contains a list of frame_numbers in the ith scene
    type: 2D list of ints
"""
def frames_to_scenes(localized):
    scenes = []
    curr_scene = 0
    prev = None
    for frame_nr, plates in localized.items():
        bbs = []
        for _, bb in plates:
            bbs.append(bb)
        if prev is None:
            scenes.append([frame_nr])
        else:
            any_similar = False
            # check if any of the bounding boxes in current frame are similar to any in the previous frame
            for bb in bbs:
                if prev is None:
                    break
                for bb_prev in prev:
                    if similar_bounding_boxes(bb, bb_prev):
                        any_similar = True
                        break
            if any_similar: # if similar add it to current scene
                scenes[curr_scene].append(frame_nr)
            else: # else create new scene
                curr_scene += 1
                scenes.append([frame_nr])
        prev = bbs
    return scenes


"""
Get all of the plates in a scene based on the frame numbers in that scene

Inputs:(Two)
    1. frame_nrs: list of frame numbers the plates are in
    type: list(int)
    2. recognized: map of frame numbers to plates recognized in that frame
    type: dict(int to list(string))
Outputs:(One)
    1. res: list of plates
    type: list(Plate)
"""
def get_plates(frame_nrs, recognized):
    res = []
    for frame_nr in frame_nrs:
        plate = recognized.get(frame_nr, None)
        if plate is not None:
            res.append(Plate(frame_nr, plate))
    return res


"""
Check how similar two strings are, gives a number between 0 and 1.

Inputs:(Two)
    1. s1: first string
    type: string
    2. s2: second string
    type: string
Ouputs:(One)
    1. similarity: similarity between the strings
    type: float (0-1)
"""
def similarity(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


"""
Given a list of recognized plates, group them together based on similarity

Inputs:(One)
    1. plates: list of plate objects
    type: list(Plate)
Outputs:(One)
    1. groups: list of groups created
    type: list(Group)
"""
def group_similar_strings(plates):
    groups = []
    for plate in plates:
        added = False
        for group in groups:
            for group_plate in group.plates:
                sim = similarity(group_plate.content, plate.content)
                if sim > SIMILARITY_THRESHOLD:
                    group.add_plate(plate)
                    added = True
                    break
        if not added:
            groups.append(Group([plate]))
    return groups


"""
Find most common plate in group

Inputs:(One)
    1. group: list of groups to find the most common plate in
    type: list(Plate)
Outputs:
    1. max_str: most common plate in group as string
    type: string
"""
def most_common_plate(group):
    counts = {}
    max_count = 0
    max_str = ''
    for plate in group.plates:
        content = plate.content
        new_count = counts.get(content, 0) + 1
        counts[content] = new_count
        if new_count > max_count:
            max_count = new_count
            max_str = content
    return max_str


"""
Given localized plates and frames divided into scenes, do a majority vote on scenes

Inputs:(Two)
    1. recognized: recognized plates
    type: dictionary (int to list of strings)
    2. scenes: frames divided into scenes
    type: 2D list of ints
Outputs:(One)
    1. voted_recognized: recognized plates after voting
    type: dictionary (int to list of strings)
"""
def majority_vote(recognized, scenes):
    voted_recognized = {}
    # perform majority vote for each scene
    for i in range(len(scenes)):
        frame_nrs = scenes[i]
        # get all recognized plates in that scene
        plates = get_plates(frame_nrs, recognized)
        # group by similarity
        groups = group_similar_strings(plates)
        # for each group do majority vote and add to result
        for group in groups:
            group_frame_nr = group.get_frame()
            current_recognized = voted_recognized.get(group_frame_nr, [])
            current_recognized.append(most_common_plate(group))
            voted_recognized[group_frame_nr] = current_recognized
    return voted_recognized
