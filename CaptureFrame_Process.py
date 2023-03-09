import time
import cv2
import pandas as pd
import Localization
import Recognize
import Scenes


"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(six)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""
def CaptureFrame_Process(file_path, sample_frequency, save_path):
    # load video as map of frame number to image
    frames, fps = loadFrames(file_path, sample_frequency)

    # for each frame, locate list of plate images
    # map of frame number to list of images
    tic = time.perf_counter()
    localized = localize_plates(frames)
    toc = time.perf_counter()
    print(f"Completed localization in {toc - tic:0.4f} seconds")

    # divide frames into scenes based on bounding box locations
    scenes = Scenes.frames_to_scenes(localized)

    # for each plate image, segment into characters and recognize them
    # map of frame number to list of strings
    recognized = recognize_plates(localized)

    # majority vote for each scene
    recognized = Scenes.majority_vote(recognized, scenes)

    # save plates to csv
    if len(recognized.items()) > 0:
        save_csv(recognized, save_path, fps)


"""
Load video as list of frames, given file path and sampling frequency

Inputs:(Two)
    1. file_path: path to video file
    type: string
    2. sample_frequency: how often a frame should be taken (2 would mean every other frame)
    type: int
Outputs:(Two)
    1. frames: map of frame numbers to images
    type: dictionary (int to 3D array)
    2. timestamps: map of frame numbers to timestamps
    type: dictionary (int to float)
"""
def loadFrames(file_path, sample_frequency = 1):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Could not load video!!!")
    frames = {}
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    while cap.isOpened():
        retval, frame = cap.read()
        if not retval:
            break
        if frame_count % sample_frequency == 0:
            frames[frame_count] = frame
        frame_count += 1

    return frames, fps


"""
Given map of frame numbers to images, localize plates in them

Inputs:(One)
    1. frames: map of frame numbers to images
    type: dictionary (int to 3D array)
Outputs:(One)
    1. localized: map of frame numbers to list of images of plates detected
    type: dictionary (int to 4D array)
"""
def localize_plates(frames):
    localized = {}
    for frame_nr, frame in frames.items():
        localized_plates = Localization.plate_detection(frame)
        if len(localized_plates) > 0:
            localized[frame_nr] = localized_plates
    return localized


"""
Given localized plates, segment and recognize characters in them

Inputs:(One)
    1. localized: map of frame numbers to list of images
    type: dictionary(int to list of images)
Outputs:(One)
    1. recognized: map of frame numbers to list of strings
    type: dictionary(int to list of strings)
"""
def recognize_plates(localized):
    recognized = {}
    for frame_nr, plates in localized.items():
        for plate, bb in plates:
            recognized_plates = Recognize.segment_and_recognize(plate)
            if recognized_plates is not None:
                recognized[frame_nr] = recognized_plates.upper()
    return recognized


"""
Convert data to csv file with columns:
    1. License plate: string of license plate recognized
    2. Frame no.: frame number of recognized plate
    3. Timestamp: timestamp at which plate is recognized in seconds
    
Inputs:(Three)
    1. plates: map of pair frame numbers to recognized plates
    type: dictionary (int to list of strings)
    2. save_path: file path to save the resulting csv file to
    type: string
    3. timestamps: map of frame number to time stamps
    type: dictionary (int to float)
Outputs:(Zero)
"""
def save_csv(plates, save_path, fps):
    data = []
    tpf = 1/fps
    for frame, plates in plates.items():
        frameNr = str(frame)
        timestamp = str(int(frameNr) * tpf)
        for plate in plates:
            data.append([plate, frameNr, timestamp])
    # convert to csv
    cols = ['License plate', "Frame no.", "Timestamp(seconds)"]
    table = pd.DataFrame(data, columns=cols)
    table.to_csv(save_path, index=False)
    print('Saved output file to: ' + save_path)
