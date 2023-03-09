import argparse
import os
import cv2
import CaptureFrame_Process
import json


"""
This file is used for labelling images for the evaluation of our localization
You can find a guide for this in Evaluation guide localization.pdf
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='labels')
    args = parser.parse_args()
    return args


x1 = 0
y1 = 0
x2 = 0
y2 = 0
img = None
drawing = False
window_name = None
current = []


"""
Callback function for drawing rectangle
"""
def draw_rectangle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, img, window_name, current
    if event == cv2.EVENT_LBUTTONDOWN: # start box
        drawing = True
        x1 = x
        y1 = y
    elif event == cv2.EVENT_RBUTTONUP: # clear boxes
        current = []
        cv2.imshow(window_name, img)
    elif event == cv2.EVENT_LBUTTONUP: # end box
        if drawing:
            drawing = False
            min_x = min(x1, x2)
            max_x = max(x1, x2)
            min_y = min(y1, y2)
            max_y = max(y1, y2)
            print("min_x: " + str(min_x) + ", max_x: " + str(max_x) + ", min_y: " + str(min_y) + ", max_y: " + str(max_y))
            print("press any key to confirm or draw different box")
            current.append([min_y, max_y, min_x, max_x])
    elif event == cv2.EVENT_MOUSEMOVE: # move box
        if not drawing:
            return
        x2 = x
        y2 = y
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        rectangle = cv2.rectangle(img.copy(), (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        # draw old boxes
        for box in current:
            rectangle = cv2.rectangle(rectangle, (box[2], box[0]), (box[3], box[1]), (0, 255, 0), 1, 8)
        cv2.imshow(window_name, rectangle)


"""
Create the label json file for a video.

Inputs:(One)
    1. frame_map: map of frame_numbers to frames
    type: dictionary (int to np array)
Outputs:(One)
    1. json_map: map of the resulting json
    type: string
"""
def create_label_json(frame_map):
    global img, x1, x2, y1, y2, window_name, current
    json_map = {}
    print("Welcome to our labeling application, draw a box or press any key to skip this frame.")
    for frame_nr, frame in frame_map.items():
        # Show image and draw rectangle
        img = frame
        window_name = "Label frame: " + str(frame_nr) + "/" + str(len(frame_map.items()))
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 600)
        cv2.moveWindow(window_name, 200, 200)
        cv2.imshow(window_name, frame)
        cv2.setMouseCallback(window_name, draw_rectangle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(current) > 0:
            if frame_nr not in json_map.keys():
                json_map[frame_nr] = []
            # add to result
            for bb in current:
                json_map[frame_nr].append(bb.copy())
            current = []
    return json_map


"""
Label application
"""
if __name__ == '__main__':
    args = get_args()
    input_path = args.input
    output_path = args.output

    if input_path is None or not os.path.exists(input_path):
        print("Please provide a valid input video.")

    # load frames as map of frame number to frame
    frame_map, timestamps = CaptureFrame_Process.loadFrames(input_path)

    # create json file
    json_file = create_label_json(frame_map)
    # save json file
    with open(output_path + ".json", 'w') as f:
        json.dump(json_file, f, sort_keys=True, indent=4)
