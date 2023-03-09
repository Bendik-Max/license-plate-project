from Localization import plate_detection
import cv2
import argparse
import Enhance
from Recognize import resize_image
from Morphology import denoise_plate



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='dataset/dummytestvideo.avi')
    parser.add_argument('--binarized', type=bool, default=False)
    parser.add_argument('--sharpened', type=bool, default=False)
    args = parser.parse_args()
    return args



# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
args = get_args()
binarized = get_args().binarized
sharpened = get_args().sharpened
file_path = args.file_path
cap = cv2.VideoCapture(file_path)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    #dummy arguments for sample frequency and save_path should be changed
    detections = plate_detection(frame)
    # Display the resulting frame
    if len(detections) > 0 and len(detections[0]) > 0:
        for plate, bounding_box in detections:
            if plate is None or len(plate) == 0 or len(plate[0]) == 0:
                continue
            if sharpened or binarized:
                print(plate.size)
                plate = resize_image(plate, len(plate[0]) * 4, len(plate) * 4)
            cv2.imshow('Frame', plate)
            cv2.waitKey(0)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break


  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()



