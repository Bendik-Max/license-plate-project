import cv2
import numpy as np


EPSILON = 0.001


"""
Clamp grayscale colours between 0 and 255

Inputs:(One)
    1. colour: some grayscale colour
    type: int
Outputs:(One)
    1. clamped: the grayscale colour clamped between 0 and 255
    type: int
"""
def clamp_grayscale(colour):
    clamped = max(min(colour, 255), 0)
    return clamped


"""
Calculate global isodata threshold given a grayscale image

Inputs:(One)
    1. image: grayscale image to calculate the global threshold of
    type: 2D array 
Outputs:(One)
    1. threshold: calculated isodata threshold (between 0 and 255)
    type: int
"""
def isodata_threshold(image):
    hist = np.bincount(np.floor(image.flatten()).astype(np.uint8), minlength=256)
    nonzero = np.where(hist != 0)[0]
    gmin = nonzero[0]
    gmax = nonzero[::-1][0]
    # mapping i to ti
    t = {0: np.average([gmin, gmax])}
    min_diff = np.inf
    min_i = 0
    for i in range(0, 255):
        # calculate m1
        numerator = 0
        for g in range(gmin, i + 1):
            numerator += g * hist[g]
        denominator = np.sum(hist[gmin:i+1]) + 0.000001
        m1 = numerator/denominator
        # calculate m2
        numerator = 0
        for g in range(i, gmax + 1):
            numerator += g * hist[g]
        denominator = np.sum(hist[i:gmax+1]) + 0.000001
        m2 = numerator/denominator

        t[i + 1] = np.average([m1, m2])
        curr_diff = np.abs(t[i + 1] - t[i])
        if curr_diff < min_diff:
            min_diff = curr_diff
            min_i = i
        if curr_diff <= EPSILON:
            return np.floor(t[i])
    return np.floor(t[min_i])


"""
Apply contrast stretching to some image

Inputs:(Three)
    1. img: the image to perform contrast stretching on
    type: image as numpy array (grayscale)
    2. alpha: alpha for contrast stretching
    type: float
    3. beta: the beta to use for contrast stretching
    type: float
Outputs:(One)
    1. stretched: image with contrast stretching applied to it
        type: image as numpy array (grayscale)
"""
def contrast_stretching(img, alpha = 0, beta=1):
    copy = img
    return (255 * cv2.normalize(copy, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)).astype(np.uint8)


"""
Given a binarized image, invert the colours

Inputs:(One)
    image: binarized image to invert
    type: array (2D)
Outputs:(One)
    inverted: inverted binary image
    type: array (2D)
"""
def invert_colours(image):
    inverted = image
    for i in range(0, len(image)):
        for j in range(0, len(image[i])):
            inverted[i,j] = 0 if inverted[i,j] == 255 else 255
    return inverted


"""
Given a binarized image of a license plate, make sure the characters are white and the rest of
the image black, by inverting the image if the background is white
this assumes that the background is the dominant colour in the image

Inputs:(One)
    1. image: binarized image to perform the operation on
    type: array (2D)
Outputs:(One)
    1. white_chars_plate: image where characters should be white
    type: array (2D)
"""
def white_chars(image):
    white_chars_plate = image
    dominant_colour = 255 if np.count_nonzero(image) >= len(image)/2 else 0
    # invert image if white is the dominant colour
    if dominant_colour == 255:
        white_chars_plate = invert_colours(white_chars_plate)
    return white_chars_plate


"""
Binarize an image
"""
def binarize(img, enhance_technique = 1):
    enhanced_image = img
    if enhance_technique == 1:  # contrast stretching for dark images
        enhanced_image = contrast_stretching(enhanced_image, 0, 1)
    if enhance_technique == 2:  # histogram equalization
        enhanced_image = cv2.equalizeHist(enhanced_image)
    enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    threshold = isodata_threshold(enhanced_image)
    _, binarized = cv2.threshold(enhanced_image, threshold, 255, cv2.THRESH_BINARY)
    # make sure characters are white
    return white_chars(binarized)


"""
Binarize an image
"""
def binarize_adaptive(img, enhance_technique = 1):
    enhanced_image = img
    if enhance_technique == 1:  # contrast stretching for dark images
        enhanced_image = contrast_stretching(enhanced_image, 0.2, 0.02)
    if enhance_technique == 2:  # histogram equalization
        enhanced_image = cv2.equalizeHist(enhanced_image)
    #enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    binarized = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
    # make sure characters are white
    return white_chars(binarized)


"""
Sharpen an unsharp mask

Inputs:(Four)
    1. image: image to sharpen
    type: np array 2D
    2. kernel_size: size of gaussian kernel
    type: pair of ints
    3. sigma: sigma for gaussian blur
    type: float
    4. sharp_factor: multiplication factor for sharpening
    type: float
Outputs:(One)
    1. sharpened: sharpened image
    type: np array 2D
"""
def unsharp_mask(image, kernel_size=(5, 5), sigma=1, sharp_factor=2.0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(sharp_factor + 1) * image - float(sharp_factor) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    return sharpened
