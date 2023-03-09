import cv2

# Structuring elements
# rectangles
RECT_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
RECT_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
RECT_4 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
RECT_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# ellipses
ELLIPSE_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
ELLIPSE_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
ELLIPSE_4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
ELLIPSE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ELLIPSE_6 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
ELLIPSE_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
ELLIPSE_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
ELLIPSE_12 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
# crosses
CROSS_3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
CROSS_4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
CROSS_5 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
CROSS_6 = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))
CROSS_7 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
CROSS_10 = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))


"""
Denoise an image by morphology

Inputs:(One)
    1. image: image to denoise
    type: array with dimension >= 2
Outputs:(One)
    1. denoised: denoised image
    type: array with dimension >= 2 (same as original image)
"""
def denoise(image):
    denoised = image
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, ELLIPSE_7)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, ELLIPSE_12)
    denoised = cv2.dilate(denoised, ELLIPSE_12)
    return denoised


"""
Denoise a license plate after binarization. This is done based on the size
of the image.

Inputs:(One)
    1. image: the image to denoise
    type: np array 2D
Outputs:(One)
    1. denoised: the denoised image
    type: np array 2D
"""
def denoise_plate(image):
    denoised = image
    size = image.size
    if 1000 <= size <= 10000:
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, ELLIPSE_2)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, ELLIPSE_2)
    if 10000 < size <= 21000:
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, ELLIPSE_3)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, ELLIPSE_3)
    if 21000 < size <= 70000:
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, ELLIPSE_4)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, ELLIPSE_4)
    if 70000 < size <= 150000:
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, ELLIPSE_5)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, ELLIPSE_5)
    if 150000 < size:
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, ELLIPSE_6)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, ELLIPSE_6)
    return denoised
