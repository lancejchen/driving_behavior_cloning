import numpy as np
import cv2

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillConvexPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def detect_lane_line(image):
    gray_img = grayscale(image)
    blur_gray = gaussian_blur(gray_img, kernel_size=5)
    edges = canny(blur_gray, low_threshold=50, high_threshold=150)
    imshape = image.shape
    # vertices = np.array([[(100,imshape[0]),(480, 310), (490, 315), (900,imshape[0])]])
    vertices = np.array([[(0, imshape[0]),(0, 70), (imshape[1], 70), (imshape[1], imshape[0])]])
    marked_edges = region_of_interest(edges, vertices)
    clipped_img = marked_edges[70:]
    resized_img = cv2.resize(clipped_img, (0, 0), fx=0.25, fy=0.25)
    resized_img = np.expand_dims(resized_img, axis=-1)
    # print('normalized image is: ', np.max(img_norm))
    # print(time.time())
    return resized_img 

