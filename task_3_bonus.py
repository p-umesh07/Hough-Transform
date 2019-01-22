import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy


def gaussian_smoothing(input_img):
    gaussian_filter = np.array([[0.109, 0.111, 0.109], [0.111, 0.135, 0.111], [0.109, 0.111, 0.109]])
    return cv2.filter2D(input_img, -1, gaussian_filter)


def canny_edge_detection(input):
    input = input.astype('uint8')
    # Using OTSU thresholding - bimodal image
    otsu_threshold_val, ret_matrix = cv2.threshold(input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lower_threshold = otsu_threshold_val * 0.8
    # upper_threshold = otsu_threshold_val * 1.7
    lower_threshold = otsu_threshold_val * 0.4
    upper_threshold = otsu_threshold_val * 1.3
    # print(lower_threshold,upper_threshold)
    # print(lower_threshold,upper_threshold)
    edges = cv2.Canny(input, lower_threshold, upper_threshold)
    return edges


def conv_transform(image):
    image_copy = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j] = image[image.shape[0] - i - 1][image.shape[1] - j - 1]
    return image_copy


def conv(image, kernel):
    kernel = conv_transform(kernel)
    image_h = image.shape[0]
    image_w = image.shape[1]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros(image.shape)

    for i in range(h, image_h - h):
        for j in range(w, image_w - w):
            sum = 0

            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = (sum + kernel[m][n] * image[i - h + m][j - w + n])

            image_conv[i][j] = sum

    return image_conv


def norm(img1, img2):
    img_copy = np.zeros(img1.shape)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            q = (img1[i][j] * 2 + img2[i][j] * 2) * (1 / 2)
            if (q > 90):
                img_copy[i][j] = 255
            else:
                img_copy[i][j] = 0

    return img_copy


def HoughCircles(input, circles):
    rows = input.shape[0]
    cols = input.shape[1]

    # initializing the angles to be computed
    sinang = dict()
    cosang = dict()

    # initializing the angles
    for angle in range(0, 360):
        sinang[angle] = np.sin(angle * np.pi / 180)
        cosang[angle] = np.cos(angle * np.pi / 180)

        # initializing the different radius
    # For Test Image <----------------------------PLEASE SEE BEFORE RUNNING------------------------------->
    # radius = [i for i in range(10,70)]
    # For Generic Images
    # length=int(rows/2)
    length = 23
    radius = [i for i in range(19, 25)]

    for r in radius:
        # Initializing an empty 2D array with zeroes
        acc_cells = np.full((rows, cols), fill_value=0, dtype=np.uint64)
        # print(acc_cells.shape)
        # Iterating through the original image
        for x in range(rows):
            for y in range(cols):
                if input[x][y] == 255:  # edge
                    # increment in the accumulator cells
                    for angle in range(0, 360):
                        b = y - round(r * sinang[angle])
                        a = x - round(r * cosang[angle])
                        if a >= 0 and a < rows and b >= 0 and b < cols:
                            a1 = int(a)
                            b1 = int(b)
                            # print(a1,b1)
                            acc_cells[a1][b1] += 1

        # print('For radius: ',r)
        acc_cell_max = np.amax(acc_cells)
        # print('max acc value: ',acc_cell_max)

        if (acc_cell_max > 150):

            # print("Detecting the circles for radius: ",r)

            # Initial threshold
            acc_cells[acc_cells < 150] = 0

            # find the circles for this radius
            for i in range(rows):
                for j in range(cols):
                    if (i > 0 and j > 0 and i < rows - 1 and j < cols - 1 and acc_cells[i][j] >= 150):
                        avg_sum = np.float32((acc_cells[i][j] + acc_cells[i - 1][j] + acc_cells[i + 1][j] +
                                              acc_cells[i][j - 1] + acc_cells[i][j + 1] + acc_cells[i - 1][j - 1] +
                                              acc_cells[i - 1][j + 1] + acc_cells[i + 1][j - 1] + acc_cells[i + 1][
                                                  j + 1]) / 9)
                        # print("Intermediate avg_sum: ",avg_sum)
                        if (avg_sum >= 50):
                            # print("For radius: ",r,"average: ",avg_sum,"\n")
                            circles.append((i, j, r))
                            acc_cells[i:i + 5, j:j + 7] = 0


def main():
    img_path = "input_images/hough.jpg"
    # sample_inp1_path = 'circle_sample_1.jpg'

    orig_img = cv2.imread(img_path)

    sobel_x = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ])
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    # Reading the input image and converting to gray scale
    inpu = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Create copy of the orignial image
    input_img = deepcopy(inpu)

    # Steps
    # 1. Denoise using Gaussian filter and detect edges using canny edge detector
    smoothed_img = gaussian_smoothing(input_img)

    # 2. Detect Edges using Canny Edge Detector
    # edged_image = canny_edge_detection(smoothed_img)

    imgx = conv(input_img, sobel_x)
    imgy = conv(input_img, sobel_y)
    edged_image = norm(imgx, imgy)
    # 3. Detect Circle radius
    # 4. Perform Circle Hough Transform
    circles = []

    # cv2.imshow('Circle Detected Image',edged_image)

    # Detect Circle
    HoughCircles(edged_image, circles)

    # Print the output
    for vertex in circles:
        cv2.circle(orig_img, (vertex[1], vertex[0]), vertex[2], (0, 255, 255), 1)
        # cv2.rectangle(orig_img,(vertex[1]-2,vertex[0]-2),(vertex[1]-2,vertex[0]-2),(0,0,255),3)

    # print(circles)

    # cv2.imshow('Circle Detected Image',orig_img)
    cv2.imwrite("coins.jpg", orig_img)


if __name__ == '__main__':
    main()