import numpy as np
import cv2


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


def hough_lines_draw(img, img1, img2, outfile, outfile2, peaks, rhos, thetas):
    for peak in peaks:
        rho = rhos[peak[0]]
        theta = thetas[peak[1]] * np.pi / 180.0
        a = np.cos(theta)
        b = np.sin(theta)
        pt0 = rho * np.array([a,b])
        pt1 = tuple((pt0 + 1000 * np.array([-b, a])).astype(int))
        pt2 = tuple((pt0 - 1000 * np.array([-b, a])).astype(int))
        if pt1[0] < 0:
            cv2.line(img, pt1, pt2, (255, 255, 0), 3)
        else:
            print(a, b)
            cv2.line(img2, pt1, pt2, (255, 255, 255), 3)
    cv2.imwrite(outfile, img)
    cv2.imwrite(outfile2, img2)
    return img


def hough_lines_draw_final(img, img1, outfile, peaks, rhos, thetas):
    for peak in peaks:
        rho = rhos[peak[0]]
        theta = thetas[peak[1]] * np.pi / 180.0
        a = np.cos(theta)
        b = np.sin(theta)
        pt0 = rho * np.array([a, b])
        pt1 = tuple((pt0 + 1000 * np.array([-b, a])).astype(int))
        pt2 = tuple((pt0 - 1000 * np.array([-b, a])).astype(int))
        cv2.line(img, pt1, pt2, (255, 255, 255), 3)
    cv2.imwrite(outfile, img)
    return img


def hough_lines_acc(img, rho_res=1, thetas = np.arange(-90, 90, 1)):
    rho_max = int(np.linalg.norm(img.shape-np.array([1, 1]), 2))
    rhos = np.arange(-rho_max, rho_max, rho_res)
    thetas -= min(min(thetas),0)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    yis, xis = np.nonzero(img)  # use only edge points
    for idx in range(len(xis)):
        x = xis[idx]
        y = yis[idx]
        temp_rhos = x * np.cos(np.deg2rad(thetas)) + y * np.sin(np.deg2rad(thetas))
        temp_rhos = temp_rhos / rho_res + rho_max
        m, n = accumulator.shape
        valid_idxs = np.nonzero((temp_rhos < m) & (thetas < n))
        temp_rhos = temp_rhos[valid_idxs]
        temp_thetas = thetas[valid_idxs]
        c = np.stack([temp_rhos,temp_thetas], 1)
        cc = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1])))
        _, idxs, counts = np.unique(cc, return_index=True, return_counts=True)
        uc = c[idxs].astype(np.uint)
        accumulator[uc[:, 0], uc[:, 1]] += counts.astype(np.uint)
    accumulator = cv2.normalize(accumulator, accumulator, 0, 255,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return accumulator, thetas, rhos


def clip(idx):
    return int(max(idx, 0))


def hough_peaks(H, numpeaks=1, threshold=100, nhood_size=5):
    peaks = np.zeros((numpeaks, 2), dtype=np.uint64)
    temp_H = H.copy()
    for i in range(numpeaks):
        _, max_val, _, max_loc = cv2.minMaxLoc(temp_H)  # find maximum peak
        if max_val > threshold:
            peaks[i] = max_loc
            (c, r) = max_loc
            t = nhood_size//2.0
            temp_H[clip(r-t):int(r+t+1), clip(c-t):int(c+t+1)] = 0
        else:
            peaks = peaks[:i]
            break
    return peaks[:, ::-1]


image = cv2.imread("input_images/hough.jpg", cv2.IMREAD_GRAYSCALE)
image_1 = cv2.imread("input_images/hough.jpg", cv2.IMREAD_GRAYSCALE)

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
img_x = conv(image, sobel_x)
img_y = conv(image, sobel_y)
edge_img = norm(img_x, img_y)

cv2.imwrite("Hough_edges.jpg", edge_img)
H, thetas, rhos = hough_lines_acc(edge_img)
peaks = hough_peaks(H, numpeaks=30, threshold=150, nhood_size=20)

cv2.imwrite("Sine_wave.jpg", H)
color_img1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
color_img2 = cv2.cvtColor(image_1, cv2.COLOR_GRAY2BGR)
res_img = hough_lines_draw(image, color_img1, color_img2, "Diagonal_lines.jpg", "vertical_lines.jpg", peaks, rhos, thetas)

final = hough_lines_draw_final(image, color_img2, "Hough_final.jpg", peaks, rhos, thetas)
