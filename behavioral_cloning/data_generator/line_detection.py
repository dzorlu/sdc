import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

LOW_THRESHOLD_CANNY = 100
HIGH_THRESHOLD_CANNY = 150

def define_vertices(img):
    imshape = img.shape
    vertices = np.array([[(0,imshape[0]), (imshape[1]/2., imshape[0]/2.), (imshape[1],imshape[0])]], dtype=np.int32)
    if vertices.shape[1]:
        vertices = [vertices]
    return vertices

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size=7):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    """
    Take lines as given. Next iteration goes throug some heuristics to merge the lines.
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines_image(img, rho=1, theta=np.pi/90, threshold=25, min_line_len=25, max_line_gap=10):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def hough_lines(img, rho=1, theta=np.pi/90, threshold=25, min_line_len=25, max_line_gap=10):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def draw_lanes(image, lines, color= [255, 0, 0], thickness=2):
    image_shape  = image.shape
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    points_left, points_right = fit_lanes(lines, image_shape)

    l, r = retrieve_min_max_points(points_left, points_right)
    cv2.line(line_img, (l['x1'], l['y1']), (l['x2'], l['y2']), color, thickness)
    cv2.line(line_img, (r['x1'], r['y1']), (r['x2'], r['y2']), color, thickness)
    return line_img


def fit_lanes(lines, image_shape):
    # determine the mid point
    mid_point = image_shape[1]/2

    left_xs = np.arange(0, mid_point, 1).reshape(-1,1)
    right_xs = np.arange(mid_point, image_shape[1], 1).reshape(-1,1)
    # flatten the lines to the points (x1,y1, x2, y2) -> (x1,y1), (x2,y2)
    lines_df = pd.DataFrame([l[0] for l in lines])
    a_, b_ = lines_df.ix[:,2:], lines_df.ix[:,:1]
    a_.columns = b_.columns
    points = np.array(pd.concat([a_,b_]))
    # linear regression for left and right space
    left_points = np.array(list(filter(lambda x: x[0] < mid_point, points )))
    right_points = np.array(list(filter(lambda x: x[0] >= mid_point, points )))
    lr_left, lr_right = LinearRegression(), LinearRegression()

    lr_right.fit(right_points[:,0].reshape(-1,1), right_points[:,1].reshape(-1,1))
    right_ys = lr_right.predict(right_xs).reshape(-1,)
    right_xs = right_xs.reshape(-1,)
    points_right = np.array(list(filter(lambda p: p[1] < image_shape[0], zip(right_xs,right_ys))))

    lr_left.fit(left_points[:,0].reshape(-1,1), left_points[:,1].reshape(-1,1))
    # prediction for left and right space
    left_ys = lr_left.predict(left_xs).reshape(-1,)
    left_xs = left_xs.reshape(-1,)
    points_left = np.array(list(filter(lambda p: p[1] < image_shape[0], zip(left_xs,left_ys))))

    return points_left, points_right

def retrieve_min_max_points(left, right):
    l = {}
    l['x1'], l['y2'] = int(left[:,0].min()), int(left[:,1].min())
    l['x2'], l['y1'] = int(left[:,0].max()), int(left[:,1].max())

    r = {}
    r['x1'], r['y1'] = int(right[:,0].min()), int(right[:,1].min())
    r['x2'], r['y2'] = int(right[:,0].max()), int(right[:,1].max())

    return l, r

def line_detection(x,padding=0):
    x = x.astype(np.uint8)
    # padding = x.shape[0] // 4
    try:
        #crop more
        _x = x[padding:,:,:]
        image = grayscale(_x)
        # Smoothing
        image = gaussian_blur(image)
        # Edge Detection
        image_edges = canny(image,LOW_THRESHOLD_CANNY, HIGH_THRESHOLD_CANNY)
        # Mask Image
        #vertices = define_vertices(image_edges)
        #masked_image = region_of_interest(image_edges, vertices)
        # Find the Lines
        lines = hough_lines(image_edges)
        # Fit the Lanes
        line_image = draw_lanes(_x, lines)
        #Pad it back
        #print(line_image.shape)
        line_image = pad(line_image, padding)
        #print(line_image.shape)
        x = weighted_img(x, line_image)
        ##print("line appended")
    except:
        pass
    return x

def pad(x, padding):
    npad = ((padding,0),(0,0),(0,0))
    return np.pad(x, npad, mode='constant')
