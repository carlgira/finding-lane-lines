import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from sklearn import linear_model


def canny(img, low_threshold, high_threshold):
	"""Applies the Canny transform"""
	return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
	"""Applies a Gaussian Noise kernel"""
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
	""" Applies an image mask to keep the region of the image defined
	by the polygon formed from `vertices`. The rest of the image is set to black.
	"""
	mask = np.zeros_like(img)

	if len(img.shape) > 2:
		channel_count = img.shape[2]
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	cv2.fillPoly(mask, vertices, ignore_mask_color)

	masked_image = cv2.bitwise_and(img, mask)
	return masked_image


def hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap):
	""" Returns an image with hough lines drawn."""

	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	draw_lines(line_img, lines, thickness=15)

	return line_img


def convert_hls(image):
	"""Convert an image to HSL color space"""
	return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def select_white_yellow(image):
	"""Create a mask to select white and yellow colors"""
	converted = convert_hls(image)
	# white color mask
	lower = np.uint8([0, 200,   0])
	upper = np.uint8([255, 255, 255])
	white_mask = cv2.inRange(converted, lower, upper)
	# yellow color mask
	lower = np.uint8([10, 0, 100])
	upper = np.uint8([40, 255, 255])
	yellow_mask = cv2.inRange(converted, lower, upper)
	# combine the mask
	mask = cv2.bitwise_or(white_mask, yellow_mask)
	return cv2.bitwise_and(image, image, mask = mask)


def draw_line(img, color, thickness, y1, y2, slope, intercept):
	""" Function to calculate and draw line based on slope and intercept"""
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
	"""Function to draw the lane lines.

	1. Separate the lines between the right and left lane
	2. Use linear regression models to calculate slope and intercept of each lane
	3. Calculate coordinates of the lane using the slope and intersect
	4. Draw lane line over image
	"""
	imshape = img.shape

	# Divide the points between left and right lane
	left_lines = lines[lines[:, :, 0] < imshape[1]/2]
	right_lines = lines[lines[:, :, 0] > imshape[1]/2]

	# Use a linear regression to get the slope and intersect of left lane
	left_regr = linear_model.LinearRegression()
	left_regr.fit(np.reshape(left_lines[:, [0, 2]], (-1, 1)), np.reshape(left_lines[:, [1, 3]], (-1, 1)))

	# Use a linear regression to get the slope and intersect of right lane
	right_regr = linear_model.LinearRegression()
	right_regr.fit(np.reshape(right_lines[:, [0, 2]], (-1, 1)), np.reshape(right_lines[:, [1, 3]], (-1, 1)))

	# Check if there are right_lines and draw the lane on the image
	if len(right_lines) > 0:
		draw_line(img, color, thickness,  imshape[0], int(imshape[0]*5/8), right_regr.coef_, right_regr.intercept_)

	# Check if there are left_lines and draw the lane on the image
	if len(left_lines) > 0:
		draw_line(img, color, thickness, imshape[0], int(imshape[0]*5/8), left_regr.coef_, left_regr.intercept_)


def process_image(image):
	"""Main function to process image and draw lane lines

	1. Filter colors white and yellow on HSL color space image.
	2. Smooth edges with gaussian blur
	3. Canny edge detection
	4. Create a region of interest using polygon
	5. Get lane lines
		5.1 Hough Line Detection
		5.2 Separate points between left and right lane
		5.3 Use regression models to calculate slope and intersect of left and right lanes
	6. Draw lane line with transparency

	"""
	imshape = image.shape

	# Filter white and yellow color
	filter_img = select_white_yellow(image)

	# Smooth edges with gaussian blur
	blur_img = gaussian_blur(filter_img, kernel_size=9)

	# Canny edges detection
	edges = canny(blur_img, low_threshold=50, high_threshold=100)

	# Create region of interest
	vertices = np.array([[(60, imshape[0]), (imshape[1]*7/16, imshape[0]*5/8), (imshape[1]*9/16, int(imshape[0]*5/8)), (imshape[1]-60, imshape[0])]], dtype=np.int32)
	masked_edges = region_of_interest(edges, vertices)

	# Get lane lines
	line_img = hough_lines(masked_edges, rho=1, theta=np.pi/180, threshold=1, min_line_length=8, max_line_gap=1)

	# Draw lane lines on image with transparency
	copy_image = np.copy(image)
	lines_edges = cv2.addWeighted(copy_image, 0.8, line_img, 1, 0)

	return lines_edges


def process_video(video, video_output):
	""" Proccess frames of video using the process image function to draw lane lines"""
	clip1 = VideoFileClip(video)
	clip = clip1.fl_image(process_image)
	clip.write_videofile(video_output, audio=False, verbose=False, progress_bar=False)

# Process and save all images
mpimg.imsave('output/solidWhiteCurve.jpg', process_image(mpimg.imread('test_images/solidWhiteCurve.jpg')))
mpimg.imsave('output/solidWhiteRight.jpg', process_image(mpimg.imread('test_images/solidWhiteRight.jpg')))
mpimg.imsave('output/solidYellowCurve.jpg', process_image(mpimg.imread('test_images/solidYellowCurve.jpg')))
mpimg.imsave('output/solidYellowCurve2.jpg', process_image(mpimg.imread('test_images/solidYellowCurve2.jpg')))
mpimg.imsave('output/solidYellowLeft.jpg', process_image(mpimg.imread('test_images/solidYellowLeft.jpg')))
mpimg.imsave('output/whiteCarLaneSwitch.jpg', process_image(mpimg.imread('test_images/whiteCarLaneSwitch.jpg')))

# Process and save all videos
process_video("test_videos/solidWhiteRight.mp4", 'output/solidWhiteRight.mp4')
process_video("test_videos/solidYellowLeft.mp4", 'output/solidYellowLeft.mp4')
process_video("test_videos/challenge.mp4", 'output/challenge.mp4')
