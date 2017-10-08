import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.interpolate import interp1d


def grayscale(img):
	"""Applies the Grayscale transform
	This will return an image with only one color channel
	but NOTE: to see the returned image as grayscale
	(assuming your grayscaled image is called 'gray')
	you should call plt.imshow(gray, cmap='gray')"""
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Or use BGR2GRAY if you read an image with cv2.imread()
	# return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
	"""Applies the Canny transform"""
	return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
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


def cluster_slopes(x):
	X = np.array(list(zip(x,np.zeros(len(x)))), dtype=np.int)
	bandwidth = estimate_bandwidth(X, quantile=0.05)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit(X)
	labels = ms.labels_

	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)

	for k in range(n_clusters_):
		my_members = labels == k
		#print ("cluster {0}: {1}".format(k, X[my_members, 0]))

	return n_clusters_, labels, X

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
	"""
	NOTE: this is the function you might want to use as a starting point once you want to
	average/extrapolate the line segments you detect to map out the full
	extent of the lane (going from the result shown in raw-lines-example.mp4
	to that shown in P1_example.mp4).

	Think about things like separating line segments by their
	slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
	line vs. the right line.  Then, you can average the position of each of
	the lines and extrapolate to the top and bottom of the lane.

	This function draws `lines` with `color` and `thickness`.
	Lines are drawn on the image inplace (mutates the image).
	If you want to make the lines semi-transparent, think about combining
	this function with the weighted_img() function below
	"""

	# Remove vertical and horizontal lines
	lines = lines[np.logical_and(lines[:,:,2]-lines[:,:,0] != 0, lines[:,:,3]-lines[:,:,1] != 0 )]

	# Slopes and Incetercept of lines
	slopes = (lines[:,3]-lines[:,1])/(lines[:,2]-lines[:,0])
	intercept = lines[:,1] - slopes*lines[:,0]

	# Calculate each line size
	weights = np.linalg.norm(lines[:,[2,3]] - lines[:,[0,1]], axis=1)

	# Creating tuples of slopes and intercept for right and left lane
	left_lines = np.array(list(zip(slopes[slopes > 0].tolist(), intercept[slopes > 0].tolist())))
	right_lines = np.array(list(zip(slopes[slopes < 0].tolist(), intercept[slopes < 0])))

	# Size array of right and left lane
	left_weights = weights[slopes > 0]
	right_weights = weights[slopes < 0]

	# Calculate average slope and intercept using lines sizes as weights (bigger line, more weight)
	left_lane = np.dot(left_weights,  left_lines) /np.sum(left_weights)
	right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights)

	def draw_line(y1, y2, slope, intercept):
		x1 = int((y1 - intercept)/(slope))
		x2 = int((y2 - intercept)/(slope))
		y1 = int(y1)
		y2 = int(y2)
		cv2.line(img, (x1, y1), (x2, y2), color, thickness)

	imshape = img.shape

	if len(right_lines)>0:
		draw_line(imshape[0], imshape[0]*5/8,right_lane[0],right_lane[1])

	if len(left_lines):
		draw_line(imshape[0], imshape[0]*5/8,left_lane[0],left_lane[1])



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	"""
	`img` should be the output of a Canny transform.

	Returns an image with hough lines drawn.
	"""
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	draw_lines(line_img, lines, thickness=15)
	return line_img

# Python 3 has support for cool math symbols.

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

def process_image(image):
	imshape = image.shape
	gray = grayscale(image)

	kernel_size = 9
	blur_gray = gaussian_blur(gray, kernel_size)

	low_threshold = 50
	high_threshold = 100
	edges = canny(blur_gray, low_threshold, high_threshold)


	vertices = np.array([[(80, imshape[0]),(460, imshape[0]/2), (470, int(imshape[0]/2)), (imshape[1]-80,imshape[0])]], dtype=np.int32)
	masked_edges = region_of_interest(edges, vertices)


	rho = 1
	theta = np.pi/180
	threshold = 1
	min_line_length = 8
	max_line_gap = 1
	line_img = hough_lines(masked_edges,rho, theta, threshold, min_line_length, max_line_gap)



	copy_image = np.copy(image)

	# Draw the lines on the edge image
	lines_edges = cv2.addWeighted(copy_image, 0.8, line_img, 1, 0)

	#plt.imshow(lines_edges)
	#plt.show()

	return lines_edges


#process_image(mpimg.imread('test_images/solidWhiteCurve.jpg'))
#process_image(mpimg.imread('test_images/solidWhiteRight.jpg'))
#process_image(mpimg.imread('test_images/solidYellowCurve.jpg'))
#process_image(mpimg.imread('test_images/solidYellowCurve2.jpg'))
#process_image(mpimg.imread('test_images/solidYellowLeft.jpg'))
#process_image(mpimg.imread('test_images/whiteCarLaneSwitch.jpg'))

def process_video(video, video_output):
	clip1 = VideoFileClip(video)
	clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
	clip.write_videofile(video_output, audio=False)

process_video("test_videos/solidWhiteRight.mp4", 'test_videos_output/solidWhiteRight.mp4')
process_video("test_videos/solidYellowLeft.mp4", 'test_videos_output/solidYellowLeft.mp4')
process_video("test_videos/challenge.mp4", 'test_videos_output/challenge.mp4')
