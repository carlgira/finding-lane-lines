print(len(lines), lines.shape)
	lines = lines[lines[:,:,2]-lines[:,:,0] != 0]
	print(len(lines), lines.shape)
	lines = np.reshape(lines, (-1, 1, 4))
	print(len(lines), lines.shape)

	slopes = (lines[:,:,3]-lines[:,:,1])/(lines[:,:,2]-lines[:,:,0])
	intercept = lines[:,:,1] - slopes*lines[:,:,0]

	print("righ_slope",slopes[slopes < 0])
	print("left_slope",slopes[slopes > 0])


	righ_slope = np.mean(slopes[slopes < 0])
	left_slope = np.mean(slopes[slopes > 0])



	print(slopes[slopes < 0])

	print(slopes[slopes > 0])

	print(len(slopes), len(slopes[slopes < 0]), len(slopes[slopes > 0]))

	righ_intercept = np.mean(intercept[slopes < 0])
	left_intercept = np.mean(intercept[slopes > 0])

	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(img, (x1, y1), (x2, y2), color, thickness)

	def draw_line(y1, y2, slope, intercept):
		print(y1, y2, slope, intercept)
		x1 = int((y1 - intercept)/(slope+ 1.e-5))
		x2 = int((y2 - intercept)/(slope++ 1.e-5))
		y1 = int(y1)
		y2 = int(y2)
		cv2.line(img, (x1, y1), (x2, y2), color, thickness)

	imshape = img.shape
	draw_line(imshape[0], imshape[0]/2,righ_slope,righ_intercept)
	draw_line(imshape[0], imshape[0]/2,left_slope,left_intercept)



print(len(lines), lines.shape)
	lines = lines[lines[:,:,2]-lines[:,:,0] != 0]
	print(len(lines), lines.shape)
	lines = np.reshape(lines, (-1, 1, 4))
	print(len(lines), lines.shape)

	slopes = (lines[:,:,3]-lines[:,:,1])/(lines[:,:,2]-lines[:,:,0])
	intercept = lines[:,:,1] - slopes*lines[:,:,0]



	righ_slopes = slopes[slopes < 0]
	left_slopes = slopes[slopes > 0]

	print("righ_slope",len(righ_slopes), np.std(righ_slopes), np.mean(righ_slopes) ,righ_slopes)
	print("left_slope",len(left_slopes), np.std(left_slopes), np.mean(left_slopes), left_slopes)

	righ_slopes = righ_slopes[np.abs(righ_slopes) + np.mean(righ_slopes) < np.std(righ_slopes)]
	left_slopes = left_slopes[np.abs(left_slopes) - np.mean(left_slopes) < np.std(left_slopes)]

	print("righ_slope",len(righ_slopes))
	print("left_slope",len(left_slopes))

	righ_slope = np.mean(righ_slopes)
	left_slope = np.mean(left_slopes)

	print(len(slopes), len(slopes[slopes < 0]), len(slopes[slopes > 0]))

	righ_intercept = np.mean(intercept[slopes < 0])
	left_intercept = np.mean(intercept[slopes > 0])

	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(img, (x1, y1), (x2, y2), color, thickness)

	def draw_line(y1, y2, slope, intercept):
		print(y1, y2, slope, intercept)
		x1 = int((y1 - intercept)/(slope+ 1.e-5))
		x2 = int((y2 - intercept)/(slope++ 1.e-5))
		y1 = int(y1)
		y2 = int(y2)
		cv2.line(img, (x1, y1), (x2, y2), color, thickness)

	imshape = img.shape
	draw_line(imshape[0], imshape[0]/2,righ_slope,righ_intercept)
	draw_line(imshape[0], imshape[0]/2,left_slope,left_intercept)


imshape = img.shape
	lines = lines[np.logical_and(lines[:,:,2]-lines[:,:,0] != 0, lines[:,:,3]-lines[:,:,1] != 0 )]
	slopes = (lines[:,3]-lines[:,1])/(lines[:,2]-lines[:,0])

	n_clusters, labels, clusters = cluster_slopes([int(k*1e5) for k in slopes])

	group_lines = []

	for k in range(n_clusters):
		my_members = labels == k
		if sum(my_members) > 2: # Filtering small groups
			group_lines.append(lines[my_members])

	y_min_left_lane = imshape[0]
	y_min_right_lane = imshape[0]



	left_list_slopes = []
	rigth_list_slopes = []
	for group in group_lines:
		slope = np.mean((group[:,3]-group[:,1])/(group[:,2]-group[:,0]))


		if slope > 0:
			left_list_slopes.append(slope)
			slope = np.average(left_list_slopes)
		else:
			rigth_list_slopes.append(slope)
			slope = np.average(rigth_list_slopes)

		intercept = np.mean(group[:,1] - slope*group[:,0])


		xmin = np.mean(group[:,[0,2]])
		xmax = np.mean(group[:,[0,2]])
		ymin = np.mean(group[:,[1,3]])

		def draw_line(y1, y2, slope, intercept):
			x1 = int((y1 - intercept)/(slope))
			x2 = int((y2 - intercept)/(slope))
			y1 = int(y1)
			y2 = int(y2)
			cv2.line(img, (x1, y1), (x2, y2), color, thickness)

		if slope > 0:
			draw_line(y_min_left_lane, ymin,slope,intercept)
			y_min_left_lane = ymin
		else:
			draw_line(y_min_right_lane, ymin,slope,intercept)
			y_min_right_lane = ymin


g = np.reshape(group, (-1, 2))
		g = g[g[:,0].argsort()]


		xp = g[:,0]
		yp = g[:,1]

		f = interp1d(xp, yp)

		x1 = np.min(xp)
		y1 = np.max(yp)

		x2 = np.max(xp)

		if slope < 0:
			y1 = np.min(yp)

		#y2 = int(np.interp(x2, xp, yp))
		y2 = f(x2)

		cv2.line(img, (x1, y1), (x2, y2), color, thickness)

