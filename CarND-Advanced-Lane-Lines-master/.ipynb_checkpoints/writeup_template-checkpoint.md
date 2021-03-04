## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

All the code found in this step can be found in `camera_cal/cam_cal.py` (Note: code taken from Udacity lessons). What the script does is that it calibrates the camera by using open-cv's `findChessboardCorners()` function as it loops through the images in that directory. We feed the output of the function right into the `calibrateCamera()`. From there, I choose to simply use pickle to output the values (specifically the mtx and dist columns) for further usage in the future. Take a look at the example below:

![alt text](camera_cal/calibration3.jpg)

After undistortion:

![alt text](camera_cal/calibrated.jpg)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Because I saved the mtx and dist values to the pickle file, I can simply import them back and apply them on the image.

![alt text](image1.jpg)

Output (pay close attention to the white car):

![alt text](image.jpg)




#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

How I took the gradients of the image:

- Took the absolute value of the Sobal operator (taken from the Udacity lesson) to get the gradients of the directions of the x and y points. Take a look at image_gen.py and the abs_sobel_thersh() function. We identify gradients in range (12,255) for x and (25,255) for y (look at line 76 and 77). 
- After that, we apply our color_threshold function to help grab the s and v channels of our image. Using those values, we can perform the binary threshold of the image (line 78).
- From there, we do the binary earch by checking if the x and y gradients are 1. If so, we set the pixels to be white. Else if the output of the colour threshold is 1, then we also still output 255.

Input image:

![alt text](test_images/test1.jpg)

Output image:

![alt text](test_images/gradient0.jpg)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

```python
img_size = (img.shape[1], img.shape[0])
    bot_width = .75 #changed form .76
    mid_width = .1 #changed from .08
    height_pct = .625
    bottom_trim = .935
    src = np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(.5+mid_width/2),img.shape[0]*height_pct],
                      [img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim],[img.shape[1]*(.5-bot_width/2),img.shape[0]*bottom_trim]])
    offset = img_size[0]*.25
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],[offset, img_size[1]]])
    
M = cv2.getPerspectiveTransform(src, dst) 
Minv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(preprocessImage, M, img_size,flags=cv2.INTER_LINEAR)
```

What I'm doing here is that I take the image shape to extract its dimensions and then create a trapezoid using the src and dst variables. I find 4 points in the original image and then use the `cv2.getPerspectiveTransform` and `cv2.warpPerspective` to get a bird's eye view of the lane lines. I feed the dst and src values into the open-cv functions so that we cna generate the final result. Take a look below:

Input image:

![alt text](test_images/test1.jpg)

Output image:

![alt text](test_images/perspective0.jpg)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the greenboxes to get an estimation of the polynomial:

```python
window_width = 25
window_height = 80
curve_centers = tracker(Mywindow_width=window_width,Mywindow_height=window_height,Mymargin=25,My_ym=10/720,My_xm=4/384,Mysmooth_factor=15)
window_centroids = curve_centers.find_window_centroids(warped)

l_points = np.zeros_like(warped)
r_points = np.zeros_like(warped)

rightx = []
leftx = []

for level in range(0,len(window_centroids)):
    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)

    leftx.append(window_centroids[level][0]) 
    rightx.append(window_centroids[level][1])

    l_points[(l_points==255)|((l_mask==1))] = 255
    r_points[(r_points==255)|((r_mask==1))] = 255

template = np.array(r_points+l_points,np.uint8)
zero_channel = np.zeros_like(template)
template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)
warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8)
result = cv2.addWeighted(warpage,1,template,0.5,0.0)
```

What I'm doing here is that once I take in the input image, I then define the window. In the example, the rectangles will be 25x80pix. After that, I then use the window_mask() function I made in the image_gen.py to help draw the window. From there, I simply perform the binary threshold and then use cv2.addWeighted to display the green boxes.

Input image:

![alt text](test_images/perspective4.jpg)

Output image:

![alt text](test_images/greenbox4.jpg)


How I got the lane lines:

I got the lane lines by fitting the leftx and rightx points into the `cv2.fillPoly` function (line 150 in image_gen.py). What happens now is these pixel values are mapped to the 2nd degree (Ax^2 + BX + C) giving the equation of the quadratic line. Because this quadratic line is actually the lane line, I can simply save the output to get the image.

Input image:

![alt text](test_images/perspective1.jpg)

Output image:

![alt text](test_images/polylines1.jpg)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Done in line 170 of image_gen.py.

```python
curverad = ((1+(2*curve_fit_cr[0]*yvals[-1]*ym_per_pix+curve_fit_cr[1])**2)**1.5)/np.absolute(2*curve_fit_cr[0]) 
```



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text](frame_ex.jpg)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](output1_tracker.mp4)

---
