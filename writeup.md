
**Advanced Lane Finding Project**

[//]: # (Image References)

[image1]: ./output_images/test3_undistorted.jpg "Undistorted"
[image2]: ./output_images/test3_find_lines_by_threshold.jpg "Lane Lines after applying threshold"
[image3]: ./output_images/test3_prespective_transform.jpg "Prespective transform"
[image4]: ./output_images/test3_window_centriod.jpg "Finding lane lines ands window centriod"
[image5]: ./output_images/test3_identify_lane_line.jpg "Draw lane boundry"
[image6]: ./output_images/test3_camera_center_and_curve.jpg "Camera center and curve calculation"
[video1]: ./output_tracked.mp4 "Project video output"


### Camera Calibration

#### 1. Camera matrix and distortion coefficients

The code for this step is contained in the first code cell of the IPython notebook located in "./Final.ipynb".

We have 9x6 image that is 9 corners in a row and 6 corners in a column. 
I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients Using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][output_images/calibration1_undistorted.jpg]

Then i saved the mtx and dist co-efficients in the pickle file for later use.


### Pipeline (single images)

#### 1. Distortion-corrected image.

The code for this step is contained in the second code cell of the IPython notebook located in "./Final.ipynb" Line 11.
First i read saved imgpoints and objpoints using pickle. 
Then i applied the distortion correction by using camera matrix and distortion coefficients to one of the test images like this one:
![alt text][output_images/test3_undistorted.jpg]

#### 2. Color transforms and gradients to create a thresholded binary image. 

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # 78 through # 80 in `Final.ipynb`). I used sobel threshold on x and y and color threshold. In color threshold i used S in HLS and V in HSV color space for clearly extracting lane lines. 
Here is the code snippet:
```
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
S = hls[:,:,2]
s_binary = np.zeros_like(S)
s_binary[(S >= sthresh[0]) & (S <= sthresh[1])] = 1

hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
V = hsv[:,:,2]
v_binary = np.zeros_like(V)
v_binary[(V >= vthresh[0]) & (V <= vthresh[1])] = 1
```
I selected S and V values if they lies between the threshold limit only.

Here's an example of my output for this step.

![alt text][output_images/test3_find_lines_by_threshold.jpg]

#### 3. Perspective transform.

The code for my perspective transform is included in same file "./Final.ipynb" in the 3rd code cell Line 27. Here i read the image 'test3_find_lines_by_threshold.jpg' which is saved in previous step. Perspective transform is done using the inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I choose the hardcode the source and destination points in the following manner:

```   
# Use hard coded values for proper understanding as first step
bottom_left = [220,720]
bottom_right = [1110, 720]
top_left = [570, 470]
top_right = [722, 470]
src = np.float32([bottom_left,bottom_right,top_right,top_left])

# Cover main area
bottom_left = [320,720]
bottom_right = [920, 720]
top_left = [320, 0]
top_right = [920, 0]
dst = np.float32([bottom_left,bottom_right,top_right,top_left])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][output_images/test3_prespective_transform.jpg]

#### 4. Identify lane-line pixels and fit their positions with a polynomial

The code to identify lane-line pixels is included in the file "./Final.ipynb" in the last cell.
I used function 'find_lane_pixels' to identify and track lane line. I took histogram of bottom half of the image as:
Then i find the peak of left and right halves of histogram and use it as my starting point.
Also margin to search left or right is set. Moreover these hyper parameters are also set.
These parameters were tuned for better output results.
Number of windows were selected as 13. Although any value between 10 to 20 gives good output.
Marigin to set width of windows as set as 90 pixels.
Minimum pixels found to recenter the windows is set as 40.
```
nwindows = 13
margin = 90
minpix = 40
``` 

Then i identified the window boundries and drawed them on the image. Then non zero pixels were indetfied and added to the list. Then after combining the array of indices, i extracted left and right line pixel positions.
Then i fit a second order polynomial to to each pixel to extract left and right fit points. If no lines are found i perform search again. Then i draw the lane on the image. Next i check the validity of the lines and drop the bad lines and process the good lines by taking average of last five frames.


```
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    
# Find our lane pixels first
leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

# Fit a second order polynomial to each using `np.polyfit`
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

```
The output is shown in the image below:

![alt text][output_images/test3_identify_lane_line.jpg]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

The code to show example image with inner lane is filled as green is included in the file "./Final.ipynb" in the last cell. Here is the output image.
![alt text][output_images/test3_lane_boundries.jpg]

The code to identify curvature of the lane and find camera position with respect to center is included in the file "./Final.ipynb".
The radius of curvature is calculated from the formula given in the class room. For the polynomail we need value in pixels but in real world curvature is measured in meters so we convert first in to pixels and then into meters as shown in code snippet below:
```
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fit_x*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fit_x * xm_per_pix, 2)

left_curved  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curved  = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
mean_curve = (left_curved + right_curved)/2

# Calculate the offset of the car on the road
## Image mid horizontal position 
mid_imgx = img.shape[1]//2

## Car position with respect to the lane
car_pos = (left_line.line_base_pos + right_line.line_base_pos)/2

## Horizontal car offset 
offsetx = (mid_imgx - car_pos) * xm_per_pix
```
Camera center is calculated by finding base of left and right lines. Center difference is difference between camera center and half of horizontal image. We also convert it to meters.
Image is shown as under:
![alt text][output_images/test3_camera_center_and_curve.jpg]

#### 6.Example image of result

The code to show example image is included in the file "./Final.ipynb" in the 6th cell.  
After all processing here is an example of my result on a test image:

![alt text][output_images/test3_camera_center_and_curve.jpg]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The code to apply all image processing on the vidoe is included in the file "./Final.ipynb" in the 7th cell.
I applied all the techniques used above for image to video stream. Here i prepared function ```process_image``` which will take one image from a frame and will return processed image after applying all the processing. Then i save the results to video file ```output_tracked.mp4```.
Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


Techniques i used:
Line detection worked well with using image processing techniques to enhance image and extract lane line. One difficult task was to drop the wrong lines. For this i used class to store previous values and in case of invalid line detection use previous value. Moreover i used average of last five frames to draw lane lines.
Also i stored the base line positions of each frame to find offset of vehicle and used it when invalid lines are detected in frame.

Secondly i only used the sobel x and sobel y to calculate gradient and then used color threshold to extract lane lines. At this point i also used magnitude and direction threshold to increase the probability of getting fair enough results.

Improvement Points: (Both these points are addressed in my latest code. But still further improvment required)
I can see my pipeline produces wobbly results when the suddently the road condition is changed. Even though it is performing well but further improvements are required when road color truns dark from light or when there is noise i.e. short garbage lines detected in between lane lines.

Another point to improve is when no lane line is detected in frame, i should use some mechanism to fall back on the area where previously detected line was. So that there should not be any wrong line drawn when there is no lane.