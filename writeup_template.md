## Writeup

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
[video1]: ./output_images/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

After creating the 2 required matrices from my camera calibration step, I can then call cv2.undistort on any image.
`frame = cv2.undistort(img, mtx, dist, None, mtx)`
This creates a distortion-corrected frame
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used 2 color channels for thresholding, the first is the S channel in HLS to detect yellow lines, the second is the L channel in LAB to detect white lines. in the S channel I used a min of 170 and a max of 255. In the L channel I used a min of 225 and a max of 255. I then applied the sobel in the x direction on the L channel. After combining the 2 channels I have a binary image that highlights road lanes well.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I created a function called warp that returns both the warp and the M inverse which can be found in the helper function section of my notebook. It takes the 4 source points which are manually selected. The function is then called, the destination points belong in the function warp.

```python

def warp(img, pts):
    img_size = (img.shape[1], img.shape[0])
    src = pts
    offset = 100
    dst = np.float32([[384, 0], [896, 0],
                                     [896, 720],
                                     [384, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, Minv

top_left = [581, 477]
top_right = [699, 477]
bottom_right = [896,675]
bottom_left = [384,675]
pts = np.array([top_left, top_right, bottom_right, bottom_left])
warp_frame, Minv = warp(frame, np.float32([pts]))
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

My method for pixel identification begins by applying a sliding window method to find the initial lane lines on the road. Once these have been found we can store them for future use. The sliding window method consist of create 9 windows on each side of the image and centering prospective lane lines in the window. Given this we can then fit a polynomial through this selection.

In every next iteration, we use the previously fit polynomial and search within a certain margin of that polynomial for line pixels. The one case where this is not used is if there is no detection of pixels in the margin, in which case we revert back to using the sliding window method.

After each iteration we perform a check to see if the detected lines make sense. We compare the polynomial from the detected line to that of the previous line. If the polynomial is drastically different, this means something went wrong, we correct this by taking the average of our previous n fits and using this as our current fit. This helps overcome issues like shadows and sudden color changes in pavement.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Once our line is detected and we have fit a polynomial we can calculate curvature of the line using the following function, while also converting our measurement into meters.

```python
def measure_curvature_pixels(ploty, left_fit, right_fit):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    if (left_fit is not None):
        left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    else:
        left_curverad = None

    if (right_fit is not None):
        right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    else:
        right_curverad = None

    return left_curverad, right_curverad
```

Then, to calculate road offset, we use the following snippet of code to compare the distance at the bottom of the 2 detected lines. We can then compare this to the vehicles center.
```python
lane_centre = (right_fitx[0] - left_fitx[0])/2 + left_fitx[0]
xm_per_pix = 3.7/700 # meters per pixel in x dimension
offset = abs(640-lane_centre)
offset = offset*xm_per_pix
```
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](output_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A few shortcomings include cracks in the road that can be mistaken for a lane line, my solution currently does not solve for situations like this. Moreover, for roads with quicker turns, my sanity check threshold may be too rigid and lock the detected line over frames.
