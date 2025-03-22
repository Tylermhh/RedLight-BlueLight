import cv2, time
import pandas as pd
from datetime import datetime

# Initialize the static background. This is what we will be using as a mask
static_back = None

# Initializing vars for recording when motion was detected
motion_list = [ None, None ]
times_moved = []

# Just a dataframe
df = pd.DataFrame(columns=["Start", "End"])

# Select the camera. VideoCapture(0) is the default webcam device
video = cv2.VideoCapture(0)

while True:
    # Read a single frame from the camera. Video.read also returns a boolean to indicate whether the read was successful
    check, frame = video.read() 
    if not check:
        continue

    motion = 0

    # cvtColor changes the color space of an image according to the flag in the second parameter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Remove noise by blurring so we can focus on main object movement detection and not some fly in the background or sth
    # (21, 21) is the kernel size of filter that will be used to apply the blur, bigger kernel(or filter) = bigger blur 
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if in first iteration, assign static_back to first frame
    if static_back is None:
        static_back = gray
        continue

    # calculate the difference between current frame and the masking frame
    diff_frame = cv2.absdiff(static_back, gray)

    ''' Apply a threshold to make the difference pixels that are greater than 30 white while making all others black
    30 is the threshold value, 255 is the maximum pixel value to set if the difference value is > 30
    cv2.THRESH_BINARY specifies thresholding method to use: specifically, if above [defined threshold], set to [defined maximum pixel value]
    cv2.threshold returns 2 values: [the threshold value used, the thresholded image] (we want this resulting image)
    '''
    threshold_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]

    ''' Dilates the image so the diffrence gets exaggerated a little. Pixel becomes white if there is at least one pixel in its kernel area is white
    None defines the kernel size for filter
    iterations is number of times to do this dilation, more iterations = more expansion of white area
    '''
    threshold_frame = cv2.dilate(threshold_frame, None, iterations = 2)

    ''' Finding contour of detected movement
    need to pass in copy since findContours modifies the image provided. 
    RETR_EXTERNAL is the contour retrieval mode, specifies what kind of contours to retrieve
        -> only retrieves outermost contours. useful for when you want boundaries of objects
    CHAIN_APPROX_SIMPLE is the contour approximation method. determines how contour points are stored.
        -> Compresses horizontal, vertical, and diagonal segments to save memory. stores only the end points of straight lines.
    '''
    cnts,_ = cv2.findContours(threshold_frame.copy(),
                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        # Filtering out small object movements
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1

        # boundingRect returns 4 values: starting coords x,y and the width and height
        (x, y, w, h) = cv2.boundingRect(contour)

        '''Draw the rectangle we got from contours
        (x,y) is the coord of top-left corner and x+w,y+h is bottom right
        (0,255,0) color of rectangle border
        3 is thickness of border. if -1, the rectangle will be filled with the color instead
        '''
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 3)

    # Append the state of motion
    motion_list.append(motion)
    # Keep only the last two motions to check if the motion started or stopped.
    motion_list = motion_list[-2:]

    # Appending Start time of motion 
    # If motion is detected in the most recent timeframe and there was no motion in previous timeframe, it indicates start of the motion
    if motion_list[-1] == 1 and motion_list[-2] == 0: 
        times_moved.append(datetime.now()) 
  
    # Appending End time of motion 
    # If no motion detected in curent timeframe and there was motion before, it indicates end of motion
    if motion_list[-1] == 0 and motion_list[-2] == 1: 
        times_moved.append(datetime.now()) 

    # Show image in grayscale
    cv2.imshow("Gray Frame", gray)

    # Show difference frame
    cv2.imshow("Difference Frame", diff_frame)

    # Displaying the black and white image in which if 
    # intensity difference greater than 30 it will appear white 
    cv2.imshow("Threshold Frame", threshold_frame) 

    # Displaying color frame with contour of motion of object 
    cv2.imshow("Color Frame", frame) 

    # Just a way to end the program.
    key = cv2.waitKey(1) 
    # if q entered whole process will stop 
    if key == ord('q'): 
        # if something is movingthen it append the end time of movement 
        if motion == 1: 
            times_moved.append(datetime.now()) 
        break

for i in range(0, len(times_moved), 2):
    df = df.append({"Start": times_moved[i], "End": times_moved[i+1]}, ignore_index=True)

df.to_csv("Time_of_Movements.csv")

video.release()
cv2.destroAllWindows()