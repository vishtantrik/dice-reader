# Using Android IP Webcam video .jpg stream (tested) in Python2 OpenCV3

import urllib.request
import cv2
import numpy as np
import time
import cam_constants as cam

# TODO: Add type checking to all functions

def get_image_from_ip_cam(url):
  # Return OpenCV Image from URL
  with urllib.request.urlopen(url) as imgResp:
    # Convert response to an array
    imgNp = np.array(bytearray(imgResp.read()), dtype = np.uint8)
    # Convert array to OpenCV image 
    img = cv2.imdecode(imgNp, -1)
    return img

def show_till_key_press(img, window_name = 'Image', k = 'q', every_n_millisec = 25):
  # Show an image in OpenCV Window untill a key is pressed
  while True:
    cv2.imshow(window_name, img)
    if cv2.waitKey(every_n_millisec) & 0xFF == ord(k):
      break

def scale_img(img, factor = 0.5):
  # Scale image up or down without changing aspect ratio
  new_height =  int(img.shape[0] * factor)
  new_width =  int(img.shape[1] * factor)
  resized = cv2.resize(img, (new_width, new_height))
  return resized

def process_img(img):
  """
  Ad-hoc processing of an image used for testing.
  Current does all of the following:
    1. Convert to grayscale
    2. Shrink size to 50%
    3. Set pixel above 200 and below 20 to zero
  """
  # Convert OpenCV Image to grayscale
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Reduce size to 50%
  img = scale_img(img, 0.5)
  # Below 20 set to zero
  ret, img = cv2.threshold(img, 20, 10000, cv2.THRESH_TOZERO)
  # truncate above 200
  ret, img = cv2.threshold(img, 200, 10000, cv2.THRESH_TRUNC)
  return img

def detect_blob(img):
  """
  Returns keypoints of detected blobs in image
  """
  min_threshold = 10                      # these values are used to filter our detector.
  max_threshold = 200                     # they can be tweaked depending on the camera distance, camera angle, ...
  min_area = 50                          # ... focus, brightness, etc.
  max_area = 200                          # ... focus, brightness, etc.
  min_circularity = .3
  min_inertia_ratio = .5
  params = cv2.SimpleBlobDetector_Params()                # declare filter parameters.
  params.filterByArea = True
  params.filterByCircularity = True
  params.filterByInertia = True
  params.minThreshold = min_threshold
  params.maxThreshold = max_threshold
  params.minArea = min_area
  params.maxArea = max_area
  params.minCircularity = min_circularity
  params.minInertiaRatio = min_inertia_ratio

  detector = cv2.SimpleBlobDetector_create(params)        # create a blob detector object.
  keypoints = detector.detect(img)                         # keypoints is a list containing the detected blobs.

  return keypoints

def draw_keypoints (img, keypoints):
  img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255, 100, 0),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  return img_with_keypoints

def insert_text(img, text):
  img_width = img.shape[1]
  img_height = img.shape[0]
  txt_org = (int(img_width * 0.9), int(img_height * 0.15))
  img = cv2.putText(img, text, txt_org, cv2.FONT_HERSHEY_DUPLEX, 3, (256, 100, 50), 3)
  return img

# show_till_key_press(process_img(get_image_from_ip_cam()))
img_org = get_image_from_ip_cam(cam.URL)
img_detector_inp = scale_img(img_org, 0.50)
keypoints = detect_blob(img_detector_inp)
img = draw_keypoints(img_detector_inp, keypoints)
num_blobs = len(keypoints)
insert_text(img, "{}".format(num_blobs))
show_till_key_press(img)