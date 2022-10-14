import cv2
import numpy as np
MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.95

import matplotlib.pyplot as plt


img_template = cv2.imread('assets/template_cmnd.jpg')
img_need_aligned = cv2.imread('assets/testCmnd.jpg')

im1Gray = cv2.cvtColor(img_need_aligned, cv2.COLOR_BGR2GRAY)
im1 = img_need_aligned
im2Gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
im2 = img_template

orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

# # Sort matches by score
# matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]
 
# Draw top matches
imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
cv2.imwrite("matches.jpg", imMatches)

  # Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

# Use homography
height, width, channels = im2.shape
im1Reg = cv2.warpPerspective(im1, h, (width, height))
imgplot = plt.imshow(imMatches)
plt.show()