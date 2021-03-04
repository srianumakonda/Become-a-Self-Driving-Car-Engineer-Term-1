import numpy as np
import cv2
import glob
import pickle

objp = np.zeros((6*9,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob('./calibration*.jpg')

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    if ret == True:
        print('working on, ', fname)
        objpoints.append(objp)
        imgpoints.append(corners)
        
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        write_name = 'corners_found'+str(idx)+'.jpg'
        cv2.imwrite(write_name,img)
        
img = cv2.imread('./calibration1.jpg')
img_size = (img.shape[1],img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)

dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump(dist_pickle,open('./calibration_pickle.p','wb'))

# img = cv2.imread('calibration3.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img_size = (img.shape[1], img.shape[0])
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# dst = cv2.undistort(img, mtx, dist, None, mtx)
# write_name = 'calibrated.jpg'
# cv2.imwrite(write_name,dst)