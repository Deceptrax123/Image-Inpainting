import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import os
from skimage import measure 

def create_mask(image):
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    blurred = cv2.GaussianBlur( gray, (9,9), 0 )
    _,thresh_img = cv2.threshold( blurred, 170, 255, cv2.THRESH_BINARY)

    thresh_img = cv2.erode( thresh_img, None, iterations=2 )
    thresh_img  = cv2.dilate( thresh_img, None, iterations=4 )
    
    labels = measure.label( thresh_img,connectivity=2, background= 0)
    mask = np.zeros( thresh_img.shape, dtype="uint8" )

    for label in np.unique( labels ):
        if label == 0:
            continue

        labelMask = np.zeros( thresh_img.shape, dtype="uint8" )
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero( labelMask )
    
        if numPixels > 300:
            mask = cv2.add( mask, labelMask )
    return mask


def main():
    images=sorted(os.listdir("./errors/"))
    images=images[1:]
    for i in images:
        img=cv2.imread("./errors/"+i)
        mask=create_mask(img)

        dst=cv2.inpaint(img,mask,inpaintRadius=3,flags=cv2.INPAINT_TELEA)
        cv2.imwrite("./glare_removed/"+i,dst) 

if __name__=='__main__':
    main()