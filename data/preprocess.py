import cv2
import numpy as np

imageName = "dataSet_Unexperienced_0001.jpg"
directory = "dataSet_cropped/"

################################################ DESKEW #####################################################

def detect(image):
    height, width = image.shape

    edges = cv2.Canny(image, 150, 200, 3, 5)
    cv2.imwrite('preprocess/preprocessed/edges.png', edges)
    cv2.imshow('EDGE', edges)
    cv2.waitKey(0)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=width*0.25, maxLineGap=100)
    angle = 0.0
 
    for i in range(0,len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            angle = np.arctan2(y2 - y1 , x2 - x1 )
            cv2.line(hough,(x1,y1),(x2,y2),(0,0,255),1)

    cv2.imshow('HOUGH LINES', hough)
    cv2.waitKey(0)

    return np.math.degrees(angle)

def correct(image, angle):
    non_zero_pixels = cv2.findNonZero(image)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)

    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = image.shape
    rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)

    return cv2.getRectSubPix(rotated, (cols, rows), center)

################################################ STAFF REMOVAL #####################################################

def removestaffline( image ):
    binary = cv2.bitwise_not(image)
    cv2.imshow("BINARIZED", binary)
    cv2.waitKey(0)

    vertical = binary.copy()
    verticalsize, horizontalsize = binary.shape

    verticalsize = 7.3
    horizontalsize = 3

    verticalstructure = np.ones((verticalsize,1),np.uint8)
    erosion = cv2.erode(vertical,verticalstructure,iterations = 1)
    horizontalstructure = np.ones((1,horizontalsize),np.uint8)
    cv2.imshow('EROSION', erosion)
    cv2.waitKey(0)
    dilate = cv2.dilate(erosion,horizontalstructure,iterations = 1)
    cv2.imshow("DILATE", dilate)
    cv2.waitKey(0)

    staffremoved = cv2.bitwise_not(dilate)

    cv2.imwrite("preprocess/preprocessed/staffremoved.png", staffremoved)

    return staffremoved

################################################ BOUNDING BOX #####################################################

def boundingbox( image ):

    gray = cv2.imread("preprocess/preprocessed/staffremoved.png", 0)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    _, contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    i=1

    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>8 and w>8 and h<80 and w<70:
            cv2.rectangle(image,(x-2,y-2),(x+w+2,y+h+2),(0,0,255),1)
            note = image[y:y+h,x:x+w]
            noteResize = cv2.resize(note,(20,20))
            cv2.imwrite('preprocess/notes/note' + str(i) + '.png', note)
            i=i+1
    return image

################################################ LOAD INPUT IMAGE #####################################################

img = cv2.imread('%s' % (directory) + '%s' % (imageName), 0)

th, binary = cv2.threshold(img, 130, 255, cv2.THRESH_OTSU)
cv2.imwrite('%s' % (directory) + 'preprocessed/binarized.png', binary)

hough = cv2.imread('%s' % (directory) + '%s' % (imageName), 1)
cv2.imwrite('%s' % (directory) + 'preprocessed/hough.png', hough)

########################################################################################################################

cv2.imshow('INPUT', img)
cv2.waitKey(0)

deskew = correct(img.copy(), detect(img))
h, w = deskew.shape
centery=h/2
centerx=w/2
deskewed = deskew[centery-71:centery+71,centerx-500:centerx+500]
cv2.imshow('DESKEWED', deskewed)
cv2.waitKey(0)

staffline_removed = removestaffline( deskewed )
cv2.imshow('REMOVED STAFFLINE', staffline_removed)
cv2.waitKey(0)

boxed = boundingbox( staffline_removed )
cv2.imshow('BOUNDING BOX', boxed)
cv2.waitKey(0)


cv2.imwrite('%s' % (directory) + 'preprocessed/output.png', deskewed)
dispedge = cv2.imread('%s' % (directory) + 'edges.png', 0)

cv2.imwrite("images/boundingbox.png", staffline_removed)

cv2.waitKey(0)
