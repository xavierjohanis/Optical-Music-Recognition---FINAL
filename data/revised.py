import cv2
import numpy as np
import csv
import subprocess
import os.path
from time import sleep

imageName = "Inexperienced_13.jpg"
directory = "testSet_Inexperienced/"

def remove(im):

    binary = cv2.bitwise_not(im)

    horizontal = binary.copy()
    vertical = binary.copy()

    verticalsize, horizontalsize = binary.shape

    verticalsize = 15
    horizontalsize = 2

    horizontalstructure = np.ones((1,horizontalsize),np.uint8)
    erosion = cv2.erode(horizontal,horizontalstructure,iterations = 1)
    dilateh = cv2.dilate(erosion,horizontalstructure,iterations = 1)

    verticalstructure = np.ones((verticalsize,1),np.uint8)
    erosion = cv2.erode(vertical,verticalstructure,iterations = 1)
    dilate = cv2.dilate(erosion,verticalstructure,iterations = 1)

    output = cv2.bitwise_not(dilate)
    cv2.imshow('EROSION', cv2.resize(erosion,(1000,700)) )
    cv2.waitKey(0)
    cv2.imshow('DILATION', cv2.resize(dilate,(1000,700)) )
    cv2.waitKey(0)

    cv2.imshow('REMOVED STAFFLINES', cv2.resize(output,(1000,700)) )
    cv2.waitKey(0)

    cv2.imwrite("preprocessed/staffremoved.jpg", output)

    return output

def box():
    ima = cv2.imread("preprocessed/staffremoved.jpg")

    gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    th, binary = cv2.threshold(blur, 130, 255, cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(binary,255,1,1,11,2)

    _, contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    i=0
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>55 and w>55:
            #cv2.rectangle(im,(x-2,y-2),(x+w+2,y+h+2),(0,0,255),1)
            note = ima[y:y+h,x:x+w]
            noteResize1 = cv2.resize(note,(20,20))
            noteResize2 = cv2.resize(note,(20,50))
            noteResize3 = cv2.resize(note,(50,20))
            cv2.imwrite("preprocessed/notes/note" + str(i) + '.png', note)
            cv2.imwrite("preprocessed/notes/20x20/note" + str(i) + '.png', noteResize1)
            cv2.imwrite("preprocessed/notes/20x50/note" + str(i) + '.png', noteResize2)
            cv2.imwrite("preprocessed/notes/50x20/note" + str(i) + '.png', noteResize3)
            i=i+1

def header():
    header = []
    for x in range(20):
        for y in range(20):
            header.append("(%d,%d)" % (x,y))
    header.append("NOTE")
    #writer = csv.writer(open("testSet_unExperienced/sep/20x20.csv",'ab'))
    writer = csv.writer(open("preprocessed/notes/20x20.csv",'ab'))
    writer.writerows([header])

    header = []
    for x in range(20):
        for y in range(50): 
            header.append("(%d,%d)" % (x,y))
    header.append("NOTE")
    #writer = csv.writer(open("testSet_unExperienced/sep/20x50.csv",'ab'))
    writer = csv.writer(open("preprocessed/notes/20x50.csv",'ab'))
    writer.writerows([header])

    header = []
    for x in range(50):
        for y in range(20):
            header.append("(%d,%d)" % (x,y))
    header.append("NOTE")
    #writer = csv.writer(open("testSet_unExperienced/sep/50x20.csv",'ab'))
    writer = csv.writer(open("preprocessed/notes/50x20.csv",'ab'))
    writer.writerows([header])

def saveCSV(im):
    if not os.path.isfile("preprocessed/notes/50x20.csv"):
        header()
    if os.path.isfile( 'preprocessed/notes/20x20/note' + '%s' % (im)+ '.png'):
        img = cv2.imread('preprocessed/notes/20x20/note' + '%s' % (im)+ '.png', 0)
        th, binary = cv2.threshold(img, 130, 255, cv2.THRESH_OTSU)
        height, width = binary.shape

        values = []
        for y in range(height):
            for x in range(width):
                if binary[y,x] == 0:
                    values.append(1)
                else:
                    values.append(0)

        values.append(note)
        writer = csv.writer(open("preprocessed/notes/20x20.csv",'ab'))
        writer.writerows([values])
    
        img = cv2.imread('preprocessed/notes/20x50/note' + '%s' % (im)+ '.png', 0)
        th, binary = cv2.threshold(img, 130, 255, cv2.THRESH_OTSU)
        height, width = binary.shape
    
        values = []
        for y in range(height):
            for x in range(width):
                if binary[y,x] == 0:
                    values.append(1)
                else:
                    values.append(0)
    
        values.append(note)
        writer = csv.writer(open("preprocessed/notes/20x50.csv",'ab'))
        writer.writerows([values])

        img = cv2.imread('preprocessed/notes/50x20/note' + '%s' % (im) + '.png', 0)
        th, binary = cv2.threshold(img, 130, 255, cv2.THRESH_OTSU)
        height, width = binary.shape
    
        values = []
        for y in range(height):
            for x in range(width):
                if binary[y,x] == 0:
                    values.append(1)
                else:
                    values.append(0)

        values.append(note)
        writer = csv.writer(open("preprocessed/notes/50x20.csv",'ab'))
        writer.writerows([values])

def convert( filename ):
    
    content = []

    with open(filename, 'rb') as csvfile:
               lines = csv.reader(csvfile, delimiter = ',')
               for row in lines:
                   content.append(row)
    csvfile.close()
    
    if filename.endswith('.csv') == True:
            filename = filename.replace('.csv', '')


    title = filename + '.arff'
    new_file = open(title, 'w')

    new_file.write('@relation ' + filename + '\n\n')

    for i in range(len(content[0])-1):
            new_file.write('@attribute \'' + str(content[0][i]) + '\' numeric \n')

    last = len(content[0])
    class_items = []

    for i in range(len(content)):
            name = content[i][last-1]
            if name not in class_items:
                class_items.append(content[i][last-1])
            else:
                pass  
    del class_items[0]


    string = '{' + ','.join(sorted(class_items)) + '}'
    new_file.write('@attribute ' + str(content[0][last-1]) + ' {whole,half,quarter,eighth,sharp,flat} \n')

    new_file.write('\n@data\n')

    del content[0]
    for row in content:
        new_file.write(','.join(row) + '\n')

    new_file.close()



note = "?"

img = cv2.imread('%s' % (directory) + '%s' % (imageName), 0)
cv2.imshow('INPUT', cv2.resize(img,(1000,700)) )
cv2.waitKey(0)

staffremoved = remove( img )


box()

for i in range(1000):
    saveCSV(i)

convert("preprocessed/notes/20x20.csv")
convert("preprocessed/notes/20x50.csv")
convert("preprocessed/notes/50x20.csv")