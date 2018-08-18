import cv2
import numpy as np
import csv
import subprocess
import os.path

import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.converters import Loader, Saver
from weka.datagenerators import DataGenerator
from weka.core.classes import Random
import weka.core.serialization as serialization


def header():
    header = []
    for x in range(20):
        for y in range(20):
            header.append("(%d,%d)" % (x,y))
    header.append("NOTE")
    writer = csv.writer(open("testset/testset20x20.csv",'ab'))
    writer.writerows([header])

    header = []
    for x in range(20):
        for y in range(50):
            header.append("(%d,%d)" % (x,y))
    header.append("NOTE")
    writer = csv.writer(open("testset/testset20x50.csv",'ab'))
    writer.writerows([header])

    header = []
    for x in range(50):
        for y in range(20):
            header.append("(%d,%d)" % (x,y))
    header.append("NOTE")
    writer = csv.writer(open("testset/testset50x20.csv",'ab'))
    writer.writerows([header])


def saveCSV(im):
    if not os.path.isfile("testset/testset50x20.csv"):
        header()
    if os.path.isfile('testset/20x20/note' + '%s' % (im)+ '.png'):
        img = cv2.imread('testset/20x20/note' + '%s' % (im)+ '.png', 0)
        th, binary = cv2.threshold(img, 130, 255, cv2.THRESH_OTSU)
        height, width = binary.shape

        values = []
        for y in range(height):
            for x in range(width):
                if binary[y,x] == 0:
                    values.append(1)
                else:
                    values.append(0)

        values.append("?")
        writer = csv.writer(open("testset/testset20x20.csv",'ab'))
        writer.writerows([values])
    
        img = cv2.imread('testset/20x50/note' + '%s' % (im)+ '.png', 0)
        th, binary = cv2.threshold(img, 130, 255, cv2.THRESH_OTSU)
        height, width = binary.shape
    
        values = []
        for y in range(height):
            for x in range(width):
                if binary[y,x] == 0:
                    values.append(1)
                else:
                    values.append(0)
    
        values.append("?")
        writer = csv.writer(open("testset/testset20x50.csv",'ab'))
        writer.writerows([values])

        img = cv2.imread('testset/50x20/note' + '%s' % (im)+ '.png', 0)
        th, binary = cv2.threshold(img, 130, 255, cv2.THRESH_OTSU)
        height, width = binary.shape
    
        values = []
        for y in range(height):
            for x in range(width):
                if binary[y,x] == 0:
                    values.append(1)
                else:
                    values.append(0)

        values.append("?")
        writer = csv.writer(open("testset/testset50x20.csv",'ab'))
        writer.writerows([values])


def load(im):
    if os.path.isfile('testset/note' + '%s' % (im)+ '.png'):
        img = cv2.imread('testset/note' + '%s' % (im)+ '.png', 0)

        resize1 = cv2.resize(img,(20,20))
        resize2 = cv2.resize(img,(20,50))
        resize3 = cv2.resize(img,(50,20))
        cv2.imwrite('testset/20x20/note' + str(im) + '.png', resize1)
        cv2.imwrite('testset/20x50/note' + str(im) + '.png', resize2)
        cv2.imwrite('testset/50x20/note' + str(im) + '.png', resize3)

        saveCSV(im)


for i in range(200):
    load(i)

