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

jvm.start(max_heap_size="512m")

def noteName(x):
    if x == 0:
        return "whole"
    elif x == 1:
        return "half"
    elif x == 2:
        return "quarter"
    elif x == 3:
        return "eighth"
    elif x == 4:
        return "sharp"
    else:
        return "flat"

def header():
    header = ["NOTE", "NN 1", "NN 2", "NN3", "MAJORITY VOTE"]
    writer = csv.writer(open("testSet_Experienced/vote_log.csv",'ab'))
    writer.writerows([header])

def classify():
    loader = Loader(classname="weka.core.converters.ArffLoader")

    dataset20x20 = loader.load_file("preprocessed/notes/20x20.arff")
    dataset20x20.class_is_last()

    dataset20x50 = loader.load_file("preprocessed/notes/20x50.arff")
    dataset20x50.class_is_last()

    dataset50x20 = loader.load_file("preprocessed/notes/50x20.arff")
    dataset50x20.class_is_last()

    model20x20 = Classifier(jobject=serialization.read("nn20x20.model"))
    model35x20 = Classifier(jobject=serialization.read("nn20x50.model"))
    model50x20 = Classifier(jobject=serialization.read("nn50x20.model"))

    nn1 = model20x20
    nn2 = model35x20
    nn3 = model50x20

    class1 = []
    class2 = []
    class3 = []

    for index, inst in enumerate(dataset20x20):
        pred1 = nn1.classify_instance(inst)
        class1.append(pred1)

    for index, inst in enumerate(dataset20x50):
        pred2 = nn2.classify_instance(inst)
        class2.append(pred2)

    for index, inst in enumerate(dataset50x20):
        pred3 = nn3.classify_instance(inst)
        class3.append(pred3)


    for i in range(len(class1)):
        if os.path.isfile('preprocessed/notes/note' + '%s' % (i)+ '.png'):
            img = cv2.imread('preprocessed/notes/note' + '%s' % (i)+ '.png', 0)
            #cv2.imshow('Note' + '%s' % (i) , cv2.resize(img,(200,200)))
            if not os.path.isfile("classified/vote_log.csv"):
                header()

            values = []

            print "NOTE", i, ":"
            print "   nn1:", noteName(class1[i])
            print "   nn2:", noteName(class2[i])
            print "   nn3:", noteName(class3[i])
            values.append("note" + str(i) + ".png")
            values.append(" ")
            values.append(noteName(class1[i]))
            values.append(noteName(class2[i]))
            values.append(noteName(class3[i]))

            cv2.imwrite( "1_%s" % noteName(class1[i]) + "/note%s" % i + ".png", img)
            cv2.imwrite( "2_%s" % noteName(class2[i]) + "/note%s" % i + ".png", img)
            cv2.imwrite( "3_%s" % noteName(class2[i]) + "/note%s" % i + ".png", img)

            if class1[i] == class2[i]:
                print "MAJORITY VOTE:", noteName(class1[i])
                cv2.imwrite( "classified/" + "%s" % noteName(class1[i]) + "/note%s" % i + ".png", img)
                values.append(noteName(class1[i]))
            elif class1[i] == class3[i]:
                print "MAJORITY VOTE:", noteName(class1[i])
                cv2.imwrite( "classified/" + "%s" % noteName(class1[i]) + "/note%s" % i + ".png", img)
                values.append(noteName(class1[i]))
            elif class2[i] == class3[i]:
                print "MAJORITY VOTE:", noteName(class2[i])
                cv2.imwrite( "classified/" + "%s" % noteName(class2[i]) + "/note%s" % i + ".png", img)
                values.append(noteName(class2[i]))
            else:
                print "No classification"
                cv2.imwrite( "noClassification/note%s" % i + ".png", img)
                values.append("no classification")
            print "\n"

            writer = csv.writer(open("classified/vote_log.csv",'ab'))
            writer.writerows([values])
            
            cv2.waitKey(0)

classify()

cv2.waitKey(0)
jvm.stop()