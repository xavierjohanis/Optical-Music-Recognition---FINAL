import weka.core.jvm as jvm

jvm.start(max_heap_size="512m")

from weka.classifiers import Classifier, Evaluation
from weka.core.converters import Loader, Saver
from weka.datagenerators import DataGenerator
from weka.core.classes import Random

import weka.core.serialization as serialization

loader = Loader(classname="weka.core.converters.ArffLoader")

dataSet20x20 = loader.load_file("trainingSet/dataSet20x20.arff")
dataSet20x20.class_is_last()

dataSet20x50 = loader.load_file("trainingSet/dataSet20x50.arff")
dataSet20x50.class_is_last()

dataSet50x20 = loader.load_file("trainingSet/dataSet50x20.arff")
dataSet50x20.class_is_last()

classifier1 = Classifier(classname="weka.classifiers.functions.MultilayerPerceptron", options=["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "9"])
classifier2 = Classifier(classname="weka.classifiers.functions.MultilayerPerceptron", options=["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "11"])
classifier3 = Classifier(classname="weka.classifiers.functions.MultilayerPerceptron", options=["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "9"])

print "\n\nTraining neural network 1"
evaluation1 = Evaluation(dataSet20x20)                   
evaluation1.crossvalidate_model(classifier1, dataSet20x20, 10, Random(42))
classifier1.build_classifier(dataSet20x20)
serialization.write("trainingSet/nn1.model", classifier1)
print "\n\n====================================================== NUERAL NETWORK 1 ======================================================"
print(evaluation1.summary())
print(evaluation1.class_details())

print "Training neural network 2"
evaluation2 = Evaluation(dataSet20x50) 
evaluation2.crossvalidate_model(classifier2, dataSet20x50, 10, Random(42))
classifier2.build_classifier(dataSet20x50)
serialization.write("trainingSet/nn2.model", classifier2)
print "\n\n====================================================== NUERAL NETWORK 2 ======================================================"
print(evaluation2.summary())
print(evaluation2.class_details())

print "Training neural network 3"
evaluation3 = Evaluation(dataSet50x20) 
evaluation3.crossvalidate_model(classifier3, dataSet50x20, 10, Random(42))
classifier3.build_classifier(dataSet50x20)
serialization.write("trainingSet/nn3.model", classifier3)
print "\n\n====================================================== NUERAL NETWORK 3 ======================================================"
print(evaluation3.summary())
print(evaluation3.class_details())





print "Training finish..."






jvm.stop()
