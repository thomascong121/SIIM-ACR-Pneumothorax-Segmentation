 #1. import packages
import cv2,os,re,numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def image_to_feature_vector(image, size):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()

def main(k=5,size=(32, 32),k_test=False):
    numbers = {"eight":8,"five":5,"four":4,"nine":9,"one":1,"seven":7,"six":6,"three":3,"two":2,"zero":0}
    data_array = []
    class_array = []
    #2. read the dataset (images and class labels)
    for filename in os.listdir("./data"):
        n = re.search('(.+?)_', filename)
        target = numbers[n.group(1)]
        image = cv2.imread("./data/{0}".format(filename))
        image_flatten = image_to_feature_vector(image,size)
        data_array.append(image_flatten)
        class_array.append(target)
        
    #3.split images dataset into 80% training and 20% test sets
    (trainData, testData, trainLabels, testLabels) = train_test_split(data_array,class_array, test_size=0.2, random_state=42)
    #4.initialize kNN model (use k=5)
    if(k_test):
        acc_list={}
        for i in range(1,15):
            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(trainData,trainLabels) 
            acc = neigh.score(testData, testLabels)
            if(k_test == "show"):
                print("{0} neighbours selected, the accuracy of the model is {1}".format(i,acc))
            acc_list[i] = acc
        max_k = max(acc_list,key=acc_list.get())
        neigh = KNeighborsClassifier(n_neighbors=max_k)
    else:
        neigh = KNeighborsClassifier(n_neighbors=k)
    
    #5.fit the KNN model using the training data (i.e. construct a search tree)
    neigh.fit(trainData,trainLabels) 
    #6. perform handwritten digit recognition using the test data
    predictions = neigh.predict(testData)
    #7.evaluate the recognition performance by calculating accuracy, confusion matrix, and precision and recall for each digit class
    acc = neigh.score(testData, testLabels)
    print("accuracy is: ",acc)
    print ("Confusion matrix")
    print(confusion_matrix(testLabels,predictions))
    print("EVALUATION ON TESTING DATA")
    print(classification_report(testLabels, predictions))

if __name__ == "__main__":
    main()