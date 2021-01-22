import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import sklearn.metrics as sk
import pandas as pd
import seaborn as sb
from NU_Project import Calculate_Distances, Read_Text  # Using Generalized Functions of the Training File


def get_confusion(testing_labels, predictions):

    Confusion_Matrix = sk.confusion_matrix(testing_labels, predictions)
    DF = pd.DataFrame(Confusion_Matrix)
    sb.heatmap(DF, annot=True, cmap="YlGnBu", fmt="d")
    plt.xlabel('Actual Vs. Predicted Labels Confusion Matrix')
    plt.show()

if __name__ == '__main__' :

    Training_Folder = "F:\Projects-Main\KNN NU-RA\Computer RA Task\Task Dataset\Train"
    Testing_Folder = "F:\Projects-Main\KNN NU-RA\Computer RA Task\Task Dataset\Test"
    TestingImgs_Distance_Values = np.zeros([200,2400])
    TestingImgs_Distance_Values = Calculate_Distances(Testing_Folder, Training_Folder, 1)
    #3rd Function Argument --> 1: For Testing Images, 0: For Leave one out Cross-Validation
    Testing_length = TestingImgs_Distance_Values.shape[0]

    Testing_Labels_Path = "F:\Projects-Main\KNN NU-RA\Computer RA Task\Task Dataset\Test Labels.txt"
    Testing_Labels = Read_Text(Testing_Labels_Path)
    Training_Labels_Path = "F:\Projects-Main\KNN NU-RA\Computer RA Task\Task Dataset\Training Labels.txt"
    Training_Labels = Read_Text(Training_Labels_Path)

    K = 4
    Min_K_Indices = np.zeros([K])
    Min_K_Values = np.zeros([K])
    Nearest_Labels = np.zeros([K])
    Img_Error_Count = 0
    Predictions = []
    Confusion_Matrix = np.zeros((10,10))
    for index in range(Testing_length):
        Sorted_Distances = np.sort(TestingImgs_Distance_Values[index])
        Sorted_Indices = np.argsort(TestingImgs_Distance_Values[index])
        for i in range(K):
            Min_K_Values[i] = Sorted_Distances[i + 1]
            Min_K_Indices[i] = Sorted_Indices[i + 1]
        i = 0
        for indx in Min_K_Indices:
            Nearest_Labels[i] = Training_Labels[int(indx)]
            i += 1
        Count = Counter(Nearest_Labels)
        Predictions.append(Count.most_common(1)[0][0])
        print("Prediction of KNN = ", Count.most_common(1)[0][0])
        print("True Value of Image = ", Testing_Labels[index])
        print("--------------------End of a Testing Image-----------------------")
        if Count.most_common(1)[0][0] != Testing_Labels[index]:
            Img_Error_Count += 1
    print("Total Error of K = %d  is " % K, Img_Error_Count)

    get_confusion(Testing_Labels, Predictions)
