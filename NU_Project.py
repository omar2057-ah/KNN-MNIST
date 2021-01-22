import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cv2
import os
import natsort
from collections import Counter

def load_images(folder):
    images = []
    dirFiles = os.listdir(folder)
    Sorted_Dir = natsort.natsorted(dirFiles, reverse=False)
    for filename in Sorted_Dir:
        grey_image = cv2.imread(os.path.join(folder,filename),0)
        #ret, bw_image = cv2.threshold(grey_image, 127, 255, cv2.THRESH_BINARY)
        filtered_img = cv2.medianBlur(grey_image, 3)
        norm_filtered_image = filtered_img / 255.0
        #blur = cv2.blur(grey_image, (3, 3))
        #blur = cv2.bilateralFilter(grey_image, 9, 75, 75)
        #edge_img = cv2.Canny(blur, 100, 200)
        if grey_image is not None:
            images.append(norm_filtered_image)
    return images
def Calculate_Distances(TestingPath, TrainingPath,fn):

    Training_Imgs = load_images(TrainingPath)
    Training_ImgsArray = np.asarray(Training_Imgs)
    Testing_Imgs = load_images(TestingPath)
    Testing_ImgsArray = np.asarray(Testing_Imgs)
    Testing_Imgs_Length = Testing_ImgsArray.shape[0]
    Training_Imgs_Length = Training_ImgsArray.shape[0]

    if fn == 1:

        Diff_Norm = np.zeros([Training_Imgs_Length, 1])
        Imgs_All_Diffs = np.zeros([Testing_Imgs_Length, Training_Imgs_Length])
        for index in range(Testing_Imgs_Length):
            for z in range(0, Training_Imgs_Length):
                    Diff_Norm[z] = LA.norm(Testing_ImgsArray[index] - Training_ImgsArray[z])
                    Imgs_All_Diffs[index][z] = Diff_Norm[z]
        return Imgs_All_Diffs

    elif fn == 0:
        Diff_Norm = np.zeros([Training_Imgs_Length, 1])
        Imgs_All_Diffs = np.zeros([Training_Imgs_Length, Training_Imgs_Length])
        for index in range(Training_Imgs_Length):
            for z in range(0, Training_Imgs_Length):
                if index != z:
                    Diff_Norm[z] = LA.norm(Training_ImgsArray[index] - Training_ImgsArray[z])
                    Imgs_All_Diffs[index][z] = Diff_Norm[z]
        return Imgs_All_Diffs
def Read_Text(filename):
    lines = np.loadtxt(filename, comments="#", delimiter="\n", unpack=False)
    labels_arr = np.asarray(lines)
    labels_arr = labels_arr.astype(int)
    return labels_arr
def Finding_K(training_distances,training_tabels):
    Error_of_Each_K = np.zeros(100)
    for K in range(1,101):
        Min_K_Indices = np.zeros([K])
        Min_K_Values = np.zeros([K])
        Nearest_Labels = np.zeros([K])
        Img_Error_Count = 0
        for index in range(Imgs_Length):
            Sorted_Distances = np.sort(training_distances[index])
            Sorted_Indices = np.argsort(training_distances[index])
            for i in range(K):
                Min_K_Values[i] = Sorted_Distances[i+1]
                Min_K_Indices[i] = Sorted_Indices[i+1]
            i = 0
            for indx in Min_K_Indices:
                Nearest_Labels[i] = training_tabels[int(indx)]
                i += 1
            Count = Counter(Nearest_Labels)
            print("Prediction of KNN = ", Count.most_common(1)[0][0])
            print("True Value of Image = ", Training_Labels[index])
            print("--------------------End of a Testing Image-----------------------")
            if Count.most_common(1)[0][0] != Training_Labels[index]:
                Img_Error_Count += 1
        Error_of_Each_K[K-1] = Img_Error_Count
        print("Total Error of K = %d  is " % K, Img_Error_Count)
        print("----------------------------------End of K---------------------------------------------")
    Minimum_Error = np.min(Error_of_Each_K)
    Minimum_Error_Index = np.argmin(Error_of_Each_K)
    print("Minimum Error = %d of K = %d"% (Minimum_Error, Minimum_Error_Index+1))
    return Error_of_Each_K, Minimum_Error_Index
def Plot_Errors(error_k, best_k):
    K = np.arange(1,101,1)
    plt.plot(K, error_k*100/2400, label = 'K Vs. Total Error')
    plt.xlabel("K")
    plt.ylabel("Error in %")
    plt.text(4,error_k[best_k],'Best K of Minimum Error')
    plt.show()
if __name__ == '__main__' :

    Training_Folder = "F:\Projects-Main\KNN NU-RA\Computer RA Task\Task Dataset\Train"
    Testing_Folder = "F:\Projects-Main\KNN NU-RA\Computer RA Task\Task Dataset\Test"
    TrainingImgs_Distance_Values = np.zeros([2400, 2400])

    TrainingImgs_Distance_Values = Calculate_Distances(Testing_Folder, Training_Folder,0)
    #3rd Function Argument --> 1: For Testing Images, 0: For Leave one out Cross-Validation

    Imgs_Length = TrainingImgs_Distance_Values.shape[0]
    Training_Labels_Path = "F:\Projects-Main\KNN NU-RA\Computer RA Task\Task Dataset\Training Labels.txt"
    Training_Labels = Read_Text(Training_Labels_Path)

    Errors_of_K, Best_K = Finding_K(TrainingImgs_Distance_Values, Training_Labels)
    Plot_Errors(Errors_of_K, Best_K)







