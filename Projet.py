###########################################################################################################################################
#
#                        CODE USE TO GENERATE THE FILTERS AND THE MODEL, THIS IS NOT THE SOURCE CODE OF THE PROJECT
#
###########################################################################################################################################

from fileinput import filename
from tabnanny import filename_only
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import glob
from skimage import io
from skimage import filters
from skimage.io import imsave
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

dataset = [] 
labels = []



def resize (img, scale):   
    
    dim = (scale, scale)
  
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

"""
# Function to save the picture with filter

def filtreFolder(path):
    i = 0
    if path == './chest_xray/train/NORMAL/*.jpeg':
        os.mkdir("./filter-otsu-200/train-normal")
        for filename in glob.glob('./chest_xray/train/NORMAL/*.jpeg'):
            print (filename)
            image = resize (cv2.imread(filename, -1).astype(np.float64), 200)
            print(image.shape)
            i = i+1
            filters.threshold_otsu( image)
            #save the output image with the new threshold
            out = image > filters.threshold_otsu( image)
            # save the image with a different name for each image
            imsave('./filter-otsu-200/train-normal/filter-export'+str(i)+'.jpeg'.format(i), out)
            print(i)
    elif path == './chest_xray/train/PNEUMONIA/*.jpeg':
        os.mkdir("./filter-otsu-200/train-pneumonia")
        for filename in glob.glob('./chest_xray/train/PNEUMONIA/*.jpeg'):
            print (filename)
            image = resize (cv2.imread(filename, -1).astype(np.float64), 200)
            print(image.shape)
            i = i+1
            filters.threshold_otsu( image)
            #save the output image with the new threshold
            out = image > filters.threshold_otsu( image)
            # save the image with a different name for each image
            imsave('./filter-otsu-200/train-pneumonia/filter-export'+str(i)+'.jpeg'.format(i), out)
            print(i)
    
    elif path == './chest_xray/test/NORMAL/*.jpeg':
        os.mkdir("./filter-otsu-200/test-normal")
        for filename in glob.glob('./chest_xray/test/NORMAL/*.jpeg'):
            print (filename)
            image = resize (cv2.imread(filename, -1).astype(np.float64), 200)
            print(image.shape)
            i = i+1
            filters.threshold_otsu( image)
            #save the output image with the new threshold
            out = image > filters.threshold_otsu( image)          
            # save the image with a different name for each image
            imsave('./filter-otsu-200/test-normal/filter-export'+str(i)+'.jpeg'.format(i), out)
            print(i)
    elif path == './chest_xray/test/PNEUMONIA/*.jpeg':
        os.mkdir("./filter-otsu-200/test-pneumonia")
        for filename in glob.glob('./chest_xray/test/PNEUMONIA/*.jpeg'):
            print (filename)
            image = resize (cv2.imread(filename, -1).astype(np.float64), 200)
            print(image.shape)
            i = i+1
            filters.threshold_otsu( image)
            #save the output image with the new threshold
            out = image > filters.threshold_otsu( image)
            # save the image with a different name for each image
            imsave('./filter-otsu-200/test-pneumonia/filter-export'+str(i)+'.jpeg'.format(i), out)
            print(i)

def filtreFolder1(path):
    i = 0
    if path == './chest_xray/train/NORMAL/*.jpeg':
        os.mkdir("./filter-yen-200/train-normal")
        for filename in glob.glob('./chest_xray/train/NORMAL/*.jpeg'):
            print (filename)
            image = resize (cv2.imread(filename, -1).astype(np.float64), 200)
            print(image.shape)
            i = i+1
            filters.threshold_yen( image)
            #save the output image with the new threshold
            out = image > filters.threshold_yen( image)
            # save the image with a different name for each image
            imsave('./filter-yen-200/train-normal/filter-export'+str(i)+'.jpeg'.format(i), out)
            print(i)
    elif path == './chest_xray/train/PNEUMONIA/*.jpeg':
        os.mkdir("./filter-yen-200/train-pneumonia")
        for filename in glob.glob('./chest_xray/train/PNEUMONIA/*.jpeg'):
            print (filename)
            image = resize (cv2.imread(filename, -1).astype(np.float64), 200)
            print(image.shape)
            i = i+1
            filters.threshold_yen( image)
            #save the output image with the new threshold
            out = image > filters.threshold_yen( image)
            # save the image with a different name for each image
            imsave('./filter-yen-200/train-pneumonia/filter-export'+str(i)+'.jpeg'.format(i), out)
            print(i)
    
    elif path == './chest_xray/test/NORMAL/*.jpeg':
        os.mkdir("./filter-yen-200/test-normal")
        for filename in glob.glob('./chest_xray/test/NORMAL/*.jpeg'):
            print (filename)
            image = resize (cv2.imread(filename, -1).astype(np.float64), 200)
            print(image.shape)
            i = i+1
            filters.threshold_yen( image)
            #save the output image with the new threshold
            out = image > filters.threshold_yen( image)          
            # save the image with a different name for each image
            imsave('./filter-yen-200/test-normal/filter-export'+str(i)+'.jpeg'.format(i), out)
            print(i)
    elif path == './chest_xray/test/PNEUMONIA/*.jpeg':
        os.mkdir("./filter-yen-200/test-pneumonia")
        for filename in glob.glob('./chest_xray/test/PNEUMONIA/*.jpeg'):
            print (filename)
            image = resize (cv2.imread(filename, -1).astype(np.float64), 200)
            print(image.shape)
            i = i+1
            filters.threshold_yen( image)
            #save the output image with the new threshold
            out = image > filters.threshold_yen( image)
            # save the image with a different name for each image
            imsave('./filter-yen-200/test-pneumonia/filter-export'+str(i)+'.jpeg'.format(i), out)
            print(i)

"""
"""
def filtreFolder2(path):
    i = 0
    if path == './chest_xray/train/NORMAL/*.jpeg':
        os.mkdir("./filter-mean-200/train-normal")
        for filename in glob.glob('./chest_xray/train/NORMAL/*.jpeg'):
            print (filename)
            image = resize (cv2.imread(filename, -1).astype(np.float64), 200)
            print(image.shape)
            i = i+1
            filters.threshold_mean( image)
            #save the output image with the new threshold
            out = image > filters.threshold_mean( image)
            # save the image with a different name for each image
            imsave('./filter-mean-200/train-normal/filter-export'+str(i)+'.jpeg'.format(i), out)
            print(i)
    elif path == './chest_xray/train/PNEUMONIA/*.jpeg':
        os.mkdir("./filter-mean-200/train-pneumonia")
        for filename in glob.glob('./chest_xray/train/PNEUMONIA/*.jpeg'):
            print (filename)
            image = resize (cv2.imread(filename, -1).astype(np.float64), 200)
            print(image.shape)
            i = i+1
            filters.threshold_mean( image)
            #save the output image with the new threshold
            out = image > filters.threshold_mean( image)
            # save the image with a different name for each image
            imsave('./filter-mean-200/train-pneumonia/filter-export'+str(i)+'.jpeg'.format(i), out)
            print(i)
    
    elif path == './chest_xray/test/NORMAL/*.jpeg':
        os.mkdir("./filter-mean-200/test-normal")
        for filename in glob.glob('./chest_xray/test/NORMAL/*.jpeg'):
            print (filename)
            image = resize (cv2.imread(filename, -1).astype(np.float64), 200)
            print(image.shape)
            i = i+1
            filters.threshold_mean( image)
            #save the output image with the new threshold
            out = image > filters.threshold_mean( image)          
            # save the image with a different name for each image
            imsave('./filter-mean-200/test-normal/filter-export'+str(i)+'.jpeg'.format(i), out)
            print(i)
    elif path == './chest_xray/test/PNEUMONIA/*.jpeg':
        os.mkdir("./filter-mean-200/test-pneumonia")
        for filename in glob.glob('./chest_xray/test/PNEUMONIA/*.jpeg'):
            print (filename)
            image = resize (cv2.imread(filename, -1).astype(np.float64), 200)
            print(image.shape)
            i = i+1
            filters.threshold_mean( image)
            #save the output image with the new threshold
            out = image > filters.threshold_mean( image)
            # save the image with a different name for each image
            imsave('./filter-mean-200/test-pneumonia/filter-export'+str(i)+'.jpeg'.format(i), out)
            print(i)


#create folders 
import os
os.mkdir("./filter-otsu-200")

# Save the picture with the filter in the previous folder created
filtreFolder('./chest_xray/train/NORMAL/*.jpeg')
filtreFolder('./chest_xray/train/PNEUMONIA/*.jpeg')
filtreFolder('./chest_xray/test/NORMAL/*.jpeg')
filtreFolder('./chest_xray/test/PNEUMONIA/*.jpeg')


#create folders 
import os
os.mkdir("./filter-yen-200")

# Save the picture with the filter in the previous folder created
filtreFolder1('./chest_xray/train/NORMAL/*.jpeg')
filtreFolder1('./chest_xray/train/PNEUMONIA/*.jpeg')
filtreFolder1('./chest_xray/test/NORMAL/*.jpeg')
filtreFolder1('./chest_xray/test/PNEUMONIA/*.jpeg')

#create folders 
import os
os.mkdir("./filter-mean-200")

# Save the picture with the filter in the previous folder created
filtreFolder2('./chest_xray/train/NORMAL/*.jpeg')
filtreFolder2('./chest_xray/train/PNEUMONIA/*.jpeg')
filtreFolder2('./chest_xray/test/NORMAL/*.jpeg')
filtreFolder2('./chest_xray/test/PNEUMONIA/*.jpeg')
"""
# Function that transform an image to an array of his pixels
def image_transform(path):
    dataset = [] 
    labels = []

    if path == './chest_xray/test/NORMAL':
        test_normal = 0
        for filename in glob.glob('./chest_xray/test/NORMAL/*.jpeg'):
            image = resize(cv2.imread(filename, -1).astype(np.float64), 200)
            # add the pixels corresponding to the image to the dataset 
            dataset.append(image)
            #ajout de l'index du dossier correspondant a la liste labels
            labels.append(0)
            
            test_normal = test_normal + 1
            #if test_normal == 100:
            #    break

    elif path == './chest_xray/test/PNEUMONIA':
        test_pneu = 0
        for filename in glob.glob('./chest_xray/test/PNEUMONIA/*.jpeg'):
            image = resize(cv2.imread(filename, -1).astype(np.float64), 200)
            # add the pixels corresponding to the image to the dataset 
            dataset.append(image)
            #ajout de l'index du dossier correspondant a la liste labels
            labels.append(1)
            
            test_pneu = test_pneu + 1
            #if test_pneu == 100:
            #    break
    
    elif path == './chest_xray/train/NORMAL':
        train_normal = 0
        for filename in glob.glob('./chest_xray/train/NORMAL/*.jpeg'):
            image = resize(cv2.imread(filename, cv2.COLOR_BGR2GRAY).astype(np.float64), 200)
            #ajout des pixels correspondantes à l'image au dataset 
            dataset.append(image)
            #ajout de l'index du dossier correspondant a la liste labels
            labels.append(0)
            
            train_normal = train_normal+1
            #if train_normal == 600:
            #    break
    
    elif path == './chest_xray/train/PNEUMONIA':
        train_pneu = 0
        for filename in glob.glob('./chest_xray/train/PNEUMONIA/*.jpeg'):
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = resize(image.astype(np.float64), 200)
            #ajout des pixels correspondantes à l'image au dataset 
            dataset.append(image)
            #ajout de l'index du dossier correspondant a la liste labels
            labels.append(1)
            
            train_pneu = train_pneu + 1
            #if train_pneu == 1200:
            #    break

    elif path == './filter-mean-200/train-normal':
        train_normal = 0
        for filename in glob.glob('./filter-mean-200/train-normal/*.jpeg'):
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = resize(image.astype(np.float64), 200)
            #ajout des pixels correspondantes à l'image au dataset 
            dataset.append(image)
            #ajout de l'index du dossier correspondant a la liste labels
            labels.append(0)
            
            train_normal = train_normal + 1
            #if train_normal == 600:
            #    break
    
    elif path == './filter-mean-200/train-pneumonia':
        train_pneu = 0
        for filename in glob.glob('./filter-mean-200/train-pneumonia/*.jpeg'):
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = resize(image.astype(np.float64), 200)
            #ajout des pixels correspondantes à l'image au dataset 
            dataset.append(image)
            #ajout de l'index du dossier correspondant a la liste labels
            labels.append(1)
            
            train_pneu = train_pneu + 1
            #if train_pneu == 1200:
            #    break
    
    elif path == './filter-mean-200/test-normal':
        test_normal = 0
        for filename in glob.glob('./filter-mean-200/test-normal/*.jpeg'):
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = resize(image.astype(np.float64), 200)
            #ajout des pixels correspondantes à l'image au dataset 
            dataset.append(image)
            #ajout de l'index du dossier correspondant a la liste labels
            labels.append(0)
            
            test_normal = test_normal + 1
            #if test_normal == 100:
            #    break
    
    elif path == './filter-mean-200/test-pneumonia':
        test_pneu = 0
        for filename in glob.glob('./filter-mean-200/test-pneumonia/*.jpeg'):
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = resize(image.astype(np.float64), 200)
            #ajout des pixels correspondantes à l'image au dataset 
            dataset.append(image)
            #ajout de l'index du dossier correspondant a la liste labels
            labels.append(1)
            
            test_pneu = test_pneu + 1
            #if test_pneu == 100:
            #    break

    #Création des array à l'aide de la bib numpy
    dataset = np.array(dataset) #Les variables indépendantes 
    labels = np.array(labels).reshape(-1, 1) #La variable dépendantepy
    
    return dataset,labels

# Function to concatenate the dataset with pneumonia and without
def concatenate (dataset,labels, dataset1,labels1):
    data = np.concatenate((dataset, dataset1), axis=0)
    lab = np.concatenate((labels, labels1), axis=0)
    return data,lab

# Function to split the data
def split(train_data, train_labels):
    #Création du dataframe
    df = pd.DataFrame(data = np.concatenate((train_data, train_labels), axis=0))
    print(df)

    #Découpage des données en test set et train set
    test_data = df.iloc[::2]
    train_data = df.iloc[1::2]
    #print(test_data)
    #print(train_data)
    
    Xtest = test_data.iloc[:, :-1].values
    ytest = test_data.iloc[:, -1].values
    
    Xtrain = train_data.iloc[:, :-1].values
    ytrain = train_data.iloc[:, -1].values
    
    sc = StandardScaler()
    Xtrain = sc.fit_transform(Xtrain)
    Xtest = sc.transform(Xtest)
    
    return Xtest,ytest,Xtrain,ytrain

#Model KNN
def KNN(Xtest,ytest,Xtrain,ytrain):
 
    classifier = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski', p = 2)
    modelKNN = classifier.fit(Xtrain, ytrain)
    
    #Prediction des résultats
    ypred = classifier.predict(Xtest)
    
    #Accuracy du model KNN
    print(f'accuracy for KNN with 5 neighbors = {(ypred.flatten() == ytest.flatten()).sum() / len(ytest) * 100.0}%')

    return modelKNN

#Model decision tree
def Tree(Xtest,ytest,Xtrain,ytrain):
    classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
    modelTree = classifier.fit(Xtrain, ytrain)
    
    #Prediction des résultats
    ypred = classifier.predict(Xtest)
    
    #Accuracy du model tree
    print(f'accuracy for decision tree using gini = {(ypred.flatten() == ytest.flatten()).sum() / len(ytest) * 100.0}%')

    return modelTree

#Model Bayes
def Bayes(Xtest,ytest,Xtrain,ytrain):
    classifier = GaussianNB()
    modelBayes = classifier.fit(Xtrain, ytrain)
    
    #Prediction des résultats
    ypred = classifier.predict(Xtest)
    
    #Accuracy du model Bayes
    print(f'accuracy for Bayes model = {(ypred.flatten() == ytest.flatten()).sum() / len(ytest) * 100.0}%')

    return modelBayes

#Model LDA
def LDA(Xtest, ytest, Xtrain, ytrain):
    classifier = LinearDiscriminantAnalysis()
    modelLDA = classifier.fit(Xtrain, ytrain)
    
    #Prediction des résultats
    ypred = classifier.predict(Xtest)
    
    #Accuracy du model LDA
    print(f'accuracy for LDA model = {(ypred.flatten() == ytest.flatten()).sum() / len(ytest) * 100.0}%')

    return modelLDA
    
# Function to load the data without filter   
def load_data ():
    dataset_train_normal,labels_train_normal = image_transform('./chest_xray/train/NORMAL')
    dataset_train_sick,labels_train_sick = image_transform('./chest_xray/train/PNEUMONIA')

    dataset_test_normal,labels_test_normal = image_transform('./chest_xray/test/NORMAL')
    dataset_test_sick,labels_test_sick = image_transform('./chest_xray/test/PNEUMONIA')

    Xtrain, ytrain = concatenate(dataset_train_normal, labels_train_normal, dataset_train_sick, labels_train_sick)
    Xtest, ytest = concatenate(dataset_test_normal, labels_test_normal, dataset_test_sick, labels_test_sick)

    dataset_train_normal,labels_train_normal = image_transform('./chest_xray/train/NORMAL')
    dataset_train_sick,labels_train_sick = image_transform('./chest_xray/train/PNEUMONIA')

    dataset_test_normal,labels_test_normal = image_transform('./chest_xray/test/NORMAL')
    dataset_test_sick,labels_test_sick = image_transform('./chest_xray/test/PNEUMONIA')

    Xtrain, ytrain = concatenate(dataset_train_normal, labels_train_normal, dataset_train_sick, labels_train_sick)
    Xtest, ytest = concatenate(dataset_test_normal, labels_test_normal, dataset_test_sick, labels_test_sick)
    
    a=[]
    for e in Xtrain:
        a.append(e.flatten())
    #print(a)

    b=[]
    for e in Xtest:
        b.append(e.flatten())
    #print(b)
    
    return a, b, ytrain, ytest

#Fucntion to load the data with the filter mean (the picture are save in local)
def load_data_filtre_mean_200 ():
    dataset_train_normal,labels_train_normal = image_transform('./filter-mean-200/train-normal')
    dataset_train_sick,labels_train_sick = image_transform('./filter-mean-200/train-pneumonia')

    dataset_test_normal,labels_test_normal = image_transform('./filter-mean-200/test-normal')
    dataset_test_sick,labels_test_sick = image_transform('./filter-mean-200/test-pneumonia')

    Xtrain, ytrain = concatenate(dataset_train_normal, labels_train_normal, dataset_train_sick, labels_train_sick)
    Xtest, ytest = concatenate(dataset_test_normal, labels_test_normal, dataset_test_sick, labels_test_sick)

    dataset_train_normal,labels_train_normal = image_transform('./filter-mean-200/train-normal')
    dataset_train_sick,labels_train_sick = image_transform('./filter-mean-200/train-pneumonia')

    dataset_test_normal,labels_test_normal = image_transform('./filter-mean-200/test-normal')
    dataset_test_sick,labels_test_sick = image_transform('./filter-mean-200/test-pneumonia')

    Xtrain, ytrain = concatenate(dataset_train_normal, labels_train_normal, dataset_train_sick, labels_train_sick)
    Xtest, ytest = concatenate(dataset_test_normal, labels_test_normal, dataset_test_sick, labels_test_sick)
    
    a=[]
    for e in Xtrain:
        a.append(e.flatten())
    #print(a)

    b=[]
    for e in Xtest:
        b.append(e.flatten())
    print(len(b[0]))
    
    return a, b, ytrain, ytest


### Part to load the data and execute the model

#Xtrain, Xtest, ytrain, ytest = load_data()
"""
#Model selection based on accuracy

#KNN(Xtest, ytest, Xtrain, ytrain)
modelTree = Tree(Xtest, ytest, Xtrain, ytrain)
"""
#modelBayes = Bayes(Xtest, ytest, Xtrain, ytrain)
"""
modelLDA = LDA(Xtest, ytest, Xtrain, ytrain)
#cm = metrics.confusion_matrix(ytest,ypred)
"""
print ("---------------------------------------------------------------------------------------------------------------")

Xtrain, Xtest, ytrain, ytest = load_data_filtre_mean_200 ()
modelKNN = KNN(Xtest, ytest, Xtrain, ytrain)
#Tree(Xtest, ytest, Xtrain, ytrain)
#Bayes(Xtest, ytest, Xtrain, ytrain)
#ypred, classifier = LDA(Xtest, ytest, Xtrain, ytrain)

print ("---------------------------------------------------------------------------------------------------------------")

## Part use to save the model on a joblib file

filename = "KNN_model.joblib"
joblib.dump(modelKNN, filename)

"""
filename1 = "Tree_model.joblib"
joblib.dump(modelTree, filename1)

filename2 = "Bayes_model.joblib"
joblib.dump(modelBayes, filename2)

filename3 = "LDA_model.joblib"
joblib.dump(modelLDA, filename3)
"""

"""
dataset = []
file = "filter-mean-200/test-normal/filter-export2.jpeg"
loaded_model = joblib.load('KNN_model.joblib')
image = resize(cv2.imread(file, cv2.COLOR_BGR2GRAY).astype(np.float64), 200)
dataset.append(image)

b=[]
for e in dataset:
    b.append(e.flatten())
print(loaded_model.predict_proba(b))


dataset = []
file = "chest_xray/test/NORMAL/NORMAL-8876914-0001.jpeg"
loaded_model = joblib.load('LDA_model.joblib')
image = resize(cv2.imread(file, cv2.COLOR_BGR2GRAY).astype(np.float64), 200)
dataset.append(image)

b=[]
for e in dataset:
    b.append(e.flatten())
print(loaded_model.predict_proba(b))


dataset = []
file = "chest_xray/test/NORMAL/NORMAL-8876914-0001.jpeg"
loaded_model = joblib.load('Tree_model.joblib')
image = resize(cv2.imread(file, cv2.COLOR_BGR2GRAY).astype(np.float64), 200)
dataset.append(image)

b=[]
for e in dataset:
    b.append(e.flatten())
print(loaded_model.predict_proba(b))
"""
"""
dataset = []
file = "chest_xray/test/NORMAL/NORMAL-8876914-0001.jpeg"
loaded_model = joblib.load('Bayes_model.joblib')
image = resize(cv2.imread(file, cv2.COLOR_BGR2GRAY).astype(np.float64), 200)
dataset.append(image)

b=[]
for e in dataset:
    b.append(e.flatten())
print(loaded_model.predict_proba(b))
"""