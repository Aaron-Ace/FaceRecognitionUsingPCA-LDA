import numpy as np
import os, glob, random, cv2
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def loadImageSet(folder='./att_faces', sampleCount=5):  # 載入圖像集，隨機選擇sampleCount張圖片用於訓練
    trainData = []
    testData = []
    yTrain = []
    yTest = []
    for k in range(40):
        folder2 = os.path.join(folder, 's%d' % (k + 1))
        # print('debug:{}'.format(folder2))
        # print('debug:{}'.format(glob.glob(os.path.join(folder2, '*.pgm'))))
        data = [cv2.imread(d, 0) for d in glob.glob(os.path.join(folder2, '*.pgm'))]
        sample = random.sample(range(10), sampleCount)
        trainData.extend([data[i].ravel() for i in range(10) if i in sample])
        testData.extend([data[i].ravel() for i in range(10) if i not in sample])
        yTest.extend([k] * (10 - sampleCount))
        yTrain.extend([k] * sampleCount)
    return np.array(trainData), np.array(yTrain), np.array(testData), np.array(yTest)


def PCA_model(dimension, xTrain, xTest, yTrain): # PCA 模型訓練
    pca = PCA(n_components=dimension)
    pca.fit(xTrain, yTrain)
    xTrain = pca.transform(xTrain)
    xTest = pca.transform(xTest)
    return xTrain, xTest


def LDA_model(xTrain, xTest, yTrain): # LDA 模型訓練
    lda = LDA()
    xTrain = lda.fit_transform(xTrain, yTrain)
    xTest = lda.transform(xTest)
    return xTrain, xTest


def plot_confusion_matrix(name, dimension, confusion_mat): #繪製混淆矩陣
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
    plt.title('{} Dimension {} Confusion matrix'.format(name, dimension))
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def recognizer(dimension, xTrain, xTest, yTrain, yTest):
    svm = SVC(kernel='linear')  # 支援向量機方法
    svm.fit(np.array(xTrain), np.float32(yTrain))
    yPredict = svm.predict(np.float32(xTest))
    # print('really: {}'.format(np.array(yTest)))
    # print('predict: {}'.format(yPredict))
    print('維度%d: SVM向量機識別率: %.2f%%' % (dimension, (yPredict == np.array(yTest)).mean() * 100))
    return yTest.tolist(), yPredict


def main(dimension):
    xTrain_, yTrain, xTest_, yTest = loadImageSet()
    # num_train, num_test = xTrain_.shape[0], xTest_.shape[0]
    # print(num_train)
    # print(num_test)

    xTrain, xTest = PCA_model(dimension, xTrain_, xTest_, yTrain)

    result, ans = recognizer(dimension, xTrain, xTest, yTrain, yTest)

    confusion_mat = confusion_matrix(ans, result)
    plot_confusion_matrix("PCA", dimension, confusion_mat)

    xTrain, xTest = PCA_model(dimension, xTrain_, xTest_, yTrain)

    xTrain, xTest = LDA_model(xTrain, xTest, yTrain)

    result, ans = recognizer(dimension, xTrain, xTest, yTrain, yTest)

    confusion_mat = confusion_matrix(ans, result)
    plot_confusion_matrix("PCA & LDA", dimension, confusion_mat)


if __name__ == '__main__':
    main(10)
    main(20)
    main(30)
    main(40)
    main(50)
