import numpy as np
import os, glob, random, cv2
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Within class scatter matrix:
        # SW = sum((X_c - mean_X_c)^2 )

        # Between class scatter:
        # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            # (4, 1) * (1, 4) = (4,4) -> reshape
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)


def PCA_model(dimension, xTrain, xTest): # PCA 模型訓練
    pca = PCA(dimension)
    pca.fit(xTrain)
    xTrain = pca.transform(xTrain)
    xTest = pca.transform(xTest)
    return xTrain, xTest


def LDA_model(xTrain, xTest): # LDA 模型訓練
    lda = LDA()
    xTrain = lda.fit_transform(xTrain)
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

    xTrain, xTest = PCA_model(dimension, xTrain_, xTest_)

    result, ans = recognizer(dimension, xTrain, xTest, yTrain, yTest)

    confusion_mat = confusion_matrix(ans, result)
    plot_confusion_matrix("PCA", dimension, confusion_mat)

    xTrain, xTest = PCA_model(dimension, xTrain_, xTest_)

    xTrain, xTest = LDA_model(xTrain, xTest)

    result, ans = recognizer(dimension, xTrain, xTest, yTrain, yTest)

    confusion_mat = confusion_matrix(ans, result)
    plot_confusion_matrix("PCA & LDA", dimension, confusion_mat)


if __name__ == '__main__':
    main(10)
    main(20)
    main(30)
    main(40)
    main(50)
