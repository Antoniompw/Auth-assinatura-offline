import cv2
import glob
import time
import sklearn.svm as svm
from hmmlearn import hmm
# Importando o MLP
from sklearn.neural_network import MLPClassifier
# Validação cruzada
from sklearn.model_selection import cross_val_score
# Hold Out
from sklearn import model_selection
# Importando as metricas de predição
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

def createTrainingInstances(images):
		start = time.time()

		charateristics = []

		cell_size = (40,40)   # h x w in pixels
		block_size = (2, 2)  # h x w in cells
		nbins = 9            # number of orientation bins

		for img in images:
			# Image reading
			img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
			# winSize is the size of the image cropped to an multiple of the cell size
			hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
			                                  img.shape[0] // cell_size[0] * cell_size[0]),
			                        _blockSize=(block_size[1] * cell_size[1],
			                                    block_size[0] * cell_size[0]),
			                        _blockStride=(cell_size[1], cell_size[0]),
			                        _cellSize=(cell_size[1], cell_size[0]),
			                        _nbins=nbins)
			n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
			hog_feats = hog.compute(img).ravel()
			charateristics.append(hog_feats)
			print(hog_feats)
		end = time.time() - start
		print("Time took for image description: " + str(end))
		return charateristics

positive = glob.glob("Base de Dados Organizada/True/3/*.png")
# Positive descriptions
print("Creating descriptions for positive")
positive_d = createTrainingInstances(positive)
t_target = [1 for _ in range(len(positive))]

negative = glob.glob("Base de Dados Organizada/False/3/*.png")
# Negative descriptions
print("Creating descriptions for negative")
negative_d = createTrainingInstances(negative)
f_target = [0 for _ in range(len(negative))]


charateristics = positive_d+negative_d
target = t_target+f_target
print("Number of positive elements: "+str(len(positive_d)))
print("Number of negative elements: "+str(len(negative_d)))
print("Number of charateristics for all images: "+str(len(positive_d[0])))

validation_size = 0.2
seed = 4
x_train, x_test, y_train, y_test = model_selection.train_test_split(positive_d, t_target, test_size=validation_size, random_state=seed)


lr = hmm.GaussianHMM(n_components=2, covariance_type="diag")
lr.startprob_ = np.array([0.5, 0.5])
lr.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
lr.fit(positive_d)
predictions = lr.predict(positive_d+negative_d)

print(predictions)
print(classification_report(t_target+f_target,predictions))