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

		cell_size = (20, 20)   # h x w in pixels
		block_size = (1, 1)  # h x w in cells
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

		end = time.time() - start
		print("Time took for image description: " + str(end))
		return charateristics


for i in range(1, 2):

	f_target = [0 for _ in range(12)]
	t_target = [1 for _ in range(36)]


	first_base = glob.glob("Base de Dados Organizada/renatha_originalforjadahabilidosaaleatoria/"+str(i)+"/*.png")
	print("Creating descriptions for first base")
	first_base_desc = createTrainingInstances(first_base)

	second_base = glob.glob("Base de Dados Organizada/renatha_tamanhosdiferentes_basenormalmenorrmaior/"+str(i)+"/*.png")
	print("Creating descriptions for second base")
	second_base_desc = createTrainingInstances(second_base)

	##############################################################
	#Fit model, auto set probabilities
	model = hmm.GaussianHMM(n_components=2)
	model.fit(first_base_desc)
	print("Iter ============ "+str(i))
	predictions = model.predict(t_target+f_target)
	###############################################################

	#print(predictions)
	print(classification_report(t_target+f_target,predictions))