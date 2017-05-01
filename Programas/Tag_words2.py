from joblib import Parallel, delayed
import os
import nltk
from os import listdir
from textblob import TextBlob

#Function 
def OnlyNouns(filename):
	print(filename)
	path_from="/home/alejandro/Documentos/ArquitecturadeDatos/Proyecto/Modelos/Asuntos"
	path_to="/home/alejandro/Documentos/ArquitecturadeDatos/Proyecto/Modelos/Asuntos_Nuevos"
	path_to_file = os.path.join(path_from, filename)
	fd = open(path_to_file, 'r')
	lines = fd.read()
	blob = TextBlob(lines)
	noun = [n for n,t in blob.tags if t == 'NN']
	path_to_export = os.path.join(path_to, filename)
	thefile = open(path_to_export, 'w')
	for item in noun:
	  thefile.write("%s\n" % item)


filenames=listdir("/home/alejandro/Documentos/ArquitecturadeDatos/Proyecto/Modelos/Asuntos")
results = Parallel(n_jobs=-1, verbose=0, backend="multiprocessing")(
             map(delayed(OnlyNouns), filenames))


