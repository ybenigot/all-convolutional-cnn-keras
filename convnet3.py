# adapted from keras examples
from keras.optimizers import SGD, Nadam
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, activity_l2
from keras.callbacks import EarlyStopping,ModelCheckpoint

from trace import TraceWeights

import numpy as np
import sys
import imageutils as im
import time
import cifar10labels as cl
import random as rn
import datetime as dt
import os
import plot as pl
import modelallcnn as mo

############### Parameters ################
image_size=32
image_border=16
input_size=image_size+2*image_border
batch_size_param=32
learn_rate=0.0003
#decay_param=4e-5 # 1 - 0.1**(1/(100*45000/256))) 
training_set_ratio=0.9
translation_augmentation=0.2
flip_augmentation=True

class PreProcessor:

	@staticmethod
	def load_datasets():
		''' load and normalize data from dataset files '''
		(X, y), (X_test, y_test) = cifar10.load_data()
		n = X.shape[0]
		n1 = int(n * training_set_ratio)
		n2 = n-n1
		#randomize dataset split
		index_val = rn.sample(range(n), n2)
		X_val = X[index_val,:]
		y_val = y[index_val]
		index_train=[i for i in range(n) if i not in index_val]		
		X_train=X[index_train,:]
		y_train=y[index_train]
		return X_train, y_train, X_val, y_val, X_test, y_test

	@staticmethod
	def compute_average(data):
		''' compute mean per color channel for all data '''
		m = np.zeros(data.shape[1])
		for j in range(0,data.shape[1]):
			m[j] = np.mean(data[:,j,:])
		return m

	@staticmethod
	def scale_data(data,avg_per_channel):
		''' scale the image pixel values into interval 0,2, mean will be substrated later '''
		scale=128
		n=data.shape[0]
		data = data.astype('float32')
		data = data.reshape((n,3,image_size,image_size))
		if input_size>image_size:
			# extend image size with zeroes
			data2 = np.zeros((n,3,input_size,input_size),dtype=np.float32)
			for i in range(0,n):
				for j in range(0,3):
					#substract mean, per sample and per color channel 
					data2[i,j,image_border:image_size+image_border,image_border:image_size+image_border] =\
						data[i,j,:,:] - avg_per_channel[j]
			data2 /= scale
			return data2	
		else:
			data /= scale
			return data

	@staticmethod
	def augment(X):
		''' compute pseudo-random translation, flip etc of X data to augment the dataset inputs '''
		''' pseudo-random augmentation means that multiple augmenation on the same data will yield the same result '''
		n=X.shape[0]
		rn.seed(a=1, version=2)
		max_translation=int(image_size*translation_augmentation)
		X2=np.zeros(X.shape)
		x_max=X.shape[2]
		y_max=X.shape[3]
		x_range=range(0,x_max)
		y_range=range(0,y_max)
		# loop on sample, channel, x coord, y coord
		for i in range(0,n):
			flip=bool(rn.randrange(0,1,1))
			x_translation=rn.randrange(0, max_translation,1)
			y_translation=rn.randrange(0, max_translation,1)
			for k in x_range:
				for l in y_range:
					if   k+x_translation in x_range and l+y_translation in y_range:
						if flip:
							for j in range(0,3): # same augmentation translation/flip for all channels
								X2[i,j,x_max-k,y_max-l]=image=X[i,j,k+x_translation,l+y_translation]
						else :
							for j in range(0,3):
								X2[i,j,k,l]=image=X[i,j,k+x_translation,l+y_translation]
		return X2

	def process_data_batch(self,X,y,avg_per_channel):
		''' preprocess a batch of data in memory, same algorithm for train, validation, and test data '''
		''' note : the same batch of data is "re augmented" again for each pass '''

		X=self.scale_data(X,avg_per_channel)
		X=self.augment(X)
		y = to_categorical(y,10)
		return X,y

class Engine:

	X_batch_current=0
	y_batch_current=0

	def __init__(self,preprocessor):
		self.preprocessor = preprocessor

	def dataGenerator(self,X,y,batch_size,avg_per_channel):
		''' a python 3 generator for producing batches of data '''
		print("new generator for %s \n" % (X.shape,) )
		while(True):
			N=int(X.shape[0]/batch_size)
			for i in range(N):
				X_batch = X[i*batch_size:(i+1)*batch_size,:,:,:]
				y_batch = y[i*batch_size:(i+1)*batch_size]
				self.X_batch_current,self.y_batch_current = self.preprocessor.process_data_batch(X_batch, y_batch, avg_per_channel)
				#print(i,'X_.shape ',X2.shape,'y.shape ',y2.shape,'  y ',y2)
				yield self.X_batch_current,self.y_batch_current
			# random shuffle the data set after each batch

	def fit(self,model , X_train, y_train, X_val, y_val, epochs,avg_per_channel):
		''' train the model '''
		earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath="record/weights.{epoch:02d}-{val_loss:.2f}.hdf", verbose=1, save_best_only=True)
		traceWeightsTrain=TraceWeights(1,self)
		g_train=self.dataGenerator(X_train, y_train, batch_size=batch_size_param,avg_per_channel=avg_per_channel)
		g_valid=self.dataGenerator(X_val, y_val,     batch_size=batch_size_param,avg_per_channel=avg_per_channel)
		history=model.fit_generator(g_train,callbacks=[checkpointer,\
			#earlyStopping,\
			#traceWeightsTrain\
			],\
			samples_per_epoch=len(X_train),nb_epoch=epochs,verbose=1,validation_data=g_valid,nb_val_samples=y_val.shape[0])
		return history

	@staticmethod
	def predict(model,X,y):
		''' predict Y given X using model '''
		pred = model.predict(X, batch_size=batch_size_param, verbose=0)
		#g.fit(X)
		#pred = model.predict_generator(g.flow(X, y, batch_size=512), X.shape[0])
		return pred

	@staticmethod
	def compute_accuracy(pred,Y):
		'''compute prediction accuracy by matching pred and Y'''
		comparison = np.argmax(pred,1)==np.argmax(Y,1)
		accuracy = sum(comparison)/pred.shape[0]
		return accuracy


def show_results(pred,X,Y):
	classification=np.argmax(pred,1)	
	for i in rn.sample(range(X.shape[0]), 1):
		im.display_normalized_image(X[i,:],input_size)
		print('prediction:',cl.labels[classification[i]],'actual value:',cl.labels[np.argmax(Y[i])])
		time.sleep(5)

def main():

	epochs=int(sys.argv[1])
	print(epochs,' epochs')

	try:
		reload_model=sys.argv[3]
	except:
		reload_model="NO"


	preprocessor = PreProcessor()
	engine = Engine(preprocessor)

	X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.load_datasets()

	avg_per_channel=preprocessor.compute_average(X_train)

	X_batch = X_train[0:batch_size_param,:,:,:]
	y_batch = y_train[0:batch_size_param]
	#X1,y1 = preprocessor.process_data_batch(X_batch,y_batch,avg_per_channel)
	# for i in range(0,3):
	# 	im.display_normalized_image(X1[i,:],input_size,avg_per_channel)
	# 	im.display_image(X_batch[i,:],image_size)

	print('X_train.shape ',X_train.shape,'y_train.shape ',y_train.shape)
	print('X_val.shape ',  X_val.shape,  'y_val.shape ',  y_val.shape)
	print('X_test.shape ', X_test.shape, 'y_test.shape ', y_test.shape)

	# prepare the model
	model = mo.make_model(input_size)
	#opt = SGD(lr=learn_rate, decay=decay_param, momentum=0.9, nesterov=True)
	opt = Nadam(lr=learn_rate)#,clipvalue=100)
	model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=["accuracy"])

	if reload_model != "NO":
		print('load model weights:',reload_model)
		model.load_weights(reload_model)

	print('model parameters:',model.count_params())
	print('model characteristics:',model.summary())
	print('----------------------------------------------------------------------------------------')

	hist=engine.fit(model , X_train, y_train, X_val, y_val, epochs,avg_per_channel)
	print(hist.history)

	# test the model
	pred = engine.predict(model,X_test,y_test)
	accuracy=engine.compute_accuracy(pred,y_test)
	print('accuracy on test data: ',accuracy*100, '%')
	show_results(pred,X_test,y_test)

	# save learned weights
	f="%d-%m-%y"
	filename='record/weights-'+dt.date.today().strftime(f)
	model.save_weights(filename,overwrite=True)

	pl.plot(hist.history,len(hist.history['acc']))
	os.system('./plot.sh')


if __name__ == "__main__":
    main()


