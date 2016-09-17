from keras.callbacks import Callback
from keras import backend as K
import numpy as np

class TraceWeights(Callback):

	def __init__(self,mode,engine):
		''' mode should be 0 for training, 1 for testing , engine is used to get current batch data'''
		self.mode = mode
		self.engine = engine

	def on_train_begin(self, logs={}):
		print('train begin')

	def print_ndarray_stats(self, s, i, X):
		''' i layer number,
			s data name,
			X data array '''
		#print("layer ",i, s, " shape : ", X.shape," max : ", np.amax(X)," min : ",np.amin(X)," avg : ", np.mean(X)\
		#		   ,"NaN count : ", np.count_nonzero(np.isnan(X)), "non Nan count : ", np.count_nonzero(~np.isnan(X)) )
		print("L:",i, ":",s, ":", X.shape,":", np.amin(X),":",np.amax(X),":", np.mean(X),":", \
			   np.count_nonzero(np.isnan(X)), ":", np.count_nonzero(~np.isnan(X)) )

	def on_batch_begin(self, batch, logs={}):
		''' on batch begin we display the statistics of the weights and the outputs to see how NaN propagate '''
		number_of_layers= len(self.model.layers)
		for i in range(1,number_of_layers):
			weights=self.model.layers[i].get_weights()
			if len(weights)>0:
				self.print_ndarray_stats("W", i,abs(weights[0]))	# trace gradient scale
			get_layer_output = K.function([self.model.layers[0].input,K.learning_phase()],[self.model.layers[i].output])
			X = self.engine.X_batch_current
			layer_output = get_layer_output([X,self.mode])[0]
			self.print_ndarray_stats("Y", i,abs(layer_output))		# trace activation scale


#IDEA : use a much smaller lambda