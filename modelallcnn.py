# adapted from keras examples
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import LeakyReLU

def make_model(input_size):

	maps_count_param=24 # a higher value like 96 ends up with a Nan loss on GPU
	lambda_param=0.00001 # a very low value since we have so much parameters
	alpha_param=0.3

	''' define the model'''
	model = Sequential()

	model.add(Convolution2D(maps_count_param, 3, 3, border_mode='same', input_shape=(3, input_size, input_size),init='orthogonal', W_regularizer=l2(lambda_param)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Dropout(0.1))
	model.add(Convolution2D(maps_count_param, 3, 3, border_mode='same', init='orthogonal', W_regularizer=l2(lambda_param)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Convolution2D(maps_count_param, 3, 3, border_mode='same', init='orthogonal', W_regularizer=l2(lambda_param),subsample=(2,2)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Dropout(0.2))
	#model.add(BatchNormalization(mode=1))

	model.add(Convolution2D(maps_count_param*2, 3, 3, border_mode='same', init='orthogonal', W_regularizer=l2(lambda_param)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Convolution2D(maps_count_param*2, 3, 3, border_mode='same', init='orthogonal', W_regularizer=l2(lambda_param)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Convolution2D(maps_count_param*2, 3, 3, border_mode='same', init='orthogonal', W_regularizer=l2(lambda_param),subsample=(2,2)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Dropout(0.3))
	#model.add(BatchNormalization(mode=1))

	model.add(Convolution2D(maps_count_param*3, 3, 3, border_mode='same', init='orthogonal', W_regularizer=l2(lambda_param)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Convolution2D(maps_count_param*3, 3, 3, border_mode='same', init='orthogonal', W_regularizer=l2(lambda_param)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Convolution2D(maps_count_param*3, 3, 3, border_mode='same', init='orthogonal', W_regularizer=l2(lambda_param),subsample=(2,2)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Dropout(0.4))
	#model.add(BatchNormalization(mode=1))

	model.add(Convolution2D(maps_count_param*4, 3, 3, border_mode='same', init='orthogonal', W_regularizer=l2(lambda_param)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Convolution2D(maps_count_param*4, 3, 3, border_mode='same', init='orthogonal', W_regularizer=l2(lambda_param)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Convolution2D(maps_count_param*4, 3, 3, border_mode='same', init='orthogonal', W_regularizer=l2(lambda_param),subsample=(2,2)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Dropout(0.5))
	#model.add(BatchNormalization(mode=1))	

	#print('model characteristics:',model.summary())

	model.add(Flatten())

	model.add(Dense(maps_count_param*16, W_regularizer=l2(lambda_param)))
	model.add(LeakyReLU(alpha=alpha_param))
	model.add(Dense(maps_count_param*2, W_regularizer=l2(lambda_param)))
	model.add(LeakyReLU(alpha=alpha_param))

	model.add(Dense(10, W_regularizer=l2(lambda_param)))
	model.add(Activation('softmax'))

	return model
