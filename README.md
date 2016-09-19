this an all convolutional CNN using the cirfar10 dataset
as of 17092016 performance, without regularization is : accuracy  92% validation 82% (so, overfitting)

the files are :
- convnet3.py runs the keras model, started by learn.sh
- modelallcnn.py the keras model
- trace.py a keras callback for tracing activation values in case of NaN
- imageutils.py image utilities
- plot.py plot accuracy during training (used by plot.sh)
- plot.sh plot accuracy
- learn.sh start training, an optional first argument would be a saved weight files for restarting training from a saved point in time

modification log :
- shuffle batch samples using skleanr shuffle()
- set a small lambda for L2 regularization

future modifications :
- recheck input dataset

