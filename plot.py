
def plot(history,size):
	f=open('plot.data','w')
	for i in range(0,size):
		line=str(i)+' '+str(history['acc'][i])+' '+str(history['val_acc'][i])+'\n'
		f.write(line)
	f.close()
