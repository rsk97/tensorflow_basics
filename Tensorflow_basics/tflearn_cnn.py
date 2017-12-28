import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

x,y,test_x,test_y = mnist.load_data(one_hot = True)

x,test_x = x.reshape([-1,28,28,1]),test_x.reshape([-1,28,28,1])

convnet = input_data(shape = [None,28,28,1],name = 'input')
#                                      output,window_size        window_size
convnet =  max_pool_2d(conv_2d(convnet,34,2,activation = 'relu'),2)
convnet =  max_pool_2d(conv_2d(convnet,64,2,activation = 'relu'),2)

convnet = fully_connected(convnet,1024,activation = 'relu')
convnet = dropout(convnet,0.8)

convnet = fully_connected(convnet,10,activation = 'softmax')
convnet = regression(convnet,optimizer = 'adam',learning_rate = 0.001,loss = 'categorical_crossentropy',name = 'targets')

model = tflearn.DNN(convnet)

model.fit({'input':x},{'targets':y},n_epoch = 10,validation_set = ({'input':test_x},{'targets':test_y}),snapshot_step = 500,show_metric = True,run_id = 'mnist')

model.save('tflearnCNN.model')
'''
model,load('tflearnCNN.model')
print(model.predict([test_x[1]]))
'''