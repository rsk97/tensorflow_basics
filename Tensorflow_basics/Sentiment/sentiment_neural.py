import tensorflow as tf
from sentiment import create_feature_label
import numpy as np
# can also be loaded from the pickle stored as well
train_x,train_y,test_x,test_y = create_feature_label('pos.txt','neg.txt')

n_nodes_hidden1 = 500
n_nodes_hidden2 = 500
n_nodes_hidden3 = 500

n_class = 2
batch_size = 100

# Variables taken as input

x = tf.placeholder('float32',[None,len(train_x[0])])
y = tf.placeholder('float32',[None,2])

#Computation graph Model
def neural_model(data):
	hidd_lay_1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hidden1])),
				  'biases' :tf.Variable(tf.random_normal([n_nodes_hidden1]))}

	hidd_lay_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hidden1,n_nodes_hidden2])),
				  'biases' :tf.Variable(tf.random_normal([n_nodes_hidden2]))}

	hidd_lay_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hidden2,n_nodes_hidden3])),
				  'biases' :tf.Variable(tf.random_normal([n_nodes_hidden3]))}

	output     = {'weights':tf.Variable(tf.random_normal([n_nodes_hidden3,n_class])),
			      'biases' :tf.Variable(tf.random_normal([n_class]))}

	l1 = tf.nn.relu(tf.add(tf.matmul(data,hidd_lay_1['weights']),hidd_lay_1['biases']))
	l2 = tf.nn.relu(tf.add(tf.matmul(l1,hidd_lay_2['weights']),hidd_lay_2['biases']))
	l3 = tf.nn.relu(tf.add(tf.matmul(l2,hidd_lay_3['weights']),hidd_lay_3['biases']))

	out = tf.add(tf.matmul(l3,output['weights']),output['biases'])

	return out

def train(x):
	pred = neural_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	nm_epochs = 20

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(nm_epochs):
			epoch_loss = 0
			i = 0
			while(i<len(train_x)):
				end = i + batch_size
			#for _ in range(int(mnist.train.num_examples/batch_size)):
				epo_x,epo_y = np.array(train_x[i:end]),np.array(train_y[i:end])
				_,epo_cost =sess.run([optimizer,cost],feed_dict={x:epo_x,y:epo_y})
				epoch_loss+=epo_cost
				i+=batch_size
			print('Epoch',epoch+1,'done of',nm_epochs,'loss_calculated:',epoch_loss)

		corr = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

		accu = tf.reduce_mean(tf.cast(corr,'float32'))

		print('Accuracy on test set is :',accu.eval({x:test_x,y:test_y}))	

train(x)

	 