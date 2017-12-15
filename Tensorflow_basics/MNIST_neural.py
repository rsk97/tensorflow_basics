import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# one_hot => 0 is [1,0,0,0,0,0,0,0,0,0] and 1 is [0,1,0,0,0,0,0,0,0,0]
mnist = input_data.read_data_sets("/tmp/data",one_hot = True)

n_nodes_hidden1 = 500
n_nodes_hidden2 = 500
n_nodes_hidden3 = 500

n_class = 10
batch_size = 100

# Variables taken as input

x = tf.placeholder('float32',[None,28*28])
y = tf.placeholder('float32',[None,10])

#Computation graph Model
def neural_model(data):
	hidd_lay_1 = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hidden1])),
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
	optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

	nm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(nm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epo_x,epo_y = mnist.train.next_batch(batch_size)
				_,epo_cost =sess.run([optimizer,cost],feed_dict={x:epo_x,y:epo_y})
				epoch_loss+=epo_cost
			print('Epoch',epoch,'done of',nm_epochs,'loss_calculated:',epoch_loss)

		corr = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

		accu = tf.reduce_mean(tf.cast(corr,'float32'))

		print('Accuracy on test set is :',accu.eval({x:mnist.test.images,y:mnist.test.labels}))	

train(x)

	 