import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# one_hot => 0 is [1,0,0,0,0,0,0,0,0,0] and 1 is [0,1,0,0,0,0,0,0,0,0]
mnist = input_data.read_data_sets("MNIST_data",one_hot = True)



n_class = 10
batch_size = 128

# Variables taken as input

x = tf.placeholder('float32',[None,28*28])
y = tf.placeholder('float32',[None,10])

keep_rate = 0.8
# can pass it in dicionary if wanted as an arg
#keep_prob = tf.placeholder('float32')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
	    #                    frame size            movement index     to append pixels in the end like a hangover frame.
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#Computation graph Model
def neural_model(data):
	weights = {'w_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
			   'w_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
			   'w_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
			   'w_out':tf.Variable(tf.random_normal([1024,n_class]))}

	biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
			  'b_conv2':tf.Variable(tf.random_normal([64])),
			  'b_fc':tf.Variable(tf.random_normal([1024])),
			  'b_out':tf.Variable(tf.random_normal([n_class]))}

	data = tf.reshape(data,shape = [-1,28,28,1])

	conv1 =  maxpool2d(tf.nn.relu(tf.add(conv2d(data,weights['w_conv1']),biases['b_conv1'])))
	conv2 =  maxpool2d(tf.nn.relu(tf.add(conv2d(conv1,weights['w_conv2']),biases['b_conv2'])))

	fc = tf.reshape(conv2,shape = [-1,7*7*64])
	fc = tf.nn.relu(tf.add(tf.matmul(fc,weights['w_fc']),biases['b_fc']))
	fc = tf.nn.dropout(fc,keep_rate)
	output = tf.add(tf.matmul(fc,weights['w_out']),biases['b_out'])

	return output
	

def train(x):
	pred = neural_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
	optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)      # rate 1e-4 is the best rate -----  info frm docs

	nm_epochs = 10

	with tf.Session(config=tf.ConfigProto(log_device_placement = False )) as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(nm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epo_x,epo_y = mnist.train.next_batch(batch_size)
				_,epo_cost =sess.run([optimizer,cost],feed_dict={x:epo_x,y:epo_y})
				epoch_loss+=epo_cost
			print('Epoch',epoch + 1,'done of',nm_epochs,'loss_calculated:',epoch_loss)

		corr = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

		accu = tf.reduce_mean(tf.cast(corr,'float32'))

		print('Accuracy on test set is :',accu.eval({x:mnist.test.images,y:mnist.test.labels}))	

train(x)

# 98,79% accuracy