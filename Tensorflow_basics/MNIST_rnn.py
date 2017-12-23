import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.python.ops import rnn,rnn_cell -------- older version
from tensorflow.contrib import rnn
# one_hot => 0 is [1,0,0,0,0,0,0,0,0,0] and 1 is [0,1,0,0,0,0,0,0,0,0]
mnist = input_data.read_data_sets("MNIST_data",one_hot = True)

nm_epochs = 2
n_class = 10
batch_size = 128
ch_size = 28
no_ch = 28
rnn_size =128 # 256 or 512 .....

# Variables taken as input

x = tf.placeholder('float32',[None,no_ch,ch_size])
y = tf.placeholder('float32',[None,10])

#Computation graph Model
def recurrent_neural_model(data):
	lay = {'weights':tf.Variable(tf.random_normal([rnn_size,n_class])),
		   'biases' :tf.Variable(tf.random_normal([n_class]))}

	data = tf.transpose(data,[1,0,2])
	data = tf.reshape(data,[-1,ch_size])
	data = tf.split(axis = 0,num_or_size_splits = no_ch,value = data,name = 'split')

	lstm_cell = rnn.BasicLSTMCell(rnn_size)
	output,states = rnn.static_rnn(lstm_cell,data,dtype = tf.float32)

	out = tf.add(tf.matmul(output[-1],lay['weights']),lay['biases'])

	return out

def train(x):
	pred = recurrent_neural_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
	optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)


	with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(nm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epo_x,epo_y = mnist.train.next_batch(batch_size)
				epo_x = epo_x.reshape((batch_size,no_ch,ch_size))
				_,epo_cost =sess.run([optimizer,cost],feed_dict={x:epo_x,y:epo_y})
				epoch_loss+=epo_cost
			print('Epoch',epoch + 1,'done of',nm_epochs,'loss_calculated:',epoch_loss)

		corr = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

		accu = tf.reduce_mean(tf.cast(corr,'float32'))

		print('Accuracy on test set is :',accu.eval({x:mnist.test.images.reshape((-1,no_ch,ch_size)),y:mnist.test.labels}))	

train(x)

	 
