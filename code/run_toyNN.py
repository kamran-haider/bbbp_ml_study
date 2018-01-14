training_data = os.path.abspath("toyNN/tests/test_datasets/train_catvnoncat.h5")
test_data = os.path.abspath("toyNN/tests/test_datasets/test_catvnoncat.h5")
train_x_orig, train_y, test_x_orig, test_y, classes = load_test_data(training_data, test_data)
# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Preprocess
# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

input_layer_nodes = num_px * num_px * 3
layer_sizes = [input_layer_nodes, 20, 7, 5, 1]
layers = [Input(input_layer_nodes), ReLU(20), ReLU(7), ReLU(5), Sigmoid(1)]
np.random.seed(1)

nn = Network(train_x, train_y, layers, weight_initialization="custom")
nn.train(learning_rate=0.0075, n_epochs=2500)
"""
A = nn.X
print(A.shape)
for i in xrange(1, len(nn.layer_sizes)):
    print("Layer: ", i)
    #print(nn.W[i].shape)
    #print(nn.b[i].shape)
    L = nn.layer_types[i]()
    A = L.forward_pass(A)
    print(A.shape)
"""

