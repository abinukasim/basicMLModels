import numpy as np

"""
We are going to use the California housing dataset provided by sklearn
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
to train a 2-layer fully connected neural net. We are going to build the neural network from scratch.
"""


class NeuralNet:
    def __init__(
        self,
        y,
        use_dropout,
        use_momentum,
        lr=0.01,
        batch_size=64,
        momentum=0.5,
        dropout_prob=0.3,
    ):
        """
        This method initializes the class, it is implemented for you.
        Args:
                y (np.ndarray): labels
                use_dropout (bool): flag to enable dropout
                use_momentum (bool): flag to use momentum
                lr (float): learning rate
                batch_size (int): batch size to use for training
                momentum (float): momentum to use for training
                dropout_prob (float): dropout probability
        """
        self.y = y  # ground truth labels

        # OTHER HYPERPARAMTERS
        self.y_hat = np.zeros((self.y.shape[0], 3))  # estimated labels
        self.dimensions = [8, 15, 7, 3]  # dimensions of different layers
        self.alpha = 0.05

        # DROPOUT
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob

        # PARAMETERS
        self.parameters = {}  # dictionary for different layer variables
        self.cache = (
            {}
        )  # cache for holding variables during forward propagation to use them in back prop
        self.loss = []  # list to store loss values
        self.batch_y = []  # list of y batched numpy arrays

        # TRAINING HYPERPARAMETERS
        self.iteration = 0  # iterator to index into data for making a batch
        self.batch_size = batch_size  # batch size

        # NEURAL NETWORK INFORMATION
        self.learning_rate = lr  # learning rate
        self.sample_count = self.y.shape[0]  # number of training samples we have
        self._estimator_type = "regression"
        self.neural_net_type = "SiLU -> SiLU -> Softmax"

        # MOMENTUM
        self.use_momentum = use_momentum
        self.momentum = momentum  # momentum factor
        self.change = {}  # dictionary for previous changes for momentum

        # TRACKING UPDATES
        self.weight_updates = 0  # Initialize counter for weight updates

    def init_parameters(self, param=None):
        """
        This method initializes the neural network variables, it is already implemented for you.
        Check it and relate to the mathematical description above.
        You are going to use these variables in forward and backward propagation.

        Args:
                param (dict): Optional dictionary of parameters to use instead of initializing.
        """
        if param is None:
            np.random.seed(0)
            self.parameters["theta1"] = np.random.randn(
                self.dimensions[0], self.dimensions[1]  # (8,15)
            ) / np.sqrt(self.dimensions[0])
            self.parameters["b1"] = np.zeros((self.dimensions[1]))  # (15,)
            self.parameters["theta2"] = np.random.randn(
                self.dimensions[1], self.dimensions[2]
            ) / np.sqrt(
                self.dimensions[1]
            )  
            self.parameters["b2"] = np.zeros((self.dimensions[2]))  # (7,)
            self.parameters["theta3"] = np.random.randn(
                self.dimensions[2], self.dimensions[3]
            ) / np.sqrt(
                self.dimensions[2]
            ) 
            self.parameters["b3"] = np.zeros((self.dimensions[3]))  # (3,)

        else:
            self.parameters = param
            self.parameters["theta1"] = self.parameters["theta1"]
            self.parameters["theta2"] = self.parameters["theta2"]
            self.parameters["theta3"] = self.parameters["theta3"]
            self.parameters["b1"] = self.parameters["b1"]
            self.parameters["b2"] = self.parameters["b2"]
            self.parameters["b3"] = self.parameters["b3"]

        for layer in self.parameters:
            self.change[layer] = np.zeros_like(self.parameters[layer])

    def silu(self, u):
        """
        The SiLU (Sigmoid Linear Unit) activation function, also known as the Swish function.
        It is defined as: SiLU(x) = x * sigmoid(x), where sigmoid(x) = 1 / (1 + exp(-x)).

        Args:
            u (np.ndarray): input array with any shape.

        Returns:
           o (np.ndarray): output, same shape as input u.
        """
        return u * 1 / (1 + np.exp(-u))

    def derivative_silu(self, x):
        """
        Derivative of the SiLU (Sigmoid Linear Unit) activation function.

        Args:
                x (np.ndarray): Input array.

        Returns:
                np.ndarray: Derivative of SiLU.
        """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid * (1 + x * (1 - sigmoid))

    def softmax(self, u):
        """
        Performs softmax function function element-wise.
        To prevent overflow, begin by subtracting each row in u by its maximum!
        Input:
                u (np.ndarray: (N, D)): logits
        Output:
                o (np.ndarray: (N, D)): N probability distributions over D classes
        """
        u_shifted = u - np.max(u, axis=1, keepdims=True)
    
        exp_u = np.exp(u_shifted)
        
        sum_exp_u = np.sum(exp_u, axis=1, keepdims=True)
        
        o = exp_u / sum_exp_u
        return o

    def cross_entropy_loss(self, y, y_hat):
        """
        Computes cross entropy loss.
        Refer to the description in the notebook and implement the appropriate mathematical equation.
        To avoid log(0) errors, add a small constant 1e-15 to the input to np.log
        Args:
                y (np.ndarray: (N, D)): one-hot ground truth labels
                y_hat (np.ndarray: (N, D)): predictions
        Returns:
                loss (float): average cross entropy loss
        """
        return -1 * (np.sum((y*np.log(y_hat + 1e-15)))/np.shape(y)[0])

    @staticmethod
    def _dropout(u, prob):
        """
        Implement the dropout layer. Refer to the description for implementation details.
        NOTE: Make sure you only use dropout on the first hidden layer
        Args:
                u (np.ndarray: (N, D)): input to dropout layer
                prob: the probability of dropping an unit
        Returns:
                u_after_dropout (np.ndarray: (N, D)): output of dropout layer
                dropout_mask (np.ndarray: (N, D)): dropout mask indicating which units were dropped

        Hint: scale the units after dropout
                  use np.random.choice to sample from Bernoulli(prob) the inactivated nodes for each iteration
        """
        if prob == 0.0:
                return u
        elif prob == 1.0:
                return np.zeros((len(u), len(u[0])))
        
        dropout_mask = np.random.choice([0,1],np.shape(u),p=[prob, 1-prob])
        u_after_dropout = (u * dropout_mask) / (1 - prob)
        
        return u_after_dropout, dropout_mask

    def forward(self, x, use_dropout):
        """
        Fill in the missing code lines, please refer to the description for more details.
        Check init_parameters method and use variables from there as well as other implemented methods.
        Refer to the description above and implement the appropriate mathematical equations.
        Do not change the lines followed by #keep.

        Args:
                x (np.ndarray: (N, M)): input to neural network
                use_dropout (bool): true if using dropout in forward.
        Returns:
                o3 (np.ndarray: (N, D)): output of neural network

        Note: The shapes of the variables in self.cache should be:
                u1 (np.ndarray: (N, K)): output after first linear layer
                o1 (np.ndarray: (N, K)): output after applying activation and dropout (if specified)
                u2 (np.ndarray: (N, R)): output after second linear layer
                o2 (np.ndarray: (N, R)): output of nueral network
                u3 (np.ndarray: (N, D)): output after third linear layer
                o3 (np.ndarray: (N, D)): output of nueral network

        where,
                N: Number of datapoints
                M: Number of input features
                K: Size of the first hidden layer
                R: Size of the second hidden layer
                D: Number of output features

        HINT 1: Refer to this guide: https://static.us.edusercontent.com/files/gznuqr6aWHD8dPhiusG2TG53 for more detail on the forward pass.
        HINT 2: Here's an outline of the function you can use. Fill in the "..." with the appropriate code:

        Note: Implement dropout only on the first layer!

        self.cache["X"] = x
        u1 = ...
        o1 = ...

        if use_dropout:
                o1 = ...
                dropout_mask = ...
                self.cache["mask"] = dropout_mask

        self.cache["u1"], self.cache["o1"] = u1, o1

        u2 = ...
        o2 = ...
        self.cache["u2"], self.cache["o2"] = u2, o2

        u3 = ...
        o3 = ...
        self.cache["u3"], self.cache["o3"] = u3, o3

        return o3
        """
        self.cache["X"] = x
        u1 = np.dot(x, self.parameters["theta1"]) + self.parameters["b1"]
        o1 = self.silu(u1)

        if use_dropout:
                o1, dropout_mask = self._dropout(o1, self.dropout_prob)
                self.cache["mask"] = dropout_mask

        self.cache["u1"], self.cache["o1"] = u1, o1

        u2 = np.dot(o1, self.parameters["theta2"]) + self.parameters["b2"]
        o2 = self.silu(u2)
        self.cache["u2"], self.cache["o2"] = u2, o2

        u3 = np.dot(o2, self.parameters["theta3"]) + self.parameters["b3"]
        o3 = self.softmax(u3)
        self.cache["u3"], self.cache["o3"] = u3, o3

        return o3

    def compute_gradients(self, y, yh, use_dropout):
        """
        Compute the gradients for each layer given the predicted outputs and ground truths.
        The dropout mask you stored at forward may be helpful.

        Args:
                y (np.ndarray: (N, D)): ground truth values
                yh (np.ndarray: (N, D)): predicted outputs

        Returns:
                gradients (dict): dictionary that maps layer names (strings) to gradients (numpy arrays)

        Note: The shapes of the derivatives in gradients should be as follows:
                dLoss_theta3 (np.ndarray: (R, D)): gradients for theta3
                dLoss_b3 (np.ndarray: (D,)): gradients for b3
                dLoss_theta2 (np.ndarray: (K, R)): gradients for theta2
                dLoss_b2 (np.ndarray: (R,)): gradients for b2
                dLoss_theta1 (np.ndarray: (M, K)): gradients for theta1
                dLoss_b1 (np.ndarray: (K,)): gradients for b1
        where,
                N: Number of datapoints
                M: Number of input features
                K: Size of the first hidden layer
                R: Size of the second hidden layer
                D: Number of output features

        Note: You will have to use the cache (self.cache) to retrieve the values
        from the forward pass!

        HINT 1: Refer to this guide: https://static.us.edusercontent.com/files/gznuqr6aWHD8dPhiusG2TG53 for more detail on computing gradients.

        HINT 2: Division by N only needs to occur ONCE for any derivative that requires a division
        by N. Make sure you avoid cascading divisions by N where you might accidentally divide your
        derivative by N^2 or greater.

        HINT 3: Here's an outline of the function you can use. Fill in the "..." with the appropriate code:

        Note: Implement drop out function only on the first hidden layer!

        dLoss_u3 = yh - y

        dLoss_theta3 = ...
        dLoss_b3 = ...
        dLoss_o2 = ...


        dLoss_u2 = ...

        dLoss_theta2 = ...
        dLoss_b2 = ...
        dLoss_o1 = ...

        if use_dropout:
                dLoss_u1 = ...
        else:
                dLoss_u1 = ...

        dLoss_theta1 = ...
        dLoss_b1 = ...

        """
        N = y.shape[0]
        dLoss_u3 = yh - y
        dLoss_theta3 = np.dot(self.cache["o2"].T, dLoss_u3) / N
        dLoss_b3 = np.sum(dLoss_u3, axis=0) / N
        
        dLoss_o2 = np.dot(dLoss_u3, self.parameters["theta3"].T)
        dLoss_u2 = dLoss_o2 * self.derivative_silu(self.cache["u2"])
        dLoss_theta2 = np.dot(self.cache["o1"].T, dLoss_u2) / N
        dLoss_b2 = np.sum(dLoss_u2, axis=0) / N
        
        dLoss_o1 = np.dot(dLoss_u2, self.parameters["theta2"].T)
        dLoss_u1 = dLoss_o1 * self.derivative_silu(self.cache["u1"])
        
        if use_dropout:
                dLoss_u1 *= self.cache["mask"] / (1 - self.dropout_prob)
        
        dLoss_theta1 = np.dot(self.cache["X"].T, dLoss_u1) / N
        dLoss_b1 = np.sum(dLoss_u1, axis=0) / N
        
        return {
                "theta3": dLoss_theta3,
                "b3": dLoss_b3,
                "theta2": dLoss_theta2,
                "b2": dLoss_b2,
                "theta1": dLoss_theta1,
                "b1": dLoss_b1,
        }
        
    def update_weights(self, dLoss, use_momentum):
        """
        1. Update weights of neural network based on learning rate given gradients for each layer.
        Can also use momentum to smoothen descent.
        2. Increment counter (self.weight_updates) every time weights are updated

        Args:
                dLoss (dict): dictionary that maps layer names (strings) to gradients (numpy arrays)
                use_momentum (bool): flag to use momentum or not

        Return:
                None

        HINT: both self.change and self.parameters need to be updated for use_momentum=True and only self.parameters needs to be updated when use_momentum=False
                  momentum records are kept in self.change
        """
        for param in self.parameters:
                if use_momentum:
                        if param not in self.change:
                                self.change[param] = np.zeros_like(self.parameters[param])
                        self.change[param] = self.momentum * self.change[param] + dLoss[param]
                        self.parameters[param] -= self.learning_rate * self.change[param]
                else:
                        self.parameters[param] -= self.learning_rate * dLoss[param]
        
        self.weight_updates += 1

    def backward(self, y, yh, use_dropout, use_momentum):
        """
        Fill in the missing code lines, please refer to the description for more details.
        You will need to use cache variables, some of the implemented methods, and other variables as well.
        Refer to the description above and implement the appropriate mathematical equations.

        Args:
                y (np.ndarray: (N, D)): ground truth labels
                yh (np.ndarray: (N, D)): neural network predictions
                use_dropout (bool): flag to use dropout
                use_momentum (bool): flag to use momentum

        Return:
                A tuple containing:
                - dLoss_theta3: gradients for theta3
                - dLoss_b3: gradients for b3
                - dLoss_theta2: gradients for theta2
                - dLoss_b2: gradients for b2
                - dLoss_theta1: gradients for theta1
                - dLoss_b1: gradients for b1

        Hint: make calls to compute_gradients and update_weights
        """
        dLoss = self.compute_gradients(y, yh, use_dropout)
        self.update_weights(dLoss, use_momentum)
        return (
            dLoss["theta3"],
            dLoss["b3"],
            dLoss["theta2"],
            dLoss["b2"],
            dLoss["theta1"],
            dLoss["b1"],
        )

    def gradient_descent(self, x, y, iter=60000, use_momentum=False, local_test=False):
        """
        This function is an implementation of the gradient descent algorithm.
        Notes:
        1. GD considers all examples in the dataset in one go and learns a gradient from them.
        2. One iteration here is one round of forward and backward propagation on the complete dataset.
        3. Append loss at multiples of 1000 i.e. at 0th, 1000th, 2000th .... iterations to self.loss
        **For LOCAL TEST append and print out loss at every iteration instead of every 1000th multiple.

        Args:
                x (np.ndarray: N x M): input
                y (np.ndarray: N x D): ground truth labels
                iter (int): number of iterations to train for
                use_momentum (bool): flag to use momentum or not
                local_test (bool): flag to indicate if local test is being run or not

        HINT: Here's an outline of the function you can use.

                self.init_parameters()

                for i in range(iter):
                        # TODO: implement training loop

                        # Print every one iteration for local test, and every 1000th iteration for AG and 1.3
                        print_multiple = 1 if local_test else 1000
                        if i % print_multiple == 0:
                                print("Loss after iteration %i: %f" % (i, loss))
                                self.loss.append(loss)
        """
        
        self.init_parameters()

        for i in range(iter):
                yh = self.forward(x, self.use_dropout)
                
                loss = self.cross_entropy_loss(y, yh)
                
                self.backward(y, yh, use_dropout=self.use_dropout, use_momentum=use_momentum)

                print_multiple = 1 if local_test else 1000
                if i % print_multiple == 0:
                        print("Loss after iteration %i: %f" % (i, loss))
                        self.loss.append(loss)

    # bonus for undergraduate students
    def batch_gradient_descent(self, x, y, use_momentum, iter=60000, local_test=False):
        """
        This function is an implementation of the batch gradient descent algorithm

        Notes:
        1. Batch GD loops over all mini batches in the dataset one by one and learns a gradient
        2. One iteration here is one round of forward and backward propagation on one minibatch.
           You will use self.iteration and self.batch_size to index into x and y to get a batch. This batch will be
           fed into the forward and backward functions.

        3. Append and printout loss at multiples of 1000 iterations i.e. at 0th, 1000th, 2000th .... iterations.
           **For LOCAL TEST append and print out loss at every iteration instead of every 1000th multiple.

        4. Append the y batched numpy array to self.batch_y at every 1000 iterations i.e. at 0th, 1000th,
           2000th .... iterations. We will use this to determine if batching is done correctly.
           **For LOCAL TEST append the y batched array at every iteration instead of every 1000th multiple

        5. We expect a noisy plot since learning on a batch adds variance to the
           gradients learnt
        6. Be sure that your batch size remains constant (see notebook for more detail). Please
           batch your data in a wraparound manner. For example, given a dataset of 9 numbers,
           [1, 2, 3, 4, 5, 6, 7, 8, 9], and a batch size of 6, the first iteration batch will
           be [1, 2, 3, 4, 5, 6], the second iteration batch will be [7, 8, 9, 1, 2, 3],
           the third iteration batch will be [4, 5, 6, 7, 8, 9], etc...

        Args:
                x (np.ndarray: N x M): input data
                y (np.ndarray: N x D): ground truth labels
                use_momentum (bool): flag to use momentum or not
                iter (int): number of BATCHES to iterate through
                local_test (bool): True if calling local test, default False for autograder and Q1.3
                                this variable can be used to switch between autograder and local test requirement for
                                appending/printing out loss and y batch arrays
        HINT:  Here's an outline of the function you can use.

                self.init_parameters()

                for i in range(iter):
                        # TODO: implement training loop

                        # Print every one iteration for local test, and every 1000th iteration for AG and 1.3
                        print_multiple = 1 if local_test else 1000
                        if i % print_multiple == 0:
                                print("Loss after iteration %i: %f" % (i, loss))
                                self.loss.append(loss)
                                self.batch_y.append(y_batch)
        """
        self.init_parameters()

        for i in range(iter):
                start_idx = self.iteration % x.shape[0]
                end_idx = (self.iteration + self.batch_size) % x.shape[0]

                if end_idx > start_idx:
                        x_batch = x[start_idx:end_idx]
                        y_batch = y[start_idx:end_idx]
                else:
                        x_batch = np.concatenate([x[start_idx:], x[:end_idx]])
                        y_batch = np.concatenate([y[start_idx:], y[:end_idx]])

                yh = self.forward(x_batch, self.use_dropout)
                
                loss = self.cross_entropy_loss(y_batch, yh)
                
                self.backward(y_batch, yh, use_dropout=self.use_dropout, use_momentum=use_momentum)

                print_multiple = 1 if local_test else 1000
                
                if i % print_multiple == 0:
                        print("Loss after iteration %i: %f" % (i, loss))
                        self.loss.append(loss)
                        self.batch_y.append(y_batch)

                self.iteration += self.batch_size

    def predict(self, x):
        """
        This function predicts new data points
        It is implemented for you

        Args:
                x (np.ndarray: (N, M)): input data
        Returns:
                y (np.ndarray: (N,)): predictions
        """
        yh = self.forward(x, False)  # (N, D) = forward((N, M))
        pred = np.argmax(yh, axis=1)  # (N,) = argmax((N, D), axis=1)
        return pred
