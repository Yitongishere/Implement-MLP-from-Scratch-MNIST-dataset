import numpy as np
import struct
import os
import matplotlib.pyplot as plt
import random


class MLP:
    
    def __init__(self, num_inputs=3, num_hidden=[3,5], num_outputs=2, init='rand'):
        
        # intitialize the number of layers
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        layers = [num_inputs] + num_hidden + [num_outputs]
        
        # initialize the weights, with different methods
        self.weights = []
        for i in range(0, len(layers)-1):
            if init == 'rand':
                w = np.random.rand(layers[i], layers[i+1])
            elif init == 'xavier':
                w = self._xavier_uniform(layers[i], layers[i+1])
            elif init == 'gauss':
                w = np.zeros((layers[i], layers[i+1]))
                for p in range(layers[i]):
                    for q in range(layers[i+1]):
                        w[p][q] = random.gauss(0,1)
            self.weights.append(w)

        # initialize the bias, all zero
        self.bias = []
        for i in range(0, len(layers)-1):
            b = np.zeros(layers[i+1])
            self.bias.append(b)

        # initialize the activations
        self.activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            self.activations.append(a)

        # initialize the derivatives
        self.derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            self.derivatives.append(d)

        # initialize the derivatives of bias
        self.derivatives_b = []
        for i in range(0, len(layers)-1):
            d_b = np.zeros(layers[i+1])
            self.derivatives_b.append(d_b)


            
    def forward_propagation(self, inputs):

        activation = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
        # calculate the net inputs
            net_inputs = np.dot(activation, w) + + self.bias[i]
        # calculate the activations
            if i == (len(self.weights) - 1):          # softmax for the last activation
                activation = self._softmax(net_inputs)
            else:
                activation = self._relu(net_inputs)
            self.activations[i+1] = activation
        
        return activation


    def back_propagation(self, target, reg_method, lambd, verbose=False):
        # loop the weights from back to front
        for i in reversed(range(len(self.derivatives))):
            current_activation = self.activations[i]
            next_activation = self.activations[i+1]
            # last layer with softmax as activation function
            if i == len(self.derivatives) - 1:          # last layer
                error = next_activation - target        # (d_E / d_predictin[L]) * (d_predictin[L] / softmax'(z[L]))
                delta = error
                delta_reshaped = delta.reshape(-1, delta.shape[0])
                current_activation_reshaped = current_activation.reshape(current_activation.shape[0], -1)
                if reg_method=='L2':
                    self.derivatives[i] = np.dot(current_activation_reshaped, delta_reshaped) + lambd * self.weights[i]
                elif reg_method=='L1':
                    self.derivatives[i] = np.dot(current_activation_reshaped, delta_reshaped) + lambd * np.sign(self.weights[i])
                else:
                    self.derivatives[i] = np.dot(current_activation_reshaped, delta_reshaped)
                self.derivatives_b[i] = delta
                error = np.dot(delta, self.weights[i].T)
            else:
                delta = error * self._relu_derivatives(next_activation)
                delta_reshaped = delta.reshape(-1, delta.shape[0])
                current_activation_reshaped = current_activation.reshape(current_activation.shape[0], -1)
                if reg_method=='L2':
                    self.derivatives[i] = np.dot(current_activation_reshaped, delta_reshaped) + lambd * self.weights[i]
                elif reg_method=='L1':
                    self.derivatives[i] = np.dot(current_activation_reshaped, delta_reshaped) + lambd * np.sign(self.weights[i])
                else:
                    self.derivatives[i] = np.dot(current_activation_reshaped, delta_reshaped)
                self.derivatives_b[i] = delta
                error = np.dot(delta, self.weights[i].T)
                if verbose == True:
                    print("The derivatives for W{} is: {}".format(i, self.derivatives[i]))


    def gradient_descent(self, learning_rate, verbose=False):
        # update the all weights
        for i in range(len(self.weights)):
            if verbose:
                print("Original W{} is: {}".format(i, self.weights[i]))
                print("Original b{} is: {}".format(i, self.bias[i]))
            self.weights[i] -= learning_rate * self.derivatives[i]
            self.bias[i] -= learning_rate * self.derivatives_b[i]  
            if verbose:
                print("Updated W{} is: {}".format(i, self.weights[i]))
                print("Updated b{} is: {}".format(i, self.bias[i]))


    def report_loss(self, loss_history, loss_history_val, train_accuracy_history, val_accuracy_history, epochs, image_save_path="./learning_curve.png"):
       # get the plot of learning curve and save
        x = np.linspace(1, epochs, epochs)

        plt.figure(figsize=(15,6))
        plt.subplot(121)
        plt.plot(x, loss_history, linewidth=1, c='blue', label='training loss')
        plt.plot(x, loss_history_val, linewidth=1, c='red', label='validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.xticks(np.arange(0,epochs,5))
        plt.yticks(np.arange(0, 2, 0.2))
        plt.legend(["Training Loss", "Validation Loss"])

        plt.subplot(122)
        plt.plot(x, train_accuracy_history, linewidth=1, c='blue', label='training accuracy')
        plt.plot(x, val_accuracy_history, linewidth=1, c='red', label='training accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(np.arange(0,epochs,5))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.legend(["Training Accuracy", "Validation Accuracy"])

        plt.savefig(image_save_path)
        # plt.show()



    def train(self, inputs, targets, val_images, val_labels, epochs, reg_method, lambd, 
                learning_rate, lr_decay=[False, 'linear', 10, 0.8],
                batch_size=1, image_save_path="./learning_curve.png"):
        """
            Implement the train function. 


            Arguments:

            inputs      --      training images.
            targets     --      training labels.
            val_images  --      validation images.
            val_labels  --      validation label.
            epochs      --      number of epoch we train
            learning rate --    step size of gradient descent
            lr_decay    --      A list with:
                                lr_decay[0]: apply learning rate decay or not
                                lr_decay[1]: which method for linear decay, 'linear' or 'non-linear'
                                lr_decay[2]: apply learning_rate decay every lr_decay_para[0] epoch
                                lr_decay[3]: learning_rate -= original_learning_rate * lr_decay_para[1],  for  'linear' method
                                                learning_rate = learning_rate * lr_decay_para[1],  for  'non-linear' method

        """

        print("Train starts: ......")
        # for recording Loss and accuracy
        loss_history = []
        loss_history_val = []
        train_accuracy_history = []
        val_accuracy_history = []
        original_lr = learning_rate

        # feed data from here!
        for i in range(epochs):

            # apply learning rate decay
            if lr_decay[0] and (lr_decay[1] == 'linear') and (i % lr_decay[2] == lr_decay[2] - 1):
                learning_rate -= original_lr * lr_decay[3]
                if learning_rate < 0:
                    learning_rate = original_lr * 0.0001
            if lr_decay[0] and (lr_decay[1] == 'non-linear') and (i % lr_decay[2] == lr_decay[2] - 1):
                learning_rate = learning_rate * lr_decay[3]

            # shuffle the dataset before every epoch training to ensure (Mini-Batch) stochastic gradient descent, SGD
            state = np.random.get_state()       # ensure the images and the labels are shuffled in the same sequence
            np.random.shuffle(inputs)
            np.random.set_state(state)
            np.random.shuffle(targets)

            outputs = []
            derivatives = []
            derivatives_b = []
            epoch_sum_error = 0
            epoch_sum_error_val = 0
            num_true_output = 0
            for j, (input, target) in enumerate(zip(inputs, targets)):
                # forward propagation
                output = self.forward_propagation(input)
                outputs.append(output)
                # back propagation
                self.back_propagation(target, reg_method, lambd)

                # calculating the average derivative of the batch
                derivatives.append(self.derivatives)
                derivatives_b.append(self.derivatives_b)
                if j % batch_size == batch_size - 1:
                    self.derivatives = self._tool(derivatives, batch_size)
                    self.derivatives_b = self._tool(derivatives_b, batch_size)
                    # gradient descent
                    self.gradient_descent(learning_rate)
                    derivatives = []
                    derivatives_b = []

                # sum error of each piece of data
                error_for_each = self._cross_entropy_Regularization(target, output, self.weights, reg_method, lambd)
                epoch_sum_error += error_for_each

            # report the loss and train accuracy of the epoch
            epoch_mean_error = epoch_sum_error / len(inputs)
            loss_history.append(epoch_mean_error)

            # calculating the accuracy on training set
            for k in range(len(inputs)):
                output_k = outputs[k].tolist()
                target_k = targets[k].tolist()
                if output_k.index(max(output_k)) == target_k.index(max(target_k)):
                    num_true_output += 1
            train_accuracy = float(num_true_output) / len(inputs)
            train_accuracy_history.append(train_accuracy)

            # evaluating the outcome on validation set
            preds = []
            num_true_pred = 0
            for image, label in zip(val_images, val_labels):
                pred = self.forward_propagation(image)
                preds.append(pred)
                error_for_each_val = self._cross_entropy_Regularization(label, pred, self.weights, reg_method, lambd)
                epoch_sum_error_val += error_for_each_val
            mean_error_val = epoch_sum_error_val / len(val_images)
            loss_history_val.append(mean_error_val)

            # calculating the accuracy on validation set
            for l in range(len(val_images)):
                pred_i = preds[l].tolist()
                label_i = val_labels[l].tolist()
                if pred_i.index(max(pred_i)) == label_i.index(max(label_i)):
                    num_true_pred += 1
            val_accuracy = float(num_true_pred) / len(val_images)
            val_accuracy_history.append(val_accuracy)

            print(" Epoch {}:  with learning rate of {}\n     Train loss: {}, Val Loss: {},\n     Train accuracy: {}, Val accuracy: {}"
            .format(i, learning_rate, round(epoch_mean_error,6), round(mean_error_val,6), round(train_accuracy,4), round(val_accuracy,4))) 
            print("-------------------------------")
            
        # plot the epoch-loss figure
        self.report_loss(loss_history, loss_history_val, train_accuracy_history, val_accuracy_history, epochs, image_save_path)
        print("------------Train finished!------------")

    
    def evaluate_model(self, images, labels, reg_method, lambd):
        # calculate loss
        sum_error = 0
        num_true_pred = 0
        preds = []
        for image, label in zip(images, labels):
            pred = self.forward_propagation(image)
            preds.append(pred)
            error_for_each = self._cross_entropy_Regularization(label, pred, self.weights, reg_method, lambd)
            sum_error += error_for_each
        mean_error = sum_error / len(images)
        print("Loss on Validation set is: {}".format(mean_error))

        # calculate accuracy
        for i in range(len(images)):
            pred_i = preds[i].tolist()
            label_i = labels[i].tolist()
            if pred_i.index(max(pred_i)) == label_i.index(max(label_i)):
                num_true_pred += 1
        accuracy = float(num_true_pred) / len(images)
        print("Accuracy on Validation set is: {}".format(accuracy))


    def _xavier_uniform(self, row, col):
        a = np.sqrt(6. / (row + col))
        return np.random.uniform(low=-a, high=a, size=[row,col]).astype(np.float32)

    def _relu(self, x):
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = 0
        return x 

    def _relu_derivatives(self, x):
        return np.where(x > 0, 1, 0)

    def _softmax(self, x):
        max = np.max(x)
        return np.exp(x-max) / np.sum(np.exp(x-max))

    # def _cross_entropy(self, target, output):
    #     Loss = 0
    #     for i in range(target.shape[-1]):
    #         Loss += target[i] * np.log(output[i] + 1e-5)
    #     return -Loss

    def _cross_entropy_Regularization(self, target, output, weights, reg_method, lambd):
        Loss = 0
        Reg_cost = 0
        for i in range(target.shape[-1]):
            Loss += target[i] * np.log(output[i] + 1e-5)
        for item in weights:
            if reg_method == 'L2':
                Reg_cost += np.sum(np.square(item))
            if reg_method == 'L1':
                Reg_cost += np.sum(np.abs(item))
        Reg_cost = (lambd/2) * Reg_cost
        return -Loss + Reg_cost

    def _tool(self, d, batch_size):
        """
        the function for data type converting
        """
        d_len = len(d[0])
        d_new = []
        for i in range(d_len):
            d_new.append(np.zeros_like(d[0][i]))
        for i in range(len(d_new)):
            for item in d:
                d_new[i] += item[i]
            d_new[i] = d_new[i] / batch_size
        return d_new


    

# Functions for Loading MNIST DATASET
def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def normalize_data(train_images):
    train_images = train_images[:][:] / 255.0
    mean = np.sum(train_images) / (len(train_images) * 28 * 28)
    std = np.sqrt(np.sum((train_images - mean)**2) / (len(train_images) * 28 * 28))
    train_images = (train_images - mean) / std
    return train_images


def sep_validation(train_images, train_labels, val_ratio=0.1):
    """
    seperate the validation set from the whole training set
    """
    val_images = train_images[int(np.shape(train_images)[0] * (1-val_ratio)) :][:]
    val_labels = train_labels[int(np.shape(train_labels)[0] * (1-val_ratio)) :]
    train_images = train_images[0 : int(np.shape(train_images)[0] * (1-val_ratio))][:]
    train_labels = train_labels[0 : int(np.shape(train_labels)[0] * (1-val_ratio))]
    return train_images, train_labels, val_images, val_labels


def trans_labels(original_labels):
    """
    trans original labels to one-hot vector labels
    """
    new_labels = np.zeros((np.shape(original_labels)[0] ,10))
    for i in range(np.shape(original_labels)[0]):
        new_labels[i][original_labels[i]] = 1
    return new_labels




if __name__ == "__main__":
    
    # unpack and load the dataset
    path='../section1/dataset'
    train_images,train_labels=load_mnist_train(path)
    test_images,test_labels=load_mnist_test(path)

    # shuffle the dataset
    state = np.random.get_state()       # ensure the images and the labels are shuffled in the same sequence
    np.random.shuffle(train_images)
    np.random.set_state(state)
    np.random.shuffle(train_labels)

    # normalize data
    train_images = normalize_data(train_images)

    # seperate the validation set from training set
    train_images, train_labels, val_images, val_labels = sep_validation(train_images, train_labels, val_ratio=0.1)

    # convert labels to 2-d array for dimension consistence
    train_labels = train_labels.reshape(np.shape(train_labels)[0], -1)
    val_labels = val_labels.reshape(np.shape(val_labels)[0], -1)
    test_labels = test_labels.reshape(np.shape(test_labels)[0], -1)

    # transfer the label to one-hot vector
    train_labels = trans_labels(train_labels)
    val_labels = trans_labels(val_labels)
    test_labels = trans_labels(test_labels)

    # construct the MLP
    mlp_BGD = MLP(784, [32, 16], 10,  init='rand')
    # mlp_SGD = MLP(784, [32, 16], 10,  init='rand')

    # Regularization method and lambd for L1 and L2 Regularization
    reg_method = 'no'
    lambd = 0.02

    # train the MLP
    mlp_BGD.train(train_images, train_labels, val_images, val_labels, epochs=100, 
                        reg_method=reg_method, lambd=lambd,
                        learning_rate=0.0003, lr_decay=[False, 'non-linear', 5, 0.95],
                        batch_size=10, image_save_path="./BGD_learning_curve.png")
    # mlp_SGD.train(train_images, train_labels, val_images, val_labels, epochs=100, 
    #                     reg_method=reg_method, lambd=lambd,
    #                     learning_rate=0.0003, lr_decay=[False, 'non-linear', 5, 0.95],
    #                     batch_size=1, image_save_path="./SGD_learning_curve.png")

    # predictions on validation set
    mlp_BGD.evaluate_model(val_images, val_labels, reg_method, lambd)
    # mlp_SGD.evaluate_model(val_images, val_labels, reg_method, lambd)


    """
    
    To config the MPL:
        "mlp = MLP(784, [32, 16], 10, init='rand')"
            config the number of layers and number of neurons in each layer.
            init:    choose different ways for initializing the weights, with available options as:
                        'rand' for randomly initializing between [0,1)
                        'xavier' for initializing with Xavier method
                        'Gauss' for Gauss distribution with 0 as mean, and 1 as std
    
    To train the MLP, set the following parameters for different configs:
        "mlp.train()"
            train_images:   the training images
            train_labels:   the training labels
            val_images:     the validation images
            val_labels:     the validation labels
            epochs:         the number of epoch to train
            reg_method:     regularization method to apply, with available options as:
                                'L1' for L1 regularization
                                'L2' for L2 regularization
                                ''   for no regularization applied
            lambd:          the lambd for L1 and L2 regularization
            learning_rate:  learning rate as you know
            lr_decay:       a list for changing learning rate in training process, with elements as:
                                lr_decay[0]: apply learning rate decay or not
                                lr_decay[1]: which method for linear decay, 'linear' or 'non-linear'
                                lr_decay[2]: apply learning_rate decay every lr_decay_para[0] epoch
                                lr_decay[3]: learning_rate -= original_learning_rate * lr_decay_para[1],  for  'linear' method
                                            learning_rate = learning_rate * lr_decay_para[1],  for  'non-linear' method
            batch_size:     batch size as you know :)
            image_save_path:    the path for saving the learning curve
            
    
    """
    
