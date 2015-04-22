import random
import numpy as np
import pdb
from scipy.special import expit

def softmax(x):
    """ Softmax function """
    ###################################################################
    # Compute the softmax function for the input here.                #
    # It is crucial that this function is optimized for speed because #
    # it will be used frequently in later code.                       #
    # You might find numpy functions np.exp, np.sum, np.reshape,      #
    # np.max, and numpy broadcasting useful for this task. (numpy     #
    # broadcasting documentation:                                     #
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)  #
    # You should also make sure that your code works for one          #
    # dimensional inputs (treat the vector as a row), you might find  #
    # it helpful for your later problems.                             #
    ###################################################################
    
    ### YOUR CODE HERE
    
    ### END YOUR CODE
    if len(np.shape(x)) <= 1:
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / np.sum(ex)
    x = x - np.max(x, -1)[:, np.newaxis]
    ex = np.exp(x)
    
    return ex / np.sum(ex, -1)[:, np.newaxis]


def sigmoid(x):
    """ Sigmoid function """
    ###################################################################
    # Compute the sigmoid function for the input here.                #
    ###################################################################
    
    ### YOUR CODE HERE

    ### END YOUR CODE
    MIN = 10e-7
    s = expit(x)
    s[s < MIN] = MIN
    s[s > 1 - MIN] = 1 - MIN
    
    return s

def sigmoid_grad(f):
    """ Sigmoid gradient function """
    ###################################################################
    # Compute the gradient for the sigmoid function here. Note that   #
    # for this implementation, the input f should be the sigmoid      #
    # function value of your original input x.                        #
    ###################################################################
    
    ### YOUR CODE HERE

    ### END YOUR CODE
    
    return sigmoid(x)*(1-sigmoid(x))

def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
    
        ### YOUR CODE HERE: try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it 
        ### possible to test cost functions with built in randomness later
    
        x2 = np.array(x)
        x2[ix] += h
        random.setstate(rndstate)
        fx2, _ = f(x2)

        x1 = np.array(x)
        x1[ix] -= h
        random.setstate(rndstate)
        fx1, _ = f(x1)

        numgrad = (fx2-fx1)/(2*h)
        ### END YOUR CODE
        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            #return
    
        it.iternext() # Step to next dimension

    print "Gradient check passed!"

# Implement a function that normalizes each row of a matrix to have unit length
def normalizeRows(x):
    """ Row normalization function """
    
    ### YOUR CODE HERE
    x2 = x*x
    ### END YOUR CODE
    if len(np.shape(x)) <= 1:
        return np.sqrt(x2 / np.sum(x2))
    
    return np.sqrt(x2/(np.sum(x2, -1)[:, np.newaxis]))

if __name__ == "__main__":
    print softmax(np.array([[1001,1002],[3,4]]))
    print softmax(np.array([[-1001,-1002]]))
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print "=== For autograder ==="
    print f
    print g
    # First implement a gradient checker by filling in the following functions

    # Sanity check for the gradient checker
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "=== For autograder ==="
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test

    # Test this function
    print "=== For autograder ==="
    print normalizeRows(np.array([[3.0,4.0],[1, 2]]))  # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
