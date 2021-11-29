# Fruits-Classifier
In this project, a feed forward fully connected nueral network is implemented.
Forward and backward propagation will work considering size of the hidden layers, which can be set manually and without limit.
The algorithm used for learning the network is **Stochastic Gradient Descent**, which is implemented in tow forms:

- ***Iterative SGD:***
<br /> In the iterative approach, the back-propagation algorithm and the derivative of network parameters is computed using 
loops and iterating ervery each element of gradient matrix of weights and biases in each layer. Abviously, this may take 
time and the learning may be slow.

- ***Vectorized SGD:***
<br /> In the vectorized form, gradient matrixes in back-propagation algorithm can be computed directly and without using 
loops. Actually, instead of using loops, we use matrix multiplications which can be computed much faster using matrix libraries 
such as numpy, due to parallelism of computations.

*The network is used to classify 4 or 6 types of fruits: Apple, Banana, Kiwi, Lemon, Mango, and Raspberry.*
<br /> *The dataset used for training the network is Kaggle Fruits-360 Dataset.*
