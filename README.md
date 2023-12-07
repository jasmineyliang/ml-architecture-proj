# ml-architecture-proj
neural networks

1. Given the convolutional neural network block as below
Given the input feature maps $\boldsymbol X \in
    \mathbb{R}^{64\times 64 \times 128}$, all convolutional
    layers perform zero-padding of $1$ on each side of $H$ and
    $W$ dimensions.

(a) What is the total number of parameters in the block (you can skip bias terms)?

(b) What is the total number of multi-add operations in the block?

(c) What is memory requirement change to store the input and output features of this block (Use percentage)?

2. Using batch normalization in neural networks requires computing the mean and variance of a tensor. Suppose a batch normalization layer takes vectors $z_1,z_2,\cdots,z_m$ as input, where $m$ is the mini-batch size. It computes $\hat z_1,\hat z_2,\cdots,\hat z_m$ according to $$\hat z_i=\frac{z_i-\mu}{\sqrt{\sigma^2+\epsilon}}$$
    where $$\mu=\frac{1}{m}\sum_{i=1}^m z_i,\,\,\,\sigma^2=\frac{1}{m}\sum_{i=1}^m(z_i-\mu)^2.$$ It then applies a second transformation to obtain $\tilde z_1,\tilde
    z_2,\cdots,\tilde z_m$ using learned parameters $\gamma$ and $\beta$ as $$\tilde z_i=\gamma \hat z_i+\beta.$$ In this question, you can assume that $\epsilon=0$.
   
(a) You forward-propagate a mini-batch of $m=4$ examples in your
    network. Suppose you are at a batch normalization layer, where the
    immediately previous layer is a fully connected layer with $3$
    units. Therefore, the input to this batch normalization layer can be
    represented as the below matrix:
    
![image](https://github.com/jasmineyliang/ml-architecture-proj/assets/150869870/01c8f088-24c4-4fc9-9dfe-f22176733acf)

    
What are $\hat z_i$? Please express your answer in a $3\times 4$ matrix.

(b) Continue with the above setting. Suppose
    $\gamma=(1,1,1)$, and $\beta=(0,-10,10)$. What are $\tilde
    z_i$? Please express your answer in a $3\times 4$ matrix.

(c) Describe the differences of computations required for batch normalization during training and testing.

(d) Describe how the batch size during testing affect testing results.

3. We investigate the back-propagation of the convolution using a simple example. In this problem, we focus on the
    convolution operation without any normalization and
    activation function. For simplicity, we consider the
    convolution in 1D cases. Given 1D inputs with a spatial size
    of $4$ and $2$ channels
   
![image](https://github.com/jasmineyliang/ml-architecture-proj/assets/150869870/f8c540c9-d730-4b6f-8103-15ba86188c2e)
we perform a 1D convolution with a kernel size of $3$ to produce output $Y$ with $2$ channels. No padding is involved.
    It is easy to see
    
![image](https://github.com/jasmineyliang/ml-architecture-proj/assets/150869870/cab5920f-3129-4376-ad52-e2f2d1b3ca36)

where each row corresponds to a channel. There are 12
    training parameters involved in this convolution, forming 4
    different kernels of size $3$:

where $W^{ij}$ scans the $i$-th channel of inputs and contributes to the $j$-th channel of outputs.

(a) Now we flatten $X$ and $Y$ to vectors as
![image](https://github.com/jasmineyliang/ml-architecture-proj/assets/150869870/ccdd6a05-a351-43be-ad47-845aead5c88b)

Please write the convolution in the form of fully connected layer as $\tilde Y=A\tilde X$ using the notations above. You can assume there is no bias term.

(b) Next, for the back-propagation, assume we've already computed the gradients of loss $L$ with respect to $\tilde Y$:

![image](https://github.com/jasmineyliang/ml-architecture-proj/assets/150869870/8beaf200-b89d-4ffa-82f8-d7668e239853)

Please write the back-propagation step of the convolution in
    the form of $\frac{\partial L}{\partial \tilde
    X}=B\frac{\partial L}{\partial \tilde Y}.$ Explain the
    relationship between $A$ and $B$.
  
(c) While the forward propagation of the
    convolution on $X$ to $Y$ could be written into $\tilde
    Y=A\tilde X$, could you figure out whether $\frac{\partial
    L}{\partial \tilde X}=B\frac{\partial L}{\partial \tilde Y}$
    also corresponds to a convolution on $\frac{\partial
    L}{\partial Y}$ to $\frac{\partial L}{\partial X}$?

4. LeNet for Image Recognition:In this coding assignment, you will need to complete the implementation of LeNet (LeCun Network) using PyTorch and
    apply the LeNet to the image recognition task on Cifar-10 (10-classes classification). You will need to install the python packages ``tqdm'' and ``pytorch''. Please read the installation guides of PyTorch here (https://pytorch.org/get-started/locally/).

(a) Complete the class LeNet().
    In particular, define operations in function
    __init\_\_() and use them in function
    forward(). The input of forward() is an image. The paper for LeNet
    can be found here
    (http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

The network architecture is shown in the figure below.

![image](https://github.com/jasmineyliang/ml-architecture-proj/assets/150869870/565c5ea8-b2c9-4372-84f2-7f66f5245bd0)

The sub-sampling is implemented by using the max pooling. And the kernel size for all the convolutional layers are $5\times5$. Please use \emph{\textbf{ReLU}} function to activate the outputs of convolutional layers and the first two fully-connected layers. The sequential layers are:

![image](https://github.com/jasmineyliang/ml-architecture-proj/assets/150869870/ec7bd540-bc5c-4168-aaa7-ef29b09be1ab)

(b) Add batch normalization operations after
    each max pooling layer. Run the model by
    python main.py and report the testing
    performance as well as a short analysis of the results.

(c) Based on (b), add dropout operations with
    drop rate of 0.3 after the first two fully-connected layers.
    Run the model by "python main.py" and
    report the testing performance as well as a short analysis of
    the results.
