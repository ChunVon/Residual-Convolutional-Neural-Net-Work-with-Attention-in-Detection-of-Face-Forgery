# Residual Convolutional Neural Net Work with Attention in Detection of Face Forgery

## Introduction

This project complete a task in detecting deepfake photos, using a residual CNN with attention,  inspired by the paper:

[FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)

[Deep Residual Learning for Image Recognition](https://ieeexplore.ieee.org/document/7780459/figures)

[CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521v2)

The idea in this project is that we can divide the deepfake photos into two groups, which are *True* and *Fake*. Then, we can use the network in classification to complete this task .The code is implemented in pytorch, with an accuracy of 82% in the dataset *deepfake_in_the_wild*. 

## Dataset

The dataset is [deepfake-in-the-wild](https://github.com/deepfakeinthewild/deepfake-in-the-wild). I use nearly 200,000 fake and true photos each. The photos in the dataset is 224 by 224 pixels RGB pictures. Some samples are as follows:

![image](https://github.com/ChunVon/Residual-Convolutional-Neural-Net-Work-with-Attention-in-Detection-of-Face-Forgery/tree/main/images/2.png)

![image](https://github.com/ChunVon/Residual-Convolutional-Neural-Net-Work-with-Attention-in-Detection-of-Face-Forgery/tree/main/images/5.png)

## Structure

The basic block is as follow:

![image](https://github.com/ChunVon/Residual-Convolutional-Neural-Net-Work-with-Attention-in-Detection-of-Face-Forgery/tree/main/images/Basic_Block.png)

The whole network is concatenation of four basic blocks like this. The CBAM part is similar to the module mentioned in the paper. 

![image](https://github.com/ChunVon/Residual-Convolutional-Neural-Net-Work-with-Attention-in-Detection-of-Face-Forgery/tree/main/images/CBAM.png)

And the convolutional layers are as followed:

![image](https://github.com/ChunVon/Residual-Convolutional-Neural-Net-Work-with-Attention-in-Detection-of-Face-Forgery/tree/main/images/Convolution.png)

At the end of the basic block the two results from CBAM and convolution are added bitwise. Then, it serves as the input of the next basic block after a max pooling layer with kernel 2 by 2. When it goes forward to the end of the eight basic blocks, there is full connect layer with one hidden layer and two output layers. More details are available in the code(including parameters of every layer). 

## Results

The loss in training is as follow:

![image](https://github.com/ChunVon/Residual-Convolutional-Neural-Net-Work-with-Attention-in-Detection-of-Face-Forgery/tree/main/images/loss.png)

I use *Cross-Entropy* loss function. Obviously the model are of convergence when the epco reaches 100. And the results are in the following table. I also conduct two other methods. 

| Method                | Accuracy |      |
| --------------------- | :------: | ---- |
| Cozzolino et al.      |  85.45   |      |
| Rahmouni et al.       |  84.27   |      |
| ResNet with Attention |  81.11   |      |

I report precision results for deepfakes. The last one was the method of this project. And the other two methods are from 

[Recasting Residual-based Local Descriptors as Convolutional Neural Networks: an Application to Image Forgery Detection](https://www.semanticscholar.org/paper/Recasting-Residual-based-Local-Descriptors-as-an-to-Cozzolino-Poggi/8b443b98099f4d713dcdc6cd706a7010b457a586) and [Distinguishing computer graphics from natural images using convolution neural networks](https://ieeexplore.ieee.org/document/8267647). 

Also, I conduct ablation experiment to test the contribution of the attention part. The loss in training are as follows:

![image](https://github.com/ChunVon/Residual-Convolutional-Neural-Net-Work-with-Attention-in-Detection-of-Face-Forgery/tree/main/images/loss_without_attention.png)

Evidently the speed of convergence is much lower than the counterpart with attention module. 
