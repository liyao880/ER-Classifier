# Embedding Regularized Classifier
Recent studies have demonstrated the vulnerability of deep convolutional neural networks against adversarial examples. Inspired by the observation that the intrinsic dimension of image data is much smaller than its pixel space dimension and the vulnerability of neural networks grows with the input dimension, we propose to embed high-dimensional input images into a low-dimensional space to perform classification. However, arbitrarily projecting the input images to a low-dimensional space without regularization will not improve the robustness of deep neural networks. Leveraging optimal transport theory, we propose a new framework, Optimal Transport Classifier (OT-Classifier), and derive an objective that minimizes the discrepancy between the distribution of the true label and the distribution of the OT-Classifier output. Experimental results on several benchmark datasets show that, our proposed framework achieves state-of-the-art performance against strong adversarial attack methods.

# Required Packages:  
- numpy  
- torch  
- torchvision
- tqdm

# Implementation Details:

CIFAR10:

- Train ER-Classifier on CIFAR10:  

``` Python3 train_er.py -dataset "cifar10" -file_name "/er_" -epochs 30 -root "/home/er"```

Note that the root corresponds to the directory where you save data folder

- Train ER-CLA+Adv on CIFAR10:  

``` Python3 train_er.py -dataset "cifar10" -delay 0 -file_name "/er_adv_" -epochs 20 -root '/home/er'```

By setting delay<epochs, the program will do adversarial training for |epochs-delay| number of epochs.

STL10:

- Train ER-CLA on STL10:  

``` Python3 train_er.py -dataset "stl10" -file_name "/er_" -epochs 30 -root "/home/er"```

Note that the root corresponds to the directory where you save data folder

- Train ER-CLA+Adv on STL10:  

``` Python3 train_er.py -dataset "stl10" -delay 0 -file_name "/er_adv_" -epochs 20 -root '/home/er'```

By setting delay<epochs, the program will do adversarial training for |epochs-delay| number of epochs.


