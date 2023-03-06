# Details
**Datasets:**

- Size: 250x250 images
- Preprocessing: 224x224 random crop for training and a 224x224 center crop for evaluation. Flip and rotations are already included in the datasets.
- Classes: 2
- Criterion: Binary Cross Entropy

**Backbone:**

ResNet family trained on ImageNet.

**Strategy:**

- Transfer Learning
- Weight Initialization: Pretrained weights provided by CNN layers
- Modification: Fully Connected Classifier and the ResNet version (18, 34, 50, 101, 152)

## ResNet 18 and classifiers
    self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    self.resnet.fc = nn.Sequential()
### Simple perceptron layer

    self.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=1, bias=True),
        nn.Sigmoid()
    )

| Optimizer | Learning rate | Momentum | Epochs | Batch size | Test loss | Test acc |
| --- | --- | --- | --- | --- | --- | --- |
| Adam | 1e-6 | (0.9, 0.999) | 100 | 32 | 0.534 | 77.74% |
| Adam | 5e-6 | (0.9, 0.999) | 100 | 32 | 0.458 | 80.73% |
| Adam | 1e-5 | (0.9, 0.999) | 100 | 32 | 0.459 | 78.55% |
| Adam | 5e-5 | (0.9, 0.999) | 100 | 32 | 0.440 | 82.02% |
| Adam | 1e-4 | (0.9, 0.999) | 100 | 32 | / | / |
| SGD | 1e-6 | 0.9 | 100 | 32 | 0.595 | 71.64% |
| SGD | 5e-6 | 0.9 | 100 | 32 | 0.434 | 82.71% |
| SGD | 1e-5 | 0.9 | 100 | 32 | 0.425 | 81.03% |
|**SGD**|**5e-5**|**0.9**|**100**|**32**|**0.395**|**83.23%**|
| SGD | 1e-4 | 0.9 | 100 | 32 | / | / |

### Two FC layers
Two fully connected layers with an intermediate non linear activation.

#### ReLU

    self.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=256, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=1, bias=True),
        nn.Sigmoid()
    )

| Optimizer | Learning rate | Momentum | Epochs | Batch size | Test loss | Test acc |
| --- | --- | --- | --- | --- | --- | --- |
| SGD | 1e-6 | 0.9 | 100 | 32 | 0.631 | 79.68% |
| SGD | 5e-6 | 0.9 | 100 | 32 | 0.463 | 80.47% |
| SGD | 1e-5 | 0.9 | 100 | 32 | 0.479 | 80.46% |
|**SGD**|**5e-5**|**0.9**|**100**|**32**|**0.431**|**80.54%**|
| SGD | 1e-4 | 0.9 | 100 | 32 | / | / |

#### Dropout and LeakyReLU
Dropout is a regularization technique that randomly sets a fraction of the inputs to a layer to zero during training, to prevent overfitting.

LeakyReLU prevents the "dying ReLU" problem.

    self.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=256, bias=True),
        nn.LeakyReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=256, out_features=1, bias=True),
        nn.Sigmoid()
    )

| Optimizer | Learning rate | Momentum | Epochs | Batch size | Test loss | Test acc |
| --- | --- | --- | --- | --- | --- | --- |
| SGD | 1e-6 | 0.9 | 100 | 32 | Local | Local |
| SGD | 5e-6 | 0.9 | 100 | 32 | 0.490 | 81.80% |
|**SGD**| **1e-5** |**0.9**|**100**|**32** |**0.415**|**85.19%**|
| SGD | 5e-5 | 0.9 | 100 | 32 | / | / |
| SGD | 1e-4 | 0.9 | 100 | 32 | / | / |


### Conclusion about ResNet 18 tests
*It seems unnecessary to make the Fully Connected classifier more complex as the loss and accuracy measured on the test dataset are equivalent.* --> not sure with the brand new results.

The next step is to use ResNet with more layers with a minimalist training classifier. The objective is to determine if the addition of layers on the feature extraction part gives richer information to the classifier.



## ResNet versions with a minimalist classifier
    self.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=1, bias=True),
        nn.Sigmoid()
    )
### Performances on ImageNet
Source: https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights

| ResNet | v. weights | acc @1 | acc @5 | #Params | Size |
| --- | --- | --- | --- | --- | --- |
| 18  | v.1 | 69.76% | 89.08% | 11.7M | 44.7 Mo |
| 34  | v.1 | 73.31% | 91.42% | 21.8M |
| 50  | v.2 | 80.86% | 95.43% | 25.6M |
| 101 | v.2 | 81.89% | 95.78% | 44.5M |
| 152 | v.2 |**82.28%**|**96.00%**| 60.2M |

### Fine-tuning
**Optimizer**: SGD with a momentum of 0.9 <br>
**Epochs**: 100 <br>
**Batch size**: 32

| ResNet | Learning rate | Test loss | Test acc |
| --- | --- | --- | --- |
| 18 v.1 | 1e-6 | 0.595 | 71.64% |
| 18 v.1 | 5e-6 | 0.434 | 82.71% |
| 18 v.1 | 1e-5 | 0.425 | 81.03% |
|**18 v.1**|**5e-5**|**0.395**|**83.23%**|
| 18 v.1 | 1e-4 | / | / |
| 34 v.1 | | / | / |
