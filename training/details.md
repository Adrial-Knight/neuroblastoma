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

### Simple perceptron layer

    self.resnet.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=1, bias=True),
        nn.Sigmoid()
    )

| Optimizer | Learning rate | Momentum | Epochs (best) | Batch size | Test loss | Test acc |
| --- | --- | --- | --- | --- | --- | --- |
| Adam | 1e-6 | (0.9, 0.999) | 100 (97) | 32 | 0.534 | 77.74% |
| Adam | 5e-6 | (0.9, 0.999) | 100 (58) | 32 | 0.458 | 80.73% |
| Adam | 1e-5 | (0.9, 0.999) | 100 (44) | 32 | 0.459 | 78.55% |
| Adam | 5e-5 | (0.9, 0.999) | 100 (8) | 32 | 0.440 | 82.02% |
| Adam | 1e-4 | (0.9, 0.999) | 25 (6) | 32 | 0.460 | 79.33% |
| Adam | 5e-4 | (0.9, 0.999) | 10 (2) | 32 | 0.418 | 82.22% |
| Adam | 1e-3 | (0.9, 0.999) | 5 (1) | 32 | 0.394 | 84.19% |
| SGD | 1e-6 | 0.9 | 100 (84) | 32 | 0.595 | 71.64% |
| SGD | 5e-6 | 0.9 | 100 (100) | 32 | 0.434 | 82.71% |
| SGD | 1e-5 | 0.9 | 100 (63) | 32 | 0.425 | 81.03% |
| SGD | 5e-5 | 0.9 | 100 (21) | 32 | 0.395 | 83.23% |
| SGD | 1e-4 | 0.9 | 25 (5) | 32 |**0.380**| **84.55%**|
| SGD | 5e-4 | 0.9 | 10 (1) | 32 | 0.421 | 84.02% |

### Two FC layers
Two fully connected layers with an intermediate non linear activation.

#### ReLU

    self.resnet.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=256, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=1, bias=True),
        nn.Sigmoid()
    )

| Optimizer | Learning rate | Momentum | Epochs (best) | Batch size | Test loss | Test acc |
| --- | --- | --- | --- | --- | --- | --- |
| SGD | 1e-6 | 0.9 | 100 (100) | 32 | 0.631 | 79.68% |
| SGD | 5e-6 | 0.9 | 100 (95) | 32 | 0.463 | 80.47% |
| SGD | 1e-5 | 0.9 | 100 (66) | 32 | 0.479 | 80.46% |
| SGD | 5e-5 | 0.9 | 100 (18) | 32 | 0.431 | 80.54% |
| SGD | 1e-4 | 0.9 | 25 (8) | 32 | 0.439 | 82.95% |
| SGD | 5e-4 | 0.9 | 10 (2) | 32 | **0.391**  |**83.59%**|
| SGD | 1e-3 | 0.9 | 5 (1) | 32 | 0.436 | 79.17% |

#### Dropout and LeakyReLU
Dropout is a regularization technique that randomly sets a fraction of the inputs to a layer to zero during training, to prevent overfitting.

LeakyReLU prevents the "dying ReLU" problem.

    self.resnet.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=256, bias=True),
        nn.LeakyReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=256, out_features=1, bias=True),
        nn.Sigmoid()
    )

| Optimizer | Learning rate | Momentum | Epochs (best) | Batch size | Test loss | Test acc |
| --- | --- | --- | --- | --- | --- | --- |
| SGD | 1e-6 | 0.9 | 100 (90) | 32 | 0.641 | 73.88% |
| SGD | 5e-6 | 0.9 | 100 (90) | 32 | 0.490 | 81.80% |
| SGD | 1e-5 | 0.9 | 100 (50) | 32 | 0.415 |**85.19%**|
| SGD | 5e-5 | 0.9 | 100 (16) | 32 |0.409| 82.12% |
| SGD | 1e-4 | 0.9 | 100 (9) | 32 |**0.393**| 82.45% |
| SGD | 5e-4 | 0.9 | 25 (1) | 32 | 0.474  | 82.84% |


### Conclusion about ResNet 18 tests
It seems unnecessary to make the Fully Connected classifier more complex as the loss and accuracy measured on the test dataset are equivalent.

The next step is to use ResNet with more layers with a minimalist training classifier. The objective is to determine if the addition of layers on the feature extraction part gives richer information to the classifier.



## ResNet versions with a minimalist classifier
    resnet_out = 512 if resnet_version <= 34 else 2048
    self.resnet.fc = nn.Sequential(
        nn.Linear(in_features=resnet_out, out_features=1, bias=True),
        nn.Sigmoid()
    )
### Performances on ImageNet
Source: https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights

| ResNet | v. weights | acc @1 | acc @5 | #Params | Size |
| --- | --- | --- | --- | --- | --- |
| 18  | v.1 | 69.76% | 89.08% | 11.7M | 44.7 Mo |
| 34  | v.1 | 73.31% | 91.42% | 21.8M | 83.3 Mo
| 50  | v.2 | 80.86% | 95.43% | 25.6M | 97.8 Mo
| 101 | v.2 | 81.89% | 95.78% | 44.5M | 171 Mo
| 152 | v.2 | 82.28% | 96.00% | 60.2M | 230 Mo

### Fine-tuning
**Optimizer**: SGD with a momentum of 0.9 <br>
**Batch size**: 32 <br>
**Max epoch**: 100

#### ResNet 18
| Learning rate | Epochs (best) | Test loss | Test acc |
| --- | --- | ---  | --- |
| 1e-6 | 100 (84)  |  0.595  |  71.64%  |
| 5e-6 | 100 (100) |  0.434  |  82.71%  |
| 1e-5 | 100 (63)  |  0.425  |  81.03%  |
| 5e-5 | 100 (21)  |  0.395  |  83.23%  |
| 1e-4 | 25 (5)    |**0.380**|**84.55%**|
| 5e-4 | 10 (1)    |  0.421  |  84.02%  |

#### ResNet 34

| Learning rate | Max epoch | Best epochs | Test Loss | Test Acc (%) |
| ------------- | --------- | ------------ | --------- | -------- |
|     1e-6      |    100    | [95, 92, 88, 91, 84, 74, 81, 72, 80, 98] | [0.527, 0.586, 0.649, 0.550, 0.603, 0.535, 0.537, 0.579, 0.599, 0.528]  | [79.07, 72.83, 62.86, 78.60, 67.88, 78.11, 79.32, 71.09, 70.39, 80.37] |
|     5e-6      |    100    | [41, 45, 39, 45, 24, 78, 66, 26, 69, 33] | [0.477, 0.479, 0.514, 0.479, 0.472, **0.411**, 0.482, 0.549, 0.427, 0.499]  | [86.02, 82.64, 79.77, 81.57, 86.55, 85.94, 81.58, 77.77, **87.67**, 79.49] |
|     1e-5      |     75    | [10, 36, 6, 41, 49, 32, 42, 17, 53, 6] | [0.510, 0.507, 0.560, 0.504, 0.443, 0.455, 0.449, 0.529, 0.509, 0.530]  | [82.90, 79.50, 77.78, 77.85, 84.46, 86.81, 84.72, 80.12, 80.30, 74.74] |
|     5e-5      |     50    | [2, 5, 8, 5] | [0.528, 0.450, 0.468, 0.522]  | [82.97, 86.46, 80.00, 81.08] |
|     1e-4      |     10    | [1, 5, 6] | [0.651, 0.490, 0.454]  | [60.07, 79.25, 81.34] |



#### ResNet 50

| Learning rate | Epochs (best) | Test loss | Test acc |
| --- | --- | --- | --- |
| 1e-6 | 100 (xx) | /       | /        |
| /    | /        | /       | /        |
| /    | /        | /       | /        |
| /    | /        | /       | /        |

#### ResNet 101

| Learning rate | Epochs (best) | Test loss | Test acc |
| --- | --- | --- | --- |
| 1e-6 | 100 (xx) | /       | /        |
| /    | /        | /       | /        |
| /    | /        | /       | /        |
| /    | /        | /       | /        |

#### ResNet 152

| Learning rate | Epochs (best) | Test loss | Test acc |
| --- | --- | --- | --- |
| 1e-6 | 100 (xx) | /       | /        |
| /    | /        | /       | /        |
| /    | /        | /       | /        |
| /    | /        | /       | /        |

#### Best learning

| ResNet | Learning rate | Best epoch | Test loss | Test acc |
| --- | --- | --- | --- | --- |
| 18 v.1 | 1e-4 | 5 | 0.380 | 84.55% |
| 34 v.1 | / | / | / | / |
