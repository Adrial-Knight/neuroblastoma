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
- Modification: Fully Connected Classifier

## 512_1
Simple perceptron layer

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=1, bias=True),
        nn.Sigmoid()
    )

| Optimizer | Learning rate | Momentum | Epochs | Batch size | Test loss | Test acc |
| --- | --- | --- | --- | --- | --- | --- |
| Adam | 1e-6 | (0.9, 0.999) | 100 | 32 | 0.534 | 77.74% |
| Adam | 5e-6 | (0.9, 0.999) | 100 | 32 | 0.458 | 80.73% |
| Adam | 1e-5 | (0.9, 0.999) | 100 | 32 | 0.459 | 78.55% |
| Adam | 5e-5 | (0.9, 0.999) | 100 | 32 | 0.440 | 82.02% |
| SGD | 1e-6 | 0.9 | 100 | 32 | 0.595 | 71.64% |
| SGD | 5e-6 | 0.9 | 100 | 32 | 0.434 | 82.71%
| SGD | 1e-5 | 0.9 | 100 | 32 | 0.425 | 81.03%
| SGD | 5e-5 | 0.9 | 100 | 32 | 0.395 | 83.23% |

## 512_256_1
Two fully connected layers with an intermediate non linear activation.

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    model.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=256, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=1, bias=True),
        nn.Sigmoid()
    )

| Optimizer | Learning rate | Momentum | Epochs | Batch size | Test loss | Test acc |
| --- | --- | --- | --- | --- | --- | --- |
| SGD | 1e-6 | 0.9 | 100 | 32 | / | / |
| SGD | 5e-6 | 0.9 | 100 | 32 | 0.463 | 80.47% |
| SGD | 1e-5 | 0.9 | 100 | 32 | Pending | Pending |
| SGD | 5e-5 | 0.9 | 100 | 32 | / | / |
