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

## ResNet versions with a minimalist classifier
    resnet_out = 512 if resnet_version <= 34 else 2048
    self.resnet.fc = nn.Sequential(
        nn.Linear(in_features=resnet_out, out_features=1, bias=True),
        nn.Sigmoid()
    )

## Performances on ImageNet
Source: https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights

| ResNet | v. weights | acc @1 | acc @5 | #Params | Size |
| --- | --- | --- | --- | --- | --- |
| 18  | v.1 | 69.76% | 89.08% | 11.7M | 44.7 Mo |
| 34  | v.1 | 73.31% | 91.42% | 21.8M | 83.3 Mo
| 50  | v.2 | 80.86% | 95.43% | 25.6M | 97.8 Mo
| 101 | v.2 | 81.89% | 95.78% | 44.5M | 171 Mo
| 152 | v.2 | 82.28% | 96.00% | 60.2M | 230 Mo

## Fine-tuning
**SGD Optimizer**: with a momentum of 0.9 <br>
**Adam Optimizer**: with betas=(0.9, 0.999) <br>
**Max epoch**: 100, could be less if unreachable

In the following tables, the _loss_ and _accuracy_ metrics correspond to the validation dataset, and the epoch column represents the optimal _epoch_ achieved on the validation dataset. The grid search is performed over a range of learning rates and batch sizes, while keeping the backbone, optimizer, and momentum fixed. Several attempts are made for a given hyper-parameter selection.

### ResNet 18
#### SGD
| | 16 | 32 | 64 | 128 |
|-|----|----|----|-----|
| **5e-6** | epoch = [] <br> loss = [] <br> acc = [] | epoch = [] <br> loss = [] <br> acc = [] | epoch = [] <br> loss = [] <br> acc = [] | epoch = [] <br> loss = [] <br> acc = [] |
| **1e-5** | epoch = [] <br> loss = [] <br> acc = [] | epoch = [] <br> loss = [] <br> acc = [] | epoch = [] <br> loss = [] <br> acc = [] | epoch = [] <br> loss = [] <br> acc = [] |
| **5e-5** | epoch = [] <br> loss = [] <br> acc = [] | epoch = [] <br> loss = [] <br> acc = [] | epoch = [] <br> loss = [] <br> acc = [] | epoch = [] <br> loss = [] <br> acc = [] |
| **1e-4** | epoch = [] <br> loss = [] <br> acc = [] | epoch = [] <br> loss = [] <br> acc = [] | epoch = [] <br> loss = [] <br> acc = [] | epoch = [] <br> loss = [] <br> acc = [] |

#### Adam

### ResNet 34
