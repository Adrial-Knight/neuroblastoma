from torch import nn

def get_classifier(kind: str, in_features: int):
    layers = []
    if kind.startswith("CNL"):
        nb_ReLU = int(kind[-1])
        for i in range(nb_ReLU):
            layers += [
                nn.Linear(in_features >> i, in_features >> (i+1), bias=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Dropout(0.5, inplace=False)
            ]
        in_features = in_features >> (i+1)

    layers += [
        nn.Linear(in_features, 1, bias=True),
        nn.Sigmoid()
    ]
    classifier = nn.Sequential(*layers)

    return classifier
