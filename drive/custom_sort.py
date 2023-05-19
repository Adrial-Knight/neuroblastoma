def alphanum(item):
    split = []
    wasdigit = False
    number, word = "", ""
    for char in item:
        if char.isdigit():
            if wasdigit:
                number += char
            else:
                split.append(word)
                number = char
            wasdigit = True
        else:
            if wasdigit:
                split.append(int(number))
                word = char
            else:
                word += char
            wasdigit = False
    if wasdigit:
        split.append(int(number))
    else:
        split.append(word)
    return split

if __name__ == "__main__":
    lst  = [f"Inception3_SGD_CNL{i}" for i in range(1, 6)]
    lst += [f"VGG{i}_SGD"    for i in [11, 13, 16, 19]]
    lst += [f"ResNet{i}_SGD" for i in [18, 32, 50, 101, 152]]
    lst += ["Inception3_SGD", "ResNet18_SGD_CNL1"]

    lst = sorted(lst, key=alphanum)
    for model in lst:
        print(model)
