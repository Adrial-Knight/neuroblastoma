import matplotlib.pyplot as plt

inception3 = [0.322, 0.295, 0.288, 0.282, 0.278, 0.296, 0.320]
resnet18   = [0.488, 0.450, 0.513]
vgg19      = [0.479]

random_max = 0.7056 + 0.0444
random_min = 0.7056 - 0.0444

N = max(len(inception3), len(resnet18), len(vgg19))


plt.figure()
plt.plot(inception3, "-s", color="C0", label="Inception3")
plt.plot(resnet18,   "-s", color="C1", label="ResNet18")
plt.plot(vgg19,      "-s", color="C2", label="VGG19")
plt.ylabel("Loss")
plt.xlabel("Number of non-linearities added")
plt.legend(title="Backbone", loc="center right")
plt.grid(True)


plt.fill_between(range(N), random_min, random_max, color="red", alpha=0.3, linestyle="dashed", edgecolor="black")
text_x = N // 2
text_y = (random_max + random_min) / 2
plt.text(text_x, text_y, "Loss without training", ha="center", va="center", color="black", fontsize=11)

plt.show()
