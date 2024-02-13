# Binary Classification of Neuroblastoma using Convolutional Neural Networks

This repository contains the code and documentation for my Final Year Project conducted at the eVida laboratory in Bilbao, between February and June 2023. The project focuses on binary classification of Neuroblastoma using Convolutional Neural Networks.

## Table of Contents

- [Introduction](#introduction)
- [Folders](#folders)
- [Results](#results)
- [Contact](#contact)

## Introduction

Neuroblastoma is a type of cancer that affects the nerve cells in infants and young children. This project aims to develop a binary classification of the Grade of Differenciation of cancerous cells. According to the [Shimada System](https://www.researchgate.net/figure/A-Simplified-diagram-of-the-International-Neuroblastoma-pathology-Classification-the_fig1_261093136), a higher level of differentiation is associated with a more favourable prognosis.

## Folders

The repository is organized into the following folders:

1. **dashboard**: Contains a graphical interface for monitoring the training process.
2. **database**: Provides functionality to partition a given image database into three independent datasets (train, valid, test) based on diversity and target proportions.
3. **doc**: Contains project documentation such as the project report, internship specifications document, PowerPoint presentations, and miscellaneous illustrations.
4. **drive**: Implements interaction with Google Drive. To use this functionality, you need to generate an API key from [Google Cloud Console](https://console.cloud.google.com/apis).
5. **image_augmentation**: Provides image augmentation capabilities through a minimalist cropping interface and rotation operations.
6. **models**: Contains the trained models and a notebook used in Colab.
7. **plot_results**: Offers functionality to visualize metrics based on data from Google Drive.

## Results

The following [Google Drive link](https://drive.google.com/drive/folders/1bOLNcIhzC4OfU5d9aMTXEzl3niaEn1g8?usp=sharing) contains a summary of all the training sessions organized in JSON files stored within a ZIP archive, along with the best weights obtained from the trained networks.

## Contact
Email: pierre.minier@ims-bordeaux.fr \
eVida: [http://evida.deusto.es/](http://evida.deusto.es/)
