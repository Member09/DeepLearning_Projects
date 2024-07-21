# DeepLearning_Projects

Here are few of the projects that I've worked on and their results!

---
---
# 1. Leaf Disease Classification
---
This is an attempt to work of the problem stated on [Kaggle](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) platform.


## Table of Contents
1. [Project Overview](#project-overview)
2. [Motivation](#motivation)
3. [Dataset Description](#dataset-description)
4. [Files](#files)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Model Training](#model-training)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)

## Project Overview
This project aims to classify cassava leaf images into four disease categories or a healthy leaf category using machine learning techniques. Cassava is a crucial crop in Africa, providing a major source of carbohydrates. However, viral diseases significantly affect yields. This project leverages data science to help farmers quickly and accurately identify diseased plants using photos taken with mobile-quality cameras.

## Motivation
Current disease detection methods are labor-intensive, costly, and require the involvement of government-funded agricultural experts. Given the constraints faced by African farmers, including access to mobile-quality cameras and low-bandwidth internet, this project aims to provide a practical, automated solution for rapid disease diagnosis.

## Dataset Description
The dataset consists of 21,367 labeled images collected from a survey in Uganda. Most images were crowdsourced from farmers' gardens and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. The dataset realistically represents the diagnostic conditions that farmers face.

## Model Training
The model training process involves the following steps:
1. Preprocessing the images and augmenting the training data.
2. Defining a convolutional neural network (CNN) architecture suitable for image classification.
3. Compiling the model with appropriate loss functions and optimizers.
4. Training the model on the training dataset.
5. Evaluating the model on a validation set.

## Evaluation
The model's performance is evaluated based on its ability to correctly classify the cassava leaf images into the respective disease categories or as healthy. The primary metric used for evaluation is accuracy. Other metrics like precision, recall, and F1-score can also be considered for a more comprehensive evaluation.





---
---

# 2. Dogs and Cats Classification
---
This project is part of the competition on [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/) platform.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Motivation](#motivation)
3. [Dataset Description](#dataset-description)
4. [Files](#files)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Model Training](#model-training)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)

## Project Overview
This project involves building a machine learning model to classify images as either containing a dog or a cat. While this task is straightforward for humans, it poses significant challenges for computers. The project is based on the Asirra (Animal Species Image Recognition for Restricting Access) dataset, which consists of images provided by Petfinder.com.

## Motivation
Asirra was originally created as a CAPTCHA to distinguish between humans and automated systems by asking users to identify images of cats and dogs. The goal of this project is to benchmark the latest computer vision and deep learning techniques in image classification, aiming to achieve high accuracy in distinguishing between dogs and cats.

## Dataset Description
The dataset used in this project is a subset of the Asirra dataset provided by Petfinder.com. It contains a total of 25,000 labeled images of dogs and cats for training purposes. The test set includes an additional 12,500 images that need to be classified.
