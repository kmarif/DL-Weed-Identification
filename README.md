# DL-Weed-Identification
### based on DeepWeeds (https://github.com/AlexOlsen/DeepWeeds)
##
This repository makes available the source code and public dataset for the work, "Weed Identification by Single-Stage and Two-Stage Neural Networks: A Study on the Impact of Image Resizers and Weights Optimization Algorithms", being published with open access by Frontiers in Plant Science: . 

It contains annotated files for DeepWeeds dataset for various deep learning models using TensorFlow object detection API and YOLO/Darknet neural network framework. Also, the inference graph from the final/optimized DL model (Faster RCNN ResNet-101) is available.

It also contains configuration files for the deep learning models including SSD MobileNet, SSD Inception-v2, Faster RCNN ResNet-50, Faster RCNN ResNet-101, Faster RCNN Inception, Yolo-v4, RetinaNet, CenterNet ResNet-50, EfficientDet, and Yolo-v4.

The annotation files, inference graph, and source code are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. The contents of this repository are released under an [Apache 2](LICENSE) license.

<!--
## Download the dataset images and our trained models

* [images.zip](https://drive.google.com/file/d/1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj) (468 MB)
* [models.zip](https://drive.google.com/file/d/1MRbN5hXOTYnw7-71K-2vjY01uJ9GkQM5) (477 MB)
-->
## Download the annotated images and trained models

* [300x300.rar](https://drive.google.com/file/d/1NODFubk6AxeY6Xpy9HKLRJ0oRSDUwsc-) (1.1 GB)
* [600x600.rar](https://drive.google.com/file/d/1zULm7sLoQQOuYPxjYwmjYg_C8-zxTChI) (2.8 GB)
* [640x640.rar](https://drive.google.com/file/d/1hNbGPJnKOK2kn4tLman0TVqayNSIF_NL) (3.3 GB)
* [512x512.rar](https://drive.google.com/file/d/1iBhh6WYwwsKALQucxCkG9fVWrRdhqOQw) (2.0 GB)
* [yolo-v4.rar](https://drive.google.com/file/d/1HkdV631NV2rNYgk1WlA6QgyM8tw7dbtc) (2.8 GB)
* [Inference graph_final optimized model.rar](https://drive.google.com/file/d/1BVTyypnWzxTvZnA7H2zPQqr-WzOHxJoa) (560 MB)

Due to the size of the images and models they are hosted outside of the Github repository. 

The images were resized with annotations using pascal_voc (https://github.com/italojs/resize_dataset_pascalvoc).

<!--
## TensorFlow Datasets
Alternatively, you can access the DeepWeeds dataset with [TensorFlow Datasets](https://www.tensorflow.org/datasets), TensorFlow's official collection of ready-to-use datasets. [DeepWeeds](https://www.tensorflow.org/datasets/catalog/deep_weeds) was officially added to the TensorFlow Datasets catalog in August 2019.

## Weeds and locations
The selected weed species are local to pastoral grasslands across the state of Queensland. They include: "Chinee apple", "Snake weed", "Lantana", "Prickly acacia", "Siam weed", "Parthenium", "Rubber vine" and "Parkinsonia". The images were collected from weed infestations at the following sites across Queensland: "Black River", "Charters Towers", "Cluden", "Douglas", "Hervey Range", "Kelso", "McKinlay" and "Paluma". The table and figure below break down the dataset by weed, location and geographical distribution.

**Table 1.** The distribution of *DeepWeeds* images by weed species (row) and location (column).
![alt text](https://i.imgur.com/2e0ow8l.png "Distribution of DeepWeeds images by species and location.")

![alt text](https://i.imgur.com/scmJcS3.jpg "Geographical distribution of DeepWeeds images.")
**Figure 2.** The geographical distribution of *DeepWeeds* images across northern Australia  (Data: Google, SIO, NOAA, U.S. Navy, NGA, GEBCO; Image © 2018 Landsat / Copernicus; Image © 2018 DigitalGlobe; Image © 2018 CNES / Airbus).

## Data organization

Images are assigned unique filenames that include the date/time the image was photographed and an ID number for the instrument which produced the image. The format is like so: ```YYYYMMDD-HHMMSS-ID```, where the ID is simply an integer from 0 to 3. The unique filenames are strings of 17 characters, such as 20170320-093423-1.

## labels

The labels.csv file assigns species labels to each image. It is a comma separated text file in the format:
```
Filename,Label,Species
...
20170207-154924-0,jpg,7,Snake weed
20170610-123859-1.jpg,1,Lantana
20180119-105722-1.jpg,8,Negative
...
```

*Note: The specific label subsets of training (60%), validation (20%) and testing (20%) for the five-fold cross validation used in the paper are also provided here as CSV files in the same format as "labels.csv".*

## models

We provide the most successful ResNet50 and InceptionV3 models saved in Keras' hdf5 model format. The ResNet50 model, which provided the best results, has also been converted to UFF format in order to construct a TensorRT inference engine.
```
resnet.hdf5
inception.hdf5
resnet.uff
```

## deepweeds.py

This python script trains and evaluates Keras' base implementation of ResNet50 and InceptionV3 on the DeepWeeds dataset, pre-trained with ImageNet weights. The performance of the networks are cross validated for 5 folds. The final classification accuracy is taken to be the average across the five folds. Similarly, the final confusion matrix from the associated paper aggregates across the five independent folds. The script also provides the ability to measure the inference speeds within the TensorFlow environment.

The script can be executed to carry out these computations using the following commands.

* To train and evaluate the ResNet50 model with five-fold cross validation, use `python3 deepweeds.py cross_validate --model resnet`.
* To train and evaluate the InceptionV3 model with five-fold cross validation, use `python3 deepweeds.py cross_validate --model inception`.
* To measure inference times for the ResNet50 model, use `python3 deepweeds.py inference --model models/resnet.hdf5`.
* To measure inference times for the InceptionV3 model, use `python3 deepweeds.py inference --model models/inception.hdf5`.

## Dependencies

The required Python packages to execute deepweeds.py are listed in requirements.txt.

## tensorrt

This folder includes C++ source code for creating and executing a ResNet50 TensorRT inference engine on an NVIDIA Jetson TX2 platform. To build and run on your Jetson TX2, execute the following commands:
```
cd tensorrt/src
make -j4
cd ../bin
./resnet_inference
```

## Citations

If you use the DeepWeeds dataset in your work, please cite it as:

IEEE style citation: “A. Olsen, D. A. Konovalov, B. Philippa, P. Ridd, J. C. Wood, J. Johns, W. Banks, B. Girgenti, O. Kenny, J. Whinney, B. Calvert, M. Rahimi Azghadi, and R. D. White, “DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning,” *Scientific Reports*, vol. 9, no. 2058, **2** 2019. [Online]. Available: https://doi.org/10.1038/s41598-018-38343-3 ”

## BibTeX
```
@article{DeepWeeds2019,
  author = {Alex Olsen and
    Dmitry A. Konovalov and
    Bronson Philippa and
    Peter Ridd and
    Jake C. Wood and
    Jamie Johns and
    Wesley Banks and
    Benjamin Girgenti and
    Owen Kenny and 
    James Whinney and
    Brendan Calvert and
    Mostafa {Rahimi Azghadi} and
    Ronald D. White},
  title = {{Weed Identification by Single-Stage and Two-Stage Neural Networks: A Study on the Impact of Image Resizers and Weights Optimization Algorithms}},
  journal = {Frontiers in Plant Science},
  year = 2022,
  number = 2058,
  month = 2,
  volume = 9,
  issue = 1,
  day = 14,
  url = "",
  doi = ""
}
-->

##
This repository is a part of the PhD research of Muhammad Hammad Saleem (H.Saleem@massey.ac.nz; engr.hammadsaleem@gmail.com)

In case of any query, please contact Dr. Khalid Mahmood Arif (K.Arif@massey.ac.nz), Muhammad Hammad Saleem (H.Saleem@massey.ac.nz; engr.hammadsaleem@gmail.com)
