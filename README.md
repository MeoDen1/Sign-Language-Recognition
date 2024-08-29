# Sign Language using Transformer

A Research Project on creating a python application that are able to predict the gloss (text) and generate sequence from videos of a person performing Sign Language or through live camera. 

Currently, the application utilizes MediaPipe for holistic landmark extraction and Transformer-based model to predict a single gloss<br>
The dataset for classification is Kaggle competition public dataset: [Google Isolated Sign Lanugage Recognition](https://www.kaggle.com/competitions/asl-signs)

In the current approach, the model can only archieve 78% accuracy on validation and test data. This is due to the model is replied on MediaPipe's landmark extraction result, which sometimes failed to fully captured holistic landmark. <br>
The next improvement of model is aimed at extracting and generating sequences from the video or through live camera without relying on MediaPipe or other pre-trained extraction models.

Details of implementation and model training are documented in **build.ipynb**
