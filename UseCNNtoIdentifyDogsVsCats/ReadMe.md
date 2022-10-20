# Use CNN to Identify Dogs Vs Cats

Here are two codes for building CNN classifier for dogs v.s. cats. The earlier work uses Tflearn and the more later work uses Keras. While Tflearn is a transparent deep learning library built on top of Tensorflow, Keras has more pre-trained models and more documentations.  <br><br>


### [CNN classifier using Keras](UseCNNtoIdentifyDogsVsCatsKeras.ipynb)

* Original dataset splited into training dataset 75% and testing dataset 25% <br>
CNN model based on VGG architecture <br>
Three stacks of convolutional layer followed by max pooling layer <br>
Kernel size: 3 x 3. 32 filters, 64 filters, and 128 filters. <br>
Max Pooling size: 2 x 2 <br>
Batch size: 64 <br>
Optimization method SGD <br>

* Training results without data augmentation. Fig. 1(a) shows the results of the baseline model. The accuracy acheived is 79.6%. Fig. 1(b) shows the results of using more convolution layers and has two fully connected layers. The accuracy acheived is 81%. Both show evident overfitting after 5~6th epoch. 

<p align="center">
<img width="400" alt="VGG3_20epoch" src="https://user-images.githubusercontent.com/86133411/196612255-63cbf6c0-9a82-4979-8034-ec3fa6efc187.png">
<img width="400" alt="VGG3_plus_20epoch" src="https://user-images.githubusercontent.com/86133411/196612191-96fd7125-a2f6-43cb-97e0-2b16a7e2b049.png">
<p align="center">Fig. 1 (a) and (b)</p>
</p>
<br>

* Training results with data augmentation. Fig. 2(a) the baseline model acheives 87% of accuracy. Fig. 2(b) the model with more layers acheived 92% of accuracy. 

<p align="center">
<img width="400" alt="VGG3_augmented_80epoch" src="https://user-images.githubusercontent.com/86133411/196612215-169306aa-ac02-44f6-8d59-f91abb7f9e24.png">
<img width="400" alt="VGG3_plus_augmented_80epoch" src="https://user-images.githubusercontent.com/86133411/196612202-e6511512-676d-47d0-940e-62547acee8af.png">
<p align="center">Fig. 2 (a) and (b)</p>
</p>
<br>

* Summary table of results of different trained model 
<p align="center">
<img width="800" src="https://user-images.githubusercontent.com/86133411/196864158-bfcf8d25-a81e-44d3-9be6-64f06c4e40fd.png">
</p>
<br>

### [CNN classifier using Tflearn](UseCNNtoIdentifyDogsVsCatsTflearn.ipynb)
* Results of applying the prediction model to 12 testing data images. 
<p align="center">
<img width="360" alt="Screenshot 2022-03-10 093110" src="https://user-images.githubusercontent.com/86133411/157817551-24923a66-14c5-4836-bf01-decdc2ec4d21.png">
</p>
<br/>

* Summary table comparing accuracy and loss of the training data w.r.t different parameters. 
<p align="center">
<img width="500" src="https://user-images.githubusercontent.com/86133411/157908361-70a480d8-89bd-4374-9ea5-27a7cd4b23b3.png">
</p>
<br/>

* Accuracy and loss curve of the training data w.r.t different parameters. 
<p align="center">
<img width="300" alt="legends" src="https://user-images.githubusercontent.com/86133411/157910152-84a66623-78e8-43bb-9c44-8350510b3a9c.png"> <br/>
<img width="360" alt="accuracy" src="https://user-images.githubusercontent.com/86133411/157910119-5fd8da30-c877-4c58-b339-e7bd2375b3d9.png">    <img width="360" alt="loss2" src="https://user-images.githubusercontent.com/86133411/157910135-a6aae590-1e6d-431b-9128-f4bbc1f0f720.png">
</p>
<br/>

## References 
[1] Jason Brownlee, https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/ <br/>
[2] Rohit G., https://www.kaggle.com/code/rohitgadhwar/image-classification-using-cnn-transfer-learning <br/>
[3] Sentdex, https://www.youtube.com/c/sentdex. <br/>
