# Breast Cancer Prediction using Deep Learning (Keras) 

## Project Overview
**There are two objectives of this project is to create an algorithm to predict whether or not a breast tumor is benign (non-cancerous) or malignant (cancerous) based off the characteristics of the cell nuclei that are present. My algorithm is able to predict whether the tumors are benign or malignant with a 98% overall acccuracy and a 97% F1-score for benign, and 98% F-1 score for malignant.** The F1-score is a measure of a model's accuracy that considers both the precision (the number of true positive results divided by the number of all positive results, including those not identified correctly) and the recall (the number of true positive results divided by the number of all samples that should have been identified as positive).

The data is from the Breast Cancer Wisconsin (Diagnostic) dataset provided by the University of Wisconsin (1995). 

The characteristics of the cell nuclei within the data set include:
- radius (mean of distances from center to points on the perimeter)
- texture (standard deviation of gray-scale values)
- perimeter
- area
- smoothness (local variation in radius lengths)
- compactness (perimeter^2 / area - 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry
- fractal dimension ("coastline approximation" - 1)


## Libraries Used
numpy (numerical operations), pandas (data manipulation), matplotlib (visualization), seaborn (visualization), sklearn (machine learning tasks), tensorflow (deep learning tasks)

## Correlation of the Features
Below is the correlation of the characteristics of the tumors. The lighter the color is the more positively correlated, and vise versa.

## Creating the Model
After splitting the data into a test and training set, I scaled the data. On top of the benefits of scaling for machine learning and deep learning models, I specifically chose "MinMaxScaler" to mitigate the varying ranges oo the values for each feature, to account for the Non-Gaussian distribution, and for optimization of the Gradient Descent. 

### Training the Model with Early Stopping nad Dropout Layers
To train the model, I implemented a deep learning neural network using TensorFlow and Keras. The model architecture included dense layers with ReLU activations and dropout layers to prevent overfitting. **The dropout rate was set at 0.5, meaning there was a 50% chance that the output of each neuron in the dropout layer would be set to zero during training.** This randomness helps the model to generalize better and not rely too heavily on any one feature.

**One of the key techniques used in training was Early Stopping, which is used to avoid overfitting when training a learner with an iterative method, such as gradient descent.** By monitoring the model's performance on a validation set, **training can be stopped at the point when performance on the validation set starts to decrease**, indicating the model is beginning to memorize the training data rather than learning patterns. The EarlyStopping callback was configured to monitor the val_loss, which is the model's loss on the validation set. The training process was set to stop if the validation loss did not improve for 20 consecutive epochs, with the idea being that if the model is no longer improving on unseen data, further training would only lead to overfitting.

The model was initially set to train for a maximum of 600 epochs, but with Early Stopping, the training stopped automatically at epoch 127. This indicates that at this point, further training would not have resulted in more accurate predictions (for unforeseen data).

After training, we plotted the training loss (loss) and the validation loss (val_loss) over epochs. The plot showed how the model's performance improved over time and confirmed that training was stopped to prevent overfitting.



###Evaluation
The EarlyStopping mechanism, along with the implementation of dropout layers, played a crucial role in achieving the high accuracy and F1-scores reported for this model. This approach to training ensures that our model is robust, generalizes well to unseen data, and is indicative of the careful consideration given to avoiding common pitfalls such as overfitting
