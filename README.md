# Convolutional-Encoder-Decoder-for-Hand-Segmentation
This Repo makes use of an Encoder-Decoder Network to Segment Hand in the given Images

This model was trained on images that had hand in the 'Fist Closed' Gesture, and the background was almost
similar in all the training examples.

100 such images were augmented using rotation and translation to expand the training set to 12500.
Resorted to this since i had to manually go and segment images in the training set which is time consuming.
So in order to save time had to do augmentation.

## Todo
- [ ] To Train the Model on other Class and Background to get better results

## Observations
### Almost closed fist/ fingers images
Though model was trained on only 'Fist closed' gesture, the model was able to perform very well for the images that were somewhere close to 'Fist closed'.

![B-train017.jpg](https://github.com/sharnam19/Convolutional-Encoder-Decoder-for-Hand-Segmentation/blob/master/New%20Test/B-train017.jpg "B-train017.jpg")
![B-train017.jpg](https://github.com/sharnam19/Convolutional-Encoder-Decoder-for-Hand-Segmentation/blob/master/New%20Test%20Output/B-train017.jpg "B-output017.jpg")


![C-train087.jpg](https://github.com/sharnam19/Convolutional-Encoder-Decoder-for-Hand-Segmentation/blob/master/New%20Test/C-train087.jpg "C-train087.jpg")
![C-train087.jpg](https://github.com/sharnam19/Convolutional-Encoder-Decoder-for-Hand-Segmentation/blob/master/New%20Test%20Output/C-train087.jpg "C-output087.jpg")


![Point-train009.jpg](https://github.com/sharnam19/Convolutional-Encoder-Decoder-for-Hand-Segmentation/blob/master/New%20Test/Point-train0009.jpg "Point-train009.jpg")
![Point-train009.jpg](https://github.com/sharnam19/Convolutional-Encoder-Decoder-for-Hand-Segmentation/blob/master/New%20Test%20Output/Point-train0009.jpg "Point-output009.jpg")

Although there seems to be some errors since some of the background is also coloured in a few output of the decoder, but relatively the performance is better than what i expected in unseen gestures.

### Open fist images
The model's performance on images with open fist and spreaded out fingers was very terrible. Though the performance can be improved by training on such images.

![Five-train073.jpg](https://github.com/sharnam19/Convolutional-Encoder-Decoder-for-Hand-Segmentation/blob/master/New%20Test%20Output/Five-train073.jpg "Five-train073.jpg")
![Five-train073.jpg](https://github.com/sharnam19/Convolutional-Encoder-Decoder-for-Hand-Segmentation/blob/master/New%20Test/Five-train073.jpg "Five-output073.jpg")

![V-train025.jpg](https://github.com/sharnam19/Convolutional-Encoder-Decoder-for-Hand-Segmentation/blob/master/New%20Test/V-train025.jpg "V-train025.jpg")
![V-train025.jpg](https://github.com/sharnam19/Convolutional-Encoder-Decoder-for-Hand-Segmentation/blob/master/New%20Test%20Output/V-train025.jpg "V-train025.jpg")

## Use cases
The output of the decoder can be used as an input to CNN. Since the output of the decoder will have uniform colour for hands it could be easier for the CNN to achieve higher accuracy.

## Alternative Approach
Instead of segmenting hand area out, we could train a Convolutional autoencoder to perform background deletion. The output of such convolutional autoencoder would be definitely more useful for a CNN in classification as it removes a lot of noise from the image, thereby allowing the model to learn without any distractions.
