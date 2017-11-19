# Convolutional-Encoder-Decoder-for-Hand-Segmentation
This Repo makes use of an Encoder-Decoder Network to Segment Hand in the given Images

This model was trained on images that had hand in the 'Fist Closed' Gesture, and the background was almost
similar in all the training examples.

100 such images were augmented using rotation and translation to expand the training set to 12500.
Resorted to this since i had to manually go and segment images in the training set which is time consuming.
So in order to save time had to do augmentation.

## Todo
- [ ] To Train the Model on other Class and Background to get better results
