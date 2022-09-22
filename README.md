# nano-net
The nice NanoNet GUI is coming soon.
To use the Jupyter notebook NanoNet download the NanNet.ipynb and follow the steps at the very bottom.

The NanoNet workflow follows the general image analysis pipeline and consist of image pre-processing, followed by a series of segmentation steps and the extraction of nanodomain (ND) features. The only input required is the size of the images in microns. First, images of equal size are pre-processed by applying a gaussian blur (𝜎=0.9), which is followed by binarization based on the pixel intensity distribution. Next, a morphological closing, consisting of dilation and erosion assures speckle noise removal inside foreground domains. Further, pixels at image edges are removed, since they would lead to a bias in various ND features such as area for example. By using Euclidean distance transform, distances are calculated from which coordinates of local maximum intensity peaks are extracted. From this, a mask is created, which is passed on to a watershed algorithm to separate touching foreground objects. Labels are then assigned to each foreground object and features from individual NDs as well as whole image properties are extracted. These features are then automatically saved as an excel file in allocation determined by the user. 

If two or more different conditions such as treatments or genotypes are given as an input, the extracted features are passed on to a TSNE to obtain a non-linear dimensionality reduced scatterplot of the initial high dimensional feature space. Further boxplots of user defined features comparing the different conditions are depicted and a correlation map between all features is generated. For unbiased estimation of differences between conditions, a Random Forest (RF) Classifier as well as a k-Nearest Neighbors (k-NN) machine learning algorithm are trained with a user defined fraction of the data and tested on the remaining data. For k-NN, a 8-fold cross validation is used to assess the accuracy score of the predicted labels as well as to estimate the optimal number of neighbors k. As an output the user receives from the k-NN the accuracy of prediction, the balanced accuracy, which takes into account imbalanced datasets in multiclass classification and is defined as average recall per class as well as two scatterplots, depicting the true condition label and the k-NN predicted labels. The output of the RF produces a confusion matrix showing the RFs performance in predicting the conditions class as percentage of correct predictions. Additionally the accuracy table is given along with the sorted feature importance, computed as the mean and standard deviation of accumulation of the impurity decrease within each tree.
