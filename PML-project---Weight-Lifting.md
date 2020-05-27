Abstract
--------

The objective of the project is to use the GroupWare Human Activity
Recognition Weight Lifting data-set to predict if a person is performing
correctly an specific exercise. The data set contains info about 6
subjects performing a weight lift while recording their motions using
on-body sensors. The training data set have 19622 observations. The ML
algorithm is random forest with 10-fold cross validation. This was run
using parallel computation in R. The model achieved an average accuracy
of 0.9951 and was able to correctly predict the 20 observations in the
testing set.

Data Loading and Pro-processing
-------------------------------

The first step is to load the necessary libraries and the training and
testing data-sets.

    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library(RANN)
    library(parallel)
    library(doParallel)

    ## Loading required package: foreach

    ## Loading required package: iterators

    training <- read.csv("./pml-training.csv")
    testing <- read.csv("./pml-testing.csv")
    dim(training)

    ## [1] 19622   160

As seen above, the training data-set contains 19622 observations of 160
different variables. Nonetheless, much of those 160 variables are
summary variables and others are subject and time indicators. For this
project, only the raw sensor data is used. Also, as observations are
independent of time and subject, the identification variables are
excluded as well. This was done using the grepl function.

    varUse <- grepl("X|user|raw|cvtd|new|num|kurtosis|skewness|max|min|amplitude|var|avg|stddev",names(training))
    trainingP <- training[,!varUse]
    dim(trainingP)

    ## [1] 19622    53

As seen above, the new training data set only contains 53 variables. If
the “classe” variable is excluded that gives a remaining of 52
predictors.  
Finally, to improve the ML algorithm, a “x” data frame is created with
the 52 predictors and a “y” data frame with the “classe” variable.

    x <- trainingP[,-53]
    y <- trainingP[,53]

    summary(y)

    ##    A    B    C    D    E 
    ## 5580 3797 3422 3216 3607

As seen above, the classe variable is a factor with 5 levels. “A” level
been the perfect weight lift performance and “B:E” levels are common
defects in the exercise.

Model Training
--------------

First, the training control is created. In this case, cross-validation
with 10-folds is used.

    modControl <- trainControl(method = "cv", number = 10)

The ML algorithm is random forest. With help of the caret::train()
function the model is fitted. Also, to make the training computation
faster, the makecluster() and registerDoParallel() functions are used.

    cluster <- makeCluster(detectCores() - 1)
    registerDoParallel(cluster)

    set.seed(9876)
    mod <- train(x, y, method = "rf", trControl = modControl)

    stopCluster(cluster)
    registerDoSEQ()

Model Accuracy and Out-of-Sample Expected Error
-----------------------------------------------

To check the model accuracy and expected out-of-sample error the
re-samples accuracy of each k-fold is checked. Also, the
confusionMatrix.train() function is used to see the average accuracy.
The expected out-of-sample accuracy es expected to be somewhat smaller
than the 10-fold average accuracy.

    mod$resample

    ##     Accuracy     Kappa Resample
    ## 1  0.9943906 0.9929023   Fold02
    ## 2  0.9928608 0.9909683   Fold01
    ## 3  0.9964340 0.9954899   Fold04
    ## 4  0.9949032 0.9935523   Fold03
    ## 5  0.9964340 0.9954903   Fold06
    ## 6  0.9949006 0.9935499   Fold05
    ## 7  0.9938869 0.9922664   Fold08
    ## 8  0.9943963 0.9929109   Fold07
    ## 9  0.9984717 0.9980670   Fold10
    ## 10 0.9964322 0.9954868   Fold09

    confusionMatrix.train(mod)

    ## Cross-Validated (10 fold) Confusion Matrix 
    ## 
    ## (entries are percentual average cell counts across resamples)
    ##  
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 28.4  0.1  0.0  0.0  0.0
    ##          B  0.0 19.2  0.1  0.0  0.0
    ##          C  0.0  0.0 17.3  0.1  0.0
    ##          D  0.0  0.0  0.1 16.2  0.0
    ##          E  0.0  0.0  0.0  0.0 18.3
    ##                             
    ##  Accuracy (average) : 0.9953

Testing
-------

The final step is to predict the testing data-set using the fitted
model.

    pred <- predict(mod, newdata = testing)
    pred

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
