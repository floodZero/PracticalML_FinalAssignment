# Practical Machine Learning - Prediction Model for Weight Lifting Exercise Dataset
Gavin Kim  
2016-12-15  



## Executive Summary
This is final report of Coursera Practical Machine Learning Course. Goal of this document is to build a prediction model of Weight Lifting Exercise. The dataset contains variables from sensor worn by users and the classes that type of exercise.
For match this goal, I used ensemble model of Generalized Boosted Regression Model, Random Forest, Conditional Inference Tree. And for reducing dimension to increase training speed, PCA is used. But the ensenble got no more performance of model than Random Forest. After all this approach, I could get 0.98% of accuracy.

## Loading data

#### Load libraries


```r
library(caret)
library(randomForest)
library(corrplot)
library(MASS)

# For increasing learning speed, use multi-core.
library(doMC)
registerDoMC(cores = 8)
```

#### Download and Load training/test data


```r
# Check existance of data directory and create it if not exist.
dataDir <- "data"
if(!dir.exists(dataDir))
  dir.create(dataDir)

# Download training and test file 
Url.train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
Url.test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
data.trainPath <- paste0(dataDir, "/", basename(Url.train))
data.testPath <- paste0(dataDir, "/", basename(Url.test))

if(!file.exists(data.trainPath))
  download.file(Url.train, data.trainPath, mode = "w")
if(!file.exists(data.testPath))
  download.file(Url.test, data.testPath, mode = "w")

data.train <- read.csv(data.trainPath)
data.test <- read.csv(data.testPath)
```

## Preprocessing data

After loaing data, we need to remove the columns what we don't use. There are columns that have only NA values and nearly zero variance. It's better that we could replace these columns from data and there's columns not related to predict weight lifting exercise like observation number or user name or timestamps. It also has to be removed.


```r
# Remove columns having NA values
cs <- colSums(is.na(data.train))
data.train <- data.train[,(cs == 0)]
data.test <- data.test[,(cs == 0)]

# Remove near zero columns
nzvColumns <- nearZeroVar(data.train,saveMetrics=TRUE)
data.train <- data.train[,!nzvColumns$nzv]
data.test <- data.test[,!nzvColumns$nzv]

# Remove unnecessary columns (Username, timesstamps)
data.train <- data.train[,-c(1:6)]
data.test <- data.test[,-c(1:6)]
```

With checking correlation plot, we can get there's no strong relation between variables to disadventage building model.


```r
corrplot(cor(data.train[,-53]), method="pie", type="upper")
```

![](ML_Final_files/figure-html/corr-1.png)<!-- -->

#### Split data to training and validation set

After cleaning data, I found the test data set has only 20 observasions. It's too small to check the accuracy of model.
Hence, I decided to split training data with training and validation. The validation data is for test model's accuracy and I'll use training data for building model with K-fold cross validation.


```r
# split data for training/validation
inTrain <- createDataPartition(y=data.train$classe, p=0.7, list=FALSE)
data.trainPart <- data.train[inTrain,]
data.validPart <- data.train[-inTrain,]
```

## Modeling

I made plan to use ensemble model with stacking gbm, rf, ctree And PCA can reduce the size of variables for training performance.


```r
# 53th column is "classe" what will used for outcome label.
pcaModel <- preProcess(data.trainPart[,-53], method=c("pca", "center", "scale"), thresh = 0.95)
data.trainPca <- predict(pcaModel, data.trainPart[,-53])
data.trainPca <- data.frame(data.trainPca, classe = data.trainPart[,53])
data.validPca <- predict(pcaModel, data.validPart[,-53])
data.validPca <- data.frame(data.validPca, classe = data.validPart[,53])
print(pcaModel)
```

```
## Created from 13737 samples and 52 variables
## 
## Pre-processing:
##   - centered (52)
##   - ignored (0)
##   - principal component signal extraction (52)
##   - scaled (52)
## 
## PCA needed 25 components to capture 95 percent of the variance
```

As you see above 25 components has captured with 95 percent of the variance.



```r
set.seed(1234)
fit_gbm <- train(classe ~ ., data = data.trainPca, method = "gbm")
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1168
##      2        1.5352             nan     0.1000    0.0799
##      3        1.4829             nan     0.1000    0.0776
##      4        1.4347             nan     0.1000    0.0628
##      5        1.3963             nan     0.1000    0.0506
##      6        1.3633             nan     0.1000    0.0455
##      7        1.3337             nan     0.1000    0.0421
##      8        1.3073             nan     0.1000    0.0368
##      9        1.2828             nan     0.1000    0.0314
##     10        1.2614             nan     0.1000    0.0335
##     20        1.1007             nan     0.1000    0.0166
##     40        0.9251             nan     0.1000    0.0082
##     60        0.8188             nan     0.1000    0.0064
##     80        0.7412             nan     0.1000    0.0056
##    100        0.6761             nan     0.1000    0.0026
##    120        0.6233             nan     0.1000    0.0020
##    140        0.5763             nan     0.1000    0.0021
##    150        0.5570             nan     0.1000    0.0016
```

```r
pred_gbm <- predict(fit_gbm, newdata = data.validPca)
confusionMatrix(pred_gbm,data.validPca$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1509  140   57   24   31
##          B   38  863   92   23   79
##          C   55   81  826  112   60
##          D   65   18   26  779   42
##          E    7   37   25   26  870
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8236          
##                  95% CI : (0.8136, 0.8333)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7765          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9014   0.7577   0.8051   0.8081   0.8041
## Specificity            0.9402   0.9511   0.9366   0.9693   0.9802
## Pos Pred Value         0.8569   0.7881   0.7284   0.8376   0.9016
## Neg Pred Value         0.9600   0.9424   0.9579   0.9627   0.9569
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2564   0.1466   0.1404   0.1324   0.1478
## Detection Prevalence   0.2992   0.1861   0.1927   0.1580   0.1640
## Balanced Accuracy      0.9208   0.8544   0.8708   0.8887   0.8921
```


```r
set.seed(1234)
fit_rf <- train(classe ~ ., data = data.trainPca, method = "rf")
pred_rf <- predict(fit_rf, newdata = data.validPca)
confusionMatrix(pred_rf,data.validPca$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1668   16    1    1    0
##          B    0 1104   21    2    3
##          C    3   14  997   36    8
##          D    3    0    3  922    3
##          E    0    5    4    3 1068
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9786          
##                  95% CI : (0.9746, 0.9821)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9729          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9693   0.9717   0.9564   0.9871
## Specificity            0.9957   0.9945   0.9874   0.9982   0.9975
## Pos Pred Value         0.9893   0.9770   0.9423   0.9903   0.9889
## Neg Pred Value         0.9986   0.9926   0.9940   0.9915   0.9971
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2834   0.1876   0.1694   0.1567   0.1815
## Detection Prevalence   0.2865   0.1920   0.1798   0.1582   0.1835
## Balanced Accuracy      0.9961   0.9819   0.9796   0.9773   0.9923
```


```r
set.seed(1234)
fit_ctree <- train(classe ~ ., data = data.trainPca, method = "ctree")
```

```
## Loading required package: party
```

```
## Loading required package: grid
```

```
## Loading required package: mvtnorm
```

```
## Loading required package: modeltools
```

```
## Loading required package: stats4
```

```
## 
## Attaching package: 'modeltools'
```

```
## The following object is masked from 'package:plyr':
## 
##     empty
```

```
## Loading required package: strucchange
```

```
## Loading required package: zoo
```

```
## 
## Attaching package: 'zoo'
```

```
## The following objects are masked from 'package:base':
## 
##     as.Date, as.Date.numeric
```

```
## Loading required package: sandwich
```

```r
pred_ctree <- predict(fit_ctree, newdata = data.validPca)
confusionMatrix(pred_ctree,data.validPca$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1447  135   60   55   33
##          B   82  797  103   54   84
##          C   62   80  753   83   51
##          D   63   53   60  724   72
##          E   20   74   50   48  842
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7754         
##                  95% CI : (0.7645, 0.786)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.7155         
##  Mcnemar's Test P-Value : 0.001367       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8644   0.6997   0.7339   0.7510   0.7782
## Specificity            0.9328   0.9319   0.9432   0.9496   0.9600
## Pos Pred Value         0.8364   0.7116   0.7318   0.7449   0.8143
## Neg Pred Value         0.9454   0.9282   0.9438   0.9512   0.9505
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2459   0.1354   0.1280   0.1230   0.1431
## Detection Prevalence   0.2940   0.1903   0.1749   0.1652   0.1757
## Balanced Accuracy      0.8986   0.8158   0.8386   0.8503   0.8691
```


```r
predDF <- data.frame(pred_gbm, pred_rf, pred_ctree, classe = data.validPca$classe)
fit_stacked <- train(classe ~ ., data = predDF, method="rf")
pred_stacked <- predict(fit_stacked, newdata = predDF)
confusionMatrix(pred_stacked,data.validPca$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1668   16    1    1    0
##          B    0 1104   21    2    3
##          C    3   14  997   36    8
##          D    3    0    3  922    3
##          E    0    5    4    3 1068
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9786          
##                  95% CI : (0.9746, 0.9821)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9729          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9693   0.9717   0.9564   0.9871
## Specificity            0.9957   0.9945   0.9874   0.9982   0.9975
## Pos Pred Value         0.9893   0.9770   0.9423   0.9903   0.9889
## Neg Pred Value         0.9986   0.9926   0.9940   0.9915   0.9971
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2834   0.1876   0.1694   0.1567   0.1815
## Detection Prevalence   0.2865   0.1920   0.1798   0.1582   0.1835
## Balanced Accuracy      0.9961   0.9819   0.9796   0.9773   0.9923
```

The model what has best performance is Random forest as you see. And the ensemble model of gbm, rf, ctree gets no benefit of accuracy.


## Testing

We need answer to predict the 20 test set with this model.
Here is the results.


```r
# Apply PCA
data.testPCA <- predict(pcaModel, data.test[,-53])

# Create predicted variales
pred_gbm_test <- predict(fit_gbm, newdata = data.testPCA)
pred_rf_test <- predict(fit_rf, newdata = data.testPCA)
pred_ctree_test <- predict(fit_ctree, newdata = data.testPCA)

predDF_test <- data.frame(pred_gbm = pred_gbm_test, pred_rf = pred_rf_test, pred_ctree = pred_ctree_test)
pred_stacked_test <- predict(fit_stacked, newdata = predDF_test)
print(t(data.frame(pred_stacked_test)))
```

```
##                   [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11]
## pred_stacked_test "B"  "A"  "B"  "A"  "A"  "E"  "D"  "B"  "A"  "A"   "B"  
##                   [,12] [,13] [,14] [,15] [,16] [,17] [,18] [,19] [,20]
## pred_stacked_test "C"   "B"   "A"   "E"   "E"   "A"   "B"   "B"   "B"
```

