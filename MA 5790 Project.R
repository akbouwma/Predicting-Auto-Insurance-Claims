############# Predictive Modelling Project #############
library(naniar)
library(dplyr)
library(moments)
library(corrplot)
library(AppliedPredictiveModeling)
library(caret)
library(e1071)
library(cplm)
library(tweedie)
library(moments)
library(fastDummies)
library(statmod)
library(HDtweedie)
library(Hmisc)  
library(pROC)
#####  AutoCaim data
data(AutoClaim)
# View(AutoClaim)

z <- AutoClaim
str(z)

z1 <- z[, -c(1, 2, 5, 16, dim(z)[2] )] #remove the columns 1,2,5, 16 and last. 
# We need to mention why we removed these columns

response_amt <- z[,5] # response variable
response_flag <- z[,16]
x <-z1
barplot(table(response_flag), main = "response Variable: CLM_FLAG" )
########################### Group JobClass Doctor With Manager ################################################

n_observations = aggregate((z$CLM_FLAG), list(z$JOBCLASS), FUN=length)
n_claims = aggregate((z$CLM_FLAG=="Yes"), list(z$JOBCLASS), FUN=sum)
claim_freq = n_claims[,2]/n_observations[,2]
dif_from_dr = abs(claim_freq[4] - claim_freq)
min(dif_from_dr[-4]) # the 7th agrument (Manager) matches doctor the cloest, Lawyer (6th argument) is the second closest
level_names = c("Unknown", "Blue Collar", "Clerical", "Doctor", "Home Maker", "Lawyer", "Manager", "Professional", "Student")

df = data.frame(level_names)
#df$n_obervations = n_observations
df$n_observations = n_observations[,2]
df$n_claims = n_claims[,2]
df$claim_freq = claim_freq
df$dif_from_dr = dif_from_dr
df

#This barplot suggests:
#  a) Job class==Doctor could have a lot of predictive power
# `b) Doctors are very similar to Managers in terms of the reponse variable
barplot(prop.table(table(AutoClaim$JOBCLASS)), ylim=c(0,0.7), main='JOB_CLASS', las=2, cex.names = 0.7, col=7)
ticks=seq(1, 10.2, length=9)
points(ticks, t(claim_freq), type='l', col=2)
points(ticks, t(claim_freq), col=2)
text(ticks, t(claim_freq)+0.03, round(t(claim_freq), 2))
legend(x = 3.5, y = 0.65, legend = c("Proportion of Observations", "Claim Frequency"),
       col = c(7, 2), lwd = 2, lty=c(1,1), pch=c(NA,1))

# re group
x <-z1
levels(x$JOBCLASS) <- c("Unknown", "Blue Collar", "Clerical", "Dr/Manager", "Home Maker", "Lawyer", "Dr/Manager", "Professional", "Student")

# barplot again after the grouping is done
x$CLM_FLAG = response_flag
n_observations = aggregate((x$CLM_FLAG), list(x$JOBCLASS), FUN=length) # length-- or count
n_claims = aggregate((x$CLM_FLAG=="Yes"), list(x$JOBCLASS), FUN=sum)
claim_freq = n_claims[,2]/n_observations[,2]
claim_freq
barplot(prop.table(table(x$JOBCLASS)), ylim=c(0,0.7), main='JOB_CLASS', las=2, cex.names = 0.7, col=7)
ticks=seq(.8, 9, length=8)
points(ticks, claim_freq, type='l', col=2)
points(ticks, t(claim_freq), col=2)
text(ticks, t(claim_freq)+0.03, round(t(claim_freq), 2))
legend(x = 3.5, y = 0.65, legend = c("Proportion of Observations", "Claim Frequency"),
       col = c(7, 2), lwd = 2, lty=c(1,1), pch=c(NA,1))

x <- x[,-dim(x)[2] ]
######################################### Group JobClass Doctor With Manager End ################################################


num_cols <- unlist(lapply(x, is.numeric)) #determine the numeric predictors
fact_cols <- unlist(lapply(x, is.factor))  # or lapply(x,  is.character):  #determine the non-numeric predictors
numCols = x[,num_cols] #sepearate out the numerical columns
factCols = x[,fact_cols] # same for factor  columns



#histograms of numerical variables
par(mfrow=c(3,5))
for (i in 1:dim(numCols)[2]) {
  hist(numCols[,i], main = " ", xlab = colnames( numCols)[i], col =i+1 )
}


#bar plots of categorical variables
par(mfrow=c(3,4))
for (i in 1:dim(factCols)[2]) {
  barplot(table( factCols[,i]), main = " ", xlab = colnames( factCols)[i], col =i+1 )
}

#boxplots plots of numerical variables
par(mfrow=c(3,5))
for (i in 1:dim(numCols)[2]) {
  boxplot(numCols[,i], main = " ", xlab = colnames( numCols)[i], col =i+1 )
}


# The following function counts the  number of outliers in a data. 
outliercount <- function(x)  {  
  length(boxplot.stats(x)$out)
}

# We then compute the number of outliers in each predictor, in a table.
outliertable <- apply(numCols, 2, outliercount ) #gives a table of the frequency 
                                                 # of outliers in a predictor 
par(mfrow= c(1,1))
tt <- barplot(outliertable, las =2, ylim = c(0,1350), 
              main = "Number of outliers", col = "purple")  # prints a barplot of 
                                                 # number of outliers in each column.
text(x = tt, y = outliertable, label = outliertable, 
                     pos = 3, cex = 0.8, col = "red")


##  Missing values analysis

library(naniar)
#which predictors have higher missing values? See plot below


gg_miss_var(x, show_pct = FALSE) # naniar package required 


vis_miss(x) # naniar package required 


gg_miss_upset(x) # to see the combination of interconnectedness of  
                  # missing variables among predictors 

x.temp <- cbind(x, response_flag)
gg_miss_case(x.temp,  x.temp$response_flag ) # to visualize how missing values relate to class.

sum(is.na(x$SAMEHOME))/length(x$SAMEHOME) # proportion of missing values in a column.

                                          # The column with highest missing values 
                                              # less than 7% missing values. 
                                              # So no need to remove the predictors


# We use knn imputation with k=5.

# During model building, we vary k to find which gives us the optimal model. 
# KNN Imputation with k=5.
imputation <- preProcess(x,method="knnImpute") ##need {caret} package
# Apply the transformations:
imputed <- predict(imputation, x)  

# undo center scale occasioned by imputation
#means = rowMeans(x, )
for(i in 1:dim(imputed)[2]){
  if (is.numeric(imputed[,i])){
    std = sd(x[,i], na.rm=TRUE)
    mean = mean(x[,i], na.rm=TRUE)
    imputed[,i] <- imputed[,i]*std+mean
  }
}

# fix the one value that is negative
imputed$SAMEHOME = abs(imputed$SAMEHOME)

dim(x)
dim(imputed)
 # 'imputed' is our updated data

xxx <- imputed


#Skewness  Before
par(mfrow=c(1,1))
skew <- round(apply(xxx[, num_cols], 2, skewness), 3)  #apply after imputation
ss <- barplot(skew, las =2, ylim = c(-1.2,4.0), main = "Skewness", col = "purple")
# Create bar plot with labels
abline(h=1, col = "red")  # cutoff line
abline(h=-1, col ="red")  # cutoff line
text(x = ss, y = skew, label = skew, pos = 3, cex = 0.8, col = "red")

# Creating dummy columns

xxx.dummies <- dummy_cols(xxx, colnames(factCols), 
                          remove_first_dummy = TRUE,  # leaves us with a 
                                                         # perfect correlation plot
                         remove_selected_columns = TRUE) 

dim(xxx.dummies)
dim(xxx)

###########################
#    We use xxx.dummies  ########

# Personal note: manual way create dummy variables for CAR_USE
#private  <- ifelse(factCols$CAR_USE == "Private", 1, 0)
#commercial <- ifelse(factCols$CAR_USE == "Commercial", 1, 0)


new.numeric <- xxx.dummies[,1:14]  # Numeric col's after imputation.


min(new.numeric+0.01)

# Boxcox
boxcox.num <- preProcess( new.numeric+0.01, method = "BoxCox")  ## need {caret} package
# the output
boxcoxed <-  predict(boxcox.num,new.numeric+1)



#before and after  plots
#############################

#### before boxcox

#histograms of numerical variables
#par(mfrow=c(3,5))
#for (i in 1:dim(numCols)[2]) {
#  hist(numCols[,i], main = " ", xlab = colnames( numCols)[i], col =i+1 )
#}
#### After boxcox

#par(mfrow=c(3,5))
#for (i in 1:dim(boxcoxed)[2]) {
#  hist(boxcoxed[,i], main = " ", xlab = colnames( boxcoxed)[i], col =i+1 )
#}
#####################

boxcoxed.xx <- cbind(boxcoxed, xxx.dummies[,-(1:14)]) # Merge the boxcoxed numeric 
                                                  # vars with the dummied vars


# Correlation plot
par(mfrow=c(1,1))
corrplot(cor(boxcoxed.xx),
      #  type="lower",
         order="hclust",
         tl.col="navy",
      #  addCoef.col="black",
         number.cex=1.0,
         tl.cex=0.8,
         title="Correlation Plot",
         # hide correlation on principal diagnal
      #  diag=FALSE,
         mar=c(0,0,1,0))

# Next we want to remove high correlations
highCorr <- findCorrelation(cor(boxcoxed.xx), cutoff = .80)    
length(highCorr)
highCorr
#filteredxx <- xx[, -c(highCorr)]  # tosses out CLM_FREQ5
#str(filteredxx)  
#str(boxcoxed.xx)

# near zero variances
#nzv<- nearZeroVar(filteredxx, freqCut = 95/5, uniqueCut = 10, saveMetrics = T); nzv
# The result affects only JOBCLASS_Doctor
#nzv.remove <- xx[-xxx.dummies$JOBCLASS_Doctor]    #this drops Doctors. But we retain doctors 
                                            # What is the significance of income
nzv<- nearZeroVar(boxcoxed.xx, freqCut = 95/5, uniqueCut = 10, saveMetrics = T); nzv

#Transformations 

#pca <- prcomp(filteredxx, scale = TRUE)
pca <- prcomp(boxcoxed.xx, scale = TRUE)

# compute total variance
variance = pca$sdev^2 / sum(pca$sdev^2)

variance

pcs = 1:length(boxcoxed.xx)
df = data.frame(pcs)
df$pecent_variance = variance
df$cumulative_percent = cumsum(variance)
df

qplot(c(1:length(boxcoxed.xx)), variance) +
  geom_line() +
  geom_point(size=4)+
  xlab("Principal Component") +
  ylab("Variance Explained") +
  ggtitle("Scree Plot") +
  ylim(0, 1)


pcbar <- barplot(df$cumulative_percent, las =2,  ylim = c(0,1.1), 
               main = "")  
text(x = pcbar, y = df$cumulative_percent, label = round(df$cumulative_percent,2), 
     pos = 3, cex = 0.5, col = "red", las =2)
abline(h=.95, col =2)
##############
#############
#trans <- preProcess(filteredxx, method = c( "center", "scale", "pca", "spatialSign"))
trans <- preProcess(boxcoxed.xx, method = c( "center", "scale", "pca", "spatialSign"))
trans

# the output
transformed <-  predict(trans, boxcoxed.xx) # our new variable after all 
                                            # necessary transformations.

# Next we check the various plots and compare the before's  with the after's.
transnumeric <- transformed[, 1:14]
#transfactor <- transformed[, fact_cols]

#Next, we do spatial sign transformation on the whole dataset. 

spatial.trans <- preProcess(transformed[,1:dim(boxcoxed.xx)[2]], method = "spatialSign")
spatial.out <- predict(spatial.trans, transformed[, 1:dim(boxcoxed.xx)[2] ])
#We also do  spatial sign transformation of the pca's

SpatialPca <- preProcess(transformed[,(dim(boxcoxed.xx)[2]+1) : (dim( transformed))[2]   ], 
                         method = "spatialSign")
spatialpca.out <- predict(SpatialPca, 
                          transformed[,(dim(boxcoxed.xx)[2]+1) : (dim( transformed))[2]   ] )
##############################################################
# Before and after  plots
##############################################################

#### Before boxcox, centering, scaling, spatial for numeric
par(mfrow=c(3,5))
for (i in 1:dim(numCols)[2]) {
  hist(numCols[,i], main = " ", xlab = colnames( numCols)[i], col =i+1 )
}


#### After boxcox
par(mfrow=c(3,5))
for (i in 1:dim(numCols)[2]) {
  hist(transnumeric[,i], main = " ", xlab = colnames(transnumeric)[i], col =i+1 )
}

#############################
### Boxplots before transformations
par(mfrow=c(3,5))
for (i in 1:dim(numCols)[2]) {
  boxplot(numCols[,i], main = " ", xlab = colnames( numCols)[i], col =i+1 )
}

#### After boxcox
par(mfrow=c(3,5))
for (i in 1:dim(transnumeric)[2]) {
  boxplot(transnumeric[,i], main = " ", xlab = colnames(transnumeric)[i], col =i+1 )
}



##############################
#Skewness  Before
par(mfrow=c(1,2))
skew <- round(apply(xxx[, num_cols], 2, skewness), 3)  #apply after imputation
ss <- barplot(skew, las =2, ylim = c(-1.2,4.0),  cex.names = 0.7, main = "Skewness Before", col = "purple")
# Create bar plot with labels
abline(h=1, col = "red")  # cutoff line
abline(h=-1, col ="red")  # cutoff line
text(x = ss, y = skew, label = skew, pos = 3, cex = 0.8, col = "red")

# After
skew2 <- round(apply(boxcoxed, 2, skewness), 3)  #apply after imputation
ss2 = barplot(skew2, las =2, cex.names = 0.7, main='skewness after', ylim=c(-1.2,4), col ="green")
# Create bar plot with labels
abline(h=1, col = "red")  # cutoff line
abline(h=-1, col ="red")  # cutoff line
text(x = ss, y = skew2, label = skew2, pos = 3, cex = 0.8, col = "red")

#############################

# Outlier counts before and after
par(mfrow= c(1,2))
# We then compute the number of outliers in each predictor, in a table.
outliertable <- apply(numCols, 2, outliercount ) #gives a table of the frequency 
                                                    # of outliers in a predictor 
tt <- barplot(outliertable, las =2, ylim = c(0,1350), main = "Number of outliers Before", col = "purple")  # prints a table of number of outliers in each column.
text(x = tt, y = outliertable, label = outliertable, pos = 3, cex = 0.8, col = "red")

outliertable2<- apply(transnumeric, 2, outliercount) #gives a table of the frequency of outliers in a predictor 
tt2 <- barplot(outliertable2, las =2,  ylim = c(0,1350),
               main = "Number of outliers After", col = "green")  # prints a table of number of outliers in each column.
text(x = tt2, y = outliertable2, label = outliertable2, pos = 3, cex = 0.8, col = "red")
##############

#####################
############################# Data Spending ########################################

# Our final cleaned data is ready, below

predictors <- spatial.out
predictors.pca <- spatialpca.out

# We now want to split out data into train-test. Stratified random 
#  sampling applies here.


# response_flag is our response variable. 
set.seed(75)
trainingRows <- createDataPartition(response_flag, p = .80, list= FALSE)  # requires caret
head(trainingRows)
nrow(trainingRows)
# Subset the data into objects for training using
# integer sub-setting

data <- xxx.dummies + 0.1
trainPredictors <- data[trainingRows, ] 
trainresponse <- response_flag[trainingRows]

# Do the same for the test set using negative integers.
testPredictors <- data[-trainingRows, ] 
testresponse <- response_flag[-trainingRows]

########################################################
############################     M  O  D  E  L  S   ##########################################################
########################################################
ctrl <- trainControl(method = "cv", number = 10, 
                     #summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

#### MODELS

############         1.   Support Vector Machine        #####################################
library(kernlab)


sigmaRangeReduced <- sigest(as.matrix(trainPredictors))
## sigest estimates the range of values for the sigma parameter
## which would return good results when used with a Support Vector Machine 
## ksvm). The estimation is based upon the 0.1 and 0.9 quantile 
## of ||x -x'||^2. Basically any value in between those two bounds 
## will produce good results.
svmRGrid <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-8, 8)))
library(doParallel)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
start <- Sys.time() 
svmRTune  <- train(x = trainPredictors,
                   y = trainresponse,
                   method = "svmRadial",
                   metric = "Kappa",
                   preProc = c("center", "scale"),    # , "BoxCox", "spatialSign"),
                   tuneGrid = svmRGrid,
                   fit = FALSE,
                   trControl = ctrl)

svmRTune
plot(svmRTune)
library(ggplot2)
ggplot(svmRTune)+coord_trans(x='log2')
end <-  Sys.time() 
duration.svm <- (as.numeric(end) - as.numeric(start)) / 60 # duration in minutes
duration.svm
predictedsvm <- predict(svmRTune, trainPredictors)
stopCluster(cl)

# Confusion matrix
c.svm <-  confusionMatrix(data =predictedsvm,
                reference = trainresponse)
roc.svm <- roc(response = svmRTune$pred$obs,
               predictor = svmRTune$pred$Yes,
                 #levels = rev(levels(svmRTune$pred$obs))
                            )
# plot(roc.svm, legacy.axes = TRUE)
auc.svm <- auc(roc.svm)
auc.svm
SVM <- c(c.svm$overall[1:2], c.svm$byClass[c(1:2,5:7) ], AUC= auc.svm) #  extracting some                                                        # important metrics
SVM
##########      2.     Random forest model           #####################################
set.seed(278)
library(randomForest)
library(doParallel)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)

start <- Sys.time() 
rf_random <- train(x = trainPredictors,
                   y = trainresponse,
                   method = 'rf',
                   metric = 'Kappa',
                   preProc = c("center", "scale" , "BoxCox", "spatialSign"),
                   tuneLength  = 15, 
                   trControl = ctrl)
print(rf_random)
plot(rf_random)
end <-  Sys.time() 
duration.rf <- (as.numeric(end) - as.numeric(start)) / 60 # duration in minutes
stopCluster(cl)
duration.rf
predictedrf <- predict(rf_random, trainPredictors)

c.rf <-  confusionMatrix(data =predictedrf,
                          reference = trainresponse)
roc.rf <- roc(response = rf_random$pred$obs,
               predictor = rf_random$pred$Yes)
# plot(roc.svm, legacy.axes = TRUE)
auc.rf <- auc(roc.rf)
auc.rf
random.forest <- c(c.rf$overall[1:2], c.rf$byClass[c(1:2,5:7)], AUC = auc.rf)
random.forest

############      3.      Neural network model        #####################################
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, 0.01,.1,0.5, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (  dim(trainPredictors)[2]   + 1) + (maxSize+1)*2) ## dim(trainPredictors)[2] is the number of predictors
set.seed(75)
start <-  Sys.time()
nnetFit <- train(x = trainPredictors,
                 y = trainresponse,
                 method = "nnet",
                 metric = "Kappa",
                 preProc = c("center", "scale",  "BoxCox", "spatialSign"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 trControl = ctrl)
nnetFit
plot(nnetFit)
end <-  Sys.time() 
duration.nn <- (as.numeric(end) - as.numeric(start)) / 60 # duration in minutes
duration.nn
stopCluster(cl)
#predict on training
predictednn <- predict(nnetFit, trainPredictors)
c.nn <-  confusionMatrix(data =predictednn,
                         reference = trainresponse)
roc.nn <- roc(response = nnetFit$pred$obs,
               predictor = nnetFit$pred$Yes)
# plot(roc.svm, legacy.axes = TRUE)
auc.nn <- auc(roc.nn)
auc.nn
NNetwork <- c(c.nn$overall[1:2], c.nn$byClass[c(1:2,5:7)], AUC = auc.nn)
NNetwork

############ 4a. Averaged Neural network model        #####################################
library(doParallel)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
#nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1, 2) , bag = TRUE)
nnetGrid <- expand.grid(.decay = c(0.001, .01, .1),
             .size = seq(1, 27, by = 2),
             .bag = FALSE)
#nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (  dim(trainPredictors)[2]   + 1) + (maxSize+1)*2) ## dim(trainPredictors)[2] is the number of predictors
set.seed(75)
start <-  Sys.time() 
avnnet <- train(x = trainPredictors,
                 y = trainresponse,
                 method = "avNNet",
                 metric = "Kappa",
                 preProc = c("center", "scale",  "BoxCox", "spatialSign"),
                 linout = TRUE,
                 trace = FALSE,
                 maxit = 1000,
                 trControl = ctrl)
avnnet


plot(avnnet)
end <-  Sys.time() 
duration.avnn <- (as.numeric(end) - as.numeric(start)) / 60 # duration in minutes
duration.avnn
stopCluster(cl)
predictedavnn <- predict(avnnet, trainPredictors)

c.avnn <-  confusionMatrix(data =predictedavnn,
                         reference = trainresponse)
averagedNN<- c(c.avnn$overall[1:2], c.avnn$byClass[c(1:2,5:7)])
averagedNN
#######


############ 5.   K-Nearest Neighbor model       #####################################
library(doParallel)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
set.seed(75)
start <-  Sys.time()
knntune <- train(x = trainPredictors,
                 y = trainresponse,
                 method = "knn",
                 metric = "Kappa",
                 preProc = c("center", "scale"  ,  "BoxCox", "spatialSign"),
                 tuneLength = 30,
                 trControl = ctrl)
knntune
plot(knntune)
end <-  Sys.time() 
duration.knn <- (as.numeric(end) - as.numeric(start)) / 60 # duration in minutes
duration.knn
stopCluster(cl)
# predict on train
predictedknn <- predict(knntune, trainPredictors)
c.knn <- confusionMatrix(data =predictedknn,
                         reference = trainresponse)
roc.knn <- roc(response = knntune$pred$obs,
              predictor = knntune$pred$Yes)
auc.knn <- auc(roc.knn)
auc.knn
KNN <- c(c.knn$overall[1:2], c.knn$byClass[c(1:2,5:7)], AUC = auc.knn)
KNN




############################### 6.  L D A  Model      ######################################
library(doParallel)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
set.seed(75)
start <-  Sys.time()
lda  <- train(x = trainPredictors,
               y = trainresponse,
               method =  "lda",
               metric = "Kappa",
              preProc = c("center", "scale" ),
                 trControl = ctrl)
lda
end <-  Sys.time() 
duration.lda <- (as.numeric(end) - as.numeric(start)) / 60 # duration in minutes
duration.lda
stopCluster(cl)
# predict on train
predictedlda <- predict(lda, trainPredictors)
c.lda <- confusionMatrix(data =predictedlda,
                         reference = trainresponse)
roc.lda <- roc(response = lda$pred$obs,
               predictor = lda$pred$Yes)
auc.lda <- auc(roc.lda)
auc.lda
LDA <- c(c.lda$overall[1:2], c.lda$byClass[c(1:2,5:7)], AUC = auc.lda)
LDA



###############################   7.  Q D A  Model      ######################################
library(doParallel)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
set.seed(75)
start <-  Sys.time()
qda  <- train(x = trainPredictors,
              y = trainresponse,
              method =  "qda",
              metric = "Kappa",
              preProc = c("center", "scale" ),
              trControl = ctrl)
qda
end <-  Sys.time() 
duration.qda <- (as.numeric(end) - as.numeric(start)) / 60 # duration in minutes
duration.qda
stopCluster(cl)
# predict on train
predictedqda <- predict(qda, trainPredictors)
c.qda <- confusionMatrix(data =predictedqda,
                         reference = trainresponse)
roc.qda <- roc(response = qda$pred$obs,
               predictor = qda$pred$Yes)
auc.qda <- auc(roc.qda)
auc.qda
QDA <- c(c.qda$overall[1:2], c.qda$byClass[c(1:2,5:7)], AUC = auc.qda)
QDA


############################### 8.  M D A  Model      ######################################
library(doParallel)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
set.seed(75)
start <-  Sys.time()
mda  <- train(x = trainPredictors,
              y = trainresponse,
              method =  "mda",
              metric = "Kappa",
              preProc = c("center", "scale" ), # ,  "BoxCox", "spatialSign"),
              tuneGrid = expand.grid(.subclasses = 1:6),
              trControl = ctrl)
mda
plot(mda)
end <-  Sys.time() 
duration.mda <- (as.numeric(end) - as.numeric(start)) / 60 # duration in minutes
duration.mda
stopCluster(cl)
# predict on train
predictedmda <- predict(mda, trainPredictors)
c.mda <- confusionMatrix(data =predictedmda,
                         reference = trainresponse)
roc.mda <- roc(response = mda$pred$obs,
               predictor = mda$pred$Yes)
auc.mda <- auc(roc.mda)
auc.mda
MDA <- c(c.mda$overall[1:2], c.mda$byClass[c(1:2,5:7)], AUC = auc.mda)
MDA

############################### 9.  F D A  Model      ######################################
library(pamr)
library(doParallel)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
set.seed(75)
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:46)
start <-  Sys.time()
fda  <- train(x = trainPredictors,
              y = trainresponse,
              method =  "fda",
              metric = "Kappa",
              preProc = c("center", "scale" ), # ,  "BoxCox", "spatialSign"),
              tuneGrid = marsGrid,
              trControl = ctrl)
fda
plot(fda)
end <-  Sys.time() 
duration.fda <- (as.numeric(end) - as.numeric(start)) / 60 # duration in minutes
duration.fda
stopCluster(cl)
# predict on train
predictedfda <- predict(fda, trainPredictors)
c.fda <- confusionMatrix(data =predictedfda,
                         reference = trainresponse)
roc.fda <- roc(response = fda$pred$obs,
               predictor = fda$pred$Yes)
auc.fda <- auc(roc.fda)
auc.fda
FDA <- c(c.fda$overall[1:2], c.fda$byClass[c(1:2,5:7)], AUC = auc.fda)
FDA

############################### 10.  P L S D A  Model      ######################################
library(klaR)
library(doParallel)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
set.seed(75)
start <-  Sys.time()
### PLSDA Model

set.seed(647)
plsda <- train(x = trainPredictors,
               y = trainresponse,
               method = "pls",
               tuneGrid = expand.grid(.ncomp = 1:15),
               preProc = c( "center","scale" ,"BoxCox", "spatialSign"),
               metric = "Kappa",
               trControl = ctrl)
plsda
plot(plsda)
# predict on train
predictedplsda <- predict(plsda, trainPredictors)
c.plsda <- confusionMatrix(data =predictedplsda,
                         reference = trainresponse)
roc.plsda <- roc(response = plsda$pred$obs,
               predictor = plsda$pred$Yes)
auc.plsda <- auc(roc.plsda)
auc.plsda
PLSDA <- c(c.plsda$overall[1:2], c.plsda$byClass[c(1:2,5:7)], AUC= auc.plsda)
PLSDA

############### 11.    Logistic Model       #####################################
library(doParallel)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
set.seed(75)
start <-  Sys.time()
logistic.glm <- train(trainPredictors,
                  trainresponse,
                  method = "glm",
                  metric = "Kappa",
                  preProcess = c( "BoxCox", "center", "scale", "spatialSign" ),
                  trControl = ctrl)
logistic.glm
summary(logistic.glm)
end <-  Sys.time() 
duration.nb <- (as.numeric(end) - as.numeric(start)) / 60 # duration in minutes
duration.nb
stopCluster(cl)
predictedglm <- predict(logistic.glm, trainPredictors)
c.glm <- confusionMatrix(data =predictedglm,
                        reference = trainresponse)
roc.glm <- roc(response = logistic.glm$pred$obs,
               predictor = logistic.glm$pred$Yes)
auc.glm <- auc(roc.glm)
auc.glm
logistic <- c(c.glm$overall[1:2], c.glm$byClass[c(1:2,5:7)], AUC = auc.glm)
logistic

############### 12. Penalized Model      ############################################
metric = "Kappa"
disp_metric = c("Kappa", "Accuracy")
preProc = c("center", "scale" , "BoxCox", "spatialSign")
preProc2 = c("center", "scale")

# Penalized
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.0, .2, length = 10))

glmnTuned <- train(x=trainX,
                   y = trainy,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = preProc,
                   metric = metric,
                   trControl = ctrl)
res = glmnTuned$results
glmnTuned$bestTune
res[c("alpha", "lambda", disp_metric)]
plot(glmnTuned)
library(pROC)
pred <- predict(glmnTuned, trainX)
pred.prob <- predict(glmnTuned, trainX, type = "prob")
c.glmnet <- confusionMatrix(data =pred,
                            reference = trainy)
ROC = roc(response = trainy ,predictor = pred.prob[,2])
Penalized <- c(c.glmnet$overall[1:2], c.glmnet$byClass[c(1:2,5:7)], "ROC"=auc(ROC))
Penalized

################13. Random GLM  ###############################

# Random GLM
ctrl_rglm <- trainControl(method = "cv", number=10,
                          summaryFunction = multiClassSummary,
                          classProbs = TRUE,
                          savePredictions = TRUE)
tunegrid <- expand.grid(maxInteractionOrder=c(1,2))
random_glm <- train(x=trainX,
                    y = trainy,
                    method="randomGLM", 
                    metric=metric,
                    preProc=preProc2,
                    tuneGrid=tunegrid,
                    trControl=ctrl_rglm)
random_glm
res <- random_glm$results
res[,c("maxInteractionOrder", disp_metric)]
plot(random_glm)
pred = predict(random_glm, testX)
vals <- data.frame(obs = testy, pred = pred)
dfSummaray = defaultSummary(vals)
dfSummaray
pred = predict(random_glm, trainX)
pred.prob <- predict(random_glm, trainX, type = "prob")
pred <- ifelse(pred.prob[,1]<0.5, "Yes", "No")
c <- confusionMatrix(data =as.factor(pred), reference = trainy)
ROC = roc(response = trainy ,predictor = pred.prob[,2])
random_glm.stats <- c(c$overall[1:2], c$byClass[c(1:2,5:7)], "ROC"=auc(ROC))
random_glm.stats



################   14. Naive Bayes #############################

nb <- train(x=trainX,
            y = trainy,
            method="naive_bayes", 
            metric=metric,
            preProc=preProc,
            tuneGrid = data.frame(laplace=1, usekernel=TRUE, adjust=1),
            trControl=ctrl)
nb
pred <- predict(nb, testX)
vals <- data.frame(obs = testy, pred = pred)
dfSummaray = defaultSummary(vals)
dfSummaray
res <- nb$results
res[, disp_metric]
pred = predict(nb, trainX)
pred.prob <- predict(nb, trainX, type = "prob")
c <- confusionMatrix(data =pred, reference = trainy)
ROC = roc(response = trainy ,predictor = pred.prob[,2])
NaiveBayes <- c(c$overall[1:2], c$byClass[c(1:2,5:7)], "ROC"=auc(ROC))
NaiveBayes



#########################################
###########           Comparison Table 
#########################################
# Comparison table
final.metric <- round( rbind(logistic, LDA, QDA, MDA,PLSDA, SVM, KNN, FDA,NNetwork),3)
penals <- round( c( 0.7893033,   0.3977279,   0.9158285,   0.4403893,   0.8186120,   0.9158285,   0.8644957,   0.8160756),3)
penalized <- setNames(penals, colnames(final.metric)) 
rand.glm <- round( c( 0.7887853,   0.3916121,   0.9193577,   0.4287105,   0.8161028,   0.9193577,   0.8646585,   0.8157560),3)
random_glm <- setNames(rand.glm, colnames(final.metric)) 

nb <- round( c(0.7160062,   0.2559735,   0.8194812,   0.4306569,   0.7987616,   0.8194812,   0.8089888,   0.7372125),3)
NaiveBayes <- setNames(nb, colnames(final.metric)) 


Final.metric <- round( rbind( logistic,penalized, LDA, QDA, MDA,PLSDA, random_glm, NaiveBayes, KNN, SVM, FDA,NNetwork),3)
Final.metric2 <- Final.metric[,-6]
m = dim(Final.metric)[1]
barplot(Final.metric2, xlab = "Performance Profiles of Models",
        main = "Performance Metrics",
        col = rainbow(m),
        beside = TRUE  , xlim = c(0,100),
        legend.text = rownames(final.metric),
        args.legend = list(title = "Model", x = "topright",
                          inset = c(-0.05, 0)), ylim = c(0,1.1)
        )



#Final.metric <- as.data.frame(final.metric)
#ggplot(data = Final.metric) +
#  geom_bar(
#    mapping = aes(x= col.names(Final.metric)), #, fill= colna  ),
#    position = "dodge"
#    )

####

# Best Two Models: Neural network
nnetFit
summary(nnetFit)
# on test test
testpredictednn <- predict(nnetFit, testPredictors)
# Confusion matrix
test_c.nn <-  confusionMatrix(data =testpredictednn,
                               reference = testresponse)
test_c.nn

testpredictedfda <- predict(fda, testPredictors)
# Confusion matrix

test_c.fda <-  confusionMatrix(data =testpredictedfda,
                              reference = testresponse)
test_c.fda


# Important Variables
plot(varImp(nnetFit), 15)


## Apendix: TEst Case With Alan
alan = data.frame("CLM_FREQ5"=0,    # How many accidents in the last 5 years
                  "CLM_AMT5"=0,     # Total cost of accidents in the last 5 years
                  "KIDSDRIV"=0,     # Number of kids who drive
                  "TRAVTIME"=10,    # Commute time to work
                  "BLUEBOOK"=4171,  # Current car price (can google)
                  "RETAINED"=8 ,    # How many years you have been with the insurance company on this policy
                  "NPOLICY"= 2,     # Number of policies
                  "MVR_PTS"=0,      # You can guess or buy this info online for $12 (plus some fees)
                  "AGE"=24,         # self explanatory
                  "HOMEKIDS"=0,     # number of kids that live with you
                  "YOJ"=4,          # years at your current job
                  "INCOME"=,        # annual income
                  "HOME_VAL"=0.0,   # value of your home (I rent so I put 0)
                  "SAMEHOME"=3,     # How many years you have lived in your house
                  ## The rest of the columns are for dummy vars and are either 0 or 1. Most are self explanatory
                  "CAR_USE_Commercial"=0, # the other option is private (so put zero unless you use your car commercially)
                  "CAR_TYPE_Pickup"=0, 
                  "CAR_TYPE_Sedan"=1,
                  "CAR_TYPE_Sports Car"=0,
                  "CAR_TYPE_SUV"=0,
                  "CAR_TYPE_Van"=0,
                  "RED_CAR_yes"=1,
                  "REVOLKED_Yes"=0, # 1 if your driver's license has ever been revoked
                  "GENDER_M"=1,
                  "MARRIED_Yes"=0,
                  "PARENT1_Yes"=0, # 1 if you are a single parent
                  "JOBCLASS_Blue Collar"=0,
                  "JOBCLASS_Clerical"=0,
                  "JOBCLASS_Dr/Manager"=0,
                  "JOBCLASS_Home Maker"=0,
                  "JOBCLASS_Lawyer"=0,
                  "JOBCLASS_Professional"=0,
                  "JOBCLASS_Student"=1,
                  "MAX_EDUC_Bachelors"=1,
                  "MAX_EDUC_High School"=0,
                  "MAX_EDUC_Masters"=0,
                  "MAX_EDUC_PhD"=0,  
                  "AREA_Urban"=0 )  # other option is rural. I think that's what houghton counts as

# I must've messed up some of the names
names(alan) <- names(testX)

dim(alan)



#### T H E    E N D!
