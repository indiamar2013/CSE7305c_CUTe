
  
#clear all environment and session variables
rm(list = ls(all = TRUE))


#Libraries

library(tidyverse)
library(caret)
library(DMwR)

## Reading data

getRawData <- function(path="", 
                       data_file="train.csv", 
                       target="", sep = ',',
                       header = TRUE) {
  if(!file.exists(data_file)) {
    return(NULL)    
  }
  setwd(path)
  raw_data <- read.csv(file = data_file, header = header, 
                       sep = sep, colClasses = 'numeric')
  
  if (!(target == "")) {
    raw_data[,colnames(raw_data) %in% target] <- as.factor(as.character(raw_data[, colnames(raw_data) %in% target]))
  }
  return(raw_data)
}

getPreProcessedData <- function(path=getwd(), 
                                data_file="train.csv", 
                                target="target", 
                                sep = ',', 
                                header = TRUE, 
                                na_threshold = 0.3,
                                zero_threshold = 0.5) {
  setwd(path)
  preproc_file <- paste('preproc_', data_file, sep="")
  
  if (file.exists(preproc_file)) {
    data <- read.csv(preproc_file)
    if (!class(data[, colnames(data) %in% target]) == 'factor') {
      data[,colnames(data) %in% target] <- as.factor(as.character(data[,colnames(data) %in% target]))
    }
    return(data)
  }
  #else
  raw_data = getRawData(path, data_file, target, sep, header)
  #remove empty columns
  emptycols <- colnames(raw_data[,sapply(raw_data[,!colnames(raw_data) %in% target], function(x) { all(is.na(x))})])
  preproc_data <- raw_data[, !colnames(raw_data) %in% emptycols]
  rm(emptycols, raw_data)
  
  #remove the redundant id column
  rownames(preproc_data) <- preproc_data$ID
  preproc_data$ID <- NULL
  
  #all the checks and standardization needs to be done on the independent variables.
  x_preproc <- preproc_data[,!colnames(preproc_data) %in% target]
  
  #remove columns with na beyond threshold
  colswithna <- sort(colSums(is.na(x_preproc)), decreasing = TRUE)
  colstoremove <- colswithna[colswithna > nrow(x_preproc)*na_threshold]
  rm(colswithna)
  #sparse columns
  sparsecols <- sort(colSums(x_preproc) == 0, decreasing = TRUE)
  append(colstoremove, sparsecols[sparsecols > round(nrow(x_preproc) * zero_threshold)])
  rm(sparsecols)
  constant_cols <- colnames(preproc_data[ , sapply(x_preproc, function(v){ var(v, na.rm=TRUE)==0})])
  append(colstoremove, constant_cols)
  rm(constant_cols)
  colstoremove
  #now we set up the data frame with just the columns  
  preproc_data <- preproc_data[, !colnames(x_preproc) %in% names(colstoremove)]
  
  write.csv(x = preproc_data, file = preproc_file, row.names = FALSE)
  return(preproc_data)
}


standardizeData <- function(preproc_data=data.frame(), target='target', for_train=TRUE) {
  train_std <- data.frame()
  test_std <- data.frame()
  filename_prefix <- deparse(substitute(preproc_data))
  print(filename_prefix)
  
  if (for_train==TRUE) {
    train_filename <- paste(filename_prefix, '_train_std.csv', sep="") 
    test_filename <- paste(filename_prefix, '_test_std.csv', sep="")
    print(train_filename)
    print(test_filename)
    if (file.exists(train_filename) & file.exists(test_filename)) {
      train_std <- read.csv(train_filename)
      if (!class(train_std[, colnames(train_std) %in% target]) == 'factor') {
        train_std[,colnames(train_std) %in% target] <- as.factor(as.character(train_std[,colnames(train_std) %in% target]))
      }
      test_std <- read.csv(test_filename)
      if (!class(test_std[, colnames(test_std) %in% target]) == 'factor') {
        test_std[,colnames(test_std) %in% target] <- as.factor(as.character(test_std[,colnames(test_std) %in% target]))
      }
    } else {
      #else
      library(RANN)
    
      #split into train and test
      #will use caret package
      train_rows <- createDataPartition(preproc_data$target, p = 0.7, list = FALSE)
      train_data <- preproc_data[train_rows,]
      test_data <- preproc_data[-train_rows,]
      prop.table(table(preproc_data$target))
      prop.table(table(train_data$target))
      prop.table(table(test_data$target))
      #remove original data to conserve memory
    
      rm(train_rows)
      
      #standardize
      set.seed(1234)
      
      #first impute only

      std_preds <- preProcess(x = train_data[,!colnames(train_data) %in% target], 
                              method = c("center", "scale", "knnImpute"))
      
      train_std <- predict(std_preds, train_data)
      test_std <- predict(std_preds, test_data)
      
      # std_preds <- preProcess(x = train_data[,!colnames(train_data) %in% target], 
      #                         method = c("center", "scale"))
      # train_std <- predict(std_preds, train_data)
      # test_std <- predict(std_preds, test_data)

      saveRDS(std_preds, file='std_preds.RDS')
      
      #the imputation takes a lot of time, so I will write it to csv in case i need
      #to resume later.
      write.csv(train_std, train_filename, row.names = FALSE)
      write.csv(test_std, test_filename, row.names = FALSE)
    }
    retlist <- list(train_std, test_std)
    return(retlist)
  } else {
    unseen_filename <- paste(filename_prefix, '_std.csv', sep="")
    print(unseen_filename)
    unseen_std <- data.frame()
    if(file.exists(unseen_filename)) {
      unseen_std <- read.csv(unseen_filename, colClass='numeric')
    } else {
      std_preds <- readRDS(file = 'std_preds.RDS')
      unseen_std <- predict(std_preds, preproc_data)
      write.csv(x = unseen_std, file = unseen_filename, row.names = FALSE)
    }
    return(unseen_std)
    
  }
}

savePredictions <- function(predictions, name='') {
  unseen_raw <- read.csv(file = 'test.csv', header = TRUE)
  submission <- data.frame('ID'=unseen_raw$ID, 'prediction'=predictions)
  rownames(submission) <- c(1:nrow(submission))
  rownames(submission)
  write.csv(x = submission, file = paste('submission',name,'.csv',sep=""))
  write.csv(x = submission, file = 'submission.csv')
}

genplot <- function(data=data.frame(), y='y') {
  require(reshape2)
  require(gridExtra)
  df <- melt(data, id.vars = y, variable.name = 'series')
  df
  plt <- ggplot(df, aes(y,value)) + geom_point(aes(colour=series)) + facet_grid(series ~ .)
  return(plt)
}

genboxplot <- function(data=data.frame) {
  require(reshape2)
  require(gridExtra)
  numericcols <- data[,sapply(data, class) == 'numeric']
  plt <- ggplot(stack(data), aes(x = ind, y = values)) +
    geom_boxplot(aes(color=values))
  return(plt)
}


setwd('~/datasets/CSE7305c_CUTe')
fin_data <- getPreProcessedData(path = getwd(), 
                                data_file = 'train.csv', 
                                target = 'target', sep = ',', 
                                header = TRUE, 
                                na_threshold = 0.3, 
                                zero_threshold = 0.5)

plt_df <- fin_data[,!colnames(fin_data) %in% c("target")]
numplts <- 5
numvars <- ncol(plt_df)
numiters <- round(numvars/numplts)
if (numvars > numiters*numplts) {
  numiters <- numiters + 1
}
numiters
ncol(plt_df)

# for (i in 1:numiters) {
#   startcol <- (((i-1)*numplts)+1)
#   ifelse (i == numiters, endcol <- ncol(plt_df), endcol <- i*numplts)
#   pltdf <- plt_df[, startcol: endcol]
#   pltdf$target <- train_data$target
#   print(genplot(data = pltdf, y='target'))
# }


for (i in 1:numiters) {
  startcol <- (((i-1)*numplts)+1)
  ifelse (i == numiters, endcol <- ncol(plt_df), endcol <- i*numplts)
  pltdf <- plt_df[, startcol: endcol]
  print(genboxplot(pltdf))
}
rm(pltdf, numiters, i, numplts, numvars, startcol, plt_df)
rm(endcol)
summary(train_data)



require(ggcorrplot)

cors <- cor(train_data[,!colnames(train_data) %in% c("target")])

ggcorrplot(cors, hc.order=TRUE, type = "upper", insig = "blank")


#standardize the data
stdlist <- standardizeData(fin_data, target = 'target')
train_std <- data.frame(stdlist[1])
test_std <- data.frame(stdlist[2])

## Model Building

#First an SVM model - we need to determine optimal C, so will use train() from caret package
library(e1071)
svm_mdl1 <- svm(target ~ ., data = train_std, kernel = "linear")
summary(svm_mdl1)

#NOTE: This completed with a warning that max iterations were reached.
#Trying with PCA to reduce #dimensions

##PCA
set.seed(123)
train_pca <- prcomp(x = train_std[,!colnames(train_std) %in% 'target'])
summary(train_pca)
#This shows that first 25 components can account for approximately 98.5% of variance in the dataset.
# So building an SVM Model with this dataset

train_pca_data <- data.frame(train_pca$x[,1:25], target=train_std$target)
test_std_pca <- predict(train_pca, newdata = test_std)
test_std_pca <- as.data.frame(test_std_pca[,1:25])


svm_mdl2 <- svm(target ~ ., data = train_pca_data, kernel = "linear")

summary(svm_mdl2)
#let's check how good this model is

#need to first convert test_std to PCA
set.seed(123)
pca_preds <- predict(object = svm_mdl2, newdata = test_std_pca)
summary(pca_preds)

confusionMatrix(pca_preds, test_std$target)

svm_mdl3 <- svm(target ~ ., data = train_pca_data, kernel = "polynomial")


svm_mdl3_preds <- predict(object = svm_mdl3, newdata = test_std_pca)
confusionMatrix(svm_mdl3_preds, test_std$target)
unseen_std_pca <- predict(train_pca, unseen_std)
unseen_std_pca <- unseen_std_pca[,1:25]



saveModelTestMetrics <- function(modelname='', confmat) {
  print(confmat)  
}

predictModel <- function(model_name = 'mdl', model, test_data, unseen_data) {
  test_preds <- predict(model, test_data)
  saveModelTestMetrics(model_name, confusionMatrix(test_preds, test_data$target, mode = 'everything'))
  unseen_preds <- predict(model, unseen_data)
  savePredictions(unseen_preds, name=model_name)
  table(unseen_preds)
}

loadModel <- function(model_name='svm_mdl') {
  
  rdsFile <- paste(model_name, ".RDS", sep="")
  print(rdsFile)
  if (file.exists(rdsFile)) {
    mdl <- readRDS(rdsFile)
    return(mdl)
  } else {
    return(NULL)
  }
}

saveModel <- function(model_name='svm_mdl', model) {
  saveRDS(object = model, file = paste(model_name, '.RDS', sep=""))
}

#tried the following

#cost = 100, 500, 1000, 0.1, 0.01, 0.001. Best was with cost=400
svm_mdl <- loadModel(model_name = 'svm_std_mdl')
if (is.null(svm_mdl)) {
  set.seed(123)
  svm_mdl <- svm(target ~ ., data = train_std, kernel = 'sigmoid', cost=400)
  saveModel(model_name = 'svm_std_mdl', svm_mdl)
}
predictModel(model_name = 'svm_std_mdl', svm_mdl, test_std, unseen_std)


# 
# 
# svm_mdl5 <- svm(target ~ ., data = train_std, kernel = 'sigmoid', cost=1000)
# summary(svm_mdl5)
# mdl5_test_preds <- predict(svm_mdl5, test_std[,!colnames(test_std) %in% 'target'])
# 
# confusionMatrix(mdl5_test_preds, test_std$target)
# 
# unseen_mdl5_preds <- predict(svm_mdl5, unseen_std)
# table(unseen_preds)
# savePredictions(unseen_preds)




# library(ROCR)
# mdl5_preds <- predict(object = svm_mdl5, type="response")
# class(mdl5_preds)
# table(mdl5_preds)
# mdl5_train_pred <- prediction(as.numeric(mdl5_preds), train_std$target)
# # The performance() function from the ROCR package helps us extract metrics such as True positive rate, False positive rate etc. from the prediction object, we created above.
# vol_perf1 <- performance(prediction.obj = mdl5_train_pred, measure = "tpr", x.measure = "fpr")
# plot(vol_perf1, col = rainbow(10), colorize=T, print.cutoffs.at=(seq(0,1,0.05)))
# 
# 
# ctrl <- trainControl(method="repeatedcv",repeats = 3)
# knnFit <- train(target ~ ., 
#                 data = train_std, 
#                 method = "knn", 
#                 trControl = ctrl)
# knnFit
# 
# plot(knnFit)
# 
# knn_preds <- predict(knnFit, newdata = test_std)
# 
# confusionMatrix(knn_preds, test_std$target)
# unseen_preds <- predict(knnFit, newdata = unseen_std)
# 
# table(unseen_preds)
# savePredictions(unseen_preds)
# 

compensateSample <- function(train_std, method='both') {
  if(!require(ROSE)) {
    install.packages('ROSE')
    require(ROSE)
  }
  # if(method == 'over') {
  #   N <- nrow(train_std)
  # } else {
  #   N <- nrow(train_std)
  # }
  N <- nrow(train_std)
  return(ovun.sample(target ~ ., data = train_std, method = method, N = N, seed = 1234)$data)
}

balanced_train_std <- compensateSample(train_std, method='both')
table(balanced_train_std$target)

set.seed(123)

mdl <- loadModel(model_name = 'svm_bal_std_mdl')
if (is.null(mdl)) {
  set.seed(123)
  mdl <- svm(target ~ ., data = balanced_train_std, kernel = 'sigmoid', cost=400)
  saveModel(model_name = 'svm_bal_std_mdl', mdl)
}
predictModel(model_name = 'svm_bal_std_mdl', mdl, test_std, unseen_std)


#RPART
mdl <- loadModel(model_name = 'rpart_std_mdl')
if (is.null(mdl)) {
  require(rpart)
  set.seed(123)
  trainCtrl <- trainControl(method = 'repeatedcv', number = 4, repeats = 2)
  rpart_grid <-expand.grid(data.frame(.cp = c(0.005, 0.004, 0.003, 0.002, 0.001)))
  mdl <- train(target~., data = train_std, method='rpart', trControl = trainCtrl, tuneGrid = rpart_grid)
  saveModel(model_name = 'rpart_std_mdl', mdl)
}
predictModel(model_name = 'rpart_std_mdl', mdl, test_std, unseen_std)
##This model got F1-stat of 16.5% on Grader
plot(mdl)
plot(mdl$finalModel)


##C5.0
mdl <- loadModel(model_name = 'C50_std_mdl')
if (is.null(mdl)) {
  require(C50)
  set.seed(123)
  mdl <- C5.0(formula=target~., data = train_std)
  saveModel(model_name = 'C50_std_mdl', mdl)
}
predictModel(model_name = 'C50_std_mdl', mdl, test_std, unseen_std)
#This model got score of 28% on grader

#C5.0 on balanced data
mdl <- loadModel(model_name = 'C50_bal_std_mdl')
if (is.null(mdl)) {
  require(C50)
  set.seed(123)
  mdl <- C5.0(formula=target~., data = balanced_train_std)
  saveModel(model_name = 'C50_bal_std_mdl', mdl)
}
predictModel(model_name = 'C50_bal_std_mdl', mdl, test_std, unseen_std)
#this model got a score of 28%

##Random Forest
mdl <- loadModel(model_name = 'random_forest_std_mdl')
if (is.null(mdl)) {
  require(randomForest)
  set.seed(123)
  mdl <- randomForest(formula = target ~ ., data = train_std, keep.forest = TRUE, ntree = 500)
  saveModel(model_name = 'random_forest_std_mdl', mdl)
}
predictModel(model_name = 'random_forest_std_mdl', mdl, test_std, unseen_std)
#this got a score of 9%



##Random Forest with balanced data-set
mdl <- loadModel(model_name = 'random_forest_bal_std_mdl')
if (is.null(mdl)) {
  require(randomForest)
  set.seed(123)
  mdl <- randomForest(formula = target ~ ., data = balanced_train_std, keep.forest = TRUE, ntree = 500)
  saveModel(model_name = 'random_forest_bal_std_mdl', mdl)
}
predictModel(model_name = 'random_forest_bal_std_mdl', mdl, test_std, unseen_std)
#this model gave a score of 29%



##XG BOOST
mdl <- loadModel(model_name = 'xgboost_std_mdl')
if(is.null(mdl)) {
  set.seed(123)
  sampling_strategy<-trainControl(method = "repeatedcv",number = 2,repeats = 2,verboseIter = T,allowParallel = T)
  param_grid <- expand.grid(.nrounds = 250, .max_depth = c(2:6), .eta = c(0.1,0.11,0.09),
                            .gamma = c(0.6, 0.3), .colsample_bytree = c(0.6),
                            .min_child_weight = 1, .subsample = c(0.5, 0.6))
  
  mdl <- train(x = train_std[ , !(names(train_std) %in% c("target"))], 
                           y = train_std$target, 
                           method = "xgbTree",
                           trControl = sampling_strategy,
                           tuneGrid = param_grid)
  saveModel(model_name = 'xgboost_std_mdl', mdl)
}
predictModel(model_name = 'xgboost_std_mdl', mdl, test_std, unseen_std)
#this gave a score of 38.96%


#what about xgb with oversampled dataset?
mdl <- loadModel(model_name = 'xgboost_bal_std_mdl')
if(is.null(mdl)) {
  set.seed(123)
  balanced_train_std <- compensateSample(train_std, method='both')
  sampling_strategy<-trainControl(method = "repeatedcv",number = 2,repeats = 2,verboseIter = T,allowParallel = T)
  param_grid <- expand.grid(.nrounds = 250, .max_depth = c(2:6), .eta = c(0.1,0.11,0.09),
                            .gamma = c(0.6, 0.3), .colsample_bytree = c(0.6),
                            .min_child_weight = 1, .subsample = c(0.5, 0.6))
  
  mdl <- train(x = balanced_train_std[ , !(names(train_std) %in% c("target"))], 
               y = balanced_train_std$target, 
               method = "xgbTree",
               trControl = sampling_strategy,
               tuneGrid = param_grid)
  saveModel(model_name = 'xgboost_bal_std_mdl', mdl)
}
predictModel(model_name = 'xgboost_bal_std_mdl', mdl, test_std, unseen_std)
#this model got a score of 47%

##Stacking
svm_mdl <- loadModel(model_name = 'svm_std_mdl')
rpart_mdl <- loadModel(model_name = 'rpart_std_mdl')
c50_mdl <- loadModel(model_name = 'C50_std_mdl')
xgb_std_mdl <- loadModel(model_name = 'xgboost_std_mdl')
xgb_bal_std_mdl <- loadModel(model_name = 'xgboost_bal_std_mdl')

svm_train_preds <- predict(svm_mdl, train_std)
rpart_train_preds <- predict(rpart_mdl, train_std)
c50_train_preds <- predict(c50_mdl, train_std)
xgb_train_preds <- predict(xgb_std_mdl, train_std)
xgb_bal_train_preds <- predict(xgb_bal_std_mdl, balanced_train_std)
ensemble_train_data <- data.frame(SVM = svm_train_preds, 
                               RPART = rpart_train_preds,
                               C50 = c50_train_preds,
                               XGB = xgb_train_preds,
                               XGB_BAL = xgb_bal_train_preds)
ensemble_train_data <- data.frame(sapply(ensemble_train_data, as.factor))
ensemble_train_data <- cbind(ensemble_train_data, target = train_std$target)
str(ensemble_train_data)
ensemble_model = glm(target ~ ., ensemble_train_data, family = binomial)
ensemble_train_preds <- predict(ensemble_model, ensemble_train_data, type = 'response')
ensemble_train_preds <- ifelse(ensemble_train_preds > 0.5, 1, 0)
confusionMatrix(ensemble_train_preds, train_std$target)
table(ensemble_train_preds)

##Now on unseen data
svm_unseen_preds <- predict(svm_mdl, unseen_std)
rpart_unseen_preds <- predict(rpart_mdl, unseen_std)
c50_unseen_preds <- predict(c50_mdl, unseen_std)
xgb_unseen_preds <- predict(xgb_std_mdl, unseen_std)
xgb_bal_unseen_preds <- predict(xgb_bal_std_mdl, unseen_std)
ensemble_unseen_data <- data.frame(SVM = svm_unseen_preds, 
                                  RPART = rpart_unseen_preds,
                                  C50 = c50_unseen_preds,
                                  XGB = xgb_unseen_preds,
                                  XGB_BAL = xgb_bal_unseen_preds)
ensemble_unseen_data <- data.frame(sapply(ensemble_unseen_data, as.factor))
ensemble_unseen_preds <- predict(ensemble_model, ensemble_unseen_data, type = 'response')
ensemble_unseen_preds <- ifelse(ensemble_unseen_preds > 0.5, 1, 0)
table(ensemble_unseen_preds)
str(ensemble_unseen_preds)
savePredictions(ensemble_unseen_preds, name = 'stack_ensemble')


