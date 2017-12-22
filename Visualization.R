
  
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
    print('i am here')
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
      
      std_preds <- preProcess(x = train_data[,!colnames(train_data) %in% target], 
                              method = c("range", "knnImpute"))
      saveRDS(std_preds, file='std_preds.RDS')
      
      train_std <- predict(std_preds, train_data)
      test_std <- predict(std_preds, test_data)

      
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

savePredictions <- function(predictions) {
  unseen_raw <- read.csv(file = 'test.csv', header = TRUE)
  submission <- data.frame('ID'=unseen_raw$ID, 'prediction'=predictions)
  rownames(submission) <- c(1:nrow(submission))
  rownames(submission)
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

#tried the following
#cost = 100, 500, 1000, 0.1, 0.01, 0.001
svm_mdl4 <- svm(target ~ ., data = train_pca_data, kernel = "sigmoid", cost=400)
summary(svm_mdl4)

svm_mdl4_preds <- predict(object = svm_mdl4, newdata = test_std_pca)
table(svm_mdl4_preds)
confusionMatrix(svm_mdl4_preds, test_std$target)

##now with validation data
unseen_data <- getPreProcessedData(path = getwd(), data_file = 'test.csv', target = 'target', na_threshold = 0.3, zero_threshold = 0.5)
unseen_std <- standardizeData(preproc_data = unseen_data, for_train = FALSE)

unseen_std_pca <- predict(train_pca, unseen_std)
unseen_std_pca <- unseen_std_pca[,1:25]

unseen_preds <- predict(svm_mdl4, newdata = unseen_std_pca)

table(unseen_preds)
savePredictions(unseen_preds)

svm_mdl5 <- svm(target ~ ., data = train_std, kernel = 'sigmoid', cost=1000)
summary(svm_mdl5)
mdl5_test_preds <- predict(svm_mdl5, test_std[,!colnames(test_std) %in% 'target'])

confusionMatrix(mdl5_test_preds, test_std$target)

unseen_mdl5_preds <- predict(svm_mdl5, unseen_std)
table(unseen_preds)
savePredictions(unseen_preds)




library(ROCR)
mdl5_preds <- predict(object = svm_mdl5, type="response")
class(mdl5_preds)
table(mdl5_preds)
mdl5_train_pred <- prediction(as.numeric(mdl5_preds), train_std$target)
# The performance() function from the ROCR package helps us extract metrics such as True positive rate, False positive rate etc. from the prediction object, we created above.
vol_perf1 <- performance(prediction.obj = mdl5_train_pred, measure = "tpr", x.measure = "fpr")
plot(vol_perf1, col = rainbow(10), colorize=T, print.cutoffs.at=(seq(0,1,0.05)))


ctrl <- trainControl(method="repeatedcv",repeats = 3)
knnFit <- train(target ~ ., 
                data = train_std, 
                method = "knn", 
                trControl = ctrl)
knnFit

plot(knnFit)

knn_preds <- predict(knnFit, newdata = test_std)

confusionMatrix(knn_preds, test_std$target)
unseen_preds <- predict(knnFit, newdata = unseen_std)

table(unseen_preds)
savePredictions(unseen_preds)


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

balanced_train_std <- compensateSample(train_std, method='over')
table(balanced_train_std$target)

svm_mdl6 <- svm(target ~ ., data = balanced_train_std, kernel = 'sigmoid', cost=400)
summary(svm_mdl6)
mdl6_test_preds <- predict(svm_mdl6, test_std[,!colnames(test_std) %in% 'target'])

confusionMatrix(mdl6_test_preds, test_std$target)

unseen_mdl65_preds <- predict(svm_mdl6, unseen_std)
table(unseen_mdl65_preds)
savePredictions(unseen_mdl65_preds)


#RPART
require(rpart)
set.seed(123)
trainCtrl <- trainControl(method = 'repeatedcv', number = 4, repeats = 2)
rpart_grid <-expand.grid(data.frame(.cp = c(0.005, 0.004, 0.003, 0.002, 0.001)))
rpart_mdl <- train(target~., data = train_std, method='rpart', trControl = trainCtrl, tuneGrid = rpart_grid)
print(rpart_mdl)
plot(rpart_mdl)
plot(rpart_mdl$finalModel)

rpart_test_preds <- predict(rpart_mdl, test_std)
confusionMatrix(rpart_test_preds, test_std$target)

rpart_unseen_preds <- predict(rpart_mdl, unseen_std)
table(rpart_unseen_preds)
savePredictions(rpart_unseen_preds)



##C5.0
require(C50)
set.seed(123)
c50_mdl <- C5.0(formula=target~., data = train_std)
plot(c50_mdl)

c50_test_preds <- predict(c50_mdl, test_std)
confusionMatrix(c50_test_preds, test_std$target)
c50_unseen_preds <- predict(c50_mdl, unseen_std)
table(c50_unseen_preds)
savePredictions(c50_unseen_preds)

#C5.0 on balanced data
set.seed(123)
c50_bal_mdl <- C5.0(target ~ ., data = balanced_train_std)
plot(c50_mdl)
c50_bal_test_preds <- predict(c50_bal_mdl, test_std)
confusionMatrix(c50_bal_test_preds, test_std$target)

c50_bal_unseen_preds <- predict(c50_bal_mdl, unseen_std)
table(c50_bal_unseen_preds)
savePredictions(c50_bal_unseen_preds)
##Random Forest
require(randomForest)
set.seed(123)
random_mdl <- randomForest(formula = target ~ ., data = train_std, keep.forest = TRUE, ntree = 500)

print(random_mdl)

random_preds <- predict(random_mdl, test_std, type = 'response')
confusionMatrix(random_preds, test_std$target)

random_unseen_preds <- predict(random_mdl, unseen_std)
table(random_unseen_preds)
savePredictions(random_unseen_preds)


##Random Forest with balanced data-set
set.seed(123)
bal_rnd_mdl <- randomForest(target ~ ., balanced_train_std, keep.forest = TRUE, ntree = 300)
print(bal_rnd_mdl)
bal_rnd_tst_prds <- predict(bal_rnd_mdl, test_std, type = 'response')
confusionMatrix(bal_rnd_tst_prds, test_std$target)

bal_unseen_preds <- predict(bal_rnd_mdl, unseen_std)
table(bal_unseen_preds)
savePredictions(random_unseen_preds)



##XG BOOST
set.seed(123)
sampling_strategy<-trainControl(method = "repeatedcv",number = 2,repeats = 2,verboseIter = T,allowParallel = T)
param_grid <- expand.grid(.nrounds = 250, .max_depth = c(2:6), .eta = c(0.1,0.11,0.09),
                          .gamma = c(0.6, 0.3), .colsample_bytree = c(0.6),
                          .min_child_weight = 1, .subsample = c(0.5, 0.6))

xgb_tuned_model <- train(x = train_std[ , !(names(train_std) %in% c("target"))], 
                         y = train_std$target, 
                         method = "xgbTree",
                         trControl = sampling_strategy,
                         tuneGrid = param_grid)

summary(xgb_tuned_model)
xgb_tuned_model

xgb_test_preds <- predict(xgb_tuned_model, test_std)
confusionMatrix(xgb_test_preds, test_std$target)

xgb_unseen_preds <- predict(xgb_tuned_model, unseen_std)
table(xgb_unseen_preds)
savePredictions(xgb_unseen_preds)
saveRDS(xgb_tuned_model, file = 'xgb_tuned_model.RDS')


#what about xgb with oversampled dataset?
set.seed(123)
balanced_train_std <- compensateSample(train_std, method='both')
table(balanced_train_std$target)

xgb_bal_mdl <- train(x = balanced_train_std[ , !(names(balanced_train_std) %in% c("target"))], 
                     y = balanced_train_std$target, 
                     method = "xgbTree",
                     trControl = sampling_strategy,
                     tuneGrid = param_grid)


xgb_bal_test_preds <- predict(xgb_bal_mdl, test_std)
confusionMatrix(xgb_bal_test_preds, test_std$target)
xgb_bal_unseen_preds <- predict(xgb_bal_mdl, unseen_std)
table(xgb_bal_unseen_preds)
savePredictions(xgb_bal_unseen_preds)

#what about xgb with pca?
set.seed(123)
xgb_pca_mdl <- train(x = train_pca_data[ , !(names(train_pca_data) %in% c("target"))], 
                     y = train_pca_data$target, 
                     method = "xgbTree",
                     trControl = sampling_strategy,
                     tuneGrid = param_grid)

xgb_pca_test_preds <- predict(xgb_pca_mdl, test_std_pca)
confusionMatrix(xgb_pca_test_preds, test_std$target)



