
  
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
                                preproc_file = 'preproc_data.csv',
                                na_threshold = 0.3,
                                zero_threshold = 0.5) {
  setwd(path)
  if (file.exists(preproc_file)) {
    data <- read.csv(preproc_file)
    if (!class(data[, colnames(data) %in% target]) == 'factor') {
      data[,colnames(data) %in% target] <- as.factor(as.character(data[,colnames(data) %in% target]))
    }
    return(data)
  }
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
  
  write.csv(x = preproc_data, file = 'preproc_data.csv', row.names = FALSE)
  return(preproc_data)
}


standardizeData <- function(preproc_data=data.frame(), target='target') {
  #Standardization
  train_std <- data.frame()
  test_std <- data.frame()
  if (file.exists('train_std.csv') & (file.exists('test_std.csv'))) {
    train_std <- read.csv('train_std.csv')
    if (!class(train_std[, colnames(train_std) %in% target]) == 'factor') {
      train_std[,colnames(train_std) %in% target] <- as.factor(as.character(train_std[,colnames(train_std) %in% target]))
    }
    test_std <- read.csv('test_std.csv')
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
    std_data <- predict(std_preds, train_data)
    train_std <- predict(std_preds, train_data)
    test_std <- predict(std_preds, test_data)
    rm(std_preds)
    
    
    #the imputation takes a lot of time, so I will write it to csv in case i need
    #to resume later.
    write.csv(train_std, "train_std.csv", row.names = FALSE)
    write.csv(test_std, "test_std.csv", row.names = FALSE)
  }
  retlist <- list(train_std, test_std)
  return(retlist)
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
                                preproc_file = 'preproc_data.csv', 
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
train_pca <- prcomp(x = train_std[,!colnames(train_std) %in% 'target'])
summary(train_pca)
#This shows that first 20 components can account for approximately 97% of variance in the dataset.
# So building an SVM Model with this dataset

train_pca_data <- data.frame(train_pca$x[,1:25], target=train_std$target)

svm_mdl2 <- svm(target ~ ., data = train_pca_data, kernel = "linear")

summary(svm_mdl2)
#let's check how good this model is
test_std_pca <- predict(train_pca, newdata = test_std)
test_std_pca <- as.data.frame(test_std_pca[,1:25])

#need to first convert test_std to PCA
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
confusionMatrix(svm_mdl4_preds, test_std$target)


#since sigmoid is giving good results, will now tune the hyper parameters
svm_fit <- trainControl(method = "repeatedcv", repeats = 3, number = 4)
svmSigmaGrid <- expand.grid(.C=c(10^-2, 10^-1, 10^1), .sigma=c(1, 10, 25))
svm_mdl5 <- train(target ~ ., data = train_pca_data, 
                  method = "svmRadialSigma",
                  tuneGrid = svmSigmaGrid, 
                  trControl = svm_fit)


svm_mdl4

svm_mdl3 <- train(Cancer ~ ., data = train_std, 
                  method = "svmLinear",
                  tuneGrid = data.frame(.C = c(0.025, 0.05, 0.075, 0.1, 0.125, 0.150)), metric = "Accuracy", 
                  trControl = svm_fit)

svm_mdl3
svm_mdl3$method
svm_mdl3$modelType
str(svm_mdl3$bestTune)
svm_mdl3$metric
svm_mdl3$control

