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

getPreprocPreds <- function(preproc_data) {
  std_preds <- preProcess(x = preproc_data,
                          method = c("range", "knnImpute"), 
                          na.remove = TRUE)
  return(std_preds)
}

standardizeData <- function(name='dataset', preproc_data=data.frame(), target='target', for_train=TRUE) {
  #filename_prefix <- deparse(substitute(preproc_data))
  filename_prefix <- name
  train_filename <- paste(filename_prefix, '_train_std.csv', sep="") 
  test_filename <- paste(filename_prefix, '_test_std.csv', sep="")
  preds_filename <- paste(filename_prefix, '_preds.RDS')
  
  if (for_train==TRUE) {
    train_std <- data.frame()
    test_std <- data.frame()
    if (file.exists(train_filename) & 
        file.exists(test_filename) &
        file.exists(preds_filename)) {
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
      
      # std_preds <- preProcess(x = train_data[,!colnames(train_data) %in% target], 
      #                         method = c("center", "scale", "knnImpute"))
      
      
      std_preds <- getPreprocPreds(train_data[,!colnames(train_data) %in% target])
      
      train_std <- predict(std_preds, train_data)
      test_std <- predict(std_preds, test_data)
      
      # std_preds <- preProcess(x = train_data[,!colnames(train_data) %in% target], 
      #                         method = c("center", "scale"))
      # train_std <- predict(std_preds, train_data)
      # test_std <- predict(std_preds, test_data)
      
      saveRDS(std_preds, file=preds_filename)
      
      #the imputation takes a lot of time, so I will write it to csv in case i need
      #to resume later.
      write.csv(train_std, train_filename, row.names = FALSE)
      write.csv(test_std, test_filename, row.names = FALSE)
    }
    retlist <- list(train_std, test_std)
    return(retlist)
  } else {
    unseen_filename <- paste(filename_prefix, '_std.csv', sep="")
    unseen_std <- data.frame()
    if(file.exists(unseen_filename)) {
      print(unseen_filename)
      unseen_std <- read.csv(unseen_filename, colClass='numeric')
    } else {
      if(file.exists(preds_filename)) {
        std_preds <- readRDS(preds_filename)
      } else {
        #we will just standardize using this data
        std_preds <- getPreprocPreds(preproc_data[,!colnames(preproc_data) %in% target])
      }
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


