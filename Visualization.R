
  
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
                                target="", 
                                sep = ',', 
                                header = TRUE, 
                                imputed_file = 'imputed_data.csv',
                                na_threshold = 0.3,
                                zero_threshold = 0.5) {
  setwd(path)
  if (file.exists(imputed_file)) {
    data <- read.csv(imputed_file)
    if (!class(data[, colnames(data) %in% target]) == 'factor') {
      data[,colnames(data) %in% target] <- as.factor(as.character(data[,colnames(data) %in% target]))
    }
    return(data)
  }
  raw_data = getRawData(path, data_file, target, sep, header)
  
  #remove empty columns
  emptycols <- colnames(raw_data[,sapply(raw_data, function(x) { all(is.na(x))})])
  preproc_data <- raw_data[, !colnames(raw_data) %in% emptycols]
  rm(emptycols, raw_data)
  #remove columns with na beyond threshold
  colswithna <- sort(colSums(is.na(preproc_data)), decreasing = TRUE)
  nacolstoremove <- colswithna[colswithna > nrow(preproc_data)*na_threshold]
  preproc_data <- preproc_data[, !colnames(preproc_data) %in% names(nacolstoremove)]
  rm(colswithna, nacolstoremove)
  
  
  sparsecols <- sort(colSums(preproc_data[,!colnames(preproc_data) %in% c("target")] == 0), decreasing = TRUE)
  colstoremove <- sparsecols[sparsecols > round(nrow(preproc_data) * zero_threshold)]
  preproc_data <- preproc_data[, !colnames(preproc_data) %in% names(colstoremove)]
  rm(colstoremove, sparsecols)
  
  
  rownames(preproc_data) <- preproc_data$ID
  preproc_data$ID <- NULL
  
  constant_cols <- colnames(preproc_data[ , sapply(preproc_data, function(v){ var(v, na.rm=TRUE)==0})])
  preproc_data <- preproc_data[, !colnames(preproc_data) %in% constant_cols]
  rm(constant_cols)
  
  #IMPUTATION
  # we have to do this before splitting because otherwise, all missing rows in test_data get imputed
  
  library(RANN)
  
  set.seed(1234)
  
  preproc_preds <- preProcess(x = subset(preproc_data, select = -c(target)), method = c("knnImpute"))
  preproc_data <- predict(preproc_preds, preproc_data)
  rm(preproc_preds)
  
  #the imputation takes a lot of time, so I will write it to csv in case i need
  #to resume later.
  write.csv(x = v, file = 'imputed_data.csv')
  return(preproc_data)
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
fin_data <- getPreProcessedData(path = getwd(), data_file = 'train.csv', target = 'target', sep = ',', header = TRUE, imputed_file = 'imputed_data.csv', na_threshold = 0.3, zero_threshold = 0.5)
getwd()

#split into train and test
#will use caret package
train_rows <- createDataPartition(fin_data$target, p = 0.7, list = FALSE)
train_data <- fin_data[train_rows,]
test_data <- fin_data[-train_rows,]
prop.table(table(fin_data$target))
prop.table(table(train_data$target))
prop.table(table(test_data$target))
#remove original data to conserve memory
rm(fin_data)
rm(train_rows)

##Visualizing the dataset


plt_df <- train_data[,!colnames(train_data) %in% c("target")]
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


## Model Building

#First an SVM model - we need to determine optimal C, so will use train() from caret package

svm_fit <- trainControl(method = "repeatedcv", repeats = 5, number = 4)
svm_mdl1 <- train(target ~ ., data = train_data, 
                  method = "svmLinear",
                  tuneGrid = data.frame(.C = c(10^-4, 10^-3, 10^-2, 10^-1, 10^1, 10^2, 10^3)),
                  trControl = svm_fit)


svm_mdl1

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

