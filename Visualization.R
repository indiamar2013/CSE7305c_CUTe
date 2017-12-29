
  
#clear all environment and session variables
rm(list = ls(all = TRUE))


#Libraries

source('~/classes/CSE7305c_CUTe/Utils.R')



setwd('~/datasets/CSE7305c_CUTe')
fin_data <- getPreProcessedData(path = getwd(), 
                                data_file = 'train.csv', 
                                target = 'target', sep = ',', 
                                header = TRUE, 
                                na_threshold = 0.3, 
                                zero_threshold = 0.5)

  
# plt_df <- fin_data[,!colnames(fin_data) %in% c("target")]
# numplts <- 5
# numvars <- ncol(plt_df)
# numiters <- round(numvars/numplts)
# if (numvars > numiters*numplts) {
#   numiters <- numiters + 1
# }
# numiters
# ncol(plt_df)
# 
# # for (i in 1:numiters) {
# #   startcol <- (((i-1)*numplts)+1)
# #   ifelse (i == numiters, endcol <- ncol(plt_df), endcol <- i*numplts)
# #   pltdf <- plt_df[, startcol: endcol]
# #   pltdf$target <- train_data$target
# #   print(genplot(data = pltdf, y='target'))
# # }
# 
# 
# for (i in 1:numiters) {
#   startcol <- (((i-1)*numplts)+1)
#   ifelse (i == numiters, endcol <- ncol(plt_df), endcol <- i*numplts)
#   pltdf <- plt_df[, startcol: endcol]
#   print(genboxplot(pltdf))
# }
# rm(pltdf, numiters, i, numplts, numvars, startcol, plt_df)
# rm(endcol)
# summary(train_data)
# 
# 
# 
require(ggcorrplot)
# 
cors <- cor(fin_data[,!colnames(fin_data) %in% c("target")])
# 
ggcorrplot(cors, hc.order=TRUE, type = "upper", insig = "blank")
# 
# 
#standardize the data

stdlist <- standardizeData(fin_data, target = 'target')

train_std <- data.frame(stdlist[1])
test_std <- data.frame(stdlist[2])

unseen_data <- read.csv('test.csv', header = TRUE)
unseen_data <- unseen_data[,colnames(unseen_data) %in% colnames(train_std)]
str(unseen_data)
unseen_std <- standardizeData(unseen_data, for_train=FALSE)

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



#tried the following

#cost = 100, 500, 1000, 0.1, 0.01, 0.001. Best was with cost=400
mdl <- loadModel(model_name = 'svm_std_mdl')
if (is.null(mdl)) {
  library(e1071)
  set.seed(123)
  mdl <- svm(target ~ ., data = train_std, kernel = 'sigmoid', cost=400)
  saveModel(model_name = 'svm_std_mdl', mdl)
}
predictModel(model_name = 'svm_std_mdl', mdl, test_std, unseen_std)
#got score of 2%. - this is a weak model

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


balanced_train_std <- compensateSample(train_std, method='both')
table(balanced_train_std$target)

set.seed(123)

mdl <- loadModel(model_name = 'svm_bal_std_mdl')
if (is.null(mdl)) {
  require(e1071)
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
print(mdl)
predictModel(model_name = 'rpart_std_mdl', mdl, test_std, unseen_std)
##This model got F1-stat of 29% on Grader

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
#This model got score of 43% on grader

#C5.0 on balanced data
mdl <- loadModel(model_name = 'C50_bal_std_mdl')
if (is.null(mdl)) {
  require(C50)
  set.seed(123)
  mdl <- C5.0(formula=target~., data = balanced_train_std)
  saveModel(model_name = 'C50_bal_std_mdl', mdl)
}
predictModel(model_name = 'C50_bal_std_mdl', mdl, test_std, unseen_std)
#this model got a score of 22%

##Random Forest
mdl <- loadModel(model_name = 'random_forest_std_mdl')
if (is.null(mdl)) {
  require(randomForest)
  set.seed(123)
  mdl <- randomForest(formula = target ~ ., data = train_std, keep.forest = TRUE, ntree = 500)
  saveModel(model_name = 'random_forest_std_mdl', mdl)
}
predictModel(model_name = 'random_forest_std_mdl', mdl, test_std, unseen_std)
#this got a score of 10%



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
#this gave a score of 38.96% with center, scale, and knn


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






transformData <- function(orig_data) {
  TotalAssets <- 10^orig_data$Attr29
  TotalLiabilities = TotalAssets*orig_data$Attr2
  Sales = TotalAssets*orig_data$Attr9
  GrossProfit=Sales*orig_data$Attr19
  ShortTermLiabilities=GrossProfit/orig_data$Attr12
  OperationalProfit=Sales*orig_data$Attr42
  WorkingCapital=orig_data$Attr55
  Inventory=(Sales*orig_data$Attr20)/365
  TotalSales=TotalAssets*orig_data$Attr36
  Equity = TotalAssets*orig_data$Attr10
  CostOfProductsSold=(Inventory*365)/orig_data$Attr47
  mod_data <- data.frame(TotalAssets = TotalAssets)
  mod_data$Sales = Sales
  mod_data$NetProfit = TotalAssets*orig_data$Attr1
  mod_data$TotalLiabilities = TotalLiabilities
  mod_data$RetainedEarnings = TotalAssets*orig_data$Attr6
  mod_data$EBIT = TotalAssets*orig_data$Attr7
  mod_data$BookValue = TotalLiabilities*orig_data$Attr8
  mod_data$Equity = Equity
  mod_data$WorkingCapital=WorkingCapital
  mod_data$GrossProfit=GrossProfit
  mod_data$Depreciation=(TotalLiabilities*orig_data$Attr16)-GrossProfit
  mod_data$Interest=(TotalAssets*orig_data$Attr14)-GrossProfit
  mod_data$ShortTermLiabilities=ShortTermLiabilities
  mod_data$CurrentAssets=ShortTermLiabilities*orig_data$Attr12
  mod_data$Inventory=Inventory
  mod_data$OperationalProfit=OperationalProfit
  mod_data$GrossProfit3Yrs=TotalAssets*orig_data$Attr24
  mod_data$FinancialExpenses=OperationalProfit/orig_data$Attr27
  mod_data$FixedAssets=WorkingCapital/orig_data$Attr28
  mod_data$Cash=TotalLiabilities-(Sales*orig_data$Attr30)
  mod_data$OperationalExpenses=ShortTermLiabilities*orig_data$Attr33
  mod_data$ProfitOnSales=TotalAssets*orig_data$Attr35
  mod_data$TotalSales=TotalSales
  mod_data$ConstantCapital=TotalAssets*orig_data$Attr38
  mod_data$Receivables=(Sales*orig_data$Attr44)/365
  mod_data$CostOfProductsSold=CostOfProductsSold
  mod_data$TotalCost=TotalSales*orig_data$Attr58
  mod_data$LongTermLiabilities=Equity*orig_data$Attr59
  mod_data$CurrentLiabilities=(CostOfProductsSold*orig_data$Attr32)/365
  mod_data$Attr43=orig_data$Attr43
  mod_data$ShareCapital=Equity-(TotalAssets*orig_data$Attr25)
  mod_data$target=orig_data$target

  return(mod_data)
}


setwd('~/datasets/CSE7305c_CUTe')
new_fin_data <- getPreProcessedData(path = getwd(), 
                                data_file = 'train.csv', 
                                target = 'target', sep = ',', 
                                header = TRUE, 
                                na_threshold = 0.3, 
                                zero_threshold = 0.5)

new_fin_data <- transformData(new_fin_data)
stdlist <- standardizeData(name='xform_fin_data', new_fin_data, target = 'target')

xform_train_std <- data.frame(stdlist[1])
xform_test_std <- data.frame(stdlist[2])

unseen_data <- read.csv('test.csv', header = TRUE)
unseen_data <- unseen_data[,colnames(unseen_data) %in% colnames(train_std)]
unseen_data <- transformData(unseen_data)
str(unseen_data)
summary(unseen_data)
unseen_std <- standardizeData(name='xform_fin_data', unseen_data, for_train=FALSE)


mdl <- loadModel(model_name = 'xgboost_bal_mod_std_mdl2')
if(is.null(mdl)) {
  set.seed(123)
  #balanced_train_std <- compensateSample(train_std, method='both')
  sampling_strategy<-trainControl(method = "repeatedcv",number = 2,repeats = 2,verboseIter = T,allowParallel = T)
  param_grid <- expand.grid(.nrounds = 250, .max_depth = c(2:6), .eta = c(0.1,0.11,0.09),
                            .gamma = c(0.6, 0.3), .colsample_bytree = c(0.6),
                            .min_child_weight = 1, .subsample = c(0.5, 0.6))
  
  mdl <- train(x = train_std[ , !(names(train_std) %in% c("target"))], 
               y = train_std$target, 
               method = "xgbTree",
               trControl = sampling_strategy,
               tuneGrid = param_grid)
  saveModel(model_name = 'xgboost_bal_mod_std_mdl2', mdl)
}
predictModel(model_name = 'xgboost_bal_mod_std_mdl2', mdl, test_std, unseen_std)

library(factoextra)
library(cluster)
library(NbClust)

#clustering
k <- 3 # Change k and observe the difference

km_clust <- kmeans(xform_train_std[,!colnames(xform_train_std) %in% 'target'], centers = k, iter.max = 10000)
# Visualize k-means clusters
fviz_cluster(km_clust, data = xform_train_std[,!colnames(xform_train_std) %in% 'target'], geom = "point",
             stand = FALSE) + theme_bw()

sum(is.na(xform_train_std))

km_clust$centers

new_data_std <- standardizeData(name='new_fin_data', new_fin_data, for_train = FALSE)
km_clust <- kmeans(new_data_std[,!colnames(new_data_std) %in% 'target'], centers = 2, iter.max = 10000)
# Visualize k-means clusters
fviz_cluster(km_clust, data = new_data_std[,!colnames(new_data_std) %in% 'target'], geom = "point",
             stand = FALSE) + theme_bw()
View(new_data_std[km_clust$cluster==2,])
rownames(new_data_std[km_clust$cluster==2,])

new_data_std <- new_data_std[!rownames(new_data_std) %in% rownames(new_data_std[km_clust$cluster==2,]),]
nrow(new_data_std)
#running kmeans again
km_clust <- kmeans(new_data_std[,!colnames(new_data_std) %in% 'target'], centers = 2, iter.max = 10000)
# Visualize k-means clusters
fviz_cluster(km_clust, data = new_data_std[,!colnames(new_data_std) %in% 'target'], geom = "point",
             stand = FALSE) + theme_bw()


new_data_std <- read.csv('new_fin_data_std.csv')
View(new_data_std[km_clust$cluster==1,])
new_data_std <- new_data_std[!rownames(new_data_std) %in% rownames(new_data_std[km_clust$cluster==1,]),]
nrow(new_data_std)
#running kmeans again
km_clust <- kmeans(new_data_std[,!colnames(new_data_std) %in% 'target'], centers = 2, iter.max = 10000)
# Visualize k-means clusters
fviz_cluster(km_clust, data = new_data_std[,!colnames(new_data_std) %in% 'target'], geom = "point",
             stand = FALSE) + theme_bw()

View(new_data_std[km_clust$cluster==2,])
new_data_std <- new_data_std[!rownames(new_data_std) %in% rownames(new_data_std[km_clust$cluster==2,]),]
nrow(new_data_std)
#running kmeans again
km_clust <- kmeans(new_data_std[,!colnames(new_data_std) %in% 'target'], centers = 2, iter.max = 10000)
# Visualize k-means clusters
fviz_cluster(km_clust, data = new_data_std[,!colnames(new_data_std) %in% 'target'], geom = "point",
             stand = FALSE) + theme_bw()

outliers <- setdiff(rownames(new_fin_data), rownames(new_data_std))

#let's see if this makes any difference
new_fin_data<- new_fin_data[!rownames(new_fin_data) %in% outliers, ]
stdlist <- standardizeData(name='xform_fin_data', new_fin_data, target = 'target')

xform_train_std <- data.frame(stdlist[1])
xform_test_std <- data.frame(stdlist[2])

unseen_data <- read.csv('test.csv', header = TRUE)
unseen_data <- transformData(unseen_data)
unseen_data <- unseen_data[,colnames(unseen_data) %in% colnames(xform_train_std)]
str(unseen_data)
summary(unseen_data)
unseen_std <- standardizeData(name='xform_fin_data', unseen_data, for_train=FALSE)
summary(unseen_std)

mdl <- loadModel(model_name = 'xgboost_bal_mod_std_mdl3')
if(is.null(mdl)) {
  set.seed(123)
  #balanced_train_std <- compensateSample(train_std, method='both')
  sampling_strategy<-trainControl(method = "repeatedcv",number = 2,repeats = 2,verboseIter = T,allowParallel = T)
  param_grid <- expand.grid(.nrounds = 250, .max_depth = c(2:6), .eta = c(0.1,0.11,0.09),
                            .gamma = c(0.6, 0.3), .colsample_bytree = c(0.6),
                            .min_child_weight = 1, .subsample = c(0.5, 0.6))
  
  mdl <- train(x = xform_train_std[ , !(names(xform_train_std) %in% c("target"))], 
               y = xform_train_std$target, 
               method = "xgbTree",
               trControl = sampling_strategy,
               tuneGrid = param_grid)
  saveModel(model_name = 'xgboost_bal_mod_std_mdl3', mdl)
}
predictModel(model_name = 'xgboost_bal_mod_std_mdl3', mdl, xform_test_std, unseen_std)

