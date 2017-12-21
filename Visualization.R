---
  title: "Team17_CSE7302c_CUTe"
author: "Amar Rao"
date: "December 16, 2017"
output: 
  html_document:
  toc: true
toc_float:
  collapsed: false
theme: united
highlight: tang
fig_width: 7
fig_height: 6
fig_caption: true
code_folding: hide

---
  
#clear all environment and session variables
rm(list = ls(all = TRUE))


Libraries

library(tidyverse)
library(lubridate)
library(caret)
library(DMwR)
library(forecast)
library(lubridate)
library(imputeTS)
library(TTR)
library(graphics)
library(zoo)

## Reading data

setwd('C:/Users/212629693/Documents/Personal/ds-classes/INSOFE_Batch35_7302c_CUTe_Team17')

fin_data_orig <- read.csv('Datasets/train_data.csv', header = TRUE, sep = ',', na.strings = "", colClasses = "double", numerals = "no.loss")


cat_attrs <- c('y2')
num_attrs <- setdiff(colnames(fin_data_orig), cat_attrs)

fin_data_orig[cat_attrs] <- data.frame(sapply(fin_data_orig[cat_attrs], as.factor))
emptycols <- colnames(fin_data_orig[,sapply(fin_data_orig, function(x) { all(is.na(x))})])
fin_data <- fin_data_orig[, !colnames(fin_data_orig) %in% emptycols]


colswithna <- sort(colSums(is.na(fin_data)), decreasing = TRUE)

nacolstoremove <- colswithna[colswithna > nrow(fin_data)*0.03]
nacolstoremove

fin_data <- fin_data[, !colnames(fin_data) %in% names(nacolstoremove)]

sparsecols <- sort(colSums(fin_data[,!colnames(fin_data) %in% c("y1", "y2")] == 0), decreasing = TRUE)
colstoremove <- sparsecols[sparsecols > round(nrow(fin_data) * 0.25)]
colstoremove

fin_data <- fin_data[, !colnames(fin_data) %in% names(colstoremove)]


rownames(fin_data) <- fin_data$timestamp
fin_data$timestamp <- NULL

constant_cols <- colnames(fin_data[ , sapply(fin_data, function(v){ var(v, na.rm=TRUE)==0})])
constant_cols
fin_data <- fin_data[, !colnames(fin_data) %in% constant_cols]

library(RANN)

set.seed(1234)

preproc_preds <- preProcess(x = subset(fin_data, select = -c(y1, y2)), method = c("knnImpute"))
fin_data <- predict(preproc_preds, fin_data)

sum(is.na(fin_data))

head(fin_data)

##Visualizing the dataset
# We need to understand the relationship between y1 and the dependent variables
# There are a lot of columns:
# d_0 - d_4
# f_0 - f_60
# t_0 - t_43


library(reshape2)
genplot <- function(data=data.frame()) {
  df <- melt(data, id.vars = 'y1', variable.name = 'series')
  df
  plt <- ggplot(df, aes(y1,value)) + geom_point(aes(colour=series)) + facet_grid(series ~ .)
  return(plt)
}


plt_df <- fin_data[,!colnames(fin_data) %in% c("y1", "y2")]
numplts <- 5
numvars <- ncol(plt_df)
numiters <- round(numvars/numplts)
if (numvars > numiters*numplts) {
  numiters <- numiters + 1
}
numiters
ncol(plt_df)

for (i in 1:numiters) {
  startcol <- (((i-1)*numplts)+1)
  ifelse (i == numiters, endcol <- ncol(plt_df), endcol <- i*numplts)
  pltdf <- plt_df[, startcol: endcol]
  pltdf$y1 <- fin_data$y1
  print(genplot(data = pltdf))
}


install.packages("ggcorrplot")
library(ggcorrplot)

cors <- cor(fin_data[,!colnames(fin_data) %in% c("y1", "y2")])
cors
ggcorrplot(cors, hc.order = TRUE, type = "upper", insig = "blank")
```
