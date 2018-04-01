library(data.table)
library(ggplot2)
library(xgboost)

#Extract predicted variable
y <- as.integer(adult$income ==" >50K")
adult$income <- NULL
adult$married <- as.integer(adult$relationship%in%c(" Husband"," Wife"))
# Factor as numeric
for (i in 1:13) {
  if(class(adult[,i])=="factor") adult[,i] <- as.numeric(adult[,i])
}

library(xgboost)
param0 <- list(
  "objective"  = "binary:logistic",
  "eval_metric" = "auc",
  "eta" = 0.01 
  ,"subsample" = 1
  , "colsample_bytree" = 1
  , "min_child_weight" = 1
  , "max_depth" = 9
) 

xgtrain <- xgb.DMatrix(as.matrix(adult), label = y)
watchlist <- list('train' = xgtrain)
cv <- xgb.cv(params = param0,data = xgtrain,nrounds = 101,nfold = 3,print.every.n = 20)



xgmod = xgb.train(
  nrounds = 100
  , params = param0
  , data = xgtrain
  , watchlist = watchlist
  , nthread = 8
  ,print.every.n = 100
)

imp <- xgb.importance(model = xgmod,feature_names = colnames(adult))

xgb.plot.importance(imp)