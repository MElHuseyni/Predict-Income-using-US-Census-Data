library(tidyverse)
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(e1071)
library(ROCR)
library(DataExplorer)

setcol <- c("age","workclass","fnlwgt","education","education-num","marital-status",
            "occupation","relationship","race","sex","capital-gain","capital-loss",
            "hours-per-week","native-country","target")
#load data
adult <- read.table("~/Desktop/Predict-Income-using-US-Census-Data/adult.data.txt",header = F,
                    sep = ",",col.names = setcol,na.strings = c(" ?"),stringsAsFactors = T)
head(adult)
adult <- dplyr::select(adult, -fnlwgt, -education.num,
                       -capital.gain, -capital.loss)

adult =  adult %>% na.omit()
colSums(sapply(adult,is.na))
plot_str(adult)
plot_missing(adult)
y <- as.integer(adult$target)

##############################################################
#Split Data into train and test
spl<-sample.split(adult$target, SplitRatio=0.8)
train<-subset(adult,spl== TRUE)
test <-subset(adult, spl== FALSE)

#############################################################################3
#All of the variables are used to build a logistic regression model to predict the variable over50k.
lg <- glm(target ~.,family = binomial(link='logit'),data=train)
Prediction2 <- predict(lg,newdata=test[-15],type = 'response')
Pred <- ifelse(Prediction2>0.5,1,0)
table(actual= test$target, predicted= Pred>0.5)

lgAcu <- (4190+823)/6033
lgAcu
ROCRpred<- prediction(Prediction2, test$target)
perf<- performance(ROCRpred, "tpr", "fpr")
plot(perf)
################################################################
census_tree<-rpart(target ~ ., data=train, method="class")
prp(census_tree)
rpart.plot(census_tree, box.col=c("red", "blue"))


predict_tree = predict(census_tree, newdata = test, type = "class")
confusionmatrix_tree<-table(test$target, predict_tree)
confusionmatrix_tree


accuract_CART <- (confusionmatrix_tree[1,1] + confusionmatrix_tree[2,2])/sum(confusionmatrix_tree)
accuract_CART


PredictROC_Tree = predict(census_tree, newdata = test)
predict2 = prediction(PredictROC_Tree[, 2], test$target)
#as.numeric(performance(predict2, "auc")@y.values)
performance2 = performance(predict2, "tpr", "fpr")
plot(performance2, main = "rpart tree")


#=====================================


set.seed(32423)
rfFit<- randomForest(target~.,data= train)
print(rfFit)
rnf_pred <- predict(rfFit,newdata = test[,-11],type = 'class')
rfAcu <-confusionMatrix(rnf_pred,test$target)$overall[1]
rfAcu
####################################################
#SVM


svm.model<- svm(target~., data = train,kernel = "radial", cost = 1, gamma = 0.1)
svm.predict <- predict(svm.model, test)
confusionMatrix(test$target, svm.predict)


##################################

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

xgtrain <- xgb.DMatrix(as.matrix(adult_omitted), label = y)
watchlist <- list('train' = xgtrain)
cv <- xgb.cv(params = param0,data = xgtrain,nrounds = 101,nfold = 3,print.every.n = 20)
