library(tidyverse)
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(e1071)

setcol <- c("age","workclass","fnlwgt","education","education-num","marital-status",
            "occupation","relationship","race","sex","capital-gain","capital-loss",
            "hours-per-week","native-country","target")
#load data
adult <- read.table("~/Desktop/Predict-Income-using-US-Census-Data/adult.data.txt",header = F,
                    sep = ",",col.names = setcol,na.strings = c(" ?"),stringsAsFactors = T)
head(adult)
adult <- dplyr::select(adult, -fnlwgt, -education.num,
                       -capital.gain, -capital.loss)

adult_omitted =  adult %>% na.omit()
colSums(sapply(adult_omitted,is.na))


spl<-sample.split(adult_omitted$target, SplitRatio=0.7)
train<-subset(adult_omitted,spl== TRUE)
test <-subset(adult_omitted, spl== FALSE)


#All of the variables are used to build a logistic regression model to predict the variable over50k.
censusglm<-glm(target ~ ., data=train, family=binomial)



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
rnf_pred <- predict(rfFit,newdata = test[,-15],type = 'class')
rfAcu <-confusionMatrix(rnf_pred,test$target)$overall[1]
rfAcu
####################################################
#SVM


svm.model<- svm(target~., data = train,kernel = "radial", cost = 1, gamma = 0.1)
svm.predict <- predict(svm.model, test)
confusionMatrix(test$target, svm.predict)