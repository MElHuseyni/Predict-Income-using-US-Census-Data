
# Input the dataset for analysis 
library(readr)
diabetes <- read_csv("../input/diabetes.csv")
library("e1071")
#Using the SVM model defined in the package with all variables considered in building the model
svm_model=svm(Outcome~.,data=diabetes,type='C-classification')
#Summary will list the respective parameters uch as cost, gamma, etc.
summary(svm_model)
#Predicting the data with the input to be the dataset itself, we can calculate the accuracy with a confusion matrix
pred=predict(svm_model,newdata=diabetes)
table(pred,diabetes$Outcome)
#The accuracy turns out to be 82.42%
#Now let's tune the SVM parameters to get a better accuracy on the training dataset
svm_tune <- tune(svm, train.x=diabetes, train.y=diabetes$Outcome, 
            kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune)
#Gives an optimal cost to be 10 and a gamma value of 0.5

svm_model_after_tune <- svm(Outcome ~ ., data=diabetes, type='C-classification',kernel="radial", cost=10, gamma=0.5)
summary(svm_model_after_tune)
#The results show us that there is an improved accuracy of about 98%, results are obtained in the form of a confusion matrix
pred <- predict(svm_model_after_tune,diabetes)
system.time(predict(svm_model_after_tune,diabetes))
table(pred,diabetes$Outcome)
