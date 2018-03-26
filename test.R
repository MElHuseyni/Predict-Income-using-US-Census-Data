library(ggplot2)
library(readr)
library(caret)
library(DataExplorer)

adult <- read.csv("~/Desktop/Predict-Income-using-US-Census-Data/adult.data")
str(adult)



hist(adult$age, breaks = 20)

# Remove rows where age of a person is greater than 65
adult <- subset(adult, age < 65)
# Plot age distributin again
hist(adult$age, breaks = 20) 


# Check for missing values
#length(which(is.na(adult)))

# NAs are given as "?". Replace ? with NA
adult[adult == "?"] <- NA
# Check for missing values
length(which(is.na(adult)))


# Compare education and income
ggplot(adult, aes(education, fill = income) ) +
  geom_bar(position = "stack")+
  ggtitle('Education and income')+
  xlab('Education level') +
  ylab('frequency')

# The following gprahs will show variables on the same scale
# Compare race and income
ggplot(adult, aes(race, fill = income) ) +
  geom_bar(position = "fill")+
  ggtitle('Race and income')+
  xlab('Race') +
  ylab('frequency')

# Compare marital status and income
ggplot(adult, aes(marital.status, fill = income) ) +
  geom_bar(position = "fill")+
  ggtitle('Marital status and income')+
  xlab('Race') +
  ylab('frequency')

# Compare marital status and race
ggplot(adult, aes(marital.status, fill = race) ) +
  geom_bar(position = "fill")+
  ggtitle('Marital status and race')+
  xlab('Race') +
  ylab('frequency')

# Compare relationship status and income
ggplot(adult, aes(relationship, fill = income) ) +
  geom_bar(position = "fill")+
  ggtitle('Relationship status and income')+
  xlab('Relationship status') +
  ylab('frequency')


library(plotly)
adult %>% select(sex,occupation,hours.per.week) %>% group_by(sex,occupation) %>% 
  summarise(work=mean(hours.per.week,na.rm=T)) %>%
  ggplot(aes(x=occupation, y=work, fill=sex)) +
  geom_bar(position="dodge",stat='identity')+
  ggtitle("Mean number of hours worked by each gender for given occupation")+
  theme(plot.title=element_text(size=10),axis.text.x = element_text(angle=90, vjust=1))

adult %>% select(sex,income,occupation) %>% mutate(income_binary=ifelse(income=='>50K',"Yes","No"))%>% filter(income_binary=='Yes') %>%
  group_by(occupation,sex) %>% summarise(n=n()) %>%
  ggplot(aes(x=occupation,y=n,fill=sex))+
  geom_bar(position='dodge',stat='identity')+
  ggtitle("Number of males and females who receive salaries greater than 50K")+
  theme(plot.title=element_text(size=10),axis.text.x=element_text(angle=90,vjust=1))

adult %>% select(sex,occupation) %>% group_by(occupation,sex) %>% summarise(n=n())%>%
  ggplot(aes(x=occupation,y=n,fill=sex))+
  geom_bar(position='dodge',stat='identity')+
  ggtitle("Number of each gender in each occupation")+
  theme(plot.title=element_text(size=10),axis.text.x=element_text(angle=90,vjust=1))

qplot(workclass, hours.per.week, data=adult, geom="boxplot", fill=workclass)+
  theme(plot.title=element_text(size=18),axis.text.x=element_text(angle=90,vjust=1))

adult %>% select(native.country,income)%>% mutate(income_binary=ifelse(income=='>50K','Yes','No')) %>% 
  group_by(native.country,income_binary)%>% summarise(n=n()) %>% filter(native.country!='United-States') %>%
  ggplot(aes(x=native.country,y=n,fill=income_binary))+
  geom_bar(position='dodge',stat='identity')+
  ggtitle("Number of individuals who earn more/less than 50K per year")+
  theme(plot.title=element_text(size=10),axis.text.x=element_text(angle=90,vjust=1))

ggplotly()


library(ggplot2)
library(GGally)
library(plyr)
library(dplyr)
library(stringr)
library(caret)
library(car)
library(class)
library(gbm)
library(randomForest)
library(knitr)

#Clean Data
colnames(AdultData)[8:15] <- c(colnames(AdultData)[9:15],"Income")
#Extract predicted variable
y <- as.integer(AdultData$Income==" >50K")
AdultData$Income <- NULL
AdultData$married <- as.integer(AdultData$relationship%in%c(" Husband"," Wife"))
# Factor as numeric
for (i in 1:15) {
  if(class(AdultData[,i])=="factor") AdultData[,i] <- as.numeric(AdultData[,i])
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

#===============================================================
# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)


# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)


# compare accuracy of models
dotplot(results)


# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

