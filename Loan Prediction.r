#Loading Training & Test data
data <- read.csv("https://raw.githubusercontent.com/aditya00kumar/AIML_dataset/master/train.csv",header=T)
test <- read.csv("https://raw.githubusercontent.com/aditya00kumar/AIML_dataset/master/test.csv",header=T)

#Loading packages
library(ggplot2)
library(readxl)
library(dplyr)
library(tidyr)
library(corrplot) #To find correlation between variables
library(caTools)#For splitting data into train & test
library(Amelia) #For data quality report
library(randomForest)
library(ROCR)
library(e1071) #For naive bayes
library(caret) #To get confusion matrix

#Exploring data
class(data)
head(data)
summary(data)
str(data)

#Understanding the missing values 
Miss<-data.frame(colMeans(is.na(data)))

#Building data quality report using Amelia package

missmap(data, legend = TRUE, col = c("indianred", "dodgerblue"))

#Finding NA's and imputing in training data

data$LoanAmount[is.na(data$LoanAmount)] <- median(data$LoanAmount[!is.na(data$LoanAmount)])
data$Loan_Amount_Term[is.na(data$Loan_Amount_Term)] <- median(data$Loan_Amount_Term[!is.na(data$Loan_Amount_Term)])
data$Credit_History[is.na(data$Credit_History)] <- median(data$Credit_History[!is.na(data$Credit_History)])

#Finding NA's and imputing in test data
summary(test)
test$LoanAmount[is.na(test$LoanAmount)] <- median(test$LoanAmount[!is.na(test$LoanAmount)])
test$Loan_Amount_Term[is.na(test$Loan_Amount_Term)] <- median(test$Loan_Amount_Term[!is.na(test$Loan_Amount_Term)])
test$Credit_History[is.na(test$Credit_History)] <- median(test$Credit_History[!is.na(test$Credit_History)])

#Finding outliers and imputing TRAIN data
x <- boxplot(data$ApplicantIncome, test$ApplicantIncome)
AIdata <- which(data$ApplicantIncome%in% x$out)
AIdata
data$ApplicantIncome[AIdata]<-median(data$ApplicantIncome,na.rm = TRUE)

x2 <- boxplot(data$CoapplicantIncome, test$CoapplicantIncome)
COdata <- which(data$CoapplicantIncome%in% x2$out)
COdata
data$CoapplicantIncome[COdata]<-median(data$CoapplicantIncome,na.rm = TRUE)

#Finding outliers and imputing Test data
x3 <- boxplot(data$ApplicantIncome, test$ApplicantIncome)
AItest <- which(test$ApplicantIncome%in% x3$out)
AItest
test$ApplicantIncome[AItest]<-median(test$ApplicantIncome,na.rm = TRUE)

x4 <- boxplot(data$CoapplicantIncome, test$CoapplicantIncome)
COtest <- which(test$CoapplicantIncome%in% x4$out)
COtest
test$CoapplicantIncome[COtest]<-median(test$CoapplicantIncome,na.rm = TRUE)

#Splitting the data to train and verify
set.seed(2)
split <- sample.split(data, SplitRatio = 0.8)

train <- subset(data, split == TRUE)

verify <- subset(data, split == FALSE)

#Verifying the model with all independent variables
model <- glm(formula = Loan_Status  ~ Gender + Married + Dependents + Education + Self_Employed + ApplicantIncome + 
               CoapplicantIncome + LoanAmount + Loan_Amount_Term + Credit_History + Property_Area, family = "binomial", data = train)
summary(model)

#Getting the best model based on Residual deviance and AIC
model <- glm(formula = Loan_Status  ~ Gender + Dependents + Education + Self_Employed +LoanAmount + Loan_Amount_Term + Credit_History + Property_Area, family = "binomial", data = train)
summary(model)

#model <- glm(formula = Loan_Status  ~ Education + ApplicantIncome +  Credit_History + Property_Area +Loan_Amount_Term, family = "binomial", data = train)
#summary(model)

verify$predictedvalue <- predict(model, verify, type = "response")


#Confusion matrix
(table("Actual value" = verify$Loan_Status, "Predicted value" = verify$predictedvalue > 0.5))

# Accuracy = 78.07%

#Plotting ROCR
roc<- predict(model, verify, type = "response")
ROCRPred = prediction(roc, verify$Loan_Status)
ROCRPref <- performance(ROCRPred, "tpr", "fpr")
plot(ROCRPref, colorize = TRUE, print.cutoffs.at = seq(0.1, by = 0.1))

#Applying model to test dataset
test$Loan_Status_perc <- predict(model, test, type = "response")
test$Loan_Status <- test$Loan_Status_perc

#Changing percentage to "Y" OR "N" based on cutoff percentage
yes <- which(test$Loan_Status >= 0.5)
test[yes,]
test$Loan_Status[yes] <- "Y"

no <- which(test$Loan_Status < 0.5)
test[no,]
test$Loan_Status[no] <- "N"

#Printing the required output
output<- select(test, c(Loan_ID, Loan_Status))

Output1 <- unite(output, loanstatus, Loan_ID, Loan_Status, sep = ", ")
names(Output1) <- NULL
Output1


write.csv(Output1,'output.csv')

################################################RANDOM Forest################################################
#Building Random Forest model
RFmodel <- randomForest(Loan_Status ~ Dependents + Self_Employed + ApplicantIncome+ CoapplicantIncome + LoanAmount + Credit_History + Property_Area, data = train, mtry =5, ntree = 400,  na.action=na.roughfix)

class(RFmodel)
str(RFmodel)
RFmodel
importance(RFmodel)

verify$predictedvalueRF <- predict(RFmodel,verify)
(table("Actual value" = verify$Loan_Status, "Predicted value" = verify$predictedvalueRF))
confusionMatrix(table(verify$predictedvalueRF, verify$Loan_Status))

# Visualizing the model
#plotRF <- ggplot(verify, aes(Loan_Status, predictedvalueRF)) + geom_point(shape=21, alpha=0.6) + stat_smooth(method = "lm", size =1, se = FALSE) + 
  labs(x = "Actual MV", y = "Predited MV") + ggtitle("Regression line using Random Forest")
#plotRF

#This plot shows the Error and the Number of Trees.We can easily notice that how the Error is dropping as we keep on adding more and more trees and average them.
plot(RFmodel) 

#Applying model to test dataset
test$Loan_Status_percRF <- predict(RFmodel, test, type = "response")


output<- select(test, c(Loan_ID, Loan_Status, Loan_Status_percRF))

#######################Naive Bayes#################################
modelnb <- naiveBayes(Loan_Status ~ ApplicantIncome+ LoanAmount + Credit_History + Property_Area, train)
modelnb

verify$predictedvalueNB <- predict(modelnb,verify)
(table("Actual value" = verify$Loan_Status, "Predicted value" = verify$predictedvalueNB))
confusionMatrix(table(verify$predictedvalueNB, verify$Loan_Status))

#Applying model to test dataset
test$Loan_Status_percNB <- predict(modelnb, test)

output<- select(test, c(Loan_ID, Loan_Status, Loan_Status_percRF, Loan_Status_percNB))
