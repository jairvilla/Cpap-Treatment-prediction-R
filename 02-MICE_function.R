#load data
library(Hmisc) 
data <- iris
#Get summary
summary(iris)
describe(iris)
fix(iris)
#Generate 10% missing values at Random 
iris.mis <- iris
#Check missing values introduced in the data
names(iris)
dim(iris.mis)
summary(iris.mis)

iris.mis <- subset(iris.mis, select = -c(Species))
data.frame (colSums(is.na(iris.mis)))

##- MICE algorithm 
library(mice)
md.pattern(iris.mis)

install.packages("VIM")
library(VIM)
mice_plot <- aggr(iris.mis, col=c('navyblue','yellow'),
                    numbers=TRUE, sortVars=TRUE,
                    labels=names(iris.mis), cex.axis=.7,
                    gap=3, ylab=c("Missing data","Pattern"))

imputed_Data <- mice(iris.mis, m=5, maxit = 50, method = 'pmm', seed = 500)
summary(imputed_Data)
# Here is an explanation of the parameters used:
# m  - Refers to 5 imputed data sets
# maxit - Refers to no. of iterations taken to impute missing values
# method - Refers to method used in imputation. we used predictive mean matching.

completeData <- complete(imputed_Data,2) 
CompleteData5 <- complete(imputed_Data,5)
#build predictive model
# dataset with missing data
model1 <- with(data=iris.mis, exp= lm ( Sepal.Width~ Sepal.Length + Petal.Width))
summary(model1)
# dataset w/o missing data,  MICE algorithm was implemented
model2 <- with(data= completeData, exp= lm ( Sepal.Width~ Sepal.Length + Petal.Width))
summary(model2)
# model with iteration 5
model3 <- with(data= CompleteData5, exp= lm ( Sepal.Width~ Sepal.Length + Petal.Width))
summary(model3)

## Conclusion 
## Cualquier iteracion que se tome del MICE presenta un resultado similar. 

#combine results of all 5 models
combine <- pool(model1)  ## Does not work!!! 
summary(combine)         


