# Title: Machine learning approaches for identifying predictors to CPAP 
#        Adherence in OSA patients
# Data set: CPAP1.SAV
# Aims:To identify potential variables to improve CPAP treatment 
# Predictive Model: SVM, Random forest, KNN.
# Author: Jair villanueva, @2019 
#...........................................................

# Install Packages for ML models ----
if(!require(caret)) install.packages("caret");
if(!require(kernlab)) install.packages("kernlab");
if(!require(mlr3)) install.packages("mlr3");
if(!require(randomForest)) install.packages("randomForest");
if(!require(superml)) install.packages("superml");
if(!require(e1071)) install.packages("e1071")

install.packages(c("pROC","glmnet","kernab","DT","psych","compareGroups","haven",
                   "klaR", "pscl","gmodels","xtable","randomForest", "ggpubr"))


# Load Packages for ML models ------
libraries <- c("dplyr", "tidyr", "caret","xtable","data.table", "lubridate","ROCR",
          "broom", "klaR","randomForest","corrplot", "nnet","DT","rio","pROC",
          "tidyverse","pscl","psych", "MASS", "ggplot2", "kernlab", "knitr", 
          "gmodels","compareGroups", "ggpubr","pROC", "glmnet","ROCR", "mlr3",
          "superml","e1071", "haven") 

sapply(libraries, require, character.only = TRUE)
#............................................................

# Load and save data ........................................
save.image(file="originalCpap_02.csv")
save.image(file="run20.RData")
load(file = "run20.RData")

# Export dataframe to cvs file 
getwd()
write.csv(psgfinal,"C:/Users/jvill/Desktop/Desktop/DataScience-2021/3-CPAP-Prediction/Cpap-Treatment-prediction-R\\psgfinal.RData", row.names = TRUE)

#............................................................


# Load and explore the dataset (cpap) from SAV Format
# cpap contains the features and information clinical patients from dataset  

# Original dataset 1 (430/335)
cpap1 <- as.data.frame(read_sav("data/originalCpap_01.sav")) # original dataset
dim(cpap1)
str(cpap1)

#--- original dataset 2 (430/50)
cpap2 <- as.data.frame(read_sav("data/originalCpap_02.sav")) # original dataset
dim(cpap2)
str(cpap2)

# Split into two Groups: PSG and HRP 
table(cpap1$grupo2)  # HPR: 218 and PSG: 212 observations

# Filter by PSG and HRP diag. method 
psg <- cpap1 %>%
  filter(grupo2=="PSG", abandono=="no")
dim(psg)  # 167 x 335

hrp <- cpap1 %>%
  filter(grupo2=="HRP", abandono=="no")
dim(hrp)  # 187 x 335

# NA Values 
missing <- as.data.frame(colSums(is.na(psg))) 
missing  


# Selecting variables from psg dataset ----
psgfinal <- psg %>%
  dplyr::select ("sexo","edad", "depresion","hta","ansiedad","cardio","enf_neuro","enf_respira",
        "dm","dislipe", "neoplasias", "acc_laboral","acc_trafico","fuma", "bebe", "epworth", 
        "talla", "peso", "p_cuello","depresion_A", "glucosa", "creatinina", "urico", "ast",   
        "colesterol", "hdl", "ldl", "trigliceridos", "hemoglobina", "hematies", "tsh",
        "hematocrito", "fibrinogeno","plaquetas", "leucocitos", "hba1c","alt","cefalina",   
        "tas", "tad", "ta_media_24", "ta_media_diurna", "ta_media_nocturna", "ta_diastolica_24",
        "ta_sistolica_24", "ta_sistolica_diurna","ta_sistolica_nocturna", "tas_valle","tad_pico",  
        "ta_diastolica_diurna", "ta_diastolica_nocturna", "tas_pico", "p_superficial_psg", 
        "tad_valle", "fcard_dia", "fcard_noche", "tiempo_total_psg", "sato2_media_psg", 
        "p_profundo_psg", "p_rem_psg", "ind_arousal_psg", "iah_psg", "somno_conducir", "uci_dias",
        "tc90_psg", "i_desatura_psg", "epworth_A", "escala_asda",  "sf36_fm", "eq50","sf36_ff",
        "acc_trafico_F","acc_laboral_F", "acc_laboral_lesiones","fosq",  
        "n_horas_dia_A")

# Anoter outcome : "n_horas_dia_B","n_horas_dia_C"

dim(psgfinal)

## This dataset constains non-redundant variables,standarize categorical varaibles as factor
# cpap3 <- dplyr::select(cpap1, glucosa,  ta_media_diurna, ta_media_nocturna, ta_media_24, 
#                        ta_sistolica_24, ta_sistolica_diurna, tas1, ta_sistolica_nocturna,
#                        tas_pico, arousal, iah, doi, enf_respira_A, class)
                    

# Otro dataset ----
# cpap3 <- dplyr::select(cpap1, tas_pico, enf_respira_A, neoplasias_A, arousal, fibrinogeno, 
#                        fosq, analogica, tas1, asda, depresion_A, quick, hematocrito, 
#                        cuello, hdl, hemoglobina, ast, class)


# Pre-processing data ----
# Convert outcome in factor
psgfinal$n_horas_dia_A <- as.factor(ifelse(psgfinal$n_horas_dia_A>=4, 1, 0))  # response variable (Cumplimiento)
dim(psgfinal)

summary(psgfinal)

## Convert chr variables to factor
psgfinal   <-  as.data.frame(unclass(psgfinal),
                stringsAsFactors = TRUE)
summary(psgfinal)
str(psgfinal)

# Select Numerical Variable
psgnum <- psgfinal %>%
  dplyr::select_if(is.numeric)
names(psgnum)

# Export initial describe stats from dataset psgfinal

sink(file = "resultspsg.csv", append = T)
describe(psgnum, na.rm = T)
sink()

# Apply MICE Algorithm 

# Plot NA Count 
psgnum %>% 
  summarise_all(funs(sum(is.na(.)))) %>%     # Return a barplot with missing data 
  gather %>% 
  ggplot(aes(x = reorder(key, value), y =value)) + geom_bar(stat = "identity") +
  coord_flip() +
  xlab("variable") +
  ylab("Absolute number of missings")

# Mice Algorithm ----
if(!require(VIM)) install.packages("VIM");
if(!require(mice)) install.packages("mice");
library(VIM);library(mice)

imp <-mice(psgnum, m=5, printFlag=FALSE, maxit = 5, seed=145)
psgnumx <- complete(imp,5)
dim(psgnumx)

# Check NA again ----
miss <- data.frame(Missdata = sapply(psgnumx, function(x) sum(is.na(x))))
print(miss)                     

sink(file = "resultspsgMice.csv", append = T)
describe(psgnumx, na.rm = T)
sink()

# Select numeric variables with MICE + Outcome 
cpapfinal  <- cbind(psgnumx,class=psgfinal$n_horas_dia_A) 
dim(cpapfinal)

# Complete case 
# Keep only complete cases for the complete case analysis
cpapfinal <- cpapfinal[which(complete.cases(cpapfinal) == TRUE), ]

# Scatterplot correlation Coeficient
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(cpapfinal, method = "color", col = col(200),  
         type = "upper", order = "hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col = "darkblue", tl.srt = 45, #Text label color and rotation
         # Combine with significance level
         p.mat = p_mat, sig.level = 0.01,  
         # hide correlation coefficient on the principal diagonal
         diag = FALSE 
)




# Exploratory Analysis  ----

# NA's Handle 
str(psgfinal)
missing <- as.data.frame(colSums(is.na(psgfinal))) 
missing  


#.............................................



# Applying Mice algorithm 


# Selecting the variables from first month
# Relevel variables 
cpap3$class <- relevel(cpap3$class, ref="1")  # Reorder levels of factor, ref = 1 then we estime the probability of fealure.

#.....................................................

# Remove reduntant variable  -----
# Correlation Matrix     
correlationMatrix <- cor(cpap1[,1:58])                              # Select only numerical Variables
hcorrelated       <- findCorrelation(correlationMatrix, cutoff=0.6) # Threshold >0.6, Find Features that are highly corrected 
print(hcorrelated)                                                  # print indexes of highly correlated attributes
highly_cor_var    <- colnames(cpap1[hcorrelated])                   # displaying highly correlated variables
data.frame(highly_cor_var)

# Removing the highly correlated variables
cpap_no_hcor <- cpap1[,-hcorrelated]

str(cpap_no_hcor)

cpap3 <- cpap_no_hcor
names(cpap3)

#............................................................
# Standarized Numeric variables 
num <- dplyr::select(cpap3,1:35)# Select categorical variables .
cat <- dplyr::select(cpap3,36:51) # Select Numeric var

# standarize Function
standarize <- function(x) {
  num   <- x - mean(x)
  denom <- sd(x)
  return (num / denom)
}

# Applying function  ............................
num   <- as.data.frame(lapply(num, standarize)) # apply function 
cpap3 <-cbind(num, cat)                         # merge dataframes 
summary(cpap3)
str(cpap3)
nums <- unlist(lapply(cpap, is.numeric))  

str(cpap)
names(cpap)

nums<- dplyr::select_if(cpap, is.numeric)
nums
table(cpap3$class)

names(cpap3)

# complete case analysis
cpap3 <- cpap3[complete.cases(cpap3),] # omit NA values from data set
names(cpap3)
# ...............................................

# Preparing for Models  ------ 
# Description of dataset -----
str(cpap3_copy)  # Num var. standiz sin HC, cat var. factor.  
str(cpap3)       # Num var. no standiz, all variables, cat var. factor.
cpap3<- cpap3_copy
str(cpap3)

#..............................................................


#..............................................
# Split dataset ----
set.seed(2019)
inTrain <- createDataPartition(y=cpap3$class, p = 0.70, list =FALSE)
train   <- cpap3[inTrain,]
test    <- cpap3[-inTrain,]
x_train <- subset(train, select= -c(class))            # only predictors from train 
x_test  <- subset(test, select= -c(class))             # only predictors from test
y_train <- as.factor(subset(train, select= c(class)))  # outcome from train data 
y_test  <- as.factor(subset(test, select= c(class)))   # outcome from test data
control <- trainControl(method = 'repeatedcv',number = 10,repeats = 3)


#......................................................
# Building Predictive Models ----
## Model 1: Logistic Model  ----- 
set.seed(2019)
glmFit <- glm(class ~.,
              data= train, 
              family=binomial(link="logit"), 
              maxit=100)

logFit <- train(class ~.,
                data = train,
                method = 'glm',
                preProc = c("center", "scale"),
                trControl = control)

# Summary model 
sink(file= "logistic21.doc")
summary(glmFit)
exp(coef(glmFit))      # coef values
summary(logFit)
caret::varImp(glmFit)  # variable is the most influential in predicting
sink()

#.......................................................
# Model 2: BACKWARD 
set.seed(2019) 
backward <- glmFit %>%   # We uses this methods for backward
  stepAIC (glmFit, direction=c("backward"))

sink(file = "back21.doc")
summary(backward)
sink()
formula(backward)  # variables include in model

#........................................................
# Model 3: Forward Model  ------
# Null model , we used for forward meth 
set.seed(2019)
null_model <- glm(class ~1 ,data=train,family=binomial(link="logit"))
null_model$aic
# full model 
full_model <- glm(class ~.,data=train,family=binomial(link="logit"))
## forward 
forward   <- step(null_model,scope=list(lower=formula(null_model),upper=formula(full_model)), direction="forward")

sink(file="forward21.doc")
summary(forward)
sink()

# For backward
#backward  <- step(null_model,scope=list(lower=formula(null_model),upper=formula(full_model)), direction="backward")
#summary(backward)


#..................................................
# Model 4: Both  ----  
set.seed(2019)
both <-step(null_model,scope=list(lower=formula(null_model),upper=formula(full_model)), direction="both")

sink(file="both21.doc")
summary(both)
sink()
#..................................................

# Model 5: Lasso Model ------
x <- model.matrix(class~., train)[ ,-which(names(train) %in% "class")]# features variable as matrix
y <- as.double(as.matrix(train[,which(names(train) %in% "class")]))
y <- as.factor(train$class) # when we tuning with caret.

#my_control <-trainControl(method="cv", number=5)

lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.005))# Lambda values 

# Tuning Lasso parameters with caret (requere que y sea factor)
# option 1: Tuning using Caret
lasso_mod <- train(x=x,
                   y=y,
                   method='glmnet',
                   trControl= control,
                   standardize= TRUE,
                   tuneGrid=lassoGrid) 
lasso_mod$bestTune
lasso_mod$results
print(bestlambda <-lasso_mod$bestTune[,2])  # Select the best lambda value

coef(lasso_mod$finalModel, lasso_mod$bestTune$lambda) # coefficients

plot (lasso_mod, xvar="lambda", xlab = "lambda",         # Show coefficients, lambda value and number of features selected 
      ylab = "Value of the coefficients", label =TRUE)

# Showing selected variables for model
lassoVarImp     <- varImp(lasso_mod,scale=F)
lassoImportance <- lassoVarImp$importance
varsSelected    <- length(which(lassoImportance$Overall!=0))
varsNotSelected <- length(which(lassoImportance$Overall==0))
cat('Lasso uses', varsSelected, 'variables in its model, and did not select', varsNotSelected, 'variables.')

# Fit model with best values ............................ 
# Fit the Lasso model with the lambda value that minimizes error (deviance)

library("glmnet")
set.seed(344)
lasso <- glmnet(x, y, 
                family = "binomial",
                alpha = 1,
                lambda = bestlambda, 
                standardize = TRUE)

lasso$df

# plot the most relevant variables ........................
library(broom)
coef(lasso, s = "lambda.min") %>%
  tidy() %>%
  filter(row != "(Intercept)") %>%
  ggplot(aes(value, reorder(row, value), color = value > 0)) +
  geom_point(show.legend = FALSE) +
  ggtitle("Influential variables") +
  xlab("Coefficient") +
  ylab(NULL)


# # extract coefficients for the best performing model (from cross validation) (only caret package)
# coef <- data.frame(coef.name = dimnames(coef(lasso_mod$finalModel,s=lasso_mod$bestTune$lambda))[[1]], 
#                    coef.value = matrix(coef(lasso_mod$finalModel,s=lasso_mod$bestTune$lambda)))
# 
# print(coef <- coef[-1,])                            # exclude the (Intercept) term
# picked_features <- nrow(filter(coef,coef.value!=0))    
# not_picked_features <- nrow(filter(coef,coef.value==0))
# 
# # Output
# cat("Lasso picked",picked_features,"variables and eliminated the other",
#     not_picked_features,"variables\n")
# 
# coef <- arrange(coef,-coef.value) # Coeficients 
# coef
# 
# # extract the top 10 and bottom 10 features
# imp_coef <- rbind(head(coef,10),
#                   tail(coef,10))
# 
# ggplot(imp_coef) +
#   geom_bar(aes(x=reorder(coef.name,coef.value),y=coef.value),
#            stat="identity") +
#   ylim(-1.5,0.6)   +
#   coord_flip()     +
#   ggtitle("Coefficents in the Lasso Model") +
#   theme(axis.title=element_blank())

#.............................................................
# Tuning model with glmnet function  
set.seed(123) 
library(glmnet)
cv.lasso2 <- cv.glmnet(x = x,
                       y = y,
                       family="binomial", 
                       standardize = TRUE,
                       alpha = 1 # LASSO 
)

coef(cv.lasso2) # coefficients selected  

plot (cv.lasso2, xvar="lambda", xlab = "lambda",         # Show coefficients, lambda value and number of features selected 
      ylab = "Value of the coefficients", label =TRUE)  # xvar="dev"

# Print the minimum lambda - regularization factor
print(bestlambda2 <- cv.lasso2$lambda.min)  # lambda for this min MSE
cv.lasso2$lambda.1se  # lambda for this MSE

# fit lasso model with glmnet function 
set.seed(134)
lasso2 <- glmnet(x, y, 
                 family = "binomial",
                 alpha = 1,
                 lambda = bestlambda2, 
                 standardize = TRUE)

lasso2$df # Number of Varibles selected
# Coefficients
coef(lasso2)

plot(lasso2, xvar="lambda",xlab=" Log of Lambda", ylab="Binomial deviance", label=TRUE)
abline(v = log(ames_ridge$lambda.1se), col = "red", lty = "dashed")

## Lasso model
lassocoefficients <- predict (lasso2, type = "coefficients", s = bestlambda)
lassocoefficients

## Sparse LASSO model  (cvlasso$lambda.1se)
lassosparse <- glmnet(x, y, family = "binomial", alpha = 1, lambda = lasso2$lambda.1se, standardize = FALSE)
lassosparse                      

## Lasso model
lassosparsecoefficients <- predict (lassosparse, type = "coefficients", s = lasso2$lambda.1se)
lassosparsecoefficients

# plot the most relevant variables ........................
library(broom)
coef(lasso2, s = "lambda.min") %>%
  tidy() %>%
  filter(row != "(Intercept)") %>%
  ggplot(aes(value, reorder(row, value), color = value > 0)) +
  geom_point(show.legend = FALSE) +
  ggtitle("Influential variables") +
  xlab("Coefficient") +
  ylab(NULL)

#..........................................................

# Model 6: Random Forest -----
# Tune parameters of model 

set.seed (2019)
## Divide the training set into 3 folds
folds   <- sample(rep(1:3, length = nrow(train)))
## Make sure folds have more or less the same size
table (folds)
features <- (ncol(train))-1 

mtryvalues <- 1:(length(train)-1) # Replace for total number of features of dataset 
set.seed(2019)     #  an arbitrary number
## Set an empty data frame with the proportion misclassified 
##(rows will be different mtry, and columns values of k) 
proportionerror <- data.frame(matrix(0, ncol = 3, nrow = features)) # change for number total of  features
colnames(proportionerror) <- c("k = 1", "k = 2", "k = 3")

## Calculate the best mtry by cross-validation
## Create a loop with 3 folds
for (k in 1:3) {
  ## Create a loop with 18 values of mtry
  for (j in 1:features) {
    ## Calculate the random forest model based on 2 of the folds
    forest <- randomForest(class ~ ., data = train[folds != k,], 
                           mtry = mtryvalues[j], ntree = 1000, importance = TRUE)
    ## Predict performance on the remaining fold not used for fitting the model
    prediction <- predict(forest, newdata = train[folds == k,-which(names(train) == "class")]) 
    ## Confusion matrix for the prediction
    confusionmatrix <- table(prediction, train[folds == k, which(names(train) == "class")]) 
    ## Misclassification error
    proportionerror[j, k] <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
      (confusionmatrix[1, 1] + confusionmatrix[1, 2] + 
         confusionmatrix[2, 1] + confusionmatrix[2, 2])
  }
}

# Calculate the mean prediction error for each mtry
proportionerror$meanerror <- rowMeans(proportionerror)
proportionerror$`k = 1`    # Expresa la proporcion de error en cada fold
mean(proportionerror$`k = 3`) # mean of error 

# Plot the proportion of errors in relation with the number of variables in mtry
plot(mtryvalues, proportionerror$meanerror, pch = 16, cex = 2, 
     col = "red", type = "o", lwd = 2,
     xlab = "Number of variables used at each node", 
     ylab = "Misclassification in the fold not used for developing the model")

## Identify the lowest mtry within 1 standard error of the mtry with minimum error
mtryminimumerror <- which(proportionerror$meanerror == min(proportionerror$meanerror))[1]
mtryminimumerror

# Fit forest model with mtry selected

forest <- randomForest(class ~ ., data = train, 
                       mtry = mtryminimumerror, ntree = 1000, importance = TRUE)

# Fit Forest2 with tunning............................
forest2 <- randomForest(class ~ ., data = train, 
                        mtry = (length(cpap3)/3), ntree = 1000, importance = TRUE)

# Top variables used into model ........................
importance    <- importance(forest2)

varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseAccuracy'],1))

rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))
rankImportance

#Use ggplot2 to visualize the relative importance of variables  ........
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  #geom_text(aes(x = Variables, y = 0.5, label = Rank),
  #hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_bw() 

# Plots relevant varaibles ............................

varImpPlot(forest, main ="Variables sorted by importance based on mean decrease accuracy and on mean decreased Gini")

# Regarding to mtry2
varImpPlot(forest2, main ="Variables sorted by importance based on mean decrease accuracy and on mean decreased Gini")


#............................................................
# Model 7:  SVM- Linear kernel ----
# Tunning Linear- SVM   .............................. 
library(e1071)
set.seed(2342)
svmparameterslinear <- tune(svm, class ~ ., data = train, 
                            kernel = "linear", 
                            ranges = list(cost = c(0.001,0.05,0.01, 0.1, 1,2, 5, 10, 100)),
                            tunecontrol = tune.control(cross = 10), 
                            standarize = TRUE, probability = TRUE)

svmlinear <- svmparameterslinear$best.model## Parameters for the best model
svmlinear
# Export results  ......................................
sink(file="svmlinear21.doc")
summary(svmparameterslinear)
svmlinear <- svmparameterslinear$best.model## Parameters for the best model
print(svmlinear)
sink()

#.................................................

# Model 8: SVM-Polynomial Kernel ----- 

# Tunning hyperparameters...............
set.seed(3232)
# svmparameterspoly <- tune(svm, class ~ ., data= train, 
#                           kernel = "polynomial", 
#                           ranges = list(cost = list(cost = c(0.001,0.05,0.01, 0.1, 1,2, 5, 10, 100)),
#                                         degree = seq(1, 6, 1)),
#                           tunecontrol = tune.control(cross = 10), 
#                           scale = TRUE, probability = TRUE)


svmparameterspoly <- tune.svm(class ~ ., data= train, 
                              kernel = "polynomial", 
                              cost = c(0.001,0.05,0.01, 0.1, 1,2, 5, 10, 100),
                              degree = seq(1, 6, 1),
                              coef0=c(0.001,0.1,0.5,1,2,3),
                              tunecontrol = tune.control(cross = 10), 
                              scale = TRUE, probability = TRUE) 

svmparameterspoly$best.model

## Select the best model................
svmpolynomial  <- svmparameterspoly$best.model
summary(svmpolynomial)

# Export results SVM poly......................
sink(file="svmpoly21.doc")
summary(svmparameterspoly)
svmparameterspoly$best.parameters # Parameters for the best model
svmpolynomial  <- svmparameterspoly$best.model
sink()

# ....................................
# Model 9: K-NN ----
set.seed(445)
knnFit <- train(class~.,
                data = train,
                method = "knn",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(.k = 1:10),
                trControl = control)

#.....................................................

# ASSESSMENT OF THE MODEL'S PERFORMANCE ------
# .....................................................

#  Model 1: Prediction with LogisticR ----
predictionglm <- predict(glmFit, newdata = x_test, type="response")  # Probabilidad de que sea 1(no cumple)
predglm01     <- ifelse(predictionglm> 0.5, 1,0)
predglm01     <- as.factor(predglm01)

# AUC and CI   .................................. 
print(aucglm   <- auc(test$class,predictionglm))
print(ciaucglm <- ci.auc(test$class, predictionglm))
rocglm          <-roc(test$class,predictionglm)$auc

# Compute error  ................................
confusionmatrix  <- table(predglm01, 
                          test[,which(names(test) == "class")])
## Missclasification --
misclassificationglm <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])

misclassificationglm

# Matrix confusion   ....................................
print(cm.glm <- confusionMatrix(data=predglm01,reference = test$class)) 

# Threshold and AUC ......................................
plot.roc(test$class, predictionglm,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE,ci.auc=TRUE) # a

## Calculate optimal threshold with Youden's index  .......
rocglm  <- roc(test$class, predictionglm)
print(bestglm <- coords(rocglm, "b", ret = "threshold", best.method = "youden"))

# Threshold Adj. fit Model ...............................
predglm01  <- ifelse( predictionglm > bestglm, 1, 0)
predglm01  <- as.factor(predglm01)

# Matrix Confusion  .....................................
sink(file="glm_prediction21")
print(cm.glm01 <- confusionMatrix(data=predglm01, reference = test$class, positive = "1")) 
sink()

# Compute error .........................................
confusionmatrix  <- table(predglm01,test[,which(names(test) == "class")]) 

# Missclasification  ....................................
misclassificationglm01 <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationglm01

#.........................................................
# Summary of GLM model---- 
sink(file="glm21.doc")
print(cm.glm01<- confusionMatrix(data=predglm01, reference = test$class, positive = "1")) 
acc.glm <- cm.glm01$overall["Accuracy"]
sen.glm <- cm.glm01$byClass["Sensitivity"]
spe.glm <- cm.glm01$byClass["Specificity"]
ppv.glm <- cm.glm01$byClass["Pos Pred Value"]
npv.glm <- cm.glm01$byClass["Neg Pred Value"]
f1.glm  <- cm.glm01$byClass["F1"]
k.glm   <- cm.glm01$overall["Kappa"]
auc.glm <- as.numeric(rocglm$auc)
cat("Error is:",misclassificationglm01, "AUC:",rocglm$auc, 
    "CI 95%:", ci.auc(test$class, predictionglm))
sink()


#.....................................................

## Model 2: Backward prediction---- 
predictionbackward <- predict(backward, newdata= x_test, type ="response") 
print(aucbackward  <- auc(test$class, predictionbackward))
print(cibackward   <- ci.auc(test$class, predictionbackward))

# Prediction based on misclassification error
predictionbackward01 <- ifelse(predictionbackward > 0.5, 1, 0)  # the output as class
predictionbackward01 <- as.factor(predictionbackward01)         # output as factor

confusionmatrix           <- table(predictionbackward01,test[ , which(names(test) == "class")]) 
misclassificationbackward <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationbackward                                      # compute the error

# summary results: confusionMatrix  ................
sink(file="backward21-1.doc")
print(cm.backward<- confusionMatrix(data=predictionbackward01 ,reference = test$class, positive = "1"))
sink()

# Compute threshold and AUC ........................................
plot.roc(test$class, predictionbackward,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) 

## Option 2: Compute optimal threshold with Youden's index  .........
rocbackward  <- roc(test$class, predictionbackward)
bestbackward <- coords(rocbackward, "b", ret = "threshold", best.method = "youden")
bestbackward                      # optimal cut-off for predictive model

## Prediction based on misclassification error .......................
predictionbackward01 <- ifelse(predictionbackward > bestbackward,1, 0)
predictionbackward01 <- as.factor(predictionbackward01)

confusionmatrix      <- table(predictionbackward01, 
                              test[,which(names(test) == "class")])

## Missclasification ..................................................
misclassificationbackward01 <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationbackward01

## Summary Results of backward model ----- 
sink(file="back21-2.doc")
print(cm.backward<- confusionMatrix(data=predictionbackward01 ,reference = test$class, positive = "1"))
acc.backward <- cm.backward$overall["Accuracy"]
sen.backward <- cm.backward$byClass["Sensitivity"]
spe.backward <- cm.backward$byClass["Specificity"]
ppv.backward <- cm.backward$byClass["Pos Pred Value"]
npv.backward <- cm.backward$byClass["Neg Pred Value"]
f1.backward  <- cm.backward$byClass["F1"]
k.backward   <- cm.backward$overall["Kappa"]
auc.backward <- as.numeric(rocbackward$auc)
cat("Error is:",misclassificationbackward01, "AUC:",rocbackward$auc, 
    "CI 95%:", ci.auc(test$class, predictionbackward))
sink()

#............................................................

## Model 3: Prediction with Forward model  ---- 
predictionforward <- predict(forward, type = "response", newdata = test[,-which(names(test)=="class")]) 
# AUC FORWARD .................................................
print(aucforward  <- auc(test$class, predictionforward))
print(ciforward   <- ci.auc(test$class, predictionforward))

## Prediction based on misclassification error  ...............
predictionforward01  <- ifelse(predictionforward > 0.5, 1, 0)
predictionforward01  <- as.factor(predictionforward01)
confusionmatrix      <- table(predictionforward01, 
                              test[ , which(names(test) == "class")])

misclassificationforward <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])

misclassificationforward

# Summary backward prediction .................................. 
sink(file="forward21-1.doc")
print(cm.forward<- confusionMatrix(data=predictionforward01 ,reference = test$class, positive = "1"))
sink()

# Threshold and AUC ............................................
plot.roc(test$class, predictionforward,
         main="Comparison of AUC backward_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a


## Compute optimal threshold with Youden's index ..............
rocforward        <- roc(test$class, predictionforward)
print(bestforward <- coords(rocforward, "b", ret = "threshold", best.method = "youden"))
bestforward

## Prediction based on misclassification error ...............
predictionforward01 <- ifelse(predictionforward > bestforward, 1, 0)
predictionforward01 <- as.factor(predictionforward01)

## Missclasification01 .........................................
confusionmatrix            <- table(predictionforward01,test[,which(names(test) == "class")]) 
misclassificationforward01 <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationforward01  # error after selecting optimal threshold. 

# Summary of Results forward  ------
sink(file="forward21-2.doc")
print(cm.forward <-confusionMatrix(data=predictionforward01,reference=test$class,positive="1"))
acc.forward <- cm.forward$overall["Accuracy"]
sen.forward <- cm.forward$byClass["Sensitivity"]
spe.forward <- cm.forward$byClass["Specificity"]
ppv.forward <- cm.forward$byClass["Pos Pred Value"]
npv.forward <- cm.forward$byClass["Neg Pred Value"]
f1.forward  <- cm.forward$byClass["F1"]
k.forward   <- cm.forward$overall["Kappa"]
auc.forward <- as.numeric(rocforward$auc)
cat("Error is:",misclassificationforward01, "AUC:",rocforward$auc, 
    "CI 95%:", ci.auc(test$class, predictionforward))
sink()

#.................................................

## Model 4: Prediction With Both model ---- 

predictionboth <- predict(both, type = "response", newdata = test[,-which(names(test)=="class")]) 
# AUC Both 
print(aucboth <- auc(test$class, predictionboth))
print(ciboth  <- ci.auc(test$class, predictionboth))

## Prediction based on misclassification error ................
predictionboth01 <- ifelse(predictionboth > 0.5, 1, 0)
predictionboth01 <- as.factor(predictionboth01)

## Misclassification  ........................................
confusionmatrix       <- table(predictionboth01,test[,which(names(test) == "class")]) 
misclassificationboth <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationboth

# Summary   ..................................................
sink(file="both21-1.doc")
print(cm.both <-confusionMatrix(data=predictionboth01,reference=test$class,positive="1"))
sink()

# Optimal threshold ........................................
plot.roc(test$class, predictionboth,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a

## Calculate optimal threshold with Youden's index ..........
rocboth        <- roc(test$class, predictionboth)
print(bestboth <- coords(rocboth, "b", ret = "threshold", best.method = "youden"))

## Prediction based on misclassification error  .............
predictionboth01 <- ifelse(predictionboth > bestboth, 1, 0)
predictionboth01 <- as.factor(predictionboth01)

## Missclasification ........................................
confusionmatrix       <- table(predictionboth01,test[,which(names(test) == "class")]) 
misclassificationboth01 <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationboth01

## Summary Both model ------ 
sink(file="both21-2.doc")
print(cm.both <-confusionMatrix(data=predictionboth01,reference=test$class, positive="1"))
acc.both <- cm.both$overall["Accuracy"]
sen.both <- cm.both$byClass["Sensitivity"]
spe.both <- cm.both$byClass["Specificity"]
ppv.both <- cm.both$byClass["Pos Pred Value"]
npv.both <- cm.both$byClass["Neg Pred Value"]
f1.both  <- cm.both$byClass["F1"]
k.both   <- cm.both$overall["Kappa"]
auc.both <- as.numeric(rocboth$auc)
cat("Error is:",misclassificationboth01, "AUC:",rocboth$auc, 
    "CI 95%:", ci.auc(test$class, predictionboth))
sink()

# ...........................................

# Model 5: Prediction with LASSO model ---- 
## Create the x matrix
x <- model.matrix(class~., data = test)[ , -which(names(test) %in% "class")]   # create matrix with predictors
y <- as.double(as.matrix(test[,which(names(test) %in% "class")]))

# Lasso predictions .......................................

predictionlasso <- predict(lasso2, newx = x, type = "response")
predictionlasso <- as.double(predictionlasso)

# Area under the curve .....................................
print(auclasso <- auc(y, predictionlasso))
print(cilasso <-  ci.auc(y, predictionlasso))

## Prediction based on misclassification error ..............
predictionlasso01 <- ifelse(predictionlasso > 0.5, 1, 0)
predictionlasso01 <- as.factor(predictionlasso01)

confusionmatrix   <- table(predictionlasso01, 
                           test[ , which(names(test) == "class")])

misclassificationlasso <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationlasso  # compute error 

# summary .............................................................
sink(file="lasso21-1.doc")
print(cm.lasso <-confusionMatrix(data=predictionlasso01, reference= test$class, positive="1"))
sink()

## Calculate optimal threshold with Youden's index .....................
roclasso  <- roc(test$class, predictionlasso)
bestlasso <- coords(roclasso, "b", ret = "threshold", best.method = "youden")
print(bestlasso)

#Compute the threshold and AUC  .........................................
plot.roc(test$class, predictionlasso,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a

## Prediction based on misclassification error  .........................
predictionlasso01 <- ifelse(predictionlasso > bestlasso, 1, 0)
predictionlasso01 <- as.factor(predictionlasso01)
confusionmatrix   <- table(predictionlasso01, 
                           test[ , which(names(test) == "class")])

misclassificationlasso01 <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationlasso01

## Summary of Prediction LASSO model  -------
sink(file ="lasso21-2.doc")
print(cm.lasso <-confusionMatrix(data=predictionlasso01,reference= test$class,positive="1"))
acc.lasso <- cm.lasso$overall["Accuracy"]
sen.lasso <- cm.lasso$byClass["Sensitivity"]
spe.lasso <- cm.lasso$byClass["Specificity"]
ppv.lasso <- cm.lasso$byClass["Pos Pred Value"]
npv.lasso <- cm.lasso$byClass["Neg Pred Value"]
f1.lasso  <- cm.lasso$byClass["F1"]
k.lasso   <- cm.lasso$overall["Kappa"]
auc.lasso <- as.numeric(roclasso$auc)
cat("Error is:",misclassificationlasso01)
cat("AUC:",roclasso$auc, "CI 95%", ci.auc(test$class, predictionlasso))
sink()

#.......................................................

# Model 6: Prediction with Random Forest model  ------
predictionrf    <- predict(forest2, newdata=x_test, type ='prob')[,2] # for succesful treatment
predictionrf01  <- predict(forest2,newdata = test[ , - which(names(test) == "class")]) # class
predictionrf01  <- as.factor(predictionrf01)   # vector as factor. 

# Compute AUC and CI .....................................
print(aucrf  <- auc(test$class, predictionrf))
print(cirf   <- ci.auc(test$class, predictionrf))

## Missclasification ......................................
confusionmatrix      <- table(predictionrf01,test[,which(names(test)=="class")]) 
misclassificationrf  <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])

misclassificationrf    # Error value

# Summary .................................................
sink(file="random21-1.doc")
print(cm.rf <- confusionMatrix(data=predictionrf01 ,reference = test$class, positive = "1"))
sink()

# Compute of Threshold and AUC ............................
plot.roc(test$class, predictionrf,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a

## Option 2: Calculate optimal threshold with Youden's index  ......
rocrf  <- roc(test$class, predictionrf)
bestrf <- coords(rocrf, "b", ret = "threshold", best.method = "youden")
bestrf

## Prediction based on misclassification error .....................
predictionrf02  <- ifelse(predictionrf > bestrf,1, 0)
predictionrf02  <- as.factor(predictionrf02)
confusionmatrix <- table(predictionrf02,test[,which(names(test)=="class")]) 

## Missclasification ...............................................
misclassificationrf01 <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationrf01

# Summary  of results Ramdom forest model ----- 
sink(file="random21-2.doc")
print(cm.rf01 <- confusionMatrix(data=predictionrf02 ,reference = test$class, positive = "1"))
acc.rf <- cm.rf01$overall["Accuracy"]
sen.rf <- cm.rf01$byClass["Sensitivity"]
spe.rf <- cm.rf01$byClass["Specificity"]
ppv.rf <- cm.rf01$byClass["Pos Pred Value"]
npv.rf <- cm.rf01$byClass["Neg Pred Value"]
f1.rf  <- cm.rf01$byClass["F1"]
k.rf   <- cm.rf$overall["Kappa"]
auc.rf <- as.numeric(rocrf$auc)
cat("Error is:",misclassificationrf01)
cat("AUC:",auc.rf, "CI 95%:", ci.auc(test$class, predictionrf))
sink()

# ..............................................................

# Model 7:  Prediction with SVM- linear model -----
predictionsvmlinear <- predict(svmlinear, probability = TRUE,
                               newdata = test[ ,- which(names(test) == "class")])

predictionsvmlinear <- attr(predictionsvmlinear, "probabilities")[,1]  # successul prediction 
predictionsvmlinear01 <- predict(svmlinear, 
                                 newdata = test[ , - which(names(test) == "class")])
predictionsvmlinear
#  Compute AUC and interval ..................................
print(aucsvmlinear <- auc(test$class, predictionsvmlinear))
print(cisvmlinear  <- ci.auc(test$class, predictionsvmlinear))

## Prediction based on misclassification error ..............
confusionmatrix     <- table(predictionsvmlinear01, test[ , which(names(test) == "class")])

misclassificationsvmlinear <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationsvmlinear

# Summary .........................................................
sink(file="svmlinear21-1.doc")
confusionMatrix(data= predictionsvmlinear01,reference = test$class, positive ="1")
sink()

# Compute of Threshold and AUC ....................................
plot.roc(test$class, predictionsvmlinear,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a

# Calculate optimal threshold with Youden's index ........................
rocsvmlinear <- roc(test$class, predictionsvmlinear)
bestsvmlinear <- coords(rocsvmlinear, "b", ret = "threshold", best.method = "youden")
bestsvmlinear

## Prediction based on misclassification error
predictionsvmlinearyouden <- ifelse(predictionsvmlinear >0.7 , 1, 0)
predictionsvmlinearyouden <- as.factor(predictionsvmlinearyouden)
confusionmatrix           <- table(predictionsvmlinearyouden, 
                                   test[ , which(names(test) == "class")])

misclassificationsvmlinear01 <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationsvmlinear01

# Summary of results SVM-Linear model -----
sink(file="svmlinear21-2.doc")
print(cm.svmlinear <- confusionMatrix(data= predictionsvmlinearyouden,reference = test$class, positive = "1"))
acc.svmlinear <- cm.svmlinear$overall["Accuracy"]
sen.svmlinear <- cm.svmlinear$byClass["Sensitivity"]
spe.svmlinear <- cm.svmlinear$byClass["Specificity"]
ppv.svmlinear <- cm.svmlinear$byClass["Pos Pred Value"]
npv.svmlinear <- cm.svmlinear$byClass["Neg Pred Value"]
f1.svmlinear  <- cm.svmlinear$byClass["F1"]
k.svmlinear   <- cm.svmlinear$overall["Kappa"]
auc.linear    <- as.numeric(rocsvmlinear$auc)
cat("Error is:",misclassificationsvmlinear01)
cat("AUC:",rocsvmlinear$auc, "CI 95%:", ci.auc(test$class, predictionrf))
sink()

#............................................

# Model 8: Prediction with SVM -Polynomial Model  -------
predictionsvmpolynomial   <- predict(svmpolynomial, probability = TRUE, 
                                     newdata = test[ , - which(names(test) == "class")]) 

predictionsvmpolynomial   <- attr(predictionsvmpolynomial, "probabilities")[, 1]  # probability of successul 

predictionsvmpolynomial01 <- predict(svmpolynomial,newdata = test[,- which(names(test) == "class")]) 


# Compute auc and CI  ............................................. 
print(aucsvmpolynomial  <- auc(test$class, predictionsvmpolynomial))
print(cisvmpolynomial   <- ci.auc(test$class, predictionsvmpolynomial))

# Prediction based on misclassification error ......................
confusionmatrix <- table(predictionsvmpolynomial01, 
                         test[ , which(names(test) == "class")])

misclasvmpolynomial <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])

print(misclasvmpolynomial)

# summary ...................................................
sink(file="polynomial21-1.doc")
confusionMatrix(data= predictionsvmpolynomial01,reference = test$class, positive = "1")
sink()

# Compute of threshold and AUC ...............................
plot.roc(test$class, predictionsvmpolynomial,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) 

# Calculate optimal threshold with Youden's index ............
rocsvmpolynomial  <- roc(test$class, predictionsvmpolynomial)
bestsvmpolynomial <- coords(rocsvmpolynomial, "b", ret = "threshold", best.method = "youden")
bestsvmpolynomial

## Prediction based on misclassification error
predictionsvmpolynomialyouden01 <- ifelse(predictionsvmpolynomial > 0.70 , 1, 0)
predictionsvmpolynomialyouden01 <- as.factor(predictionsvmpolynomialyouden01)

confusionmatrix <- table(predictionsvmpolynomialyouden01, 
                         test[ , which(names(test) == "class")])

misclassificationsvmpoly01 <-(confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
print(misclassificationsvmpoly01)


# Summary of SVM-Poly Model  --------
sink(file="poly21-2.doc")
print(cm.svmpoly <-confusionMatrix(data=predictionsvmpolynomialyouden01,
                                   reference = test$class, positive = "1"))
acc.svmpoly <- cm.svmpoly$overall["Accuracy"]
sen.svmpoly <- cm.svmpoly$byClass["Sensitivity"]
spe.svmpoly <- cm.svmpoly$byClass["Specificity"]
ppv.svmpoly <- cm.svmpoly$byClass["Pos Pred Value"]
npv.svmpoly <- cm.svmpoly$byClass["Neg Pred Value"]
k.svmpoly   <- cm.svmpoly$overall["Kappa"]
f1.svmpoly  <- cm.svmpoly$byClass["F1"]
k.svmpoly   <- cm.svmpoly$overall["Kappa"]
auc.poly    <- as.numeric(rocsvmpolynomial$auc)
cat("Error is:",misclassificationsvmpoly01)
cat("AUC:",rocsvmpolynomial$auc, "CI 95%",cisvmpolynomial )
sink()

# ............................................

# Model 10: Prediction with KNN ----
predictionknn    <- predict(knnFit, newdata=x_test, type ='prob')[,2] # for succesful treatment
predictionknn01  <- predict(knnFit,newdata = test[,- which(names(test) == "class")]) # class
predictionknn01  <- as.factor(predictionknn01)

# Compute AUC and CI .....................................
print(aucknn  <- auc(test$class, predictionknn))
print(ciknn   <- ci.auc(test$class, predictionknn))

## Missclasification ......................................
confusionmatrix      <- table(predictionknn01,test[,which(names(test)=="class")]) 
misclassificationknn <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])

misclassificationknn    # Error value

# Summary .................................................
sink(file="knn21-1.doc")
print(cm.knn <- confusionMatrix(data=predictionknn01 ,reference = test$class, positive = "1"))
sink()

# Compute of Threshold and AUC ............................
plot.roc(test$class, predictionknn,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a

## Option 2: Calculate optimal threshold with Youden's index  ......
rocknn  <- roc(test$class, predictionknn)
bestknn <- coords(rocknn, "b", ret = "threshold", best.method = "youden")
bestknn

## Prediction based on misclassification error .....................
predictionknn02  <- ifelse(predictionknn >bestknn,1, 0)
predictionknn02  <- as.factor(predictionknn02)
confusionmatrix <- table(predictionknn02,test[,which(names(test)=="class")]) 

## Missclasification ...............................................
misclassificationknn01 <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])

misclassificationknn01

# Summary  of results Random forest model ----- 
sink(file="knn21-2.doc")
print(cm.knn01 <- confusionMatrix(data=predictionknn02 ,reference = test$class, positive = "1"))
acc.knn <- round(cm.knn01$overall["Accuracy"],2)
sen.knn <- round(cm.knn01$byClass["Sensitivity"],2)
spe.knn <- round(cm.knn01$byClass["Specificity"],2)
ppv.knn <- round(cm.knn01$byClass["Pos Pred Value"],2)
npv.knn <- round(cm.knn01$byClass["Neg Pred Value"],2)
f1.knn  <- round(cm.knn01$byClass["F1"],2)
k.knn   <- round(cm.knn$overall["Kappa"],2)
auc.knn <- round(as.numeric(rocknn$auc),2)
cat("Error is:",misclassificationknn01)
cat("AUC:",auc.knn, "CI 95%:", ci.auc(test$class, predictionknn))
sink()


# ....................................................

#  Results summmarized in a table ------


# Model names .............
modelnames   <- c("LRM", "Forward","Backward", "Both", "Lasso", "Random Forest", 
                  "SVM-Linear K.", "SVM-Poly K.","K-NN")
## Outcome names  ..........
outcomenames <- c("Accuracy", "AUC", "Lower(95%CI)","Upper(95%CI)","Sensitivity", "Specificity",
                  "PPV", "NPV", "Misclassification", "Kappa", "F1")
## Accuracy  ......... 
modelacc     <- as.numeric(c(acc.glm, acc.forward, acc.backward,acc.both,acc.lasso,
                             acc.rf,acc.svmlinear,acc.svmpoly, acc.knn))

## AUC curve ............
modelauc     <- c(auc.glm, auc.forward, auc.backward,auc.both,auc.lasso,
                  auc.rf,auc.linear,auc.poly, auc.knn)     

## Model CI Lower AUC   ............
modelcilower <- c(ciaucglm[1],ciforward[1], cibackward[1], ciboth[1], cilasso[1], cirf[1], 
                  cisvmlinear[1], cisvmpolynomial[1],ciknn[1] )

## Model CI upper AUC   ..................
modelciupper <- c(ciaucglm[3],ciforward[3], cibackward[3], ciboth[3], cilasso[3], cirf[3], 
                  cisvmlinear[3], cisvmpolynomial[3], ciknn[3])

## Misclassification
modelmisclassification <- c(misclassificationglm01, misclassificationforward01, misclassificationbackward01, 
                            misclassificationboth01, misclassificationlasso01, 
                            misclassificationrf01, misclassificationsvmlinear01, 
                            misclassificationsvmpoly01, misclassificationknn01)
## Model sensitivity .....
modelsensitivity <- c(sen.glm, sen.forward, sen.backward, sen.both, 
                      sen.lasso, sen.rf, sen.svmlinear,sen.svmpoly, sen.knn) 

## Model specificity  ......
modelspecificity <- c(spe.glm, spe.forward, spe.backward, spe.both, 
                      spe.lasso, spe.rf,spe.svmlinear, spe.svmpoly, spe.knn)


## Model positive predictive value  ......
modelppv   <- c(ppv.glm, ppv.forward, ppv.backward, ppv.both, ppv.lasso, 
                ppv.rf, ppv.svmlinear, ppv.svmpoly, ppv.knn)

## Negative predictive value  .......
modelnpv   <- c(npv.glm, npv.forward, npv.backward, npv.both, npv.lasso, 
                npv.rf, npv.svmlinear, npv.svmpoly, npv.knn)

# f1 - score ........................
modelf1    <- c(f1.glm, f1.forward, f1.backward, f1.both, f1.lasso, 
                f1.rf, f1.svmlinear, f1.svmpoly, f1.knn)
# Kappa   ............................
modelkappa <- as.numeric (c(k.glm, k.forward, k.backward, k.both, k.lasso, 
                            k.rf, k.svmlinear, k.svmpoly, f1.knn))

## Final table
results <- data.frame( modelacc, modelauc, modelcilower, modelciupper,modelsensitivity, 
                       modelspecificity, modelppv,modelnpv,modelmisclassification,  
                       modelkappa,modelf1)

rownames(results) <- modelnames
colnames(results) <- outcomenames

print(results)

# Final table summary ----
library(formattable)
# Color 
customGreen0 = "#DeF7E9"
customGreen  = "#71CA97"
customRed    = "#ff7f7f"
# output table
S <- formattable(round(results,2), align =c("l","c","c","c","c", "c", "c", "c", "c","c", "c","c"), 
                 list("modelnames" = formatter("span", style = ~ style(color = "grey",font.weight = "bold")), 
                      "Accuracy"= color_tile(customGreen0,customGreen ),
                      "AUC"= color_tile(customGreen0,customGreen),
                      "Misclassification"= color_bar(customRed)))
print(S)

# Export to Excel
library(xlsx)
write.xlsx2(S, file="Run21_final_summary.xlsx", sheetName = "Sheet1",
            col.names = TRUE, row.names = TRUE, append = FALSE)


# ..............................................................

#  ROC Curves Plot  -------

par(pty="s")  # Adjust Graphs to sheets

# Backward Plot 
y1 <-plot.roc(roc(test$class, predictionbackward), col = "aquamarine4", lwd = 2, grid = TRUE,  
              main = "Comparison of the performance for ML models ", 
              xlab = "1 - Specificity", legacy.axes = TRUE, grid.v=NULL)
# forward Plot
y2 <-plot.roc(roc(test$class, predictionforward), add = TRUE,
              col = "gold2", lwd = 2) 

# BOTH Plot
y3 <- plot.roc(roc(test$class, predictionboth), add = TRUE,
               col = "darkorange", lwd = 2, pch=8) 

# Lasso Plot
y4 <-plot.roc(roc(test$class, predictionlasso), add = TRUE,
              col = "magenta3", lwd = 2)

# Forest Plot
y5 <- plot.roc(roc(test$class, predictionrf), add = TRUE,
               col = "navy", lwd = 2)

# SVM-linear Plot
y6 <- plot.roc(roc(test$class, predictionsvmlinear), add = TRUE,
               col = "red", lwd = 2) 

# SVM-POLY
y7 <- plot.roc(roc(test$class, predictionsvmpolynomial),add = TRUE,
               col = "tan4", lwd = 2) 

# GLM 
y8 <-plot.roc(roc(test$class, predictionglm),add = TRUE,
              col = "khaki3", lwd = 2, grid = FALSE) 

# KNN Plot
y9 <-plot.roc(roc(test$class, predictionknn), add = TRUE,
              col = "cyan3", lwd = 2)
# Legend
legend(x=0.4,y=0.5, legend = c("Logistic", "Forward", "Backward", "Both", 
                               "Lasso", "Random forest","SVM-Linear","SVM-Polyno.",
                               "k-NN"),pch=16, bty="n",cex= 0.8,lty=1:1,lwd=2,  
       col = c("khaki3","aquamarine4","gold2", "darkorange", 
               "magenta3", "navy", "red", "tan4","cyan3"))


# .......................................................

# ROC CURVE Plot with Smooth -------

par(pty="s")

# Backward
plot.roc(smooth(roc(test$class, predictionbackward)), col = "aquamarine4", lwd = 2,  
         main = "Comparison", xlab = "1 - Specificity", legacy.axes = TRUE)
# forward
plot.roc(smooth(roc(test$class, predictionforward)), add = TRUE,
         col = "gold2", lwd = 2) 

# BOTH 
plot.roc(smooth(roc(test$class, predictionboth)), add = TRUE,
         col = "darkorange", lwd = 2) 

# Lasso
plot.roc(smooth(roc(test$class, predictionlasso)), add = TRUE,
         col = "magenta3", lwd = 2)
# Forest 
plot.roc(smooth(roc(test$class, predictionrf)), add = TRUE,
         col = "navy", lwd = 2)
# SVM Linear
plot.roc(smooth(roc(test$class, predictionsvmlinear)), add = TRUE,
         col = "red", lwd = 2) 
# SVM POLY
plot.roc(smooth(roc(test$class, predictionsvmpolynomial)),add = TRUE,
         col = "tan4", lwd = 2) 

#  GLM 
plot.roc(smooth(roc(test$class, predictionglm)),add = TRUE,
         col = "khaki3", lwd = 2)

# knn  
plot.roc(smooth(roc(test$class, predictionknn)),add = TRUE,
         col = "cyan3", lwd = 2)

# Legend
legend(x = "bottomright", legend = c("Logistic","Forward", "Backward", "Both", 
                                     "Lasso", "SVM-linear", "Random forest", 
                                     "SVM- polynomial", "kNN"),pch=16, bty="n",cex= 0.8,lty=1:1, lwd=2,
       col = c("khaki3","aquamarine4","gold2", "darkorange", "magenta3", 
               "navy", "red", "tan4", "cyan3"))

#.......................................................................  

# Compare models using pvalues from ROC curve ----

## Only p-values  (compare logistic vs other ML models)
testobj1 <- roc.test(y8, y1)
testobj2 <- roc.test(y8, y2)
testobj3 <- roc.test(y8, y3)
testobj4 <- roc.test(y8, y4)  
testobj5 <- roc.test(y8, y5)  
testobj6 <- roc.test(y8, y6)  
testobj7 <- roc.test(y8, y7)
testobj9 <- roc.test(y8, y9)  

models <- c("log","back_log", "forw_log", "both_log", "lasso_log", "RF_log", 
            "svm-l_log", "svm-P_log", "knn_log")

pv_roc  <- c(1,round(testobj1$p.value,2),round(testobj2$p.value,2),
             round(testobj3$p.value,2),round(testobj4$p.value,2),round(testobj5$p.value,2), 
             round(testobj6$p.value,2), round(testobj7$p.value,2),round(testobj9$p.value,2))
pv_roc  <- as.numeric(pv_roc)

sink(file="run20_pvalues_curveroc.doc")
pval_ROC <- cbind(Models = models, pvalues=pv_roc)
print(pval_ROC)
sink()

#.......................................................

# McNemar Test ----
## This test compared the accuracy of prediction models 

x <- data.frame(actual=test$class, logistic =predglm01, backward=predictionbackward01,
                forward=predictionforward01,  both= predictionboth01,lasso=predictionlasso01, RF= predictionrf01,
                SVM_linear= predictionsvmlinearyouden, SVM_Poly= predictionsvmpolynomialyouden01,
                kNN= predictionknn02)

# compare actual vs classifiers  
x$act_log      <- factor(ifelse(x$actual == x$logistic,1,0))  # actual = the outcome variable from dataset. 
x$act_backward <- factor(ifelse(x$actual == x$backward,1,0))  
x$act_forward  <- factor(ifelse(x$actual == x$forward,1,0))    
x$act_both     <- factor(ifelse(x$actual == x$both,1,0))   
x$act_lasso    <- factor(ifelse(x$actual == x$lasso,1,0)) 
x$act_RF       <- factor(ifelse(x$actual == x$RF,1,0)) 
x$act_SVM_linear  <- factor(ifelse(x$actual == x$SVM_linear,1,0))
x$SVM_Poly     <- factor(ifelse(x$actual == x$SVM_Poly,1,0))
x$act_kNN      <- factor(ifelse(x$actual == x$kNN,1,0)) 

# compare two classifiers (Logistic Model vs Other classifier)

xlog_back  <- table(x$act_log, x$act_backward, dnn=c("clasifier1", "clasifier2"))
xlog_forw  <- table(x$act_log, x$act_forward, dnn=c("clasifier1", "clasifier2"))
xlog_both  <- table(x$act_log, x$act_both , dnn=c("clasifier1", "clasifier2"))
xlog_lasso <- table(x$act_log, x$act_lasso, dnn=c("clasifier1", "clasifier2"))
xlog_rf    <- table(x$act_log, x$act_RF, dnn=c("clasifier1", "clasifier2"))
xlog_svm_linear <- table(x$act_log, x$act_SVM_linear, dnn=c("clasifier1", "clasifier2"))
xlog_svm_poly   <- table(x$act_log, x$SVM_Poly , dnn=c("clasifier1", "clasifier2"))
xlog_knn   <- table(x$act_log, x$kNN, dnn=c("clasifier1", "clasifier2"))

renemar <- data.table( log=c("-"),
                       log_back  =  c(mcnemar.test(xlog_back, correct =FALSE)["p.value"]),
                       log_for   =  c(mcnemar.test(xlog_forw, correct =FALSE)["p.value"]),
                       log_both  =  c(mcnemar.test(xlog_both,  correct =FALSE)["p.value"]),
                       log_lasso = c(mcnemar.test(xlog_lasso, correct =FALSE)["p.value"]),
                       log_rf    =  c(mcnemar.test(xlog_rf, correct =FALSE)["p.value"]),
                       log_svml  =  c(mcnemar.test(xlog_svm_linear,  correct =FALSE)["p.value"]), 
                       log_svmp  =  c(mcnemar.test(xlog_svm_poly,  correct =FALSE)["p.value"]),
                       log_knn   =  c(mcnemar.test(xlog_knn,  correct =FALSE)["p.value"])) 


sink(file="run20_mcnemar.doc")                                  
print(mcneamtest <-data.frame(t(renemar)))
sink()

# Table final : including mcnemar test, P-VALUE from ROC function in R.    

final <- cbind(S,pv_Mc = mcneamtest, pv_ROC = pv_roc)
S1    <- formattable(final)
print(S1)

# Export to Excel 
write.xlsx2(S1, file="RUN20-1_final_Table_summary.xlsx", sheetName = "Sheet1",
            col.names = TRUE, row.names = TRUE, append = FALSE)


mtryminimumerror <- which(proportionerror$meanerror == min(proportionerror$meanerror))[1]
mtryminimumerror

