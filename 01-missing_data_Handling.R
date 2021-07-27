# Tema: Manejo de datos faltantes (NA)
# Identificar los datos faltantes (NA) en una dataset. 
# Bar plot the missing data
# Imputaciones de valor faltantes
#***************************************************

# Librerias requeridas ----
library(dplyr); library(mice);library(tidyr)
library(naniar); library(UpSetR);library(VIM)
library(lattice)
# ***************************************************

# Encontrar datos faltantes en un dataset(NA = Missing data)  
# Opcion 1: funcion sapply
miss <- data.frame(Missdata = sapply(data1, function(x) sum(is.na(x))))
print(miss)                      # Ver los resultados
# ??sapply                       # Buscar ayuda para la sapply funcion  

## Opcion 2:  Find the NA values ----
nan   <- colSums(is.na(data1)) # Contar Valor NA por columnas
print(nan)


# Select numeric features ---
numeric_df <- dplyr::select_if(cpap_model, is.numeric)

# Impute mean values into NA values -----
for (x in cpap_model) {
  mean_value <- mean(cpap_model[[x]],na.rm = TRUE)
  cpap_model[[x]][is.na(cpap_model[[x]])] <- mean_value
}

# Omit NA Values ------
cpap_model <- na.omit(cpap_psg)


##Remove NA of specific columns ----
data2 <- data2[!(is.na(data2$VM_A6) == TRUE), ]
dim(data2)

# remove features 
cpap_psg <- dplyr::select(complete_cpap, -c(decision_pr, hora_dormir, decision_psg, humidificador))
str(cpap_psg)


# NA Values Plots ----    
cpap_orig %>% 
  summarise_all(funs(sum(is.na(.)))) %>%     # Return a barplot with missing data 
  gather %>% 
  ggplot(aes(x = reorder(key, value), y =value)) + geom_bar(stat = "identity") +
  coord_flip() +
  xlab("variable") +
  ylab("Absolute number of missings")


# Option 2 -----  
install.packages("UpSetR")
install.packages("naniar")
library(naniar)
library(UpSetR)
gg_miss_case(cpap_orig) + labs(x= "Number of cases")

gg_miss_upset(cpap_orig)

# NA Plot 
library(VIM)
md.pattern(cpap_psg)  # Missing data pattern (solo para variables numericas)
mice_plot <- aggr(cpap_psg, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(cpap_psg), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))


# Impute values to missing values (MICE)  ----
# MICE algorithm 

data2 <- select(data1, -(19:21))
data2 <- select(data2, "edad", "bmi", "epworth", "eq50", 6:7, 14:18) # Creat new dataset with only numerical variables 
imp <-mice(data2, m=5, printFlag=FALSE, maxit = 5, seed=145)
summary(imp)
# The output imp contains m=5 completed datasets. Each dataset can be analysed

# Completed datasets (observed and imputed), for example the second one, can be extracted by
# Select an iteration (4)
data2 <- complete(imp,5)
data.frame(colSums(is.na(data2)))
names(data2)
dim(data4)
# Check for implausible imputations (values that are clearly impossible, e.g. negative values for bmi)
# The imputations, for example for epworth, are stored as
imp$imp$iah_psg

# Back a dataset (data1)
data1 <- cbind(data2, data1[,19:21], data1[,8:13])
names(data1)

## We can inspect the distributions of the original and the imputed data:
## scatterplot (of epworth and iah_total) for each imputed dataset
xyplot(imp,  epworth ~ iah_psg | .imp, pch = 20, cex = 1.4)

## pmm stands for predictive mean matching, default method of mice() for imputation of continuous incomplete variables; for each missing value, pmm finds a set of observed values with the closest predicted mean 
# as the missing one and imputes the missing values by a random draw from that set. 
imp$method

##-----------------------
# Explanation Mice code ----
# To impute the missing values, mice package use an algorithm in a such a way that use information 
# from other variables in the dataset to predict and impute the missing values. Therefore, you may
# not want to use a certain variable as predictors. For example, the ID variable does not have any predictive value.

# To skip a imputation variable
# meth[c("gender", "depresion", "ansiedad", "hta, cardio", "enf_neuro", "enf_respira")]=""

## Remove responses from dataset.
# data2 <-(data1)[,1:10] ## Just have 10 predictors variables (just numeric predictor)