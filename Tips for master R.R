
# load the packages
library(mlbench)
# load the dataset
data(PimaIndiansDiabetes)

# Outcome

y <- PimaIndiansDiabetes$diabetes
cbind(freq=table(y), percentage=prop.table(table(y))*100)

# Desviacion estandar  -----
# calculate standard deviation for all attributes
sapply(PimaIndiansDiabetes[,1:8], sd)
sapply(PimaIndiansDiabetes[,1:8], mean)

#Skewness -----
# calculate skewness for each variable
library(e1071)
skew <- apply(PimaIndiansDiabetes[,1:8], 2, skewness)
# display skewness, larger/smaller deviations from 0 show more skew
print(skew)

# Correlations -----

# load the packages
library(mlbench)
# load the dataset
data(PimaIndiansDiabetes)
# calculate a correlation matrix for numeric variables
correlations <- cor(PimaIndiansDiabetes[,1:8])
# display the correla
correlations

# calculate correlations
correlations <- cor(PimaIndiansDiabetes[,1:8])
# create correlation plot
corrplot(correlations, method = "circle")

# Scatterplot Matrix  ----
# Pair-wise scatterplots of all 4 attributes
pairs(PimaIndiansDiabetes[,1:8])
names(PimaIndiansDiabetes)

# Para dos variables
pairs(~log(mass) + log(triceps), data = PimaIndiansDiabetes, 
       main="Scatterplot simple")

# Correlacion entre los predictores y el outcome  ----
# pair-wise scatterplots colored by class
pairs(diabetes~., data=PimaIndiansDiabetes, col=PimaIndiansDiabetes$diabetes)





