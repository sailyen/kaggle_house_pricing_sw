library(ggplot2)
library(moments)
library(reshape2)
library(GGally)
library(missForest)

train <- read.csv("train.csv", header = T)
head(train)
summary(train)

#summary target
target <- summary(train$SalePrice)
target
gt <- ggplot(data = train, aes(x = train$SalePrice)) + geom_histogram(bins = 40) 
gt
gt_d <- ggplot(data = train, aes(x = train$SalePrice))+ 
  geom_density(alpha=.2, fill="#FF6666")
gt_d
p <- ggplot(train, aes(sample = SalePrice)) + stat_qq()
p

skewness(train$SalePrice)
kurtosis(train$SalePrice)

#Relationship with numerical variables
#GrLivArea
names(train)
g1 <- ggplot(train, aes(GrLivArea, SalePrice)) + geom_jitter(color = "#E69F00")
g1
p <- ggplot(train, aes(sample = GrLivArea)) + stat_qq()
p
#TotalBsmtSF
g2 <- ggplot(train, aes(TotalBsmtSF, SalePrice)) + geom_jitter(color = "#E69F00")
g2
p <- ggplot(train, aes(sample = TotalBsmtSF)) + stat_qq()
p
#
train$OverallQual <- as.factor(train$OverallQual)
train$YearBuilt <- as.factor(train$YearBuilt)
g3 <- ggplot(train, aes(OverallQual, SalePrice)) + geom_boxplot(color = "#E69F00")
g3
g4 <- ggplot(train, aes(YearBuilt, SalePrice)) + geom_boxplot(color = "#E69F00")
g4

#correlation
nums <- unlist(lapply(train, is.numeric))  
df <- train[, nums]
df_cor <- melt(cor(df))
head(df_cor)
g <- ggplot(data = df_cor, aes(x=Var1, y=Var2, fill=value)) + 
  geom_raster()
g <- g + theme(axis.text.x = element_text(angle = 90, hjust = 1))
g
summary(cor(df))
#BsmtUnfSF, TotalBsmtSF, FullBath, TotRmsAbvGrd, GarageArea highly corelated to SalePrice
ggpairs(df[,c("BsmtUnfSF", "TotalBsmtSF", "FullBath", "TotRmsAbvGrd", "GarageArea", "SalePrice")],
        mapping=ggplot2::aes(colour = "cond"))

#missing data imputing
sapply(train, function(x) sum(is.na(x)) )
nrow(train)
miss_drop <- c("Alley","MiscFeature","PoolQC","MiscFeature","Fence","FireplaceQu")
train_miss_drop <- train[, -which(names(train) %in% miss_drop)]
sapply(train_miss_drop, function(x) sum(is.na(x)) )

train_imp <- missForest(train_miss_drop)
train_imp$ximp
train_imp$OOBerror
#NRMSE is normalized mean squared error. 
#It is used to represent error derived from imputing continuous values.
#PFC (proportion of falsely classified) is used to represent error 
#derived from imputing categorical values. 
#NRMSE          PFC 
#0.0005928575 0.0419085367 
train.imp <- data.frame(train_imp$ximp)
summary(train.imp)
nrow(train.imp)
ncol(train.imp)

#

