library(ggplot2)
library(moments)
library(reshape2)
library(GGally)
library(missForest)
library(randomForest)

#####Data exploring
train <- read.csv("train.csv", header = T)
test <- read.csv("test.csv",header = T)
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
pairs <- ggpairs(df[,c("BsmtUnfSF", "TotalBsmtSF", "FullBath", "TotRmsAbvGrd", "GarageArea", "SalePrice")],
        mapping=ggplot2::aes(colour = "cond"))
pairs

#missing data imputing
sapply(train, function(x) sum(is.na(x)) )
nrow(train)
miss_drop <- c("Alley","MiscFeature","PoolQC","MiscFeature","Fence","FireplaceQu")
train_miss_drop <- train[, -which(names(train) %in% miss_drop)]
test_miss_drop <- test[, -which(names(train) %in% miss_drop)]
sapply(train_miss_drop, function(x) sum(is.na(x)) )
sapply(test_miss_drop, function(x) sum(is.na(x)) )

train_imp <- missForest(train_miss_drop)
train_imp$ximp
train_imp$OOBerror

test_imp <- missForest(test_miss_drop)
test_imp$ximp
test_imp$OOBerror
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

test.imp <- data.frame(test_imp$ximp)
summary(test.imp)
nrow(test.imp)
ncol(test.imp)

##############################################modeling
#linear: Multiple R-squared:  0.9307,	Adjusted R-squared:  0.9177 
m_linear <- lm(SalePrice~., data = train.imp)
summary(m_linear)

test_pr_linear <- predict(m_linear, newdata = test.imp)
summary(test_pr_linear)

sub <- as.data.frame(cbind(test$Id, test_pr_linear))
head(sub)
names(sub)[1] <- "ID"
names(sub)[2] <- "SalePrice"

sub$SalePrice <- ifelse(is.na(sub$SalePrice), mean(sub$SalePrice, na.rm=TRUE), sub$SalePrice)
write.csv(sub, "result1.csv")
head(sub)


#rf : % Var explained: 88.07
nums <- unlist(lapply(train.imp, is.numeric))  
df_n <- train.imp[, nums]
facs <- unlist(lapply(train.imp, is.factor))  
df_f <- train.imp[, facs]

nums_test <- unlist(lapply(test.imp, is.numeric))  
df_n_test <- test.imp[, nums_test]
facs_test <- unlist(lapply(test.imp, is.factor))  
df_f_test <- test.imp[, facs_test]

ncol(df_f_test)

for (i in 1:38) {
  levels(df_f_test[,i]) <- levels(df_f[,i])
}
#-------------------------------------------------------------------OR
for (i in 1:75) {
  levels(test.imp[,i]) <- levels(train.imp[,i])
}

names(test.imp)
names(train.imp)

#-----------------------------------------------------------------------
train_rf <- cbind(df_n,df_f)
test_rf <- cbind(df_n_test, df_f_test)


m_rf <- randomForest(SalePrice~., data = train.imp, importance = TRUE)
m_rf
rf <- predict(m_rf, newdata = test.imp, predict.all = TRUE)
summary(rf)
rf$aggregate
rf_pr <- as.data.frame(cbind(test.imp$Id, rf$aggregate))
head(rf_pr)
names(rf_pr)[1] <- "ID"
names(rf_pr)[2] <- "SalePrice"

write.csv(rf_pr,"result2.csv")

#gbm

#




#modeling (H2O)

