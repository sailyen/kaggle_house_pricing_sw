library(ggplot2)
library(moments)
library(reshape2)
library(GGally)
library(missForest)
library(randomForest)
library(gbm)
library(xgboost)
library(dplyr)


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

train$SalePrice <- log(train$SalePrice)
gt <- ggplot(data = train, aes(x = train$SalePrice)) + geom_histogram(bins = 40) 
gt

#Relationship with numerical variables
#GrLivArea
names(train)
gt1 <- ggplot(data = train, aes(x = train$GrLivArea)) + geom_histogram(bins = 40) 
gt1

train$GrLivArea <- log(train$GrLivArea)
test$GrLivArea <- log(test$GrLivArea)

g1 <- ggplot(train, aes(GrLivArea, SalePrice)) + geom_jitter(color = "#E69F00")
g1

p <- ggplot(train, aes(sample = GrLivArea)) + stat_qq()
p

#TotalBsmtSF
gt2 <- ggplot(data = train, aes(x = train$TotalBsmtSF)) + geom_histogram(bins = 40) 
gt2
g2 <- ggplot(train, aes(TotalBsmtSF, SalePrice)) + geom_jitter(color = "#E69F00")
g2
summary(train$TotalBsmtSF)
train <- train[!(train$TotalBsmtSF == 6110) ,]
p <- ggplot(train, aes(sample = TotalBsmtSF)) + stat_qq()
p
#OverallQual
g3 <- ggplot(train, aes(OverallQual, SalePrice)) + geom_jitter(color = "#E69F00")
g3
summary(train$OverallQual)
g4 <- ggplot(train, aes(YearBuilt, SalePrice)) +geom_jitter(color = "#E69F00")
g4

#BsmtUnfSF
gt5 <- ggplot(data = train, aes(x = train$BsmtUnfSF)) + geom_histogram(bins = 40) 
gt5
skewness(train$BsmtUnfSF)
g5 <- ggplot(train, aes(BsmtUnfSF, SalePrice)) + geom_jitter(color = "#E69F00")
g5
p <- ggplot(train, aes(sample = BsmtUnfSF)) + stat_qq()
p


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

sapply(train, function(x)sum(is.infinite(x)))


#missing data imputing
sapply(train, function(x) sum(is.na(x)) )
nrow(train)
miss_drop <- c("Alley","MiscFeature","PoolQC","MiscFeature","Fence","FireplaceQu")
train_miss_drop <- train[, -which(names(train) %in% miss_drop)]
test_miss_drop <- test[, -which(names(train) %in% miss_drop)]
sapply(train_miss_drop, function(x) sum(is.na(x)) )
sapply(test_miss_drop, function(x) sum(is.na(x)) )

train_imp <- missForest(train_miss_drop, )
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
#0.13430!!!!!!!!
#linear: Multiple R-squared:  0.9307,	Adjusted R-squared:  0.9177 
#after EDA Multiple R-squared:  0.9459,	Adjusted R-squared:  0.9357 
m_linear <- lm(SalePrice~., data = train.imp)
summary(m_linear)

test_pr_linear <- predict(m_linear, newdata = test.imp)
summary(test_pr_linear)
test_pr_linear_t <- exp(test_pr_linear)
summary(test_pr_linear_t)


sub <- as.data.frame(cbind(test$Id, test_pr_linear_t))
head(sub)
names(sub)[1] <- "ID"
names(sub)[2] <- "SalePrice"
sapply(sub, function(x) sum(is.na(x)))
write.csv(sub, "resultEDA_1.csv")
head(sub)

#******************************************************************************
#Grid Search

random_index <- sample(1:nrow(train.imp), nrow(train.imp))
random_ames_train <- train.imp[random_index, ]
# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)
#81
# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # train model
  gbm.tune <- gbm(
    formula = SalePrice ~ .,
    distribution = "gaussian",
    data = random_ames_train,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)

#   shrinkage interaction.depth n.minobsinnode bag.fraction optimal_trees  min_RMSE
#1       0.10                 3             15         0.65           220 0.1361317
#2       0.01                 5              5         0.80          1291 0.1362568
#3       0.01                 5             15         0.65          1459 0.1367346
#4       0.01                 3              5         0.80          2250 0.1371275


gbm.fit <- gbm(
  formula = SalePrice ~ .,
  distribution = "gaussian",
  data = train.imp,
  bag.fraction = 0.80,
  n.minobsinnode = 5,
  n.trees = 1291,
  interaction.depth = 5,
  shrinkage = 0.01,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
gbm1 <- predict(gbm.fit, newdata = test.imp, n.trees=1291)
gbm1_t <- exp(gbm1)
gbm_pr1 <- as.data.frame(cbind(test.imp$Id, gbm1_t))
head(gbm_pr1)
names(gbm_pr1)[1] <- "ID"
names(gbm_pr1)[2] <- "SalePrice"

write.csv(gbm_pr1,"resultEDA_3.csv") #0.12949!!!!!

#*****************************************************************************
gbm.fit <- gbm(
  formula = SalePrice ~ .,
  distribution = "gaussian",
  data = train.imp,
  bag.fraction = 0.65,
  n.minobsinnode = 15,
  n.trees = 220,
  interaction.depth = 3,
  shrinkage = 0.1,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
gbm1 <- predict(gbm.fit, newdata = test.imp, n.trees=220)
gbm1_t <- exp(gbm1)
gbm_pr1 <- as.data.frame(cbind(test.imp$Id, gbm1_t))
head(gbm_pr1)
names(gbm_pr1)[1] <- "ID"
names(gbm_pr1)[2] <- "SalePrice"

write.csv(gbm_pr1,"resultEDA_2.csv") #0.12695

#*****************************************************************************
gbm.fit <- gbm(
  formula = SalePrice ~ .,
  distribution = "gaussian",
  data = train.imp,
  bag.fraction = 0.65,
  n.minobsinnode = 15,
  n.trees = 1459,
  interaction.depth = 5,
  shrinkage = 0.01,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
gbm1 <- predict(gbm.fit, newdata = test.imp, n.trees=1459)
gbm1_t <- exp(gbm1)
gbm_pr1 <- as.data.frame(cbind(test.imp$Id, gbm1_t))
head(gbm_pr1)
names(gbm_pr1)[1] <- "ID"
names(gbm_pr1)[2] <- "SalePrice"

write.csv(gbm_pr1,"resultEDA_4.csv") #0.12423

#*****************************************************************************

gbm.fit <- gbm(
  formula = SalePrice ~ .,
  distribution = "gaussian",
  data = train.imp,
  bag.fraction = 0.80,
  n.minobsinnode = 5,
  n.trees = 2250,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
gbm1 <- predict(gbm.fit, newdata = test.imp, n.trees=2250)
gbm1_t <- exp(gbm1)
gbm_pr1 <- as.data.frame(cbind(test.imp$Id, gbm1_t))
head(gbm_pr1)
names(gbm_pr1)[1] <- "ID"
names(gbm_pr1)[2] <- "SalePrice"

write.csv(gbm_pr1,"resultEDA_5.csv") #0.12896

#*******************************************************************************

for (i in 1:75) {
  levels(test.imp[,i]) <- levels(train.imp[,i])
}

names(test.imp)
names(train.imp)


m_rf <- randomForest(SalePrice~., data = train.imp, ntree = 1000, importance = TRUE)
m_rf
rf <- predict(m_rf, newdata = test.imp, predict.all = TRUE)
summary(rf)
rf$aggregate
rf$aggregate <- exp(rf$aggregate)
rf_pr <- as.data.frame(cbind(test.imp$Id, rf$aggregate))
head(rf_pr)
names(rf_pr)[1] <- "ID"
names(rf_pr)[2] <- "SalePrice"

write.csv(rf_pr,"resultEDA_6.csv")

