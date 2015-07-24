setwd('C:/Users/DF/Desktop/Coursera/C8_PracticalMachineLearning')
library(caret)

x <- read.csv('pml-training.csv', stringsAsFactors=FALSE)
x$classe <- as.factor(x$classe)
x$new_window <- as.factor(x$new_window)

x[x==''] <- NA
x2 <- x[,colSums(is.na(x)) < (nrow(x)/2)]
x2 <- x2[,c(6:ncol(x2))]

set.seed(10101)
inTrain <- createDataPartition(y=x2$classe, p=0.6, list=FALSE)
training <- x2[inTrain,]
testing <- x2[-inTrain,]
dim(training); dim(testing)

nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[,c(-1,-2)]
testing <- testing[,c(-1,-2)]
########## ABOVE REMOVES NUM_WINDOW WHICH HAS ALMOST ZERO VARIANCE

########## TRY A CLASSIFICATION TREE
library(rpart); library(rattle)
set.seed(8787)
modRpart <- rpart(classe~., data=training, method="class")
predRpart <- predict(modRpart, testing, type="class")
confusionMatrix(predRpart, testing$classe)
fancyRpartPlot(modRpart)

plotcp(modRpart)
printcp(modRpart)
modRpart2 <- prune(modRpart, cp=.011)
predRpart2 <- predict(modRpart2, testing, type="class")
fancyRpartPlot(modRpart2)
confusionMatrix(predRpart2, testing$classe)


library(tree)
set.seed(8787)
modTree <- tree(classe~., data=training)
plot(modTree)
text(modTree,all=T)
plot(cv.tree(modTree))

modTree2 <- prune.misclass(modTree, best=14)
predTree <- predict(modTree, testing, type="class")
predTree2 <- predict(modTree2, testing, type="class")
confusionMatrix(predTree, testing$classe)
confusionMatrix(predTree2, testing$classe)


library(party)
set.seed(8787)
modParty <- cforest(classe~., data=training, controls=cforest_unbiased(ntree=50, mtry=3))
predParty <- predict(modParty, testing, OOB=TRUE, type="response")
confusionMatrix(predParty, testing$classe)

library(gbm)
set.seed(8787)
modGBM <- gbm(classe~., data=training, n.trees=100, verbose=F)
predictGBM <- predict(modGBM, testing, n.trees=100, type="response")
predGBMout <- apply(predictGBM, 1, which.max)
##NOT USEFUL


library(randomForest)
set.seed(8787)
modRF <- randomForest(classe~., data=training, ntree=500)
modRF
plot(modRF)

set.seed(8787)
modRF1 <- randomForest(classe~., data=training, ntree=50)
modRF1
predRF1 <- predict(modRF1, newdata=testing)
confusionMatrix(predRF1, testing$classe)
##OOB=0.94%, accuracy = .9938

set.seed(8787)
modRF2 <- randomForest(classe~., data=training, ntree=50, importance=T, proximity=T)
modRF2
predRF2 <- predict(modRF2, newdata=testing)
confusionMatrix(predRF2, testing$classe)
##OOB=0.99%, accuracy = .9954

set.seed(8787)
modRF2 <- randomForest(classe~., data=training, ntree=50, importance=F, proximity=T)
modRF2
predRF2 <- predict(modRF2, newdata=testing)
confusionMatrix(predRF2, testing$classe)
##OOB=0.94%, accuracy = .9938

set.seed(8787)
modRF2 <- randomForest(classe~., data=training, ntree=50, importance=T, proximity=F)
modRF2
predRF2 <- predict(modRF2, newdata=testing)
confusionMatrix(predRF2, testing$classe)
##OOB=0.99%, accuracy = .9954

set.seed(8787)
tc <- trainControl(method = "cv", number = 7, verboseIter=FALSE , preProcOptions="pca", allowParallel=TRUE)
##Six models are estimated: Random forest, Support Vector Machine (both radial and linear), a Neural net, a Bayes Generalized linear model and a Logit Boosted model.

rf <- train(classe ~ ., data = training, method = "rf", trControl= tc)
svmr <- train(classe ~ ., data = training, method = "svmRadial", trControl= tc)
NN <- train(classe ~ ., data = training, method = "nnet", trControl= tc, verbose=FALSE)
svml <- train(classe ~ ., data = training, method = "svmLinear", trControl= tc)
bayesglm <- train(classe ~ ., data = training, method = "bayesglm", trControl= tc)
logitboost <- train(classe ~ ., data = training, method = "LogitBoost", trControl= tc)

rfpredict <- predict(rf, testing)


