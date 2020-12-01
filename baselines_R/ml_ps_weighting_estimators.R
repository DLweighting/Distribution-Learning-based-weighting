
#install.packages("caTools")
# install.packages("caretEnsemble")
# install.packages("mlbench")
# install.packages("caret")
# install.packages("xgboost")

library(randomForest)
require(caTools)
library("MatchIt")
library("xgboost")
library("glmnet")
library("caret")
library("e1071")

ml_ps<-function(i, data_all, dim){
  ###### rf
  formula = as.formula(paste("w~",paste("X",seq(1,dim),sep="", collapse="+")))

  rf <- randomForest(formula,data=data_all)
  estimate = predict(rf, newdata = data_all[,1:dim],type="prob")[,"1"]
  bias_rf[i]<<- mean((data_all$prob_w - estimate)^2)

  weights = estimate/(1-estimate)
  rf_weights = ifelse(data_all$w==1, 1, weights)
  
  y1 = weighted.mean(data_all[data_all$w==1,]$y,rf_weights[data_all$w==1])
  y0 = weighted.mean(data_all[data_all$w==0,]$y,rf_weights[data_all$w==0])
  ATT = y1 - y0
  cat(paste("\nThe rf weighting ATT is",ATT))
  ATT_rf[i]<<-ATT
  data_all["rf_weights"] <<- rf_weights
   
  
  ###### treebag
  fit.treebag <- train(formula, data=data_all, method="treebag")
  estimate <- predict(fit.treebag, newdata = data_all[,1:dim], type = "prob")[,2]
  bias_treebag[i]<<-mean((data_all$prob_w - estimate)^2)

  weights = estimate/(1-estimate)
  treebag_weights = ifelse(data_all$w==1, 1, weights)
  
  y1 = weighted.mean(data_all[data_all$w==1,]$y,treebag_weights[data_all$w==1])
  y0 = weighted.mean(data_all[data_all$w==0,]$y,treebag_weights[data_all$w==0])
  ATT = y1 - y0
  cat(paste("\nThe treebag weighting ATT is",ATT))
  ATT_treebag[i]<<-ATT
  data_all["treebag_weights"] <<- treebag_weights
  
  ###### xgboost
  fit.xgboost <- xgboost(data = as.matrix(data_all[,1:dim]), 
                         label = as.numeric(as.character(data_all$w)), 
                         max.depth = 6, eta = 0.2, nthread = 2, 
                         verbose = 0,nrounds = 1000, objective = "binary:logistic")
  estimate <- predict(fit.xgboost, newdata = as.matrix(data_all[,1:dim]))
  bias_xgboost[i]<<-mean((data_all$prob_w - estimate)^2)

  weights = estimate/(1-estimate)
  xgboost_weights = ifelse(data_all$w==1, 1, weights)
  
  y1 = weighted.mean(data_all[data_all$w==1,]$y,xgboost_weights[data_all$w==1])
  y0 = weighted.mean(data_all[data_all$w==0,]$y,xgboost_weights[data_all$w==0])
  ATT = y1 - y0
  cat(paste("\nThe xgboost weighting ATT is",ATT))
  ATT_xgboost[i]<<-ATT
  data_all["xgboost_weights"] <<- xgboost_weights
  
  
  ### lasso
  cv.lasso <- cv.glmnet(as.matrix(data_all[,1:dim]), data_all$w, alpha = 1, 
                        nfolds = 10, family = "binomial",type.measure = "class")
  # print(cv.lasso)
  # plot(cv.lasso)
  estimate <- predict(cv.lasso,newx = as.matrix(data_all[,1:dim]),
                      s= "lambda.min",type="response")
  bias_lasso[i]<<-mean((data_all$prob_w - estimate)^2)

  weights = estimate/(1-estimate)
  lasso_weights = ifelse(data_all$w==1, 1, weights)
  
  y1 = weighted.mean(data_all[data_all$w==1,]$y,lasso_weights[data_all$w==1])
  y0 = weighted.mean(data_all[data_all$w==0,]$y,lasso_weights[data_all$w==0])
  ATT = y1 - y0
  cat(paste("\nThe lasso weighting ATT is",ATT))
  ATT_lasso[i]<<-ATT
  data_all["lasso_weights"] <<- lasso_weights
}

# # for 10-folds cross validation, use "caret" package; select the best hyper-parameter;
# # And then train on the whole training dataset
# # https://topepo.github.io/caret/available-models.html
# # https://cran.csiro.au/web/packages/caret/vignettes/caret.html
# train_control <- trainControl(method="cv", number=10, search = "random")
# model <- train(formula, data=data_all, trControl=train_control, method="ordinalRF",
#                preProcess = c("center", "scale"), metric = "Accuracy", tuneLength=6)
# print(model)
# estimate <- predict(model, newdata= data_all[,1:dim], type ="prob")[,"1"]