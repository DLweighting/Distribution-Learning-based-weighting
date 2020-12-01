# install.packages("BART")
# install.packages("matrixStats")
# install.packages("nnls")
# install.packages("R.methodsS3")
# install.packages("lcmix", repos="http://R-Forge.R-project.org")
# install.packages("D:/3_Matching/1_flow_based/1_flow_weighting/simulation/R/lcmix_0.3.tar.gz"
#                  , repos=NULL, type="source")
library("BART") 
library("grf")
library("randomForest") 
  
bart_estimate<-function(i, data_all, dim){
  
  d_obs = dim

  data_control <- data_all[which(data_all$w==0),]
  x = data_control[,1:d_obs]
  y = data_control[,(d_obs+3)]
  bart_control = wbart(x, y)
  
  y0_samples= predict(bart_control, data_all[1:d_obs])
  E0 = apply(y0_samples, 2, mean)
  
  ATT = mean(data_all[which(data_all$w==1),]$y) - mean(E0[which(data_all$w ==1)])
  cat(paste("\nThe BART estimated ATT is",ATT))
  ATT_bart[i] <<- ATT
  data_all["bart_y0"] <<- E0
  
  b = data_all[which(data_all$w==0),"y0"]
  mse = mean((E0[which(data_all$w ==0)]-b)^2)
  cat(paste("\nThe BART y0 estimated error is",mse))
  mse_bart_y0[i] <<- mse
}


rf_estimate<-function(i, data_all, dim){
  
  data_control <- data_all[which(data_all$w==0),]
  x = data_control[,1:dim]
  y = data_control[,(dim+3)]
  rf_control = randomForest(x, y, ntree=500)
  
  E0 = predict(rf_control, data_all[,1:dim])
  
  ATT = mean(data_all[which(data_all$w==1),]$y) - mean(E0[which(data_all$w ==1)])
  cat(paste("\nThe rf estimated ATT is",ATT))
  ATT_rforest[i] <<- ATT
  data_all["rf_y0"] <<- E0
  
  b = data_all[which(data_all$w==0),"y0"]
  mse = mean((E0[which(data_all$w ==0)]-b)^2)
  cat(paste("\nThe rf y0 estimated error is",mse))
  mse_rf_y0[i] <<- mse
}

cf_estimate<-function(i, data_all, dim){
  tau.forest <- causal_forest(data_all[,1:dim], data_all$y, 
                              as.numeric(as.character(data_all$w)))

  # Estimate the average treatment effect on the treated sample (ATT).
  ATT = average_treatment_effect(tau.forest, target.sample = "treated", method="AIPW")[1]
  cat(paste("\nThe cf estimated ATT is",ATT))
  ATT_cf[i] <<- ATT
}  