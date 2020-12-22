
# install.packages("WeightIt") # ensembled many weighting methods
# install.packages("cobalt") # Covariate Balance Tables and Plots
# install.packages("ebal") # entropy balance
# install.packages("CBPS") # Covariate Balancing Propensity Score
# install.packages("optweight") # Stable balancing weights
# install.packages("ATE") # Empirical Balancing Calibration Weighting
# install.packages("devtools")

# library(devtools)
# install_github("swager/balanceHD")

library("WeightIt")
library("cobalt")

library("ebal")
library("CBPS")
library("optweight")
library("ATE")
library(balanceHD)

weighting<-function(i, data_all, dim){
  formula = as.formula(paste("w~",paste("X",seq(1,dim),sep="",collapse = "+")))
  ############ for entropy balance weighting, PA
  
  W <- weightit(formula,
                data = data_all, estimand = "ATT", method = "ebal")
  # summary(W)
  # bal.tab(W)  
  y1 = weighted.mean(data_all[data_all$w==1,]$y,W$weights[data_all$w==1])
  y0 = weighted.mean(data_all[data_all$w==0,]$y,W$weights[data_all$w==0])
  ATT = y1 - y0
  cat(paste("\nThe ebal weighting ATT is",ATT))
  ATT_ebal[i] <<- ATT
  data_all["weights_ebal"] <<- W$weights

  ############ for Covariate Balancing Propensity Score, J. R. Statist. Soc. B
  
  W <- weightit(formula ,
                data = data_all, estimand = "ATT", method = "cbps")
  y1 = weighted.mean(data_all[data_all$w==1,]$y,W$weights[data_all$w==1])
  y0 = weighted.mean(data_all[data_all$w==0,]$y,W$weights[data_all$w==0])
  ATT = y1 - y0
  cat(paste("\nThe cbps weighting ATT is",ATT))
  ATT_cbps[i] <<- ATT
  data_all["weights_cbps"] <<- W$weights
  

  ############ Stable balancing weights, Optimization-Based Weighting, JASA
  
  W <- weightit(formula,
                data = data_all, estimand = "ATT", method = "optweight", tols = 0.01)
  # W <- weightit(formula,
  #               data = data_all, estimand = "ATT", method = "optweight", tols = 1,
  #               eps_abs=1, eps_rel=1) # for n=40000
  
  y1 = weighted.mean(data_all[data_all$w==1,]$y,W$weights[data_all$w==1])
  y0 = weighted.mean(data_all[data_all$w==0,]$y,W$weights[data_all$w==0])
  ATT = y1 - y0
  cat(paste("\nThe optweight weighting ATT is",ATT))
  ATT_optweight[i] <<- ATT
  data_all["weights_opt"] <<- W$weights
  
  
  ############ Empirical Balancing Calibration Weighting, J. R. Statist. Soc. B
  
  W <- weightit(formula ,
                data = data_all, estimand = "ATT", method = "ebcw")
  y1 = weighted.mean(data_all[data_all$w==1,]$y,W$weights[data_all$w==1])
  y0 = weighted.mean(data_all[data_all$w==0,]$y,W$weights[data_all$w==0])
  ATT = y1 - y0
  cat(paste("\nThe ebcw weighting ATT is",ATT))
  ATT_ebcw[i] <<- ATT
  data_all["weights_ebcw"] <<- W$weights
  
  
  ############ propensity score weights
  
  W <- weightit(formula ,
                data = data_all, estimand = "ATT", method = "ps")
  
  y1 = weighted.mean(data_all[data_all$w==1,]$y,W$weights[data_all$w==1])
  y0 = weighted.mean(data_all[data_all$w==0,]$y,W$weights[data_all$w==0])
  ATT = y1 - y0
  cat(paste("\nThe ps weighting ATT is",ATT))
  ATT_ps[i] <<- ATT
  data_all["weights_ps"] <<- W$weights
  
  ########### ATT_arb, approximate risidual balance
  tau.hat = residualBalance.ate(data_all[1:dim], data_all$y, data_all$w, target.pop=1)
  ATT = tau.hat[1]
  cat(paste("\nThe arb estimate is", ATT))
  ATT_arb[i] <<- ATT
  # tau.hat = residualBalance.ate(X, Y, W, estimate.se = TRUE)
  # print(paste("true tau:", tau))
  # print(paste("point estimate:", round(tau.hat[1], 2)))
  # print(paste0("95% CI for tau: (", round(tau.hat[1] - 1.96 * tau.hat[2], 2), ", ", round(tau.hat[1] + 1.96 * tau.hat[2], 2), ")"))  
  # 
}

