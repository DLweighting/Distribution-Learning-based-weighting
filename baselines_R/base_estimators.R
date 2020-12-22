library("MatchIt")

base <- function(i, data_all, dim){
  ATT_unadjusted = mean(data_all[which(data_all$w==1),]$y) - mean(data_all[which(data_all$w==0),]$y)
  cat(paste("\nThe base ATT is",ATT_unadjusted))
  ATT_base[i]<<- ATT_unadjusted
  
  # use true p-score for weighting
  control = data_all[which(data_all$w==0),]
  y0_adjusted = control$y*control$prob_w/(1-control$prob_w)
  ATT_weight = mean(data_all[which(data_all$w==1),]$y) - sum(y0_adjusted)/length(which(data_all$w==1))
  cat(paste("\nThe ATT_weight is",ATT_weight))
  ATT_true_weight[i]<<-ATT_weight
  
  weight = control$prob_w/(1-control$prob_w)
  ATT_weight_adjust = mean(data_all[which(data_all$w==1),]$y) - sum(control$y*weight/sum(weight))
  cat(paste("\nThe ATT_weight_adjust is",ATT_weight_adjust))
  ATT_true_weight_adjust[i]<<-ATT_weight_adjust
}



matching <-function(i, data_all, dim){
    formula = as.formula(paste("w~",paste("X",seq(1,dim),sep="",collapse = "+")))
    
    glm1 <- glm(formula, family = "binomial", data = data_all)
    estimate = glm1$fitted
    bias_pscore[i]<<-mean((data_all$prob_w - estimate)^2)

    # Using MatchIt package
    # p-score matching
    m.out <- matchit(formula, data=data_all,
                     method = "nearest", distance = "logit", replace=T)
    m.data <- match.data(m.out)
    m.data1 <- match.data(m.out, group = "treat")
    m.data0 <- match.data(m.out, group = "control")
    ATT_pscore[i]<<-mean(m.data1$y)-mean(m.data0$y *m.data0$weights)
    
    # # OLS regression
    lm1 <- lm(y~.-prob_w -y0 -y1, data = data_all)
    summary(lm1)
    ATT_ols[i]<<-lm1$coefficients[length(lm1$coefficients)]
}

