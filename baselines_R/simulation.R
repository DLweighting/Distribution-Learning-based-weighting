args <- commandArgs(TRUE)
# args = c("simu2",8,10000,0.8,2)

set.seed(2018)
class <- args[1]
dim = as.numeric(args[2])
n = as.numeric(args[3])
sc = as.numeric(args[4])
times = as.numeric(args[5])

name=paste("class_",class,"_d", dim, 
           "_n",n,"_sc",sc,sep="")
sink(paste("R_result/",name,".txt",sep=""))

source("data_generation.R")
source("base_estimators.R")
source("weighting_estimators.R")
source("ml_ps_weighting_estimators.R")
source("outcome_based_estimators.R")


if (dim == 8){
  #beta1 = runif(8, 0, 1)
  beta1 = c(0.7599586, 0.8201394, 0.6193205, 0.4586492, 0.5157641,
            0.2891778, 0.5111296, 0.0548527)
  # beta2 = runif(28, 0, 1)
  # write(paste(beta2, collapse = ','), 'beta2_d8.txt')
  beta2 = as.vector(as.matrix(read.table("beta2_d8.txt",sep=",")))
}

if (dim == 16){
  # beta1 = runif(16, 0, 1)
  beta1 =  c(0.50013665, 0.93067848, 0.85806673, 0.40690777, 0.51404529, 0.66080241,
             0.61336544, 0.65048289, 0.40307454, 0.97905294, 0.01737927, 0.31150095,
             0.15645899, 0.17930328, 0.55058790, 0.04315068)
  # beta2 = runif(120, 0, 1)
  # write(paste(beta2, collapse = ','), 'beta2_d16.txt')
  beta2 = as.vector(as.matrix(read.table("beta2_d16.txt",sep=",")))
}


#base
ATT_base<-c()
ATT_true_weight<- c()
ATT_true_weight_adjust <-c()

#matching and ols
bias_pscore<-c()
ATT_pscore<-c()
ATT_ols<-c()

# weighting
ATT_arb<- c() #without further double robust
ATT_ebal<-c()
ATT_cbps<-c()
ATT_optweight<-c()
ATT_ebcw<-c()
ATT_ps<-c()

#ml_ps
bias_rf<-c()
bias_treebag<-c()
bias_xgboost<-c()
bias_lasso<-c()

ATT_rf<-c()
ATT_treebag<-c()
ATT_xgboost<-c()
ATT_lasso<-c()

# outcome_based
ATT_bart <-c()
mse_bart_y0 <-c()
ATT_rforest <-c()
mse_rf_y0 <-c()
ATT_cf <-c()


time_start = Sys.time()
# times = 1
for (i in 1:times){
  if (class == "simu1"){
    data_all = generate_data_1(n,dim)
  } else if (class == "simu2"){
    data_all = generate_data_2(n,dim)
  } else if (class == "simu3"){
    data_all = generate_data_3(n,dim)
  }
  
  data_all$w= as.factor(data_all$w)
  
  cat(paste("\nThe treatment data has",length(which(data_all$w==1))))
  
  ## base estimators
  base(i, data_all, dim)
  ## matching estimators
  matching(i, data_all, dim)
  ## weighting estimators
  weighting(i, data_all, dim)
  ## ml_ps weighting based estimators
  ml_ps(i, data_all, dim)
  ## outcome_model based estimators
  bart_estimate(i, data_all, dim)
  rf_estimate(i, data_all, dim)
  cf_estimate(i, data_all, dim)
  
  write.csv(data_all,paste("data_simu/", name, "_data_all_",i, ".csv",sep=""),row.names = FALSE)
}
time_end = Sys.time()
time_use = difftime(time_end, time_start, units="hours")[[1]]
cat(paste('\nTime_use is', time_use,"hours"))


# write out base estimators
original_result = data.frame(ATT_base,ATT_true_weight,ATT_true_weight_adjust,
                ATT_pscore,ATT_ols, 
                ATT_arb,ATT_ebal,ATT_cbps,ATT_optweight,ATT_ebcw,ATT_ps,
                ATT_rf,ATT_treebag,ATT_xgboost,ATT_lasso,
                ATT_bart,ATT_rforest,ATT_cf,
                bias_pscore,
                bias_rf,bias_treebag,bias_xgboost,bias_lasso,
                mse_bart_y0,mse_rf_y0)
write.csv(original_result, paste("R_result/", name, "_baseline_results.csv", 
                                 sep=""), row.names = FALSE)
