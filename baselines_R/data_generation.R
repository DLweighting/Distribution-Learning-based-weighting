generate_data_1<-function(n,dim=8){
  num = rmultinom(1, size = n, prob = c(1/3,1/3,1/3))
  
  x1 <- rnorm(n = num[1],  0, 1)
  x2 <- rnorm(n = num[2], -3, 1)
  x3 <- rnorm(n = num[3],  3, 1)
  x = sample(c(x1, x2, x3))
  for (i in 1:(dim-1)){
    x1 <- rnorm(n = num[1],  0, 1)
    x2 <- rnorm(n = num[2], -3, 1)
    x3 <- rnorm(n = num[3],  3, 1)
    xx <- sample(c(x1, x2, x3))
    x = cbind(x, xx)
  }
  
  x = as.matrix(x)
  for (i in 1:ncol(x)){
    x[,i]= (x[,i] - mean(x[,i]))/sd(x[,i])
  }
  
  linear = 0
  
  for (i in 1:dim){
    linear = linear + x[,i]*sc
  }
  
  prob_w = as.vector(1/(1+exp(0.5*linear)))
  w = rbinom(n, 1, prob_w)
  
  y0 = 0
  y0 = y0 + as.vector(x %*% beta1)
  
  y1 = y0 + 1 + rnorm(n, 0, 1)
  y0 = y0 + rnorm(n, 0, 1)
  
  y = ifelse(w == 1, y1, y0)
  
  colnames(x) = paste("X",seq(1,dim), sep="")
  return(data.frame(cbind(x,w,prob_w,y,y0,y1)))
}


generate_data_2<-function(n,dim=8){
  num = rmultinom(1, size = n, prob = c(1/3,1/3,1/3))

  x1 <- rnorm(n = num[1],  0, 1)
  x2 <- rnorm(n = num[2], -3, 1)
  x3 <- rnorm(n = num[3],  3, 1)
  x = sample(c(x1, x2, x3))
  for (i in 1:(dim-1)){
    x1 <- rnorm(n = num[1],  0, 1)
    x2 <- rnorm(n = num[2], -3, 1)
    x3 <- rnorm(n = num[3],  3, 1)
    xx <- sample(c(x1, x2, x3))
    x = cbind(x, xx)
  }

  x = as.matrix(x)
  for (i in 1:ncol(x)){
    x[,i]= (x[,i] - mean(x[,i]))/sd(x[,i])
  }

  linear = 0
  
  for (i in 1:dim){
    linear = linear + (x[,i]^2 -1) *sc # 1 is constant to make data balanced
  }
  
  for (i in 1:(dim-1)){
    for (j in (i+1):dim){
      linear = linear+x[,i]*x[,j] *sc
    }
  }

  prob_w = as.vector(1/(1+exp(0.5*linear)))
  w = rbinom(n, 1, prob_w)
  
  y0 = 0
  y0 = y0 + as.vector(x^2 %*% beta1)
  
  index = 1
  for (i in 1:(dim-1)){
    for (j in (i+1):dim){
      y0 = y0+ x[,i]*x[,j] * beta2[index]
      index = index +1
    }
  }  
  
  y1 = y0 + 1 + rnorm(n, 0, 1)
  y0 = y0 + rnorm(n, 0, 1)
  
  y = ifelse(w == 1, y1, y0)
  
  colnames(x) = paste("X",seq(1,dim), sep="")
  return(data.frame(cbind(x,w,prob_w,y,y0,y1)))
}



generate_data_3<-function(n,dim=8){
  num = rmultinom(1, size = n, prob = c(1/3,1/3,1/3))
  
  x1 <- rnorm(n = num[1],  0, 1)
  x2 <- rnorm(n = num[2], -3, 1)
  x3 <- rnorm(n = num[3],  3, 1)
  x = sample(c(x1, x2, x3))
  for (i in 1:(dim-1)){
    x1 <- rnorm(n = num[1],  0, 1)
    x2 <- rnorm(n = num[2], -3, 1)
    x3 <- rnorm(n = num[3],  3, 1)
    xx <- sample(c(x1, x2, x3))
    x = cbind(x, xx)
  }
  
  x = as.matrix(x)
  for (i in 1:ncol(x)){
    x[,i]= (x[,i] - mean(x[,i]))/sd(x[,i])
  }
  
  linear = 0

  for (i in 1:dim){
    linear = linear + (log(1 + x[,i]^2) -0.5) *sc # 0.5 is a constant to make the data balanced
  }
  
  for (i in 1:(dim-1)){
    for (j in (i+1):dim){
      linear = linear + x[,i]*x[,j] *sc
    }
  }
  
  prob_w = as.vector(1/(1+exp(0.5*linear)))
  w = rbinom(n, 1, prob_w)
  
  y0 = 0
  y0 = y0 + as.vector(log(x^2+1) %*% beta1)
  
  index = 1
  for (i in 1:(dim-1)){
    for (j in (i+1):dim){
      y0 = y0+ 2*sin(x[,i]*x[,j]) * beta2[index]
      index = index+ 1
    }
  }
  
  y1 = y0 + 1 + rnorm(n, 0, 1)
  y0 = y0 + rnorm(n, 0, 1)
  
  y = ifelse(w == 1, y1, y0)
  
  colnames(x) = paste("X",seq(1,dim), sep="")
  return(data.frame(cbind(x,w,prob_w,y,y0,y1)))
}
