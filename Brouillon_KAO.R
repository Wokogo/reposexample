
source("~/Documents/ScriptsR/R/function_eric.R")
source("~/Documents/ScriptsR/R/kao_functions.R")
source("~/Documents/ScriptsR/R/kaoConv.R")
source("~/Documents/ScriptsR/R/MLkao.R")
source("~/Documents/ScriptsR/R/MLkaoL.R")
# source("~/Documents/ScriptsR/R/Brouillon_Joseph.R")


truncate1 <- function(x){
  pmin(pmax(x,exp(-700)),exp(700))
}


##############

AggKalmanGTetafix <- function(FtMat, expertMatrix, Y, eta){
  Nobs = nrow(expertMatrix); Const = Nobs/100
  pseudoLoss = FtMat  # first term of the pseudo losses
  cumul_regret_maT <- apply(pseudoLoss, 2, cumsum)
  ftMAX <- max(FtMat)/Const
  # truncate function for the 2nd term of the pseudo loss
  TrunCATE <- function(x)pmin(pmax(0,x),ftMAX)
  # computing \hat{y}_1
  yhat = numeric(); yhat[1] = mean(expertMatrix[1,])
  # error = sum of the 2nd term of the pseudo loss
  error  = numeric()
  error <- TrunCATE((expertMatrix[1,]-yhat[1])^2)
  pseudoL.now = cumul_regret_maT[1,] - error 
  yhatFunc <- function(LR, Expert, Loss){# Loss=\sum_{s=1}^{t-1}pseudoLoss
    rhoVect=truncate1(exp(-LR*Loss)); rhoVect=rhoVect/sum(rhoVect)
    res=sum(rhoVect*Expert); return(res)
  }
  yhat[2] = yhatFunc(LR=eta , Expert=expertMatrix[2,], Loss=pseudoL.now)
  for (i in 3:Nobs){
    error <- error + TrunCATE((expertMatrix[i-1,]-yhat[i-1])^2)
    pseudoL.now = cumul_regret_maT[i-1,] - error
    yhat[i] = yhatFunc(eta, Expert=expertMatrix[i,], Loss=pseudoL.now)
  }
  finalRes = list(yhat=yhat)
  return(finalRes)
}

##############################
n=50; prop=.2; win=3; seed=1
armaExperts <- function(n, prop, win, seed){
  # n is the size of the data
  # prop is the proportion of data that is used for the fitting step
  # win stands for window
  # seed is an argument of set.seed()
  require(forecast)
  pmax <- 5; qmax <- 5
  sigma <- 1; n.fit <- n*prop; n.pred <- n*(1-prop)
  orders <- unique(combn(x=c(1:pmax,1:qmax), 2, simplify = FALSE))
  names.expert <- sapply(orders, function(o)paste0(o, collapse = ""))
  names.expert <- c("y", names.expert); expert = NULL; ft = NULL
  set.seed(seed)
  eps<-rnorm(n, 0, sd=sigma)
  set.seed(seed); Nrep <- floor(n.pred/win)
  #ar = round(runif(4, min = -.5, max = .5), 2)
  ar = round(ARcoef(4), 2)
  #ma = round(runif(4, min = -.8, max = .9), 2)
  ma = round(ARcoef(4), 2)
  y1 <- as.numeric(arima.sim(n = n, list(ar = ar, ma = ma), innov=eps))
  y1.fit.list <- lapply(orders, function(o)forecast::Arima(y1[1:n.fit],order=c(o[1],0,o[2]), method="ML"))
  for (i in 1:Nrep){
    i=1
    cat(i,"\t")
    #win = ifelse(i==Nrep, min(n.pred-Nrep*win, win), win)
    y.pred.list <- lapply(y1.fit.list, function(fit)predict(fit, n.ahead=win)) # predicting win ahead
    expert.now <- matrix(sapply(y.pred.list, function(p)p$pred), ncol=length(orders))
    ft.now <- matrix(sapply(y.pred.list, function(p)p$se), ncol=length(orders))
    expert <- rbind(expert, expert.now); ft <- rbind(ft, ft.now)
    n.fit <- n.fit+win
    y1.fit.list <- lapply(orders, function(o)forecast::Arima(y1[1:n.fit],order=c(o[1],0,o[2]), method="ML"))
  }
  n.fit <- n*prop
  expert <- cbind(y1[(n.fit+1):n], expert)
  colnames(expert) <- names.expert
  res <- list(expert=expert, ft=ft)
  return(res)
}
require(doParallel)
registerDoParallel(20)

# f=function(x, v){
#   res=matrix(sapply(1:length(v), function(k)x[,k]*v[k]),ncol=length(v))
#   return(res)
# }
# xx=matrix(1:9,3,3); cc=c(-1,2,1)
# f(xx, cc)

### on migale
resList1 = list()
foreach (i = 1:10) %dopar%{cat(i, "\t");resList1[[i]]=armaExperts(n=4000, prop=.5, win=1, seed=i)}
#
resList2 = list()
foreach (i = 11:20)%dopar%{cat(i, "\t");resList2[[i]]=armaExperts(n=4000, prop=.5, win=1, seed=i)}
#
resList3 = list()
foreach (i = 21:30)%dopar%{cat(i, "\t");resList3[[i]]=armaExperts(n=4000, prop=.5, win=1, seed=i)}
#
resList4 = list()
foreach (i = 31:40)%dopar%{cat(i, "\t");resList4[[i]]=armaExperts(n=4000, prop=.5, win=1, seed=i)}
#
resList5 = list()
foreach (i = 41:50)%dopar%{cat(i, "\t");resList5[[i]]=armaExperts(n=4000, prop=.5, win=1, seed=i)}
#
resList6 = list()
foreach (i = 51:60){cat(i, "\t");resList6[[i]]=armaExperts(n=4000, prop=.5, win=1, seed=i)}
#
resList7 = list()
foreach (i = 61:70)%dopar%{cat(i, "\t");resList7[[i]]=armaExperts(n=4000, prop=.5, win=1, seed=i)}
#
resList8 = list()
foreach (i = 71:80)%dopar%{cat(i, "\t");resList8[[i]]=armaExperts(n=4000, prop=.5, win=1, seed=i)}
#
resList9 = list()
foreach (i = 81:90)%dopar%{cat(i, "\t");resList9[[i]]=armaExperts(n=4000, prop=.5, win=1, seed=i)}
#
resList10 = list()
foreach (i = 91:100)%dopar%{cat(i, "\t");resList10[[i]]=armaExperts(n=4000, prop=.5, win=1, seed=i)}
#
###

test <- armaExperts(n=50, prop=.2, win=1, seed=1)
test.agg = AggKalmanGTetafix(FtMat=test$ft, expertMatrix=test$expert[,-1], Y=test$expert[,1], eta=seq(f=1e-6,t=.8,l=1e4))
mean((test.agg$yhat-test$expert[,1])^2)

## comparison with result from opera/boa
require(opera)
loss.type <- 'square'
loss.gradient <- TRUE
# the model
model <- 'BOA'

agg.opera.BOA <- mixture(Y = test$expert[,1], experts = test$expert[,-1], model = model , loss.type = loss.type, loss.gradient = loss.gradient)

agg.opera.BOA.weightMat <- agg.opera.BOA$weights
agg.opera.BOA.expert <- as.vector(apply(agg.opera.BOA.weightMat*test$expert[,-1], 1, sum))

mean((agg.opera.BOA.expert-test$expert[,1])^2)

matplot(test$expert[,-1], col='grey', type='l', lty=1, ylim = c(-5,5))
lines(test$expert[,1], col='black', type='l', lty=2)
lines(agg.opera.BOA.expert, col='red')
legend("topleft", legend = c("experts","aggregation","yt"), col = c("grey","red","black"), fill = c("grey","red","black"))

######### extraction des mse apres le burn in 
aggKaoList2 = list()
aggBoaList2 = list()
mseKao2 = numeric()
mseBoa2 = numeric()
for (r in 14:length(expertList)){
  cat(r,"\t")
  #simul = armaExperts2(n=10000, prop=.2, seed=r) # armaExperts2(n=10000, prop=.2, win=1, seed=r)
  #expertList[[r]] = simul$expert
  #ftList[[r]] = simul$ft
  aggKaoList2[[r]] = AggKalmanGTetafix(FtMat=ftList[[r]], expertMatrix=expertList[[r]][,-1], Y=expertList[[r]][,1], eta=seq(f=1e-6,t=.8,l=1e4))$yhat
  BoaNow = mixture(Y = expertList[[r]][,1], experts = expertList[[r]][,-1], model = model , loss.type = loss.type, loss.gradient = loss.gradient)
  aggBoaList2[[r]] = as.vector(apply(BoaNow$weights*expertList[[r]][,-1], 1, sum))
  mseKao2[r] = mean((aggKaoList2[[r]]-expertList[[r]][,-1])^2); mseBoa2[r] = mean((aggBoaList2[[r]]-expertList[[r]][,-1])^2)
}

par(mfrow=c(2, 1))
plot(mseBoa2[-(1:14)], type = "b", col="red", ylim=c(min(c(mseBoa2[-(1:14)],mseKao2[-(1:14)])),max(c(mseBoa2[-(1:14)],mseKao2[-(1:14)]))))
lines(mseKao2[-(1:14)], type = "b", col="blue", pch="x")
legend("topleft", legend = c("KAO","BOA"), col = c("blue","red"), fill = c("blue","red"))

############ fonction d'Olivier
armaExperts2 <- function(n, prop, seed){ # bonne fonction
  # n is the size of the data
  # prop is the proportion of data that is used for the fitting step
  # win stands for window
  # seed is an argument of set.seed()
  require(KFAS)
  pmax <- 9; qmax <- 9
  sigma <- 1; n.fit <- n*prop; n.pred <- n*(1-prop)
  orders <- unique(combn(x=c(1:pmax,1:qmax), 2, simplify = FALSE))
  names.expert <- sapply(orders, function(o)paste0(o, collapse = ""))
  names.expert <- c("y", names.expert); expert = NULL; ft = NULL
  set.seed(seed)
  eps<-rnorm(n, 0, sd=sigma)
  set.seed(seed); 
  ar = round(ARcoef(4), 2) # round(runif(4, min = -1, max = 1), 2)
  ma = round(ARcoef(4), 2) # round(runif(5, min = -1, max = 1), 2)
  y1 <- as.numeric(arima.sim(n = n, list(ar = artransform(ar), ma = artransform(ma)), innov=eps))
  y1.fit.list <- lapply(orders, function(o)forecast::Arima(y1[1:n.fit],order=c(o[1],0,o[2]),method="ML",include.mean=FALSE))
  sigma2 <- sapply(y1.fit.list, function(p)p$sigma2)    # very bad estimation of sigma...
  SS.pred.list <- lapply(y1.fit.list,function(fit)SSModel(y1[n.fit+1:n.pred]~-1+SSMarima(ar=fit$model$phi,ma=fit$model$theta)))
  KF.pred.list <- lapply(SS.pred.list,function(pred)KFS(pred))
  expert <- matrix(sapply(KF.pred.list, function(p)p$a[-(n.pred+1),1]), ncol=length(orders))
  ft <- matrix(sapply(KF.pred.list, function(p)p$F), ncol=length(orders)) # -2 if for the confidency of the expert on itself
  ft <- ft + matrix(rep(sigma2,each=n.pred),ncol=length(orders)) # + the estimation of the variance of the noise in each model
  expert <- cbind(y1[n.fit+1:n.pred],expert)
  colnames(expert) <- names.expert
  res <- list(expert=expert, ft=ft, sigma2=sigma2)
  return(res)
}
###
start_time <- Sys.time()
test2 <- armaExperts2(n=1000, prop=.5, seed=1)
end_time <- Sys.time()
end_time - start_time

test_1000_list = lapply(1:10, function(i)tryCatch(armaExperts2(n=1000, prop=.5, seed=i),error=function()NA))
length(test_1000_list)


# an example
t=test_1000_list[[1]]
test_1000.aggKao = AggKalmanGTetafix(FtMat=test_1000_list[[1]]$ft, expertMatrix=test_1000_list[[1]]$expert[,-1], 
                                     Y=test_1000_list[[1]]$expert[,1], eta=seq(f=1e-6,t=.8,l=1e4))$yhat

test_1000.aggKao_mat = sapply(test_1000_list, function(t){
  agg=AggKalmanGTetafix(FtMat=t$ft, expertMatrix=t$expert[,-1],Y=t$expert[,1], eta=seq(f=1e-6,t=.8,l=1e4))$yhat
  res=mean((t$expert[,1]-agg)^2)
  return(res)
  }
  )#, ncol=length(test_1000_list))
#

test_1000.aggKao.cor_mat = sapply(test_1000_list, function(t){
  agg=AggKalmanGTetafix(FtMat=t$ft-min(t$sigma2), expertMatrix=t$expert[,-1],Y=t$expert[,1], eta=seq(f=1e-6,t=.8,l=1e4))$yhat
  res=mean((t$expert[,1]-agg)^2)
  return(res)
}
)

test_1000.aggBoa_mat = sapply(test_1000_list, function(t){
  agg.opera.BOA <- mixture(Y=t$expert[,1],experts=t$expert[,-1],model=model,loss.type=loss.type,loss.gradient=loss.gradient)
  agg.opera.BOA.weightMat <- agg.opera.BOA$weights
  agg.boa <- as.vector(apply(agg.opera.BOA.weightMat*t$expert[,-1], 1, sum))
  res=mean((t$expert[,1]-agg.boa)^2)
  return(res)
   })#

plot(test_1000.aggKao.cor_mat, type='b', pch="*", ylab="mse")
lines(test_1000.aggBoa_mat, type='b', col='red')
legend("topleft", legend = c("kao","boa"), col = c("black","red"), fill = c("black", "red"))

### Nouvelles simulations
N=1500
eps<-rnorm(n=N, 0, sd=1)
AR = round(ARcoef(4), 2)
MA = round(runif(4, min = -1, max = 1), 2)
y <- as.numeric(arima.sim(n = N, list(ar = AR, ma = MA), innov=eps))
n.fit = 500
y.fit <- forecast::Arima(y[1:n.fit],order=c(5,0,6), method="ML")
y.refit <- forecast::Arima(y[1:N], model = y.fit) # prevision sur les 500 suivants
y.predErrorVar <- mean((y[(n.fit+1):(2*n.fit)]-y.refit$fitted[(n.fit+1):(2*n.fit)])^2)
require(KFAS)

# require(smooth)
# y.pred <- predict(object=y.fit, n.ahead=500, se.fit=TRUE)

n=50; prop=1/3; seed=1
armaExperts <- function(n, prop, seed){
  # n is the size of the data
  # prop is the proportion of data that is used for the fitting step
  # seed is an argument of set.seed()
  require(forecast)
  pmax <- 5; qmax <- 5
  sigma <- 1; n.fit <- n*prop; n.predVar <- (n.fit+1):(2*n.fit) #n.pred <- n*(1-prop)
  orders <- unique(combn(x=c(1:pmax,1:qmax), 2, simplify = FALSE))
  names.expert <- sapply(orders, function(o)paste0(o, collapse = ""))
  names.expert <- c("y", names.expert); expert = NULL; ft = NULL
  set.seed(seed)
  eps<-rnorm(n, 0, sd=sigma)
  #ar = round(runif(4, min = -.5, max = .5), 2)
  ar = round(ARcoef(4), 2)
  ma = round(runif(4, min = -.8, max = .9), 2)
  # ma = round(ARcoef(4), 2)
  y1 <- as.numeric(arima.sim(n = n, list(ar = ar, ma = ma), innov=eps))
  y1.fit.list <- lapply(orders, function(o)forecast::Arima(y1[1:n.fit],order=c(o[1],0,o[2]), method="CSS"))
  y1.refit.list <- lapply(y1.fit.list, function(fit)forecast::Arima(y[1:n], model = fit))
  expert <- matrix(sapply(y1.refit.list, function(refit)refit$fitted[-(1:(2*n.fit))]), ncol = length(orders))
  ft <- sapply(y1.refit.list, function(refit)mean((y1[n.predVar]-refit$fitted[n.predVar])^2))
  ft <- matrix(rep(ft,n-2*n.fit),ncol=length(orders),byrow=TRUE)
  expert <- cbind(y1[-(1:(2*n.fit))], expert)
  colnames(expert) <- names.expert
  res <- list(expert=expert, ft=ft)
  return(res)
}

start_time <- Sys.time()
test=armaExperts(n=1500, prop=1/3, seed=28)
end_time <- Sys.time()
end_time - start_time

dim(test$expert)
dim(test$ft)
head(test$ft)
tail(test$ft)

##
seeds=c(2:20,23:27,30:33,35:68)
armaExperts.list <- lapply(seeds, function(s)armaExperts(n=1500, prop=1/3, seed=s))

##sans gradient trick mais avec eta adaptatif (dans une grille)
kao <- kaoGrid(Ft=test$ft, expertMatrix=test$expert[,-1], y=test$expert[,1])

mes_exp <- apply(test$expert[,-1], 2, function(x)(mean((test$expert[,1]-x)^2)))

kao_MSE <- mean((test$expert[,1]-kao$pred)^2)
kao_regret <- mean((test$expert[,1]-kao$pred)^2 - (test$expert[,1]-test$expert[,-1][, which.min(mes_exp)])^2)

regretFunction <- function(ft, expert, y, method){
  agg.method <- method(Ft=ft,expertMatrix=expert,y=y)
  mes_exp.method <- apply(expert, 2, function(x)(mean((y-x)^2)))
  method_MSE <- mean((y-agg.method$pred)^2)
  method_regret <- mean((y-agg.method$pred)^2 - (y-expert[, which.min(mes_exp.method)])^2)
  return(method_regret)
}

regretFunction(ft=test$ft, expert=test$expert[,-1], y=test$expert[,1], method=kaoGrid)


## avec gradient trick et eta adaptatif (dans une grille)
kalman.kaoConv = kaoConv(FtMat=test$ft, expertMatrix=test$expert[,-1], y=test$expert[,1])
kaoConv_MSE <-mean((test$expert[,1]-kalman.kaoConv$pred)^2)
kaoConv_regret <- mean((test$expert[,1]-kalman.kaoConv$pred)^2 - (test$expert[,1]-test$expert[,-1][, which.min(mes_exp)])^2)

## avec gradient trick, eta multiple et adaptatif (MLR)
kalman.kao_GMLR = MLkao1(FtMat=test$ft, expertMatrix=test$expert[,-1], y=test$expert[,1])
kaoGMLR_MSE <-  mean((test$expert[,1]-kalman.kao_GMLR$pred)^2)
kaoGMLR_MSE_regret <- mean((test$expert[,1]-kalman.kao_GMLR$pred)^2 - (test$expert[,1]-test$expert[,-1][, which.min(mes_exp)])^2)

## BOA

boa <- mixture(Y=test$expert[,1], experts=test$expert[,-1], model = "BOA", loss.gradient=FALSE)
boa_MSE<- mean((test$expert[,1]-boa$pred)^2)
boa_regret <- mean((test$expert[,1]-boa$pred)^2 - (test$expert[,1]-test$expert[,-1][, which.min(mes_exp)])^2)


boa_grad <- mixture(Y=Data1$Y, experts=experts, model = "BOA", loss.gradient = TRUE)
boa_grad_MSE <- mean((Data1$Y-boa$pred)^2)
boa_grad_regret <- mean((Data1$Y-boa_grad$pred)^2 - (Data1$Y-experts[, which.min(mes_exp)])^2)

regretFunctionBOA <- function(expert, y, loss.gradient=FALSE){
  require(opera)
  agg.boa <- mixture(Y=y, experts=expert, model = "BOA", loss.gradient=loss.gradient)
  boa_MSE<- mean((y-agg.boa$pred)^2)
  boa_regret <- mean((y-boa$pred)^2 - (y-expert[, which.min(mes_exp)])^2)
  return(boa_regret)
}

regretFunctionMLpoly <- function(expert, y, loss.gradient=FALSE){
  require(opera)
  agg.MLpol <- mixture(Y=y, experts=expert, model = "MLpol", loss.gradient=loss.gradient)
  mse_exp.MLpol <- apply(expert, 2, function(x)(mean((y-x)^2)))
  # MLpol_MSE<- mean((y-agg.MLpol$pred)^2)
  MLpol_regret <- mean((y-agg.MLpol$pred)^2 - (y-expert[, which.min(mse_exp.MLpol)])^2)
  return(MLpol_regret)
}


regretFunctionBOA(expert=test$expert[,-1], y=test$expert[,1], loss.gradient=FALSE)

### MC simulations

# KAO
kao.regret <- sapply(armaExperts.list, function(ae)regretFunction(ft=ae$ft, expert=ae$expert[,-1], y=ae$expert[,1], method=kaoGrid))

kaoConv.regret <- sapply(armaExperts.list, function(ae)regretFunction(ft=ae$ft, expert=ae$expert[,-1], y=ae$expert[,1], method=kaoConv))

kaoGMLR.regret <- sapply(armaExperts.list, function(ae)regretFunction(ft=ae$ft, expert=ae$expert[,-1], y=ae$expert[,1], method=MLkao1))

# BOA
boa.regret <- sapply(armaExperts.list, function(ae)regretFunctionBOA(expert=ae$expert[,-1], y=ae$expert[,1], loss.gradient=FALSE))

boa_grad.regret <- sapply(armaExperts.list, function(ae)regretFunctionBOA(expert=ae$expert[,-1], y=ae$expert[,1], loss.gradient=TRUE))

# MLpoly
MLpoly.regret <- sapply(armaExperts.list, function(ae)regretFunctionMLpoly(expert=ae$expert[,-1], y=ae$expert[,1], loss.gradient=FALSE))

MLpoly_grad.regret <- sapply(armaExperts.list, function(ae)regretFunctionMLpoly(expert=ae$expert[,-1], y=ae$expert[,1], loss.gradient=TRUE))

method=as.vector(t(replicate(length(kao.regret), c("kao", "kaoConv", "kaoGMLR", "boa", "boaGrad", "mlpol", "mlpolGrad"))))

regret=c(kao.regret, kaoConv.regret, kaoGMLR.regret, boa.regret, boa_grad.regret, MLpoly.regret, MLpoly_grad.regret)

regret.df <- data.frame(regret=regret, method=method)
str(regret.df)
head(regret.df)


require(ggplot2)
pdf("~/Dropbox/Eric/Notes_Eric/simulations/graphics/regrets.kao_boa.pdf")
# pdf("~/Documents/ScriptsR/regrets.kao_boa.pdf")
ggplot(regret.df, aes(x=method, y=regret, color=method)) + 
  geom_boxplot() + ylab("Regret") + xlab("") +
  theme(legend.position = "none")   + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(limits = c(-0.1,1.5))
dev.off()

# Astuces de Yannig (ce 06/04/2021)
agg.online<- mixture(Y = IrishAgg1$dem , experts = experts, model = 'EWA', 
                     loss.type = "square", loss.gradient = F, parameters=list(grid.eta= seq(learning.rate/100, 10*learning.rate,length=10)))

#
M <- (IrishAgg0$dem-IrishAgg0$dem336)^2%>%mean
learning.rate <-  (1/M)*sqrt(8*log(ncol(experts)))/nrow(IrishAgg1)
#





