inf.fun <- function(fit) {

  # Converts from
  #   > dd
  #    a b
  # 1  1 1
  # 2  1 2
  # 3  1 3
  # 4  1 4
  # 5  2 1
  # 6  2 2
  # 7  2 3
  # 8  2 4
  # 9  3 1
  # 10 3 2
  # 11 3 3
  # 12 3 4
  #   To
  #   > model.matrix(~ a + b, dd)
  #    (Intercept) a2 a3 b2 b3 b4
  # 1            1  0  0  0  0  0
  # 2            1  0  0  1  0  0
  # 3            1  0  0  0  1  0
  # 4            1  0  0  0  0  1
  # 5            1  1  0  0  0  0
  # 6            1  1  0  1  0  0
  # 7            1  1  0  0  1  0
  # 8            1  1  0  0  0  1
  # 9            1  0  1  0  0  0
  # 10           1  0  1  1  0  0
  # 11           1  0  1  0  1  0
  # 12           1  0  1  0  0  1


  dm <- model.matrix(fit)
  Ihat <- (t(dm) %*% (dm * fit$fitted.values * (1 - fit$fitted.values))) / nrow(dm)
  ## influence function
  infl <- (dm * resid(fit, type = "response")) %*% solve(Ihat)
}



## generate data
data1 <- data.frame(x = rbinom(1000, size = 1, 0.15))
# sensitivity 0.8 specificity 0.8
data1$a = ifelse(data1$x == 1, rbinom(1000,1,0.8), rbinom(1000,1,0.2))
# sensitivity 0.9 specificity 0.9
#data1$a = ifelse(data1$x == 1, rbinom(1000,1,0.9), rbinom(1000,1,0.1))
# sensitivity 0.9 specificity 0.8
#data1$a = ifelse(data1$x == 1, rbinom(1000,1,0.9), rbinom(1000,1,0.2))
# sensitivity 0.8 specificity 0.9
#data1$a = ifelse(data1$x == 1, rbinom(1000,1,0.8), rbinom(1000,1,0.1))

data1$z1 = runif(1000)
data1$z2 = rbinom(1000, size = 1, 0.6)

p_y <- exp(-2 + 1.5 * data1$x + data1$z1 + data1$z2)/(1 + exp(-2 + 1.5 * data1$x + data1$z1 + data1$z2))
data1$y <- rbinom(1000, size = 1, p_y)

tru.val <- c(-2, 1.5, 1, 1)

## simulation
## define strata
## define strata 
data1$stra = 1 + 4 * data1$y + 2 * data1$a + data1$z2
## population strata size
n1 =  xtabs(~stra, data1)

strata = list()
for (index in 1:length(n1)){
  strata[[index]] = which(data1$stra == index)  
}

full = glm(y ~ x + z1 + z2, data = data1, family = binomial)
impute = glm(x ~ a + z1 + z2, data = data1, family = binomial)
infl <- inf.fun(full)[, 2]

sd.stra1 = numeric()

for(ii in 1:length(strata)){
  sd.stra1 = c(sd.stra1, sd(infl[strata[[ii]]]))
}

optimal.size = integer.neyman.w2(n.strata = length(strata), NS = n1 * sd.stra1, sample.size = 300, upper = n1)




