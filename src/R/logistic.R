require(rstan)
require(arm)

data <- read.csv(file="../../data/reviewer-data.csv")

d <- data[,grep("ct_bin", colnames(data))]
d2 <- data[,!grepl("reviewerid|reviewno_reviewers", colnames(data))]
d2 <- d2[,grep("ct_bin|bin_avg|_mode",colnames(d2))]
d2 <- d2[, apply(d2,2,function(X) !any(is.na(X)))]

fit <- lapply(colnames(d2), function(i) bayesglm(paste0(i, "~ ."), data=d2, family=binomial, prior.scale=0.5, prior.df=Inf))

probs <- sapply(fit, fitted.values)
probs <- logit(probs)
colnames(probs) <- colnames(d2)
summary(probs)

pca <- prcomp(probs, scale=TRUE)
plot(pca$x[,1:2],pch=".")

write.csv(probs, file="../../data/fittedVals.csv")

p.data <- apply(d2,2,as.numeric)
p.data <- p.data[,apply(p.data,2,function(X) !any(is.na(X)))]
pca2 <- prcomp(p.data)
plot(pca2$x[,1:2],pch=".")

stan.mod <- stan_model(file="logistic.stan")
stanopt <- optimizing(stan.mod, data=list(N=nrow(d), K=ncol(d), X=d), verbose=TRUE, refresh=10)

