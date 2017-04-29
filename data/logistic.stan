functions {
  real logisticReal(matrix mu, matrix X){
    int K;
    int N;
    real lp;
    
    K = cols(X);
    N = rows(X);
    lp = 0;
    
    for(k in 1:K){
      for (n in 1:N){
        lp = lp + X[n,k] * mu[n,k] +  log1m_inv_logit(mu[n,k]);
      }
    }
    return(lp);
  }
}

data {
  int N;
  int K;
  matrix[N, K] X;
}

parameters {
  vector[K] beta;
  matrix[K,N] mu;
  cholesky_factor_corr[K] L;
}

transformed parameters{
  matrix[N,K] mu_hat;
  
  {
    matrix[K,N] beta_temp;
    
    for(n in 1:N) for(k in 1:K) beta_temp[k, n] = beta[k];
    mu_hat = (beta_temp + L * mu)';
  }
  
}

model {
  beta ~ normal(0,1);
  to_vector(mu) ~ normal(0,1);
  L ~ lkj_corr_cholesky(K+1);
  
  target += logisticReal(mu_hat, X);
}
