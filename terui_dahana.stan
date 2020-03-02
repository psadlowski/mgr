//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//
// ISSUE1: Check is beta mutliplication still makes sense (line 76)

data {
  int<lower=0> H; // # individuals
  int<lower=0> N; // # transactions (obs)
  int<lower=0> M; // # outcome levels
  int y[N]; // outcome
  int h_id[N]; // individual ID
  vector[N] ps; // Price shock for chosen brand
  int<lower=0> J_r; // # of threshold regressors
  matrix[H, J_r] z_r; // threshold regressors
  int<lower=0> J_beta; // # of beta regressors
  vector[J_beta] z_beta[H]; // beta regressors 
  int<lower=0> K; // # of brand choice regressors
  matrix[K, M] x[N]; // brand choice regressors
}

transformed data {
  real sigma_1 = (J_r + 4.0) / 2.0;
}

parameters {
  // Threshold equation
  vector[J_r] theta_lower; // lower threshold coefs
  vector[J_r] theta_upper; // upper threshold coefs
  real<lower=0> sigma_lower; // lower threshold error var
  real<lower=0> sigma_upper; // upper threshold error var
  
  vector<upper=0>[H] r_lower; // lower threshold
  vector<lower=0>[H] r_upper; // upper threshold
  
  // Beta equation
  // Normal noise for delta
  matrix[K, J_beta] z_delta_losses;
  matrix[K, J_beta] z_delta_lpa;
  matrix[K, J_beta] z_delta_gains;
  // Cov matrix scale
  vector<lower=0>[K] d_beta_losses;
  vector<lower=0>[K] d_beta_lpa;
  vector<lower=0>[K] d_beta_gains;
  // Cov matrix shape
  cholesky_factor_corr[K] L_beta_losses;
  cholesky_factor_corr[K] L_beta_lpa;
  cholesky_factor_corr[K] L_beta_gains;
  // Normal noise for beta
  vector[K] z_beta_gains[H];
  vector[K] z_beta_lpa[H];
  vector[K] z_beta_losses[H];
  
}

transformed parameters {
  // Beta equation
  matrix[K, J_beta] delta_losses;
  matrix[K, J_beta] delta_lpa;
  matrix[K, J_beta] delta_gains;
  // Brand choice equation
  // Betas as arrays for faster indexing
  vector[K] beta_losses[H];
  vector[K] beta_lpa[H];
  vector[K] beta_gains[H];
  // This may be less restrictive than original Rossi paper, but definitely should estimate
  // the covariance matrix for different demographics' impact on each beta
  // You should also check if the multiplication still makes sense as all you did was fix dims
  {
    // TBH, the entire matrix-like approach might simply be slowing everything down
    matrix[K, K] L_sigma;
    L_sigma = diag_pre_multiply(d_beta_gains, L_beta_gains);
    delta_gains = L_sigma * z_delta_gains;
    for (i in 1:H)
      beta_gains[i] = delta_gains * z_beta[i] + L_sigma * z_beta_gains[i];
    
    L_sigma = diag_pre_multiply(d_beta_lpa, L_beta_lpa);
    delta_lpa = L_sigma * z_delta_lpa;
    for (i in 1:H)
      beta_lpa[i] = delta_lpa * z_beta[i] + L_sigma * z_beta_lpa[i];
      
    L_sigma = diag_pre_multiply(d_beta_losses, L_beta_losses);
    delta_losses = L_sigma * z_delta_losses;
    for (i in 1:H)
      beta_losses[i] = delta_losses * z_beta[i] + L_sigma * z_beta_losses[i];
    
  }
  
}

model {
  // Threshold equation
  // Priors
  // Dropping multidimensionality as diagonal error matrix assumed, as suggested per manual
  theta_lower ~ normal(0, 0.01);
  theta_upper ~ normal(0, 0.01);
  // Here we stick to the original paper, even though other approaches have been proposed within the
  // Stan manual
  // Might be easier to drop multivariate approach here, even though I'm unsure if that's correct
  // ATM Univariate inverse gamma is used to simplify
  sigma_lower ~ inv_gamma(sigma_1, 1 / sigma_1);
  sigma_upper ~ inv_gamma(sigma_1, 1 / sigma_1);
  
  // Likelihood
  // Maybe you need to declare truncated normal here, dunno
  {
    vector[H] z_theta;
    
    z_theta = z_r * theta_lower;
    r_lower ~ normal(z_theta, sigma_lower);
    
    z_theta = z_r * theta_upper;
    r_upper ~ normal(z_theta, sigma_upper);
  }
  
  // Beta equation
  // A question is: do we need to draw ALL betas or only those we'll need?
  // Checking might be more demanding than just drawing each, though
  // Priors
  // You may be able to turn this into a loop if using an inverse gamma is reasonable
  // Since Stan prefers it this way, let's use the decomposition
  d_beta_losses ~ cauchy(0, 2.5);
  d_beta_lpa ~ cauchy(0, 2.5);
  d_beta_gains ~ cauchy(0, 2.5);
  
  // 1 gives uniform, maybe read more
  L_beta_losses ~ lkj_corr_cholesky(1);
  L_beta_lpa ~ lkj_corr_cholesky(1);
  L_beta_gains ~ lkj_corr_cholesky(1);

  // Std normal priors for delta noise
  to_vector(z_delta_losses) ~ std_normal();
  to_vector(z_delta_lpa) ~ std_normal();
  to_vector(z_delta_gains) ~ std_normal();
  
  // Likelihood
  // Potentially you'll need to iterate this
  for (i in 1:H) {
    z_beta_losses[i] ~ std_normal();
    z_beta_lpa[i] ~ std_normal();
    z_beta_gains[i] ~ std_normal();
  }
  
  // Brand choice equation
  // Likelihood
  // Might be easier when rewritten to a 3d array if feasible
  // ATM it's multi-logit, not multi probit as in the original paper
  // Only one issue here: are we sure the code below will ensure that error distribution across observations
  // will be based on the same prior?
  // Also, we're assuming here that the below is conditional on what we know beforehand, but well, let's see
  {
    vector[M] x_k;
    // As of Stan 2.18 categorical logit not vectorised
    for (n in 1:N) {
      // The actual price shock probably needs to be passed separately
      // Try to avoid those transpositions here, maybe?
      if (ps[n] < r_lower[h_id[n]]) // Losses condition
        x_k = x[n]' * beta_losses[h_id[n]];
      else if (ps[n] > r_upper[h_id[n]]) // Gains condition
        x_k = x[n]' * beta_gains[h_id[n]];
      else // LPA
        x_k = x[n]' * beta_lpa[h_id[n]];
      y[n] ~ categorical_logit(x_k);
    }
  }
  
}









