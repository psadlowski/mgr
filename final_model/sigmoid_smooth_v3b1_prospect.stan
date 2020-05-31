// Categorical logit
// normal prior on beta
// Heterogeneous betas
// Corr structure for beta unknown
// Std norm independent deltas
// thresholds

data {
  int<lower=0> H; // # individuals
  int<lower=0> N; // # transactions (obs)
  int<lower=0> M; // # outcome levels less one (normalisation)
  int y[N]; // outcome
  row_vector[M] ps[N]; // price shocks
  int h_id[N]; // individual ID
  int<lower=0> J_beta; // # of beta regressors
  vector[J_beta] z_beta[H]; // beta regressors
  int<lower=0> K; // # of brand choice regressors
  matrix[M, K] x[N]; // brand choice regressors, counterintuitive shape for easier multiplication
  int<lower=0, upper=1> run_estimation; // simulation switch
  real<lower=0> k; // sigmoid shape parameter
  real sd_beta; // for delta and sigma_beta priors
}

parameters {
  vector[K] z_beta_gains[H];
  vector[K] z_beta_losses[H];
  
  matrix[K,J_beta] delta_gains;
  matrix[K,J_beta] delta_losses;
  
  cholesky_factor_corr[K] L_gains;
  cholesky_factor_corr[K] L_losses;
  
  vector<lower=0>[K] sigma_gains;
  vector<lower=0>[K] sigma_losses;
}

transformed parameters {
  vector[K] beta_gains[H];
  vector[K] beta_losses[H];
  
  vector[M] xbeta[N];
  
  for (h in 1:H) {
    beta_gains[h] = delta_gains * z_beta[h] + diag_pre_multiply(sigma_gains, L_gains) * z_beta_gains[h];
    beta_losses[h] = delta_losses * z_beta[h] + diag_pre_multiply(sigma_losses, L_losses) * z_beta_losses[h];
  }
  
  for (n in 1:N) {
    int hh = h_id[n];
    row_vector[M] phi_1 = exp( -log1p_exp( -k * ps[n])); // Check unnecessary
    row_vector[M] phi_2 = 1 - phi_1;
    
    xbeta[n] = diagonal( x[n] * append_col( beta_gains[hh], beta_losses[hh]) *
    append_row(phi_1, phi_2));
  }
}

model {
  // Priors
  // You may be able to turn this into a loop if using an inverse gamma is reasonable
  // Since Stan prefers it this way, let's use the decomposition
  // 
  // https://betanalpha.github.io/assets/case_studies/fitting_the_cauchy.html
  // Check the above for an alternative specification for Cauchy in STAN
  
  to_vector(delta_gains) ~ normal(0, sd_beta);
  to_vector(delta_losses) ~ normal(0, sd_beta);
  
  sigma_gains ~ normal( sd_beta, sd_beta);
  sigma_losses ~ normal( sd_beta, sd_beta);
  
  L_gains ~ lkj_corr_cholesky(16);
  L_losses ~ lkj_corr_cholesky(16);
  
  for (h in 1:H) {
    z_beta_gains[h] ~ std_normal();
    z_beta_losses[h] ~ std_normal();
  }
  
  // Brand choice equation
  // Likelihood
  // Might be easier when rewritten to a 3d array if feasible
  // ATM it's multi-logit, not multi probit as in the original paper
  // Also, we're assuming here that the below is conditional on what we know beforehand, but well, let's see
  if (run_estimation == 1){
    // As of Stan 2.18 categorical logit not vectorised
    // Asymptote is 1 so a=1 in the sigmoid
    // or maybe logistic, not Gompertz
    for (n in 1:N) {
      y[n] ~ categorical_logit( append_row( xbeta[n], 0));
    }
  }
}

generated quantities {
  vector[N] y_sim;
  
  for (n in 1:N) {
    y_sim[n] = categorical_logit_rng( append_row( xbeta[n], 0));
  }
}

