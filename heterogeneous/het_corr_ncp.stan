// Categorical logit
// normal prior on beta
// Heterogeneous betas

data {
  int<lower=0> H; // # individuals
  int<lower=0> N; // # transactions (obs)
  int<lower=0> M; // # outcome levels
  int y[N]; // outcome
  // real ps[N]; // price shocks
  int h_id[N]; // individual ID
  // int<lower=0> J_r; // # of threshold regressors
  // matrix[H, J_r] z_r; // threshold regressors
  int<lower=0> J_beta; // # of beta regressors
  vector[J_beta] z_beta[H]; // beta regressors
  int<lower=0> K; // # of brand choice regressors
  matrix[M, K] x[N]; // brand choice regressors, counterintuitive shape for easier multiplication
  int<lower=0, upper=1> run_estimation; // simulation switch
}

parameters {
  vector[K] z_beta_n[H];
  matrix[K,J_beta] delta;
  cholesky_factor_corr[K] L;
  vector<lower=0, upper=pi()/2>[K] sigma_raw;
}

transformed parameters {
  vector[K] sigma = 2.5 * tan(sigma_raw);
  vector[K] beta[H];
  for (h in 1:H) {
    beta[h] = delta * z_beta[h] + diag_pre_multiply(sigma, L) * z_beta_n[h];
  }
}

model {
  // Priors
  // You may be able to turn this into a loop if using an inverse gamma is reasonable
  // Since Stan prefers it this way, let's use the decomposition
  // 
  // https://betanalpha.github.io/assets/case_studies/fitting_the_cauchy.html
  // Check the above for an alternative specification for Cauchy in STAN
  
  to_vector(delta) ~ std_normal();
  L ~ lkj_corr_cholesky(1);
  for (h in 1:H) {
    z_beta_n[h] ~ std_normal();
  }
  
  // Brand choice equation
  // Likelihood
  // Might be easier when rewritten to a 3d array if feasible
  // ATM it's multi-logit, not multi probit as in the original paper
  // Also, we're assuming here that the below is conditional on what we know beforehand, but well, let's see
  if (run_estimation == 1){
    // As of Stan 2.18 categorical logit not vectorised
    for (n in 1:N) {
      y[n] ~ categorical_logit(x[n] * beta[h_id[n]]);
    }
  }
}

generated quantities {
  vector[N] y_sim;
  corr_matrix[K] Omega = L * L';
  for (n in 1:N) {
    y_sim[n] = categorical_logit_rng(x[n] * beta[h_id[n]]);
  }
}

