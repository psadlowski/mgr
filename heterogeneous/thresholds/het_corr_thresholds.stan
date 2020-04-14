// Categorical logit
// normal prior on beta
// Heterogeneous betas
// Corr structure for beta unknown
// Std norm independent deltas
// thresholds

data {
  int<lower=0> H; // # individuals
  int<lower=0> N; // # transactions (obs)
  int<lower=0> M; // # outcome levels
  int y[N]; // outcome
  real ps[N]; // price shocks
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
  vector[K] z_beta_gains[H];
  vector[K] z_beta_lpa[H];
  vector[K] z_beta_losses[H];
  
  matrix[K,J_beta] delta_gains;
  matrix[K,J_beta] delta_lpa;
  matrix[K,J_beta] delta_losses;
  
  cholesky_factor_corr[K] L_gains;
  cholesky_factor_corr[K] L_lpa;
  cholesky_factor_corr[K] L_losses;
  
  vector<lower=0, upper=pi()/2>[K] sigma_raw_gains;
  vector<lower=0, upper=pi()/2>[K] sigma_raw_lpa;
  vector<lower=0, upper=pi()/2>[K] sigma_raw_losses;
  
  vector<lower=0>[H] r_lower;
  vector<upper=0>[H] r_upper;
}

transformed parameters {
  vector[K] sigma_gains = 2.5 * tan(sigma_raw_gains);
  vector[K] sigma_lpa = 2.5 * tan(sigma_raw_lpa);
  vector[K] sigma_losses = 2.5 * tan(sigma_raw_losses);
  
  vector[K] beta_gains[H];
  vector[K] beta_lpa[H];
  vector[K] beta_losses[H];
  
  for (h in 1:H) {
    beta_gains[h] = delta_gains * z_beta[h] + diag_pre_multiply(sigma_gains, L_gains) * z_beta_gains[h];
    beta_lpa[h] = delta_lpa * z_beta[h] + diag_pre_multiply(sigma_lpa, L_lpa) * z_beta_lpa[h];
    beta_losses[h] = delta_losses * z_beta[h] + diag_pre_multiply(sigma_losses, L_losses) * z_beta_losses[h];
  }
}

model {
  // Priors
  // You may be able to turn this into a loop if using an inverse gamma is reasonable
  // Since Stan prefers it this way, let's use the decomposition
  // 
  // https://betanalpha.github.io/assets/case_studies/fitting_the_cauchy.html
  // Check the above for an alternative specification for Cauchy in STAN
  r_lower ~ std_normal();
  r_upper ~ std_normal();
  
  to_vector(delta_gains) ~ std_normal();
  to_vector(delta_lpa) ~ std_normal();
  to_vector(delta_losses) ~ std_normal();
  
  L_gains ~ lkj_corr_cholesky(1);
  L_lpa ~ lkj_corr_cholesky(1);
  L_losses ~ lkj_corr_cholesky(1);
  
  for (h in 1:H) {
    z_beta_gains[h] ~ std_normal();
    z_beta_lpa[h] ~ std_normal();
    z_beta_losses[h] ~ std_normal();
  }
  
  // Brand choice equation
  // Likelihood
  // Might be easier when rewritten to a 3d array if feasible
  // ATM it's multi-logit, not multi probit as in the original paper
  // Also, we're assuming here that the below is conditional on what we know beforehand, but well, let's see
  if (run_estimation == 1){
    // As of Stan 2.18 categorical logit not vectorised
    for (n in 1:N) {
      if (ps[n] < r_lower[h_id[n]]) {
        y[n] ~ categorical_logit(x[n] * beta_losses[h_id[n]]);
      } else if (ps[n] > r_upper[h_id[n]]) {
        y[n] ~ categorical_logit(x[n] * beta_gains[h_id[n]]);
      } else {
        y[n] ~ categorical_logit(x[n] * beta_lpa[h_id[n]]);
      }
    }
  }
}

generated quantities {
  vector[N] y_sim;
  corr_matrix[K] Omega_gains = L_gains * L_gains';
  corr_matrix[K] Omega_lpa = L_lpa * L_lpa';
  corr_matrix[K] Omega_losses = L_losses * L_losses';
  
  for (n in 1:N) {
    if (ps[n] < r_lower[h_id[n]]) {
      y_sim[n] = categorical_logit_rng(x[n] * beta_losses[h_id[n]]);
    } else if (ps[n] > r_upper[h_id[n]]) {
      y_sim[n] = categorical_logit_rng(x[n] * beta_gains[h_id[n]]);
    } else {
      y_sim[n] = categorical_logit_rng(x[n] * beta_lpa[h_id[n]]);
    }
  }
}

