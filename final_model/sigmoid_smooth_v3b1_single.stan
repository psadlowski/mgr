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
  // int<lower=0> J_r; // # of threshold regressors
  // matrix[H, J_r] z_r; // threshold regressors
  int<lower=0> J_beta; // # of beta regressors
  vector[J_beta] z_beta[H]; // beta regressors
  int<lower=0> K; // # of brand choice regressors
  matrix[M, K] x[N]; // brand choice regressors, counterintuitive shape for easier multiplication
  int<lower=0, upper=1> run_estimation; // simulation switch
  real<lower=0> k; // sigmoid shape parameter
  // real g;
  real sd_beta; // for delta and sigma_beta priors
  // real scf; // threshold scaling factor
  // real<lower=0, upper=1> r_barrier; // max phi value at 0
}

parameters {
  vector[K] z_beta_gains[H];
  vector[K-1] z_beta_lpa[H];
  vector[K] z_beta_losses[H];
  
  matrix[K,J_beta] delta_gains;
  matrix[K-1,J_beta] delta_lpa;
  matrix[K,J_beta] delta_losses;
  
  cholesky_factor_corr[K] L_gains;
  cholesky_factor_corr[K-1] L_lpa;
  cholesky_factor_corr[K] L_losses;
  
  vector<lower=0>[K] sigma_gains;
  vector<lower=0>[K-1] sigma_lpa;
  vector<lower=0>[K] sigma_losses;
  
  // Price elasticity in the LPA should be independent of the rest
  // It should also have a fixed, small variance, TD suggests
  real pe_lpa[H];
  
  real<upper=0> r_lower;
  real<lower=0> r_upper;
}

transformed parameters {
  vector[K] beta_gains[H];
  vector[K] beta_lpa[H];
  vector[K] beta_losses[H];
  
  vector[M] xbeta[N];
  
  for (h in 1:H) {
    beta_gains[h] = delta_gains * z_beta[h] + diag_pre_multiply(sigma_gains, L_gains) * z_beta_gains[h];
    beta_lpa[h] = append_row(delta_lpa * z_beta[h] + diag_pre_multiply(sigma_lpa, L_lpa) * z_beta_lpa[h],
    pe_lpa[h]);
    beta_losses[h] = delta_losses * z_beta[h] + diag_pre_multiply(sigma_losses, L_losses) * z_beta_losses[h];
  }
  
  for (n in 1:N) {
    int hh = h_id[n];
    row_vector[M] phi_1 = exp( -log1p_exp( -k * (r_lower - ps[n]))); // Check unnecessary
    row_vector[M] phi_2 = exp( -log1p_exp( -k * (ps[n] - r_upper))); // Check unnecessary
    
    //row_vector[M] phi_1 = 1 ./ (1 + exp( -k * (r_lower[hh] - ps[n]))); // Check unnecessary
    //row_vector[M] phi_2 = 1 ./ (1 + exp( -k * (ps[n] - r_upper[hh]))); // Check unnecessary
    row_vector[M] phi_3 = 1 - phi_1 - phi_2; // Check unnecessary
    
    xbeta[n] = diagonal( x[n] * append_col( beta_gains[hh], append_col( beta_lpa[hh], beta_losses[hh])) *
    append_row(phi_2, append_row( phi_3, phi_1)));
    
  }
}

model {
  // Priors
  // You may be able to turn this into a loop if using an inverse gamma is reasonable
  // Since Stan prefers it this way, let's use the decomposition
  // 
  // https://betanalpha.github.io/assets/case_studies/fitting_the_cauchy.html
  // Check the above for an alternative specification for Cauchy in STAN
  
  // Couldn't decipher what their prior would imply for stds, made some assumptions as a result
  // Their d' is dim of z_beta, which is quite counterintuitive tbh
  
  r_lower ~ std_normal();
  r_upper ~ std_normal();
  
  to_vector(delta_gains) ~ normal(0, sd_beta);
  to_vector(delta_lpa) ~ normal(0, sd_beta);
  to_vector(delta_losses) ~ normal(0, sd_beta);
  
  sigma_gains ~ normal( sd_beta, sd_beta);
  sigma_lpa ~ normal( sd_beta, sd_beta);
  sigma_losses ~ normal( sd_beta, sd_beta);
  
  // Following TD
  pe_lpa ~ normal( 0, 0.01);
  
  L_gains ~ lkj_corr_cholesky(16);
  L_lpa ~ lkj_corr_cholesky(16);
  L_losses ~ lkj_corr_cholesky(16);
  
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
    // Asymptote is 1 so a=1 in the sigmoid
    // or maybe logistic, not Gompertz
    for (n in 1:N) {
      y[n] ~ categorical_logit( append_row( xbeta[n], 0));
    }
  }
}

generated quantities {
  vector[N] y_sim;
  
  // corr_matrix[K] Omega_gains = L_gains * L_gains';
  // corr_matrix[K] Omega_lpa = L_lpa * L_lpa';
  // corr_matrix[K] Omega_losses = L_losses * L_losses';
  
  for (n in 1:N) {
    y_sim[n] = categorical_logit_rng( append_row( xbeta[n], 0));
  }
}

