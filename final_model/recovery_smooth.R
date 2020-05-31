library( rstan)

fitter <- 'D:/mgr/amexpert/final_model/sigmoid_smooth_v3b1_prospect.stan'
seed3 <- 333

d <- readRDS( 'D:/mgr/amexpert/inputs/data_v3b_C_top10.rds')
compiled_model <- stan_model( fitter)

# Necessary amends
d$ps <- t( d$ps[-nrow( d$ps),])
d$k <- 30
d$sd_beta <- 0.2

################### MODEL RECOVERY ########################
options( mc.cores = parallel::detectCores())
d$run_estimation <- 1

fit <- sampling(
  compiled_model,  # Stan program
  data = d,    # named list of data
  chains = 4,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain
  cores = 4,             # number of cores (could use one per chain)
  init = 'random',
  pars = c( 'y_sim', 'pe_lpa',
            'z_beta_gains', 'z_beta_lpa', 'z_beta_losses',
            'xbeta'),
  include = F,
  seed = seed3
)

saveRDS( fit, 'fit.rds')
