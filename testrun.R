data <- readRDS( 'preproc_test.rds')
data$z_beta <- t( data$z_beta)
library( rstan)
fit <- stan(
  file = "terui_dahana.stan",  # Stan program
  data = data,    # named list of data
  chains = 1,             # number of Markov chains
  warmup = 100,          # number of warmup iterations per chain
  iter = 200,            # total number of iterations per chain
  cores = 1,             # number of cores (could use one per chain)
  init = 'random',
  control = list( max_treedepth = 12) # default of 10 was not enough
  #refresh = 0             # no progress shown
)
saveRDS( fit, 'testrun_200.rds')