
// Infer the distance and total space motion for a star assuming an
// exponentially decreasing space density prior

data {  
  real parallax; // [arcseconds]
  real pmra; // [arcseconds/yr]
  real pmdec; // [arcseconds/yr]

  real<lower=0> parallax_error; // [arcseconds]
  real<lower=0> pmra_error; // [arcseconds/yr]
  real<lower=0> pmdec_error; // [arcseconds/yr]

  // Correlation coefficients
  real<lower=-1, upper=+1> parallax_pmra_corr;
  real<lower=-1, upper=+1> parallax_pmdec_corr;
  real<lower=-1, upper=+1> pmra_pmdec_corr;

  // Radial velocity
  real vrad; // [km/s]
  real<lower=0> vrad_error; // [km/s]

  // Length scale for the exponentially decreasing space density prior [pc]
  real<lower=0> L;
}

transformed data {
  real total_parallax_error; // [arcseconds]
  vector[3] y;
  matrix[3, 3] Sigma;

  // Asserting a 0.3 mas systematic uncertainty to add in quadrature
  total_parallax_error = sqrt(pow(parallax_error, 2) + pow(0.3/1000.0, 2));

  y[1] = parallax;
  y[2] = pmra;
  y[3] = pmdec;

  Sigma[1, 1] = pow(total_parallax_error, 2);
  Sigma[2, 2] = pow(pmra_error, 2);
  Sigma[3, 3] = pow(pmdec_error, 2);

  Sigma[1, 2] = parallax_pmra_corr * total_parallax_error * pmra_error;
  Sigma[2, 1] = parallax_pmra_corr * total_parallax_error * pmra_error;

  Sigma[1, 3] = parallax_pmdec_corr * total_parallax_error * pmdec_error;
  Sigma[3, 1] = parallax_pmdec_corr * total_parallax_error * pmdec_error;

  Sigma[2, 3] = pmra_pmdec_corr * pmra_error * pmdec_error;
  Sigma[3, 2] = pmra_pmdec_corr * pmra_error * pmdec_error;
}

parameters {
  // Radial velocity
  real true_vrad; // [km/s]

  // Proper motions
  real true_pmra; // [arcseconds/yr]
  real true_pmdec; // [arcseconds/yr]

  // Distance [pc]
  real<lower=0> d;
}

model {
  // Assume an exponentially decreasing space density prior on distance
  // P(d) = 1.0/(2L^3) * d^2 * exp(-d/L)
  // log(P(d)) = L_const + 2 * log(d) - d/L
  // Note: We can drop the constant L_const term such that:
  target += 2 * log(d) - d/L;

  // Model the radial velocity distribution (independent of Gaia observables)
  vrad ~ normal(true_vrad, vrad_error);

  // Model the distance and proper motions
  {
    vector[3] true_y;
    true_y[1] = 1.0/d;
    true_y[2] = true_pmra;
    true_y[3] = true_pmdec;

    y ~ multi_normal(true_y, Sigma);    
  }
}

generated quantities {
  real speed;
  speed = sqrt(
      pow(true_vrad, 2) +
      pow(4.74 * d * sqrt(pow(true_pmra, 2) + pow(true_pmdec, 2)), 2)
  );
}