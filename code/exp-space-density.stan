
// Infer distance from a noisy parallax assuming an exponentially decreasing 
// space density prior with a prescribed length scale

data {
  real parallax; // [arcsec]
  real<lower=0> parallax_error; // [arcsec]

  // Length scale for the exponentially decreasing space density prior [pc]
  real<lower=0> L;
}

transformed data {
  real total_parallax_error; // [arcsec]
  // Asserting a 0.3 mas systematic uncertainty to add in quadrature
  total_parallax_error = sqrt(pow(parallax_error, 2) + pow(0.3/1000.0, 2));
}

parameters {
  real<lower=0> d; // Distance [pc]
}

model {
  // Exponentially decreasing space density prior with a scale length 
  // P(d) = 1.0/(2L^3) * d^2 * exp(-d/L)
  // log(P(d)) = L_const + 2*log(d) - d/L
  // Note: We can drop the constant L_const term such that:
  target += 2 * log(d) - d/L;

  parallax ~ normal(1.0/d, total_parallax_error);
}