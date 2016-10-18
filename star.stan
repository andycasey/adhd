
// Infer the distance and total space motion for a star when the radial velocity
// is not known

data {
  //real alpha; // Right ascension [degrees]
  //real delta; // Declination [degrees]
  
  // y contains: 
  //  parallax angle [mas]; 
  //  proper motion in right ascension [mas/yr];
  //  proper motion in declination [mas/yr]

  vector[3] y;
  matrix[3, 3] Sigma;
  
  // Length scale for the exponentially decreasing space density prior [kpc]
  real<lower=0> L;

  // Solar motion along the line-of-sight for this sky position [km/s]
  real solar_motion;
}

parameters {
  // Radial velocity [km/s]
  real radial_velocity;

  // True values of:
  //  parallax [mas];
  //  proper motion in right ascension [mas/yr];
  //  proper motion in declination [mas/yr]
  vector[3] true_y;

  // Distance [kpc]
  real<lower=0> d;
}

transformed parameters {
  // Total velocity [km/s]
  real<lower=0> total_velocity_sq;

  total_velocity_sq = pow(radial_velocity, 2) 
                    + pow(4.74 * d 
                        * sqrt(pow(true_y[2], 2) + pow(true_y[3], 2)), 2);
}

model {
  // Prior on radial velocity centered on the solar motion for this sky position,
  // with an uncertainty of 100 km/s.
  radial_velocity ~ normal(solar_motion, 100);

  // Include exponentially decreasing space density prior on distance
  target += log(pow(d, 2) / (2.0 * pow(L, 3))) - d/L;

  y ~ multi_normal(true_y, Sigma);

  // Model parallax
  true_y[1] ~ normal(1.0/d, sqrt(Sigma[1, 1]));
}

generated quantities {
  real total_velocity;
  
  real parallax;
  real pmra;
  real pmdec;

  total_velocity = sqrt(total_velocity_sq);
  parallax = true_y[1];
  pmra = true_y[2];
  pmdec = true_y[3];
}