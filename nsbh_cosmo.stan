functions{
    real h2n(real h, vector a) {
        return a[1] + a[2] * h + a[3] * h ^ 2 + a[4] * h ^ 3 + a[5] * h ^ 4;
    }
    real hq2n(real h, real q, vector a) {
        return a[1] + a[2] * h + a[3] * q + a[4] * h ^ 2 + 
               a[5] * q ^ 2 + a[6] * h * q + a[7] * h ^ 3 + 
               a[8] * q ^ 3 + a[9] * q * h ^ 2 + a[10] * h * q ^ 2 + 
               a[11] * h ^ 4 + a[12] * q ^ 4 + a[13] * q * h ^ 3 + 
               a[14] * h * q ^ 3 + a[15] * (h * q) ^ 2;
    }
}
data {
    int<lower=0, upper=1> fixed_n_mrg;      // assume sample size known
    int<lower=1> n_mrg;                     // total number of mergers
    int<lower=1> n_cmp_max;                 // max number of KDE/GMM components in all distance likelihoods
    int<lower=1> n_cmp[n_mrg];              // number of KDE/GMM components in each distance likelihood
    vector[n_mrg] obs_d_weights[n_cmp_max]; // distance likelihood KDE/GMM weights
    vector[n_mrg] obs_d_means[n_cmp_max];   // distance likelihood KDE/GMM means
    vector[n_mrg] obs_d_stds[n_cmp_max];    // distance likelihood KDE/GMM standard deviations
    vector[n_mrg] obs_v_pec;                // measured peculiar velocity
    vector[n_mrg] obs_z;                    // measured redshift
    real sig_v_pec;                         // std of true peculiar velocities
    real sig_obs_v_pec;                     // noise on observed peculiar velocities
    real sig_z;                             // noise on observed redshifts
    real z_max;                             // maximum prior redshift
    int<lower=0> n_coeffs;                  // number of coefficients of polynomial fit to \bar{N}(H_0,q_0)
    vector[n_coeffs] n_bar_det_coeffs;      // coefficients of polynomial fit to \bar{N}(H_0,q_0)
}
transformed data {
    real c;                                 // c in km/s
    real g;                                 // g in Mpc / M_sol / (km/s)^2
    c = 2.99792458e5;
    g = 4.301e-9;
}
parameters {
    real<lower=50.0, upper=90.0> h_0;
    real<lower=-2.0, upper=1.0> q_0;
    vector<lower=0.0, upper=z_max>[n_mrg] true_z_cos;
    vector[n_mrg] true_v_pec;
}
transformed parameters {

    vector[n_mrg] true_z;
    vector<lower=0.0>[n_mrg] true_d;
    real<lower=0.0> n_bar_det;

    // calculate total redshift and true distance
    for(i in 1:n_mrg) {
        
        true_z[i] = true_z_cos[i] + (1.0 + true_z_cos[i]) * 
                    true_v_pec[i] / c;
        true_d[i] = c * true_z_cos[i] / h_0 * 
                    (1.0 + 
                     (1.0 - q_0) * true_z_cos[i] / 2.0 + 
                     (-2.0 + q_0 + 3.0 * square(q_0)) * 
                     square(true_z_cos[i]) / 6.0);
        if (true_d[i] < 0) {
            print("D BAD! ", true_d[i], " ", true_z_cos[i], " ", 
                  h_0, " ", q_0);
        }

    }

    // calculate expected number of detections
    n_bar_det = hq2n(h_0, q_0, n_bar_det_coeffs);
    if (n_bar_det < 0) {
        print("N BAD! ", n_bar_det, " ", h_0, " ", q_0);
    }
    if (is_nan(n_bar_det)) {
        print("N BAD! ", n_bar_det, " ", h_0, " ", q_0);
    }
    
}
model {

    // priors on true parameters
    h_0 ~ normal(70.0, 20.0);
    q_0 ~ normal(-0.5, 0.5);
    true_v_pec ~ normal(0.0, sig_v_pec);

    // pick order-appropriate volume element. NB: as addition acts 
    // per vector element, the statement below (correctly) applies a 
    // factor of 1/H_0^3 per object
    target += -3.0 * log(h_0) + 2.0 * log(true_z_cos) + 
              log(1.0 - 2.0 * (1.0 + q_0) * true_z_cos + 
                  5.0 / 12.0 * (7.0 + 14.0 * q_0 - 2.0 + 9.0 * square(q_0)) * 
                  square(true_z_cos));
    target += -log(1.0 + true_z_cos);

    // Poisson exponent
    if (fixed_n_mrg) {
        target += -n_mrg * log(n_bar_det);
    } else {
        target += -n_bar_det;
    }

    // EM likelihoods
    obs_v_pec ~ normal(true_v_pec, sig_obs_v_pec);
    obs_z ~ normal(true_z, sig_z);

    // GW likelihoods. loop over mergers.
    for (i in 1:n_mrg) {

        // now loop over KDE components to build up likelihood
        vector[n_cmp[i]] log_post;
        for (j in 1:n_cmp[i]) {
            log_post[j] = log(obs_d_weights[j][i]) + 
                          normal_lpdf(true_d[i] | obs_d_means[j][i], 
                                                  obs_d_stds[j][i]);
        }
        target += log_sum_exp(log_post);

    }

}

