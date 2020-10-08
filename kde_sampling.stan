data {
    int<lower=0> n_comp;    // number of samples
    vector[n_comp] locs;    // locations
    vector[n_comp] weights; // weights
    real bw;                // common KDE bandwidth
}
parameters {
    real<lower=0.0> d;
}
model {
    vector[n_comp] lps;
    for (i in 1:n_comp)
    	lps[i] = log(weights[i]) + normal_lpdf(d | locs[i], bw);
    target += log_sum_exp(lps);
}