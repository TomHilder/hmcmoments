data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
    vector[N] u_y;
    real a_lower;
    real a_upper;
    real b_lower;
    real b_upper;
    real c_lower;
    real c_upper;
}
parameters {
    real<lower=a_lower, upper=a_upper> a;
    real<lower=b_lower, upper=b_upper> b;
    real<lower=c_lower, upper=c_upper> c;
}
model {
    y ~ normal(
        a * exp(-0.5 * square((x - b) / c)),
        u_y
    );
}