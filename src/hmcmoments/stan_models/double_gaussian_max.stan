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
    vector<lower=a_lower, upper=a_upper>[2] a;
    positive_ordered[2] b;
    vector<lower=c_lower, upper=c_upper>[2] c;
}
model {
    b ~ uniform(b_lower, b_upper);
    y ~ normal(
        fmax(a[1] * exp(-0.5 * square((x - b[1]) / c[1])), a[2] * exp(-0.5 * square((x - b[2]) / c[2]))),
        u_y
    );
}