package edu.shanghaitech.nlp.crfae.parser.util;

public class Calc {

    public static final double EPS = 1e-12;

    public static boolean equal(double a, double b) {
        return Math.abs(a - b) < EPS;
    }

    public static double[][] log(double[][] X) {
        int m = X.length;
        int n = X[0].length;
        double[][] W = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                W[i][j] = Math.log(X[i][j]);
            }
        }

        return W;
    }

    /**
     * Compute log(a + b) via log(a) and log(b).
     *
     * @param log_a: log(a)
     * @param log_b: log(b)
     * @return val: log(a + b) = log(a) + log(1 + exp(log(b) - log(a)))
     */
    public static double logSum(double log_a, double log_b) {
        if (Double.isInfinite(log_a) && Double.isInfinite(log_b)) {
            /*  log(0 + 0) = -Inf + log(1 + exp(Inf - Inf)) = -Inf,
                but "Inf - Inf" in Java produces a NaN,
                so we manually return -Inf to avoid this problem */
            return Double.NEGATIVE_INFINITY;
        }

        // Choose the smaller one as minuend for numerical stability.
        double larger = Math.max(log_a, log_b);
        double smaller = Math.min(log_a, log_b);
        return larger + Math.log(1 + Math.exp(smaller - larger));
    }


    /**
     * @param x
     * @param y
     * @return val: the relative difference of x and y.
     */
    public static double relDiff(double x, double y) {
        return Math.abs(x - y) / (Math.max(1e-8, Math.abs(x) + Math.abs(y)));
    }


}
