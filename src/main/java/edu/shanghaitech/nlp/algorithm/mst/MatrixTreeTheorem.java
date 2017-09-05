package edu.shanghaitech.nlp.algorithm.mst;

import org.ejml.simple.SimpleMatrix;
import org.ejml.factory.SingularMatrixException;

import java.util.Arrays;

public class MatrixTreeTheorem {
    /*
    * n:
    *   the number of nodes in dependency graph. #[excluding <root-node>]
    *
    * W:
    *   the weight matrix of dependency graph.
    *   size: (n + 1) * (n + 1) #[including <root-node>]
    *
    * Q:
    *   the Laplacian matrix defined in Tarry Koo(2007).
    *   size: n * n
    *
    * u:
    *   the marginal probability of dependency graph defined in Larry Koo(2007).
    *   size: (n + 1) * (n + 1)
    *
    * Z:
    *   the partition function of dependency graph.
    * */

    private SimpleMatrix Q;  // laplacian matrix
    private double[][] W;

    public double Z;        // partition function
    public double[][] u;    // marginal probability.

    public MatrixTreeTheorem(double[][] weight) {
        W = weight;
        int n = W.length - 1; // #. of nodes except <root-node>
        Q = new SimpleMatrix(n, n);

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                int x = i - 1, y = j - 1;
                if (i == 1) {
                    Q.set(x, y, W[0][j]);
                } else {
                    if (i == j) {
                        double v = 0;
                        for (int k = 1; k <= n; k++) {
                            v += W[k][j];
                        }
                        Q.set(x, y, v);
                    } else {
                        Q.set(x, y, -W[i][j]);
                    }
                }
            }
        } // filling the laplacian matrix.

        Z = Q.determinant();
    }

    public double partition() {
        return Z;
    }

    public double[][] marginal() {
        if (this.u == null) {
            int n = W.length - 1;
            this.u = new double[n + 1][n + 1];

            try {
                SimpleMatrix invQ = Q.invert();

                for (int i = 0; i <= n; i++) {
                    for (int j = 1; j <= n; j++) {
                        if (i == 0) {
                            u[0][j] = W[0][j] * invQ.get(j - 1, 1 - 1);
                        } else {
                            u[i][j] = (1 - kronDelta(1, j)) * W[i][j] * invQ.get(j - 1, j - 1) -
                                    (1 - kronDelta(i, 1)) * W[i][j] * invQ.get(j - 1, i - 1);
                        }

                        // Some problem which may caused by the stability of Matrix Inverse.
                        // In theory, the marginal of every edge can't be negative.
                        // But in our experiment, this situation occurs sometimes.
                        if (u[i][j] < 0.) {
                            u[i][j] = -u[i][j];
                        }
                        assert !Double.isNaN(u[i][j]) && u[i][j] >= 0.;
                    }
                }
            } catch (SingularMatrixException | AssertionError e) {
                e.printStackTrace();
            }
        }

        return u;
    }

    private int kronDelta(int x, int y) {
        return x == y ? 1 : 0;
    }
}

