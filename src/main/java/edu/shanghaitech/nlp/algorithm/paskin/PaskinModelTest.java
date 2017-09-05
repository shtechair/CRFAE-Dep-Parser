package edu.shanghaitech.nlp.algorithm.paskin;

import edu.shanghaitech.nlp.crfae.parser.util.Calc;

public class PaskinModelTest {
    private static boolean equal(double[][] a, double[][] b) {
        int n = a.length;

        boolean isEqual = true;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                isEqual &= Calc.equal(a[i][j], b[i][j]);
            }
        }

        return isEqual;
    }


    public static boolean case1() {
        double[][] W = {
                {0, 0.1, 0.9},
                {0, 0, 1},
                {0, 1, 0}
        };

        double[][] log_W = Calc.log(W);

        double[][] expection = {
                {0, 0.1, 0.9},
                {0, 0, 0.1},
                {0, 0.9, 0}
        };

        double[][] addans = new PaskinAddVer(log_W).expectationCount();
        double[][] mulans = new PaskinMulVer(W).expectationCount();
        boolean addRight = equal(addans, expection);
        boolean mulRight = equal(mulans, expection);

        return addRight && mulRight;
    }


    public static boolean case2() {
        double[][] W = {
                {0, 1},
                {0, 0}
        };

        double[][] log_W = Calc.log(W);

        double[][] expection = {
                {0, 1},
                {0, 0},
        };

        double[][] addans = new PaskinAddVer(log_W).expectationCount();
        double[][] mulans = new PaskinMulVer(W).expectationCount();
        boolean addRight = equal(addans, expection);
        boolean mulRight = equal(mulans, expection);

        return addRight && mulRight;
    }

    public static boolean case3() {
        double[][] W = {
                {0, 0.1, 0.3},
                {0, 0, 0.4},
                {0, 0.9, 0}
        };

        double[][] log_W = Calc.log(W);

        double[][] expection = {
                {0, 0.04 / 0.31, 0.27 / 0.31},
                {0, 0, 0.04 / 0.31},
                {0, 0.27 / 0.31, 0}
        };

        double[][] addans = new PaskinAddVer(log_W).expectationCount();
        double[][] mulans = new PaskinMulVer(W).expectationCount();
        boolean addRight = equal(addans, expection);
        boolean mulRight = equal(mulans, expection);

        return addRight && mulRight;

    }

    public static boolean case4() {
        double[][] W = {
                {0, 1, 1, 1},
                {0, 0, 1, 1},
                {0, 1, 0, 1},
                {0, 1, 1, 0}
        };

        double[][] log_W = Calc.log(W);

        double[][] expection = {
                {0, (1 + 1 + 1) / 7., 1 / 7., (1 + 1 + 1) / 7.},
                {0, 0, (1 + 1 + 1) / 7., (1 + 1) / 7.},
                {0, (1 + 1) / 7., 0, (1 + 1) / 7.},
                {0, (1 + 1) / 7., (1 + 1 + 1) / 7., 0}
        };

        double[][] addans = new PaskinAddVer(log_W).expectationCount();
        double[][] mulans = new PaskinMulVer(W).expectationCount();
        boolean addRight = equal(addans, expection);
        boolean mulRight = equal(mulans, expection);

        return addRight && mulRight;
    }


    public static boolean case5() {
        double[][] W = {
                {0, 1, 2, 3},
                {0, 0, 4, 5},
                {0, 6, 0, 7},
                {0, 8, 9, 0}
        };

        double[][] log_W = Calc.log(W);

        double[][] expection = {
                {0, (20 + 28 + 45) / 651., 84 / 651., (216 + 96 + 162) / 651.},
                {0, 0, (20 + 28 + 96) / 651., (20 + 45) / 651.},
                {0, (84 + 162) / 651., 0, (84 + 28) / 651.},
                {0, (216 + 96) / 651., (216 + 45 + 162) / 651., 0}
        };

        double[][] addans = new PaskinAddVer(log_W).expectationCount();
        double[][] mulans = new PaskinMulVer(W).expectationCount();
        boolean addRight = equal(addans, expection);
        boolean mulRight = equal(mulans, expection);

        return addRight && mulRight;
    }

    public static void main(String[] args) {
        assert case1();
        assert case2();
        assert case3();
        assert case4();
        assert case5();

        System.out.println("Passed.");
    }
}
