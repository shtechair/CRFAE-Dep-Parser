package edu.shanghaitech.nlp.algorithm.paskin;

import edu.shanghaitech.nlp.crfae.parser.util.Calc;

import java.util.HashMap;
import java.util.Map;

public class PaskinAddVer {
    private static final boolean T = true, F = false;
    private static final boolean[] BOOLS = {T, F};
    private static final double NEG_INF = Double.NEGATIVE_INFINITY;

    /*
    * We name the log(weight) as score.
    **/

    private int n;
    private double[][] W;
    private Map<Integer, Integer> parse;

    private Map<Span, Double> scores = new HashMap<>();
    private Map<Span, Operation> fromOp = new HashMap<>();

    private Map<Span, Double> insideScores = new HashMap<>();
    private Map<Span, Double> outsideScores = new HashMap<>();

    public PaskinAddVer(double[][] W) {
        /*
         *  W: W[i][j] is the log(weight) of edge (i, j).
         */
        this.n = W.length;
        this.W = W;
    }

    public Map<Integer, Integer> chartParsing() {
        for (int i = 1; i <= n - 1; i++) {
            add(Operation.newSeed(i));
            add(Operation.newCloseLeft(new Span(i, i + 1, F, F, T)));
            if (i > 1) {
                add(Operation.newCloseRight(new Span(i, i + 1, F, F, T)));
            }
        }


        for (int len = 2; len <= n - 1; len++) {
            for (int i = 1; i <= n - len; i++) {
                int j = i + len;
                for (int k = i + 1; k <= j - 1; k++) {
                    ////
                    for (boolean bL : BOOLS) {
                        for (boolean b : BOOLS) {
                            for (boolean bR : BOOLS) {
                                for (boolean s : BOOLS) {
                                    Span sig1 = new Span(i, k, bL, b, T);
                                    Span sig2 = new Span(k, j, !b, bR, s);
                                    if (isJoinDefined(sig1, sig2)) {
                                        add(Operation.newJoin(sig1, sig2));
                                    }
                                }
                            }
                        }
                    }
                    ////
                }// for of `k`
                add(Operation.newCloseLeft(new Span(i, j, F, F, F)));
                if (i > 1) add(Operation.newCloseRight(new Span(i, j, F, F, F)));
            } // for of `i`
        }

        return extractBestParse();
    }

    private Map<Integer, Integer> extractBestParse() {
        final Span[] finalSigs = {
                new Span(1, n, F, T, F),
                new Span(1, n, F, T, T)
        };

        Span bestSig = null;
        double bestScore = NEG_INF;
        for (Span finalSig : finalSigs) {
            double score = scores.getOrDefault(finalSig, NEG_INF);
            if (score > bestScore) {
                bestSig = finalSig;
                bestScore = score;
            }
        }

        assert (bestSig != null); // The sentence can't be parsed.
        return extractBestParse(bestSig);
    }

    private Map<Integer, Integer> extractBestParse(Span sig) {
        Map<Integer, Integer> result = new HashMap<>();

        Operation op = fromOp.get(sig);
        assert (op != null);

        if (op.type == Operation.Type.SEED) {
            return result;
        } else if (op.type == Operation.Type.CLOSE_LEFT) {
            int parent = op.sig1.leftIndex;
            int child = op.sig1.rightIndex;

            result.put(child - 1, parent - 1);
            Map<Integer, Integer> subResult = extractBestParse(op.sig1);
            result.putAll(subResult);
        } else if (op.type == Operation.Type.CLOSE_RIGHT) {
            int parent = op.sig1.rightIndex;
            int child = op.sig1.leftIndex;

            result.put(child - 1, parent - 1);
            Map<Integer, Integer> subResult = extractBestParse(op.sig1);
            result.putAll(subResult);
        } else if (op.type == Operation.Type.JOIN) {
            Map<Integer, Integer> subResult1 = extractBestParse(op.sig1);
            Map<Integer, Integer> subResult2 = extractBestParse(op.sig2);
            result.putAll(subResult1);
            result.putAll(subResult2);
        }

        return result;
    }


    private boolean isJoinDefined(Span sig1, Span sig2) {
        int i = sig1.leftIndex;
        int k = sig1.rightIndex;
        int j = sig2.rightIndex;

        if (i == 1) {
            return (((j == n) && (!sig1.bL && sig1.bR && sig1.isSimple) && (!sig2.bL && sig2.bR)) ||
                    ((k == 2) && (!sig1.bL && !sig1.bR && sig1.isSimple) && (sig2.bL && !sig2.bR))) &&
                    sig1.isValid() && sig2.isValid();

        }
        return sig1.isValid() && sig2.isValid();
    }

    private void add(Operation op) {
        Span result = op.result;
        double score = NEG_INF;

        if (op.type == Operation.Type.SEED) {
            score = 0.;

            // Inside
            insideScores.put(result, 0.);
        } else if (op.type == Operation.Type.CLOSE_LEFT || op.type == Operation.Type.CLOSE_RIGHT) {
            Span sig = op.sig1;
            int i = sig.leftIndex, j = sig.rightIndex;
            double arcScore = (op.type == Operation.Type.CLOSE_LEFT ? W[i - 1][j - 1] : W[j - 1][i - 1]);
            score = scores.get(sig) + arcScore;

            // Inside
            double oldInsideScore = insideScores.getOrDefault(result, NEG_INF);
            double newInsideScore = Calc.logSum(oldInsideScore, (insideScores.get(sig) + arcScore));
            insideScores.put(result, newInsideScore);
        } else if (op.type == Operation.Type.JOIN) {
            Span sig1 = op.sig1, sig2 = op.sig2;
            score = scores.get(sig1) + scores.get(sig2);

            // Inside
            double oldInsideScore = insideScores.getOrDefault(result, NEG_INF);
            double newInsideScore = Calc.logSum(
                    oldInsideScore,
                    insideScores.get(sig1) + insideScores.get(sig2)
            );
            insideScores.put(result, newInsideScore);
        }

        if (score >= scores.getOrDefault(result, NEG_INF)) {
            scores.put(result, score);
            fromOp.put(result, op);
        }
    }

    private void insideOutsideAlgorithm() {
        parse = chartParsing();
        outsideScoreCalculate();
    }

    public double partition() {
        insideOutsideAlgorithm();
        double log_sum = Calc.logSum(
                insideScores.getOrDefault(new Span(1, n, F, T, F), NEG_INF),
                insideScores.getOrDefault(new Span(1, n, F, T, T), NEG_INF)
        );

        return Math.exp(log_sum);
    }

    public double[][] expectationCount() {
        insideOutsideAlgorithm();
        double[][] ret = new double[n][n];

        double denominator = Calc.logSum(
                insideScores.getOrDefault(new Span(1, n, F, T, F), NEG_INF),
                insideScores.getOrDefault(new Span(1, n, F, T, T), NEG_INF)
        );

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (i < j) {
                    Span sig = new Span(i, j, F, T, T);
                    double inScore = insideScores.getOrDefault(sig, NEG_INF);
                    double outScore = outsideScores.getOrDefault(sig, NEG_INF);
                    ret[i - 1][j - 1] = Math.exp((inScore + outScore) - denominator);
                } else if (j < i) {
                    Span sig = new Span(j, i, T, F, T);
                    double inScore = insideScores.getOrDefault(sig, NEG_INF);
                    double outScore = outsideScores.getOrDefault(sig, NEG_INF);
                    ret[i - 1][j - 1] = Math.exp((inScore + outScore) - denominator);
                }
            }
        }
        return ret;
    }

    private void outsideScoreCalculate() {
        // Base case
        outsideScores.put(new Span(1, n, F, T, F), 0.);
        outsideScores.put(new Span(1, n, F, T, T), 0.);

        for (int len = n - 1; len >= 1; len--) {
            for (int i = 1; i <= n - len; i++) {
                int j = i + len;
                boolean s = (len == 1);

                double oldOutsideScore, newOutsideScore, arcScore;

                Span sig = new Span(i, j, F, F, s);
                oldOutsideScore = outsideScores.getOrDefault(sig, NEG_INF);
                arcScore = W[i - 1][j - 1];
                newOutsideScore = Calc.logSum(
                        oldOutsideScore,
                        outsideScores.getOrDefault(new Span(i, j, F, T, T), NEG_INF) + arcScore
                );
                outsideScores.put(sig, newOutsideScore);

                oldOutsideScore = outsideScores.getOrDefault(sig, NEG_INF);
                arcScore = W[j - 1][i - 1];
                newOutsideScore = Calc.logSum(
                        oldOutsideScore,
                        outsideScores.getOrDefault(new Span(i, j, T, F, T), NEG_INF) + arcScore
                );
                outsideScores.put(sig, newOutsideScore);

                if (len > 1) {
                    for (boolean bL : BOOLS) {
                        for (boolean bR : BOOLS) {
                            sig = new Span(i, j, bL, bR, F);
                            for (int k = i + 1; k <= j - 1; k++) {
                                for (boolean b : BOOLS) {
                                    for (boolean sR : BOOLS) {
                                        Span sig1 = new Span(i, k, bL, b, T);
                                        Span sig2 = new Span(k, j, !b, bR, sR);
                                        if (Operation.newJoin(sig1, sig2).result.equals(sig) && isJoinDefined(sig1, sig2)) {
                                            oldOutsideScore = outsideScores.getOrDefault(sig1, NEG_INF);
                                            newOutsideScore = Calc.logSum(
                                                    oldOutsideScore,
                                                    insideScores.get(sig2) + outsideScores.getOrDefault(sig, NEG_INF)
                                            );
                                            outsideScores.put(sig1, newOutsideScore);

                                            oldOutsideScore = outsideScores.getOrDefault(sig2, NEG_INF);
                                            newOutsideScore = Calc.logSum(
                                                    oldOutsideScore,
                                                    insideScores.get(sig1) + outsideScores.getOrDefault(sig, NEG_INF)
                                            );
                                            outsideScores.put(sig2, newOutsideScore);
                                        }
                                    }
                                }
                            } // for `k`
                        }
                    } // for `bR`
                } // for `bL`
            }
        }// for `i`
    } // for `len`
}

