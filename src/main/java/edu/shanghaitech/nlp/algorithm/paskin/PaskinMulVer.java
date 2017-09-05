package edu.shanghaitech.nlp.algorithm.paskin;


import java.util.HashMap;
import java.util.Map;

public class PaskinMulVer {
    private static final boolean T = true, F = false;
    private static final boolean[] BOOLS = {T, F};

    private int n;
    private double[][] W;
    private Map<Integer, Integer> parse;

    private Map<Span, Double> props = new HashMap<>();
    private Map<Span, Operation> fromOp = new HashMap<>();

    private Map<Span, Double> insideProps = new HashMap<>();
    private Map<Span, Double> outsideProps = new HashMap<>();

    public PaskinMulVer(double[][] W) {
        this.n = W.length;
        this.W = W;
    }


    public Map<Integer, Integer> chartParsing() {
        for (int i = 1; i <= n - 1; i++) {
            add(Operation.newSeed(i));
            add(Operation.newCloseLeft(new Span(i, i + 1, F, F, T)));
            if (i > 1) add(Operation.newCloseRight(new Span(i, i + 1, F, F, T)));
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
        double bestScore = 0.;
        for (Span finalSig : finalSigs) {
            double score = props.getOrDefault(finalSig, 0.);
            if (score > bestScore) {
                bestSig = finalSig;
                bestScore = score;
            }
        }

        assert (bestSig != null); // Cant be parsed.
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
        double prop = 0.;

        if (op.type == Operation.Type.SEED) {
            prop = 1.;
            insideProps.put(result, 1.);
        } else if (op.type == Operation.Type.CLOSE_LEFT || op.type == Operation.Type.CLOSE_RIGHT) {
            Span sig = op.sig1;
            int i = sig.leftIndex, j = sig.rightIndex;
            double arcProp = (op.type == Operation.Type.CLOSE_LEFT ? W[i - 1][j - 1] : W[j - 1][i - 1]);
            prop = props.get(sig) * arcProp;

            double oldInsideProp = insideProps.getOrDefault(result, 0.);
            double newInsideProp = oldInsideProp + insideProps.getOrDefault(sig, 0.) * arcProp;
            insideProps.put(result, newInsideProp);
        } else if (op.type == Operation.Type.JOIN) {
            Span sig1 = op.sig1, sig2 = op.sig2;
            prop = props.get(sig1) * props.get(sig2);

            double oldInsideProp = insideProps.getOrDefault(result, 0.);
            double newInsideProp = oldInsideProp + insideProps.getOrDefault(sig1, 0.) * insideProps.getOrDefault(sig2, 0.);
            insideProps.put(result, newInsideProp);
        }

        if (prop >= props.getOrDefault(result, 0.)) {
            props.put(result, prop);
            fromOp.put(result, op);
        }
    }

    private void insideOutsideAlgorithm() {
        parse = chartParsing();
        outsidePropCalculate();
    }

    public double partition() {
        insideOutsideAlgorithm();
        double ret = 0.;

        ret += insideProps.getOrDefault(new Span(1, n, F, T, F), 0.);
        ret += insideProps.getOrDefault(new Span(1, n, F, T, T), 0.);

        return ret;
    }

    public double[][] expectationCount() {
        insideOutsideAlgorithm();
        double[][] ret = new double[n][n];

        double denominator = (
                insideProps.getOrDefault(new Span(1, n, F, T, F), 0.) +
                        insideProps.getOrDefault(new Span(1, n, F, T, T), 0.)
        ); // partition function
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (i < j) {
                    Span sig = new Span(i, j, F, T, T);
                    double inProp = insideProps.getOrDefault(sig, 0.);
                    double outProp = outsideProps.getOrDefault(sig, 0.);
                    ret[i - 1][j - 1] = (inProp * outProp) / denominator;
                } else if (j < i) {
                    Span sig = new Span(j, i, T, F, T);
                    double inProp = insideProps.getOrDefault(sig, 0.);
                    double outProp = outsideProps.getOrDefault(sig, 0.);
                    ret[i - 1][j - 1] = (inProp * outProp) / denominator;
                }
            }
        }

        return ret;
    }

    private void outsidePropCalculate() {
        // Base case
        outsideProps.put(new Span(1, n, F, T, F), 1.);
        outsideProps.put(new Span(1, n, F, T, T), 1.);

        for (int len = n - 1; len >= 1; len--) {
            for (int i = 1; i <= n - len; i++) {
                int j = i + len;
                boolean s = (len == 1);

                double oldOutsideProp, newOutsideProp;

                Span sig = new Span(i, j, F, F, s);
                oldOutsideProp = outsideProps.getOrDefault(sig, 0.);
                newOutsideProp = oldOutsideProp + outsideProps.getOrDefault(new Span(i, j, F, T, T), 0.) * W[i - 1][j - 1];
                outsideProps.put(sig, newOutsideProp);

                oldOutsideProp = outsideProps.getOrDefault(sig, 0.);
                newOutsideProp = oldOutsideProp + outsideProps.getOrDefault(new Span(i, j, T, F, T), 0.) * W[j - 1][i - 1];
                outsideProps.put(sig, newOutsideProp);

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
                                            oldOutsideProp = outsideProps.getOrDefault(sig1, 0.);
                                            newOutsideProp = oldOutsideProp +
                                                    insideProps.get(sig2) * outsideProps.getOrDefault(sig, 0.);
                                            outsideProps.put(sig1, newOutsideProp);

                                            oldOutsideProp = outsideProps.getOrDefault(sig2, 0.);
                                            newOutsideProp = oldOutsideProp +
                                                    insideProps.get(sig1) * outsideProps.getOrDefault(sig, 0.);
                                            outsideProps.put(sig2, newOutsideProp);
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


