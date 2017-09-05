package edu.shanghaitech.nlp.crfae.parser;

import edu.shanghaitech.nlp.algorithm.mst.MatrixTreeTheorem;
import edu.shanghaitech.nlp.algorithm.paskin.PaskinAddVer;
import edu.shanghaitech.nlp.crfae.parser.optimization.KmInitObjectiveFunction;
import edu.shanghaitech.nlp.crfae.parser.optimization.ObjectiveDiffFunction;
import edu.shanghaitech.nlp.crfae.parser.util.Calc;
import edu.shanghaitech.nlp.crfae.parser.util.IO;
import edu.stanford.nlp.optimization.SGDWithAdaGradAndFOBOS;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.*;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

public class Parameters implements Serializable {
    @FunctionalInterface
    private interface Function2<T, U, R> {
        R apply(T t, U u);
    }

    @FunctionalInterface
    private interface Function3<T, U, V, R> {
        R apply(T t, U u, V v);
    }

    public enum Distribution {
        CRF, JOINT, KM, JOINT_PRIOR
    }

    static class Dir {
        final static boolean ENABLE = HyperParameter.getInstance().decoderDir;
        final static int LEFT = 0;
        final static int RIGHT = 1;

        static int getDirIndex(int parent, int child) {
            if (!ENABLE) {
                return LEFT;
            } else {
                return (parent < child) ? RIGHT : LEFT;
            }
        }

        static int getDirDim() {
            if (!ENABLE) {
                return 1; // Disable dir in decoder.
            } else {
                return 2;
            }
        }
    }

    static class Dist {
        final static boolean ENABLE = HyperParameter.getInstance().decoderDist;

        public static int[] getDistanceBins() {
            int[] bins;
            if (ENABLE) {
                bins = new int[]{1, 2, 3, 4, 5, 10};
            } else {
                bins = new int[]{1};
            }
            return bins;
        }

        public static int getDistanceIndex(int dist) {
            if (dist == 0) return 0;
            int[] bins = getDistanceBins();
            int i = 0;
            while (i < bins.length && dist >= bins[i]) {
                i++;
            }
            return i - 1;
        }

        public static int getDistanceDim() {
            return getDistanceBins().length;
        }
    }


    public DepPipe pipe;
    public Alphabet reconsParentAlphabet;
    public Alphabet reconsChildAlphabet;
    public double[] crfParam;      // lambda in Wammar(2014)
    public double[][][][] reconsParam; // theta in Wammar(2014)


    public Parameters(DepPipe pipe) {
        this.pipe = pipe;

        int featSize = pipe.featAlphabet.size();
        crfParam = new double[featSize];

        reconsParentAlphabet = pipe.posAlphabet;
        reconsChildAlphabet = pipe.wordAlphabet;
        int parentDim = reconsParentAlphabet.size();
        int childDim = reconsChildAlphabet.size();
        int distDim = Dist.getDistanceDim();
        int dirDim = Dir.getDirDim();

        reconsParam = new double[parentDim][childDim][distDim][dirDim];
    }

    public void kmInit(DepInstance[] il) {
        if (HyperParameter.getInstance().kmType == HyperParameter.KMType.JOINT) {
            KmInitObjectiveFunction kmObj = new KmInitObjectiveFunction(pipe);
            ObjectiveDiffFunction f = new ObjectiveDiffFunction(kmObj, il, this);

            SGDWithAdaGradAndFOBOS<ObjectiveDiffFunction> sgd =
                    new SGDWithAdaGradAndFOBOS<>(0.1, 1.0, 5, 200, "lasso",
                            1.0, false, false, 1e-3, 0.95);
            sgd.shutUp(); // Disable standford.optimization log.

            double functionTolerance = 0; // This variable isn't used in our code.
            this.crfParam = sgd.minimize(f, functionTolerance, this.crfParam); // Update
        }

        reconsParam = kmInitReconsParam(il);
    }

    private double[][][][] kmInitReconsParam(DepInstance[] il) {
        int parentDim = reconsParentAlphabet.size();
        int childDim = reconsChildAlphabet.size();
        int distDim = Dist.getDistanceDim();
        int dirDim = Dir.getDirDim();

        double[][][][] acc = new double[parentDim][childDim][distDim][dirDim];

        String rootPos = DepPipe.ROOT_POS;
        String rootWord = DepPipe.ROOT_WORD;
        int rootParentIndex = reconsParentAlphabet.lookupIndex(rootPos);
        int rootChildIndex = reconsChildAlphabet.lookupIndex(rootWord) != -1 ?
                reconsChildAlphabet.lookupIndex(rootWord) :
                reconsChildAlphabet.lookupIndex(rootPos);

        // Root -> * , Uniform.
        for (int j = 0; j < childDim; j++) {
            for (int k = 0; k < distDim; k++) {
                int dirIndex = Dir.getDirIndex(0, 1); // Always RIGHT.
                if (j == rootChildIndex) {
                    acc[rootParentIndex][j][k][dirIndex] = 0.;
                } else {
                    acc[rootParentIndex][j][k][dirIndex] = 1. / (distDim * dirDim * (childDim - 1));
                }
            }
        }

        // The rest POS.
        for (DepInstance inst : il) {
            int len = inst.length;
            // Skip 0 means skipping the Root pos.
            for (int i = 1; i < len; i++) {
                for (int j = 1; j < len; j++) {
                    if (i != j) {
                        String parent = reconsParent(inst, i);
                        String child = reconsChild(inst, j);

                        int parentIndex = reconsParentAlphabet.lookupIndex(parent);
                        int childIndex = reconsChildAlphabet.lookupIndex(child);
                        int distance = Math.abs(i - j);
                        int distIndex = Dist.getDistanceIndex(distance);
                        int dirIndex = Dir.getDirIndex(i, j);

                        acc[parentIndex][childIndex][distIndex][dirIndex] += 1. / distance;
                    }
                }
            }
        }

        // Normalization.

        for (int i = 0; i < parentDim; i++) {
            for (int k = 0; k < distDim; k++) {
                for (int m = 0; m < dirDim; m++) {
                    // Sum
                    double sum = 0.0;
                    for (int j = 0; j < childDim; j++) {
                        sum += acc[i][j][k][m];
                    }

                    // Normalize
                    for (int j = 0; j < childDim; j++) {
                        if (j == rootChildIndex) {
                            reconsParam[i][j][k][m] = 0.;
                        } else {
                            reconsParam[i][j][k][m] = (sum == 0. ?
                                    (1. / (distDim * dirDim * (childDim - 1))) :
                                    acc[i][j][k][m] / sum
                            );
                        }
                    }
                }
            }
        }

        return reconsParam;
    }

    public void goodInit(DepInstance[] il) {
        int parentDim = reconsParentAlphabet.size();
        int childDim = reconsChildAlphabet.size();
        int distDim = Dist.getDistanceDim();
        int dirDim = Dir.getDirDim();

        String rootPos = DepPipe.ROOT_POS;
        int rootPosIndex = reconsParentAlphabet.lookupIndex(rootPos);

        double[][][][] acc = new double[parentDim][childDim][distDim][dirDim];

        for (int i = 0; i < parentDim; i++) {
            for (int j = 0; j < childDim; j++) {
                for (int k = 0; k < distDim; k++) {
                    for (int m = 0; m < dirDim; m++) {
                        acc[i][j][k][m] = Calc.EPS;
                    }
                }
            }
        }// Smoothing

        for (DepInstance inst : il) {
            for (int child = 1; child < inst.length; child++) {
                int parent = inst.deps[child];
                int parentIndex = reconsParentAlphabet.lookupIndex(inst.pos[parent]);
                int childIndex = reconsChildAlphabet.lookupIndex(inst.pos[child]);
                int distIndex = Dist.getDistanceIndex(Math.abs(parent - child));
                int dirIndex = Dir.getDirIndex(parent, child);

                acc[parentIndex][childIndex][distIndex][dirIndex] += 1;
            }
        }

        // Normalization.

        for (int i = 0; i < parentDim; i++) {
            for (int k = 0; k < distDim; k++) {
                for (int m = 0; m < dirDim; m++) {
                    double sum = 0.0;
                    for (int j = 0; j < childDim; j++) {
                        sum += acc[i][j][k][m];
                    }
                    for (int j = 0; j < childDim; j++) {
                        if (j == rootPosIndex) {
                            reconsParam[i][j][k][m] = 0.;
                        } else {
                            reconsParam[i][j][k][m] = (sum == 0. ?
                                    1. / (distDim * dirDim * (childDim - 1)) :
                                    acc[i][j][k][m] / sum
                            );
                        }
                    }
                }
            }
        }
    }

    /*
     * The weight of a tree equals to the product of weight of its edges.
     * The score of a tree equals to the sum of score of its edges.
     * Weight = Exp(Score), Score = Log(Weight)
     */


    public double crfScore(DepInstance inst, int i, int j) {
        FeatureVector fv = pipe.featureOf(inst, i, j);
        double score = 0.;
        for (FeatureVector curr = fv; curr.index >= 0; curr = curr.next) {
            score += crfParam[curr.index] * curr.value;
        }
        return score;
    }

    public String reconsParent(DepInstance inst, int index) {
        return inst.pos[index];
    }

    public String reconsChild(DepInstance inst, int index) {
        String ret;
        int threshold = HyperParameter.getInstance().wordThreshold;
        if (pipe.wordCount.getOrDefault(inst.words[index], 0) > threshold) {
            ret = inst.words[index];
        } else {
            ret = inst.pos[index];
        }
        return ret;
    }

    public double reconsScore(DepInstance inst, int i, int j) {
        String parent = reconsParent(inst, i);
        String child = reconsChild(inst, j);

        int parentIndex = reconsParentAlphabet.lookupIndex(parent);
        int childIndex = reconsChildAlphabet.lookupIndex(child);

        int childDim = reconsChildAlphabet.size();

        double ret;
        if (parentIndex == -1) {
            // Parent POS don't occur in training corpus.
            ret = 1. / childDim;
        } else if (childIndex == -1) {
            // Child POS don't occur in training corpus.(But parent do.)
            ret = Calc.EPS;
        } else {
            int distIndex = Dist.getDistanceIndex(Math.abs(i - j));
            int dirIndex = Dir.getDirIndex(i, j);
            ret = reconsParam[parentIndex][childIndex][distIndex][dirIndex];
        }

        assert !(ret < 0.) && !Double.isNaN(ret);
        return Math.log(ret);
    }

    public double jointScore(DepInstance inst, int i, int j) {
        return crfScore(inst, i, j) + reconsScore(inst, i, j);
    }

    private double[][] buildScoreGraph(
            DepInstance inst,
            Function3<DepInstance, Integer, Integer, Double> fn) {
        int n = inst.length;
        double[][] W = new double[n][n];

        for (int i = 0; i < n; i++) {
            W[i][0] = Double.NEGATIVE_INFINITY; // (* -> Root)
            for (int j = 1; j < n; j++) {
                if (i == j) {
                    W[i][j] = Double.NEGATIVE_INFINITY;
                } else {
                    W[i][j] = fn.apply(inst, i, j);
                }
            }
        }
        return W;
    }

    private double[][] scoreGraphToWeightGraph(double[][] Gs) {
        double[][] Gw = new double[Gs.length][];
        for (int i = 0; i < Gs.length; i++) {
            Gw[i] = DoubleStream.of(Gs[i]).map(Math::exp).toArray();
        }
        return Gw;
    }

    public double[][] crfScoreGraph(DepInstance inst) {
        return buildScoreGraph(inst, this::crfScore);
    }

    public double[][] crfWeightGraph(DepInstance inst) {
        return scoreGraphToWeightGraph(crfScoreGraph(inst));
    }

    public double[][] jointScoreGraph(DepInstance inst) {
        return buildScoreGraph(inst, this::jointScore);
    }

    public double[][] jointWeightGraph(DepInstance inst) {
        return scoreGraphToWeightGraph(jointScoreGraph(inst));
    }

    public double[][] reconsScoreGraph(DepInstance inst) {
        return buildScoreGraph(inst, this::reconsScore);
    }

    public double[][] priorScoreGraph(DepInstance inst) {
        return buildScoreGraph(
                inst,
                (instance, i, j) -> {
                    int n = instance.length;
                    double priorWeight = HyperParameter.getInstance().priorWeight;
                    String parent = instance.upos[i];
                    String child = instance.upos[j];
                    return (isInLinguistRuleset(parent, child) ? (priorWeight * 1. / n) : 0.);
                });
    }

    public double[][] kmExpectedCount(DepInstance inst) {
        int n = inst.length;

        double[][] ret = new double[n][n];
        for (int i = 0; i < n; i++) {
            if (i == 0) {
                for (int j = 1; j < n; j++) {
                    ret[i][j] = 1. / (n - 1);
                }
            } else {
                for (int j = 1; j < n; j++) {
                    if (i == j) {
                        ret[i][j] = 0;
                    } else {
                        ret[i][j] = Math.abs(1. / (i - j));
                    }
                }
            }
        }

        double sum = Stream.of(ret).mapToDouble((x -> DoubleStream.of(x).sum())).sum();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                ret[i][j] *= ((n - 1) / sum);
            }
        }
        return ret;
    }


    public double[][] graphAddtion(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] ret = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ret[i][j] = a[i][j] + b[i][j];
            }
        }
        return ret;
    }

    private boolean isInLinguistRuleset(String parent, String child) {
        List<String> rulesParent = new ArrayList<>();
        List<String> rulesChild = new ArrayList<>();

        Function2<String, String, Void> addRules = (h, d) -> {
            rulesParent.add(h);
            rulesChild.add(d);
            return null;
        };

        final String rootPos = DepPipe.ROOT_POS;

        switch (HyperParameter.getInstance().rulesType) {
            case UD: {
                // Universal rules for ud data set.
                // Add this make our best performance get worse.
                // addRules.apply(rootPos, "VERB");
                // addRules.apply(rootPos, "NOUN");

                addRules.apply("VERB", "VERB");
                addRules.apply("VERB", "NOUN");
                addRules.apply("VERB", "PRON");
                addRules.apply("VERB", "ADV");
                addRules.apply("VERB", "ADP");

                addRules.apply("NOUN", "NOUN");
                addRules.apply("NOUN", "ADJ");
                addRules.apply("NOUN", "DET");
                addRules.apply("NOUN", "NUM");
                addRules.apply("NOUN", "CONJ");

                addRules.apply("ADJ", "ADV");
                addRules.apply("ADP", "NOUN");
                break;
            }
            case WSJ: {
                // Manually encode the prior used in "Using Universal Linguistic Knowledge to Guide Grammar Induction"(2010)

                addRules.apply(rootPos, "MD"); // Root -> Auxiliary
                addRules.apply(rootPos, "VB"); //Root -> Verb
                //addRules.apply("<root-POS>", "NN"); //Root -> Noun

                addRules.apply("VB", "NN"); // Verb -> Noun
                addRules.apply("VB", "WP"); // Verb -> Pronoun (1)
                addRules.apply("VB", "PR"); // Verb -> Pronoun (2)
                addRules.apply("VB", "RB"); // Verb -> Adverb
                addRules.apply("VB", "VB"); // Verb -> Verb

                addRules.apply("MD", "VB"); // Auxiliary -> Verb

                addRules.apply("NN", "JJ"); // Noun -> Adjective
                addRules.apply("NN", "NN"); // Noun -> Noun
                addRules.apply("NN", "CD"); // Noun -> Numeral

                addRules.apply("IN", "NN"); //Preposition -> Noun

                addRules.apply("JJ", "RB"); //Adjective -> Adverb
                break;
            }
        }

        for (int i = 0; i < rulesParent.size(); i++) {
            String ruleParent = rulesParent.get(i);
            String ruleChild = rulesChild.get(i);
            if (parent.startsWith(ruleParent) && child.startsWith(ruleChild)) {
                return true;
            }
        }
        return false;
    }

    private double[][] getScoreGraph(Distribution dis, DepInstance inst) {
        double[][] graph = null;
        switch (dis) {
            case CRF: {
                graph = crfScoreGraph(inst);
                break;
            }
            case JOINT: {
                graph = jointScoreGraph(inst);
                break;
            }
            case JOINT_PRIOR: {
                graph = graphAddtion(jointScoreGraph(inst), priorScoreGraph(inst));
                break;
            }
        }
        return graph;
    }

    private double[][] getWeightGraph(Distribution dis, DepInstance inst) {
        return scoreGraphToWeightGraph(getScoreGraph(dis, inst));
    }

    public double[][] expectationCount(Distribution dis, DepInstance inst) {
        HyperParameter hyperParameter = HyperParameter.getInstance();
        double[][] probs = null;
        double[][] ret = null;

        switch (dis) {
            case KM:
                ret = kmExpectedCount(inst);
                break;
            default:
                switch (hyperParameter.modelType) {
                    case NON_PROJ: {
                        probs = getWeightGraph(dis, inst);
                        ret = new MatrixTreeTheorem(probs).marginal();
                        break;
                    }
                    case PROJ: {
                        probs = getScoreGraph(dis, inst);
                        ret = new PaskinAddVer(probs).expectationCount();
                        break;
                    }
                    default: {
                        ret = null;
                        IO.error("Unsupported dependency type.");
                    }
                }
        }
        return ret;
    }

    public double partition(Distribution dis, DepInstance inst) {
        HyperParameter hyperParameter = HyperParameter.getInstance();
        double ret;
        double[][] probs = null;

        switch (hyperParameter.modelType) {
            case NON_PROJ: {
                probs = getWeightGraph(dis, inst);
                ret = new MatrixTreeTheorem(probs).partition();
                break;
            }
            case PROJ: {
                probs = getScoreGraph(dis, inst);
                ret = new PaskinAddVer(probs).partition();
                break;
            }
            default: {
                ret = -1;
                IO.error("Unsupported dependency type.");
            }
        }
        return ret;
    }

    /*
    ********************
    *   For gradient check.
    ********************
    */
    public Parameters copy() {
        Parameters ret = new Parameters(this.pipe);
        ret.crfParam = Arrays.copyOf(this.crfParam, this.crfParam.length);
        ret.reconsParam = new double
                [reconsParam.length]
                [reconsParam[0].length]
                [reconsParam[0][0].length]
                [reconsParam[0][0][0].length];
        for (int i = 0; i < reconsParam.length; i++) {
            ret.reconsParam[i] = Arrays.copyOf(reconsParam[i], reconsParam[i].length);
        }
        return ret;
    }

    /*
    ********************
    *   Serialization
    ********************
    */
    private void writeObject(ObjectOutputStream out) throws IOException {
        out.writeObject(pipe);
        out.writeObject(crfParam);
        out.writeObject(reconsParam);
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        pipe = (DepPipe) in.readObject();
        crfParam = (double[]) in.readObject();
        reconsParam = (double[][][][]) in.readObject();
        reconsParentAlphabet = pipe.posAlphabet;
        reconsChildAlphabet = pipe.wordAlphabet;
    }
}
