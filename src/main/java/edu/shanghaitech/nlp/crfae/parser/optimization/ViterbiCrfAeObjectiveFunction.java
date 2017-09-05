package edu.shanghaitech.nlp.crfae.parser.optimization;

import edu.shanghaitech.nlp.crfae.parser.*;
import edu.shanghaitech.nlp.crfae.parser.util.Calc;
import edu.shanghaitech.nlp.crfae.parser.Parameters.Distribution;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class ViterbiCrfAeObjectiveFunction implements ObjectiveFunction {
    private DepPipe pipe;

    public ViterbiCrfAeObjectiveFunction(DepPipe pipe) {
        this.pipe = pipe;
    }

    public Map<Integer, Double> calcGradient(DepInstance[] il, Parameters param) {
        Map<Integer, Double> gradient = new HashMap<>();

        for (DepInstance inst : il) {
            FeatureVector[][] fv_map = pipe.getFeatureMatrix(inst);

            double[][] crf_u = param.expectationCount(Distribution.CRF, inst);

            DepParser parser = new DepParser(param);
            Map<Integer, Integer> bestParse = parser.singleRootParse(inst);

            int n = inst.length;
            for (int i = 0; i < n; i++) {
                for (int j = 1; j < n; j++) {
                    if (i != j) {
                        FeatureVector fv = fv_map[i][j];
                        boolean isEdgeInBestParse = (bestParse.get(j) == i);

                        for (FeatureVector curr = fv; curr.index >= 0; curr = curr.next) {
                            double val = gradient.getOrDefault(curr.index, 0.)
                                    - ((isEdgeInBestParse ? 1 : 0) - crf_u[i][j]) * curr.value;
                            gradient.put(curr.index, val);
                        }
                    }
                }
            }
        }
        return gradient;
    }

    @Override
    public double[] gradAt(DepInstance[] il, Parameters param) {
        Map<Integer, Double> gradient = calcGradient(il, param);
        double[] ret = new double[param.crfParam.length];
        for (Integer x : gradient.keySet()) {
            ret[x] = gradient.get(x);
        }
        return ret;
    }

    private double valueAt(DepInstance inst, Parameters param) {
        double crf_Z = param.partition(Distribution.CRF, inst);

        DepParser parser = new DepParser(param);
        Map<Integer, Integer> bestParse = parser.singleRootParse(inst);

        double treeScore = 0.;

        for (Integer c : bestParse.keySet()) {
            Integer p = bestParse.get(c);
            treeScore += param.jointScore(inst, p, c);
        }


        return -(treeScore - Math.log(crf_Z));
    }

    @Override
    public double valueAt(DepInstance[] il, Parameters param) {
        double result = 0.0;

        for (DepInstance inst : il) {
            result += valueAt(inst, param);
        }

        return result;
    }

    private double valueAtGivenParseTree(
            DepInstance inst, Parameters param,
            Map<Integer, Integer> parse
    ) {
        double crf_Z = param.partition(Distribution.CRF, inst);
        double treeScore = 0.;

        for (Integer c : parse.keySet()) {
            Integer p = parse.get(c);
            treeScore += param.jointScore(inst, p, c);
        }

        return -(treeScore - Math.log(crf_Z));
    }

    private double valueAtGivenParseTree(
            DepInstance[] il, Parameters param,
            ArrayList<Map<Integer, Integer>> parses
    ) {
        double result = 0.0;

        for (int i = 0; i < il.length; i++) {
            DepInstance inst = il[i];
            result += valueAtGivenParseTree(inst, param, parses.get(i));
        }

        return result;
    }

    @Override
    public void gradientCheck(DepInstance[] il, Parameters param) {
        ArrayList<Map<Integer, Integer>> parses = new ArrayList<>();
        DepParser parser = new DepParser(param);
        for (DepInstance inst : il) {
            parses.add(parser.singleRootParse(inst));
        }

        Parameters paramCopy = param.copy();
        int N = Math.min(500, param.crfParam.length);

        Integer[] idxs = G.randomIndexArray(N, param.crfParam.length);

        double[] numeric_gradient = new double[param.crfParam.length];
        double eps = 1e-6;
        for (Integer i : idxs) {
            paramCopy.crfParam[i] += eps;
            double x1 = this.valueAtGivenParseTree(il, paramCopy, parses);
            paramCopy.crfParam[i] += -2 * eps;
            double x2 = this.valueAtGivenParseTree(il, paramCopy, parses);
            numeric_gradient[i] = (x1 - x2) / (2 * eps);
            paramCopy.crfParam[i] += eps;
        }

        Map<Integer, Double> gradient = this.calcGradient(il, param);

        System.out.println();
        for (Integer i : idxs) {
            double num_grad = numeric_gradient[i];
            double analy_grad = gradient.getOrDefault(i, 0.);
            double rel_err = Calc.relDiff(num_grad, analy_grad);
            System.out.format("%d\t: (numeric) %.10f == (analyze) %.10f. rel err: %f\n",
                    i, num_grad, analy_grad, rel_err);
        }

    }
}
