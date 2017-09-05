package edu.shanghaitech.nlp.crfae.parser.optimization;

import edu.shanghaitech.nlp.crfae.parser.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class SupervisedViterbiCrfAeObjectiveFunction implements ObjectiveFunction {
    private DepPipe pipe;

    public SupervisedViterbiCrfAeObjectiveFunction(DepPipe pipe) {
        this.pipe = pipe;
    }

    public Map<Integer, Double> calcGradient(DepInstance[] il, Parameters param) {
        Map<Integer, Double> gradient = new HashMap<>();

        for (DepInstance inst : il) {
            FeatureVector[][] fv_map = pipe.getFeatureMatrix(inst);

            double[][] crf_u = param.expectationCount(Parameters.Distribution.CRF, inst);

            int n = inst.length;
            for (int i = 0; i < n; i++) {
                for (int j = 1; j < n; j++) {
                    if (i != j) {
                        FeatureVector fv = fv_map[i][j];
                        boolean isEdgeInBestParse = (inst.deps[j] == i);

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
        double crf_Z = param.partition(Parameters.Distribution.CRF, inst);

        double treeScore = 0.;

        for (int i = 1; i < inst.deps.length; i++) {
            treeScore += param.jointScore(inst, inst.deps[i], i);
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

    @Override
    public void gradientCheck(DepInstance[] il, Parameters param) {
    }

}
