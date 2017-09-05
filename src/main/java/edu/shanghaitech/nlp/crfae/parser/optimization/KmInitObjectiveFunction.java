package edu.shanghaitech.nlp.crfae.parser.optimization;

import edu.shanghaitech.nlp.crfae.parser.DepInstance;
import edu.shanghaitech.nlp.crfae.parser.DepPipe;
import edu.shanghaitech.nlp.crfae.parser.FeatureVector;
import edu.shanghaitech.nlp.crfae.parser.Parameters;

import java.util.HashMap;
import java.util.Map;

public class KmInitObjectiveFunction implements ObjectiveFunction {
    private DepPipe pipe;

    public KmInitObjectiveFunction(DepPipe pipe) {
        this.pipe = pipe;
    }

    public Map<Integer, Double> calcGradient(DepInstance[] il, Parameters param) {
        Map<Integer, Double> gradient = new HashMap<>();

        for (int k = 0; k < il.length; k++) {
            DepInstance inst = il[k];
            FeatureVector[][] fv_map = pipe.getFeatureMatrix(inst);

            double[][] crf_u = param.expectationCount(Parameters.Distribution.CRF, inst);
            double[][] km_u = param.expectationCount(Parameters.Distribution.KM, inst);

            int n = inst.length;
            for (int i = 0; i < n; i++) {
                for (int j = 1; j < n; j++) {
                    if (i != j) {
                        FeatureVector fv = fv_map[i][j];
                        for (FeatureVector curr = fv; curr.index >= 0; curr = curr.next) {
                            double val = (gradient.getOrDefault(curr.index, 0.) -
                                    (km_u[i][j] - crf_u[i][j]) * curr.value);
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
        // Objective function: -log(P(\hat{x} | x))

        double[][] km_u = param.expectationCount(Parameters.Distribution.KM, inst);
        double[][] jointScore = param.jointScoreGraph(inst);

        double crf_Z = param.partition(Parameters.Distribution.CRF, inst);
        double log_km_Z = 0;

        int n = inst.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                log_km_Z += km_u[i][j] * jointScore[i][j];
            }
        }
        return -(log_km_Z - Math.log(crf_Z));
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
