package edu.shanghaitech.nlp.crfae.parser.optimization;

import edu.shanghaitech.nlp.crfae.parser.*;
import edu.shanghaitech.nlp.crfae.parser.Parameters.Distribution;

import java.util.HashMap;
import java.util.Map;

public class CrfAeObjectiveFunction implements ObjectiveFunction {
    private DepPipe pipe;

    public CrfAeObjectiveFunction(DepPipe pipe) {
        this.pipe = pipe;
    }

    public Map<Integer, Double> calcGradient(DepInstance[] il, Parameters param) {
        Map<Integer, Double> gradient = new HashMap<>();

        for (int k = 0; k < il.length; k++) {
            DepInstance inst = il[k];
            FeatureVector[][] fv_map = pipe.getFeatureMatrix(inst);

            double[][] joint_prior_u = param.expectationCount(Distribution.JOINT_PRIOR, inst);
            double[][] crf_u = param.expectationCount(Distribution.CRF, inst);

            int n = inst.length;
            for (int i = 0; i < n; i++) {
                for (int j = 1; j < n; j++) {
                    if (i != j) {
                        FeatureVector fv = fv_map[i][j];

                        for (FeatureVector curr = fv; curr.index >= 0; curr = curr.next) {
                            double val = (gradient.getOrDefault(curr.index, 0.) -
                                    (joint_prior_u[i][j] - crf_u[i][j]) * curr.value);
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
        double joint_prior_Z = param.partition(Distribution.JOINT_PRIOR, inst);
        double crf_Z = param.partition(Distribution.CRF, inst);

        return -(Math.log(joint_prior_Z) - Math.log(crf_Z));
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
        Parameters paramCopy = param.copy();
        int N = 20;

        double[] numeric_gradient = new double[param.crfParam.length];
        double eps = 1e-5;
        for (int i = 0; i < N; i++) {
            paramCopy.crfParam[i] += eps;
            double x1 = this.valueAt(il, paramCopy);
            paramCopy.crfParam[i] += -2 * eps;
            double x2 = this.valueAt(il, paramCopy);
            numeric_gradient[i] = (x1 - x2) / (2 * eps);
            paramCopy.crfParam[i] += eps;
        }

        Map<Integer, Double> gradient = this.calcGradient(il, param);

        System.out.println();
        for (int i = 0; i < N; i++) {
            System.out.format("(numeric) %f == (alg) %f\n", numeric_gradient[i], gradient.getOrDefault(i, 0.));
        }

    }
}