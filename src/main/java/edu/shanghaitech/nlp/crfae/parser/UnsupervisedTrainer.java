package edu.shanghaitech.nlp.crfae.parser;

import edu.shanghaitech.nlp.crfae.parser.HyperParameter.RegType;
import edu.shanghaitech.nlp.crfae.parser.Parameters.Distribution;
import edu.shanghaitech.nlp.crfae.parser.ds.*;
import edu.shanghaitech.nlp.crfae.parser.optimization.*;
import edu.stanford.nlp.optimization.SGDWithAdaGradAndFOBOS;

import java.util.function.Function;

public class UnsupervisedTrainer {
    public Parameters param;
    public DepPipe pipe;
    public DepInstance[] il;

    public ObjectiveFunction objectiveFunction;

    public UnsupervisedTrainer(DepInstance[] il, DepPipe pipe, Parameters param) {
        this.il = il;
        this.pipe = pipe;
        this.param = param;

        if (HyperParameter.getInstance().trainingType == HyperParameter.TrainingType.HARD) {
            objectiveFunction = new ViterbiCrfAeObjectiveFunction(pipe);
        } else {
            objectiveFunction = new CrfAeObjectiveFunction(pipe);
        }
    }

    private Parameters AdaGrad(DepInstance[] il, Parameters param,
                               double initRate, double lambda, int numPasses, int batchSize) {

        ObjectiveDiffFunction f = new ObjectiveDiffFunction(objectiveFunction, il, param);

        String regType = HyperParameter.getInstance().regType == RegType.L1 ? "lasso" : "gaussian";
        SGDWithAdaGradAndFOBOS<ObjectiveDiffFunction> sgd =
                new SGDWithAdaGradAndFOBOS<>(initRate, lambda, numPasses, batchSize, regType,
                        1.0, false, false, 1e-3, 0.95);
        sgd.shutUp(); // Disable standford.optimization log.

        double functionTolerance = 0; // This variable isn't used in our code.
        param.crfParam = sgd.minimize(f, functionTolerance, param.crfParam); // Update
        return param;
    }

    public static Parameters EM_algorithm(
            DepInstance[] il,
            Function<DepInstance, EdgeExp> exp_func,
            Parameters param
    ) {
        // Initialization
        int parentDim = param.reconsParentAlphabet.size();
        int childDim = param.reconsChildAlphabet.size();
        int distDim = Parameters.Dist.getDistanceDim();
        int dirDim = Parameters.Dir.getDirDim();

        double[][][][] U = new double[parentDim][childDim][distDim][dirDim];

        // Smoothing of U.
        for (int i = 0; i < parentDim; i++) {
            for (int k = 0; k < distDim; k++) {
                for (int m = 0; m < dirDim; m++) {
                    U[i][0][k][m] = 0.;
                }
            }
            for (int j = 1; j < childDim; j++) {
                for (int k = 0; k < distDim; k++) {
                    for (int m = 0; m < dirDim; m++) {
                        U[i][j][k][m] = HyperParameter.getInstance().smoothingPower;
                    }
                }
            }
        }

        // E-step
        for (DepInstance inst : il) {
            EdgeExp edgeExp = exp_func.apply(inst);
            for (Edge e : edgeExp.edges()) {
                int i = e.from, j = e.to;

                String parent = param.reconsParent(inst, i);
                String child = param.reconsChild(inst, j);

                int parentIndex = param.reconsParentAlphabet.lookupIndex(parent);
                int childIndex = param.reconsChildAlphabet.lookupIndex(child);
                int distIndex = Parameters.Dist.getDistanceIndex(Math.abs(i - j));
                int dirIndex = Parameters.Dir.getDirIndex(i, j);
                U[parentIndex][childIndex][distIndex][dirIndex] += edgeExp.count(i, j);
            }
        }

        // M-step
        for (int i = 0; i < parentDim; i++) {
            for (int k = 0; k < distDim; k++) {
                for (int m = 0; m < dirDim; m++) {
                    double sum = 0.0;
                    for (int j = 0; j < childDim; j++) {
                        sum += U[i][j][k][m];
                    }
                    for (int j = 0; j < childDim; j++) {
                        param.reconsParam[i][j][k][m] = U[i][j][k][m] / sum;
                        double val = U[i][j][k][m] / sum;
                        assert !(val < 0) && !Double.isNaN(val);
                    }
                }
            }
        }

        return param;
    }


    private Parameters EM(DepInstance[] il, Parameters param) {
        Function<DepInstance, EdgeExp> fn = (inst) -> {
            return new DenseEdgeExp(param.expectationCount(Distribution.JOINT_PRIOR, inst));
        };
        return EM_algorithm(il, fn, param);
    }

    private Parameters HardEM(DepInstance[] il, Parameters param) {
        DepParser parser = new DepParser(param);
        Function<DepInstance, EdgeExp> fn = (inst) -> {
            int[] parseArr = parser.singleRootParseArray(inst);
            return new SparseEdgeExp(new ParseTree(parseArr));
        };
        return EM_algorithm(il, fn, param);
    }

    private Parameters SupervisedHardEM(DepInstance[] il, Parameters param) {
        Function<DepInstance, EdgeExp> fn = (inst) -> {
            return new SparseEdgeExp(new ParseTree(inst.deps));
        };
        return EM_algorithm(il, fn, param);
    }

    public Parameters iteration() {
        double objBefore;
        double objAfter;

        objBefore = objectiveFunction.valueAt(il, param);

        HyperParameter hyperParam = HyperParameter.getInstance();
        double initRate = hyperParam.initRate;
        double lambda = hyperParam.lambda;
        int batchSize = hyperParam.batchSize;
        int numPasses = hyperParam.gdNumPasses;

        param = AdaGrad(il, param, initRate, lambda, numPasses, batchSize);
        objAfter = objectiveFunction.valueAt(il, param);
        System.out.format("GD: %f -> %f\n", objBefore, objAfter);

        objBefore = objectiveFunction.valueAt(il, param);
        int num_EM_passes = hyperParam.emNumPasses;
        for (int i = 0; i < num_EM_passes; i++) {
            if (HyperParameter.getInstance().trainingType == HyperParameter.TrainingType.HARD) {
                param = HardEM(il, param);
            } else {
                param = EM(il, param);
            }
        }
        objAfter = objectiveFunction.valueAt(il, param);
        System.out.format("EM: %f -> %f\n", objBefore, objAfter);

        return param;
    }

    public Parameters supervised_iteration() {
        double objBefore;
        double objAfter;

        objBefore = objectiveFunction.valueAt(il, param);

        HyperParameter hyperParam = HyperParameter.getInstance();
        double initRate = hyperParam.initRate;
        double lambda = hyperParam.lambda;
        int batchSize = hyperParam.batchSize;
        int numPasses = 2;

        ObjectiveFunction temp = objectiveFunction;
        objectiveFunction = new SupervisedViterbiCrfAeObjectiveFunction(pipe);

        param = AdaGrad(il, param, initRate, lambda, numPasses, batchSize);
        objAfter = objectiveFunction.valueAt(il, param);
        System.out.format("GD: %f -> %f\n", objBefore, objAfter);

        objectiveFunction = temp;

        objBefore = objectiveFunction.valueAt(il, param);
        int num_EM_passes = 2;
        for (int i = 0; i < num_EM_passes; i++) {
            param = SupervisedHardEM(il, param);
        }
        objAfter = objectiveFunction.valueAt(il, param);
        System.out.format("EM: %f -> %f\n", objBefore, objAfter);

        return param;
    }
}


