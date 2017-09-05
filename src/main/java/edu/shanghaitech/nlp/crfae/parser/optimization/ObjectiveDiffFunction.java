package edu.shanghaitech.nlp.crfae.parser.optimization;

import edu.shanghaitech.nlp.crfae.parser.DepInstance;
import edu.shanghaitech.nlp.crfae.parser.Parameters;
import edu.stanford.nlp.optimization.AbstractStochasticCachingDiffUpdateFunction;

public class ObjectiveDiffFunction extends AbstractStochasticCachingDiffUpdateFunction {
    private ObjectiveFunction function;
    private Parameters param;
    private DepInstance[] il;

    public ObjectiveDiffFunction(ObjectiveFunction function, DepInstance[] il, Parameters param) {
        this.function = function;
        this.il = il;
        this.param = param;
    }

    private Parameters constructParameter(double[] x) {
        Parameters param = this.param.copy();
        param.crfParam = x;
        return param;
    }

    private Parameters constructParameter(double[] x, double xScale) {
        Parameters param = this.param.copy();
        param.crfParam = x;
        for (int i = 0; i < param.crfParam.length; i++) {
            param.crfParam[i] *= xScale;
        }
        return param;
    }

    private DepInstance[] getBatch(int[] batch) {
        int n = batch.length;
        DepInstance il[] = new DepInstance[n];
        for (int i = 0; i < n; i++) {
            il[i] = this.il[batch[i]];
        }
        return il;
    }

    @Override
    public double valueAt(double[] x, double xScale, int[] batch) {
        Parameters param = constructParameter(x, xScale);
        DepInstance[] il = getBatch(batch);
        return function.valueAt(il, param);
    }

    @Override
    public double calculateStochasticUpdate(double[] x, double xScale, int[] batch, double gain) {
        Parameters param = constructParameter(x, xScale);
        DepInstance[] il = getBatch(batch);
        // ignore gain
        return function.valueAt(il, param);
    }

    @Override
    public void calculateStochasticGradient(double[] x, int[] batch) {
        Parameters param = constructParameter(x);
        DepInstance[] il = getBatch(batch);
        derivative = function.gradAt(il, param);
    }

    @Override
    public void calculateStochastic(double[] x, double[] v, int[] batch) {
        Parameters param = constructParameter(x);
        DepInstance[] il = getBatch(batch);
        value = function.valueAt(il, param);
        derivative = function.gradAt(il, param);
        // ignore HdotV
        HdotV = null;
    }

    @Override
    public int dataDimension() {
        return il.length;
    }

    @Override
    protected void calculate(double[] x) {
        Parameters param = constructParameter(x);
        value = function.valueAt(il, param);
        derivative = function.gradAt(il, param);
    }

    @Override
    public int domainDimension() {
        // Use GD to optimization our CRF parameter.
        return param.crfParam.length;
    }
}
