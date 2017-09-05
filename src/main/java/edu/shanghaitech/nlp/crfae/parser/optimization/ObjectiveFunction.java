package edu.shanghaitech.nlp.crfae.parser.optimization;

import edu.shanghaitech.nlp.crfae.parser.DepInstance;
import edu.shanghaitech.nlp.crfae.parser.Parameters;

/**
 * The interface Objective function.
 */
public interface ObjectiveFunction {

    /**
     * Grad at double [ ].
     *
     * @param il    the il
     * @param param the param
     * @return the double [ ]
     */
    double[] gradAt(DepInstance[] il, Parameters param);

    /**
     * Value at double.
     *
     * @param il    the il
     * @param param the param
     * @return the double
     */
    double valueAt(DepInstance[] il, Parameters param);

    void gradientCheck(DepInstance[] il, Parameters param);
}
