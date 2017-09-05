package edu.shanghaitech.nlp.crfae.parser.ds;


import java.util.Set;

public interface EdgeExp {
    /**
     * Edge Expectation.
     */

    Set<Edge> edges();

    double count(int from, int to);
}
