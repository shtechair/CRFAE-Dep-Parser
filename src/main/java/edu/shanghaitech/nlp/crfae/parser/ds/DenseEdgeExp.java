package edu.shanghaitech.nlp.crfae.parser.ds;

import java.util.HashSet;
import java.util.Set;

public class DenseEdgeExp implements EdgeExp {
    private Set<Edge> edges = new HashSet<>();
    private double[][] exp;

    public DenseEdgeExp(double[][] exp) {
        this.exp = new double[exp.length][exp[0].length];
        for (int i = 0; i < exp.length; i++) {
            for (int j = 1; j < exp[0].length; j++) {
                if (i != j) {
                    Edge e = Edge.newInstance(i, j);
                    this.edges.add(e);
                    this.exp[i][j] = exp[i][j];
                }
            }
        }
    }

    @Override
    public Set<Edge> edges() {
        return edges;
    }

    @Override
    public double count(int from, int to) {
        return edges.contains(Edge.newInstance(from, to)) ? exp[from][to] : 0.;
    }
}
