package edu.shanghaitech.nlp.crfae.parser.ds;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class SparseEdgeExp implements EdgeExp {
    private Set<Edge> edges = new HashSet<>();
    private Map<Edge, Double> exp = new HashMap<>();

    public SparseEdgeExp(ParseTree t) {
        for (int i = 1; i <= t.size(); i++) {
            Edge e = Edge.newInstance(t.parent(i), i);
            edges.add(e);
            exp.put(e, 1.);
        }
    }

    @Override
    public Set<Edge> edges() {
        return edges;
    }

    @Override
    public double count(int from, int to) {
        return exp.getOrDefault(Edge.newInstance(from, to), 0.);
    }
}
