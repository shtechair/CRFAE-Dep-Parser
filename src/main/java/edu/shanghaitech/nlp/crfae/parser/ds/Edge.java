package edu.shanghaitech.nlp.crfae.parser.ds;

public class Edge {
    public int from;
    public int to;

    public Edge(int from, int to) {
        this.from = from;
        this.to = to;
    }

    public static Edge newInstance(int from, int to) {
        return new Edge(from, to);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Edge)) return false;

        Edge edge = (Edge) o;

        if (from != edge.from) return false;
        return to == edge.to;
    }

    @Override
    public int hashCode() {
        int result = from;
        result = 31 * result + to;
        return result;
    }

    @Override
    public String toString() {
        return "Edge{" + "from=" + from + ", to=" + to + '}';
    }
}
