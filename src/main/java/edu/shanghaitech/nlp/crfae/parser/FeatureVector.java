package edu.shanghaitech.nlp.crfae.parser;

public class FeatureVector {
    public int index; // feature's index in FeatAlphabet
    public double value;
    public FeatureVector next;
    public int length;

    public FeatureVector(int i, double v, FeatureVector n) {
        index = i;
        value = v;
        next = n;
        length = n == null ? 0 : n.length + 1;
    }

    public String toString() {
        if (next == null)
            return "" + index;
        return index + " " + next.toString();
    }
}
