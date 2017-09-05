package edu.shanghaitech.nlp.crfae.parser.ds;

import java.util.HashMap;
import java.util.Map;

public class ParseTree {
    private Map<Integer, Integer> tree = new HashMap<>();
    private int[] parseArray;

    public ParseTree(int[] t) {
        // t: [root: -1] parent_of_the_first_word parent_of_the_second_word ...
        parseArray = t.clone();

        for (int child = 1; child < t.length; child++) {
            tree.put(child, t[child]);
        }
    }

    public int parent(int child) {
        return tree.get(child);
    }

    public int size() {
        // sentence length without root
        return tree.size();
    }

    public int[] getParseArray() {
        return parseArray;
    }

    public boolean[] diff(ParseTree golden) {
        boolean[] ret = new boolean[tree.size()];

        ret[0] = true; // root
        for (int i : golden.tree.keySet()) {
            ret[i] = (golden.parent(i) == this.parent(i));
        }

        return ret;
    }

    @Override
    public String toString() {
        return "ParseTree{" + "tree=" + tree + '}';
    }

}
