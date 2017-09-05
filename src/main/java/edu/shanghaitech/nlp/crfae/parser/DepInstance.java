package edu.shanghaitech.nlp.crfae.parser;

import java.util.Arrays;

public class DepInstance {
    public String[] words;
    public String[] pos;
    public String[] upos; // Universal Pos.
    public int[] deps;
    public int length;

    public DepInstance(String[] words, String[] pos, String[] upos) {
        this.words = words;
        this.pos = pos;
        this.upos = upos;
        this.length = words.length;
    }

    public DepInstance(String[] words, String[] pos, String[] upos, int[] deps) {
        this(words, pos, upos);
        this.deps = deps;
    }

    @Override
    public String toString() {
        return "DepInstance{" +
                "words=" + Arrays.toString(words) +
                ", pos=" + Arrays.toString(pos) +
                ", upos=" + Arrays.toString(upos) +
                ", deps=" + Arrays.toString(deps) +
                ", length=" + length +
                '}';
    }
}
