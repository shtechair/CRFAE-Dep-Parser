package edu.shanghaitech.nlp.algorithm.paskin;

public class Operation {
    public final Span sig1;
    public final Span sig2;
    public final Type type;
    public final Span result;

    enum Type {
        SEED, CLOSE_LEFT, CLOSE_RIGHT, JOIN
    }

    private Operation(Span sig1, Span sig2, Type type) {
        this.sig1 = sig1;
        this.sig2 = sig2;
        this.type = type;
        this.result = process();
    }

    public boolean isValid() {
        return false;
    }

    private Span process() {
        Span ret = null;

        if (type == Type.SEED) {
            int i = sig1.leftIndex;

            ret = new Span(i, i + 1, false, false, true);
        } else if (type == Type.CLOSE_LEFT) {
            int i = sig1.leftIndex;
            int j = sig1.rightIndex;

            ret = new Span(i, j, false, true, true);
        } else if (type == Type.CLOSE_RIGHT) {
            int i = sig1.leftIndex;
            int j = sig1.rightIndex;

            ret = new Span(i, j, true, false, true);
        } else if (type == Type.JOIN) {
            int i = sig1.leftIndex;
            int k = sig1.rightIndex;
            int j = sig2.rightIndex;
            boolean bL = sig1.bL;
            boolean bR = sig2.bR;
            boolean s = sig2.isSimple;

            ret = new Span(i, j, bL, bR, false);
        }

        return ret;
    }

    public static Operation newCloseLeft(Span sig) {
        return new Operation(sig, null, Type.CLOSE_LEFT);
    }

    public static Operation newCloseRight(Span sig) {
        return new Operation(sig, null, Type.CLOSE_RIGHT);
    }

    public static Operation newJoin(Span sig1, Span sig2) {
        return new Operation(sig1, sig2, Type.JOIN);
    }

    public static Operation newSeed(int i) {
        Span sig = new Span(i, i, false, false, false);
        return new Operation(sig, null, Type.SEED);
    }

    @Override
    public String toString() {
        String ret = "";
        if (type == Type.SEED) {
            ret = "Seed(" + sig1.leftIndex + ")";
        } else if (type == Type.CLOSE_LEFT) {
            ret = "CloseLeft(" + sig1 + ")";
        } else if (type == Type.CLOSE_RIGHT) {
            ret = "CloseRight(" + sig1 + ")";
        } else if (type == Type.JOIN) {
            ret = "Join(" + sig1 + ", " + sig2 + ")";
        }
        return ret;
    }
}
