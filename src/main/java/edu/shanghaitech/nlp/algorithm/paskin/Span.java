package edu.shanghaitech.nlp.algorithm.paskin;


public class Span {
    public final int leftIndex;
    public final int rightIndex;
    public final boolean bL;
    public final boolean bR;
    public final boolean isSimple;

    public Span(int left, int right, boolean bL, boolean bR, boolean s) {
        this.leftIndex = left;
        this.rightIndex = right;
        this.bL = bL;
        this.bR = bR;
        this.isSimple = s;
    }

    public boolean isValid() {
        if (rightIndex - leftIndex == 1 && !isSimple) {
            return false;
        }

        if (bL && bR) {
            return false;
        }

        if(rightIndex - leftIndex > 1 && isSimple && (!bL && !bR)){
            return false;
        }

        return true;
    }

    @Override
    public int hashCode() {
        return this.toString().hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Span) {
            return this.toString().equals(obj.toString());
        }
        return false;
    }

    @Override
    public String toString() {
        return "<" +
                leftIndex + ", " +
                rightIndex + ", " +
                (bL ? "T" : "F") + ", " +
                (bR ? "T" : "F") + ", " +
                (isSimple ? "T" : "F") +
                ">";
    }
}
