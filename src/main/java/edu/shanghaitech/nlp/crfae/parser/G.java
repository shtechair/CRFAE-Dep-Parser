package edu.shanghaitech.nlp.crfae.parser;

import net.sourceforge.argparse4j.inf.Namespace;

import java.lang.reflect.Array;
import java.util.*;

// Helper Class
public class G {
    // Some test variable.
    public static double maxValInTheta = Double.NEGATIVE_INFINITY;
    // END.

    static Namespace ns;
    static String modelInstanceName;
    static String outputDir;

    public static Integer[] randomIndexArray(int size, int bound) {
        assert size <= bound;

        if(size == 0) return new Integer[0];

        Random rng = new Random();
        Set<Integer> generated = new HashSet<>();
        while (generated.size() < size) {
            Integer next = rng.nextInt(bound);
            generated.add(next);
        }

        Integer[] ret = generated.toArray(new Integer[1]);
        Arrays.sort(ret);
        return ret;
    }

    public static <E> E[] randomSubArray(Class<E> clazz, E[] arr, int subArrSize) {
        @SuppressWarnings("unchecked")
        E[] ret = (E[]) Array.newInstance(clazz, subArrSize);

        Integer[] indexArr = randomIndexArray(subArrSize, arr.length);
        for (int i = 0; i < indexArr.length; i++) {
            ret[i] = arr[indexArr[i]];
        }
        return ret;
    }
}
