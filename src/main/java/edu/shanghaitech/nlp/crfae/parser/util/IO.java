package edu.shanghaitech.nlp.crfae.parser.util;

import edu.shanghaitech.nlp.crfae.parser.DepInstance;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public class IO {

    public static void error(String error_message) {
        System.err.println(error_message);
        System.exit(-1);
    }

    public static void outputInstance(DepInstance inst) {
        System.out.println(Arrays.toString(inst.words));
        System.out.println(Arrays.toString(inst.pos));
    }

    public static void output2dDoubleArray(double[][] arr) {
        int n = arr.length;

        System.out.println("[");
        for (int i = 0; i < n; i++) {
            System.out.println(Arrays.toString(arr[i]) + ",");
        }
        System.out.println("]");
    }

    public static double[][] convert4dTo2d(double[][][][] arr) {
        int m = arr.length;
        int n = arr[0].length;
        double[][] ret = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
               ret[i][j] = arr[i][j][0][0];
            }
        }
        return ret;
    }


    public static void output2dDoubleArrayToFile(double[][] arr, String filename) {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            int n = arr.length;
            for (int i = 0; i < n; i++) {
                out.write(Arrays.toString(arr[i]).replaceAll("\\[|\\]", "") + "\n");
            }
            out.flush();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
