package edu.shanghaitech.nlp.crfae.parser;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public class DepEvaluator {

    public static double[] evaluate(String goldFile, int[][] predictions, int maxSentSize) throws IOException {
        int total = 0;
        int corr = 0;
        int numsent = 0;
        int corrsent = 0;

        int[][] truePredictions = readParsesFromFile(goldFile, maxSentSize);

        if (truePredictions.length != predictions.length) {
            System.out.println("Lengths do not match");
            System.exit(-1);
        }

        int m = predictions.length;
        for (int i = 0; i < m; i++) {
            if (truePredictions[i].length != predictions[i].length) {
                System.out.println("Lengths do not match");
                System.exit(-1);
            }

            int n = predictions[i].length;

            boolean whole = true;
            for (int j = 1; j < n; j++) {
                if (truePredictions[i][j] == predictions[i][j]) {
                    corr++;
                } else {
                    whole = false;
                }
            }

            total += (n - 1); // Remove the root.

            if (whole) corrsent++;
            numsent++;
        }

        System.out.println("Performance: ");
        System.out.println("\tTokens: " + total);
        System.out.println("\tCorrect: " + corr);
        System.out.println("\tUnlabeled Accuracy: " + ((double) corr / total));
        System.out.println("\tUnlabeled Complete Correct: " + ((double) corrsent / numsent));
        System.out.println();

        double[] ret = {(double) corr / total, (double) corrsent / numsent};
        return ret;
    }

    private static int[][] readParsesFromFile(String filename, int maxSentSize) throws IOException {
        BufferedReader in = new BufferedReader(new FileReader(filename));
        String words;
        String pos;
        String upos;
        String emptyLine;
        String deps;

        List<List<Integer>> lt = new LinkedList<>();

        words = in.readLine();
        pos = in.readLine();
        upos = in.readLine();
        deps = in.readLine();
        emptyLine = in.readLine();

        while (deps != null) {
            String[] depsStr = deps.split("\t");

            if (depsStr.length <= maxSentSize) {
                List<Integer> list = new LinkedList<>();
                list.add(-1);

                for (String depStr : depsStr) {
                    list.add(Integer.parseInt(depStr));
                }

                lt.add(list);
            }
            words = in.readLine();
            pos = in.readLine();
            upos = in.readLine();
            deps = in.readLine();
            emptyLine = in.readLine();
        }

        int[][] ret = new int[lt.size()][];
        for (int i = 0; i < lt.size(); i++) {
            List<Integer> parse = lt.get(i);
            int[] parseArr = new int[parse.size()];
            for (int j = 0; j < parse.size(); j++) {
                parseArr[j] = parse.get(j);
            }
            ret[i] = parseArr;
        }

        return ret;
    }
}
