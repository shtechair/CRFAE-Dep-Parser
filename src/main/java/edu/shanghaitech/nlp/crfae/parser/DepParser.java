package edu.shanghaitech.nlp.crfae.parser;

import edu.shanghaitech.nlp.algorithm.mst.ChuLiuEdmond;
import edu.shanghaitech.nlp.algorithm.paskin.PaskinAddVer;

import java.util.HashMap;
import java.util.Map;

public class DepParser {
    public Parameters params;
    public DepPipe pipe;

    public DepParser(Parameters params) {
        this.params = params;
        this.pipe = params.pipe;
    }

    public Map<Integer, Integer> projectiveSingleRootParse(DepInstance inst, boolean usePrior, boolean joint) {
        double[][] W;

        if (joint) {
            W = params.jointScoreGraph(inst);
        } else {
            W = params.crfScoreGraph(inst);
        }

        //Add prior knowledge.
        if (usePrior) {
            double[][] priors = params.priorScoreGraph(inst);
            int len = inst.length;
            for (int i = 0; i < len; i++) {
                for (int j = 1; j < len; j++) {
                    W[i][j] += priors[i][j];
                }
            }
        }

        PaskinAddVer model = new PaskinAddVer(W);
        return model.chartParsing();
    }

    public Map<Integer, Integer> nonProjectiveSingleRootParse(DepInstance inst, boolean usePrior, boolean joint) {
        double[][] W = null;

        if (joint) {
            W = params.jointWeightGraph(inst);
        } else {
            W = params.crfWeightGraph(inst);
        }

        //Add prior knowledge.
        int len = inst.length;

        if (usePrior) {
            double[][] priors = params.priorScoreGraph(inst);
            for (int i = 0; i < len; i++) {
                for (int j = 1; j < len; j++) {
                    W[i][j] *= Math.exp(priors[i][j]);
                }
            }
        }

        int n = inst.length - 1;

        // Create prob 2d array without <root> node.
        double[][] subProbs2d = new double[n][n];

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                subProbs2d[i - 1][j - 1] = W[i][j];
            }
        }

        Map<Integer, Integer> bestParse = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (int dummyRootIndex = 0; dummyRootIndex < n; dummyRootIndex++) {
            Map<Integer, Integer> subRet = ChuLiuEdmond.getMaximumArborescence(dummyRootIndex, subProbs2d);

            Map<Integer, Integer> parse = new HashMap<>();

            for (int i : subRet.keySet()) {
                int child = i + 1;
                int parent = subRet.get(i) + 1;
                parse.put(child, parent);
            }
            parse.put(dummyRootIndex + 1, 0);

            double currentScore = 0.0;
            for (Integer child : parse.keySet()) {
                Integer parent = parse.get(child);
                currentScore += W[parent][child];
            }

            if (currentScore > bestScore) {
                bestScore = currentScore;
                bestParse = parse;
            }
        }

        return bestParse;
    }

    public int[] testTimeSingleRootParseArray(DepInstance inst) {
        HyperParameter hyperParameter = HyperParameter.getInstance();

        Map<Integer, Integer> parse = null;
        switch (hyperParameter.modelType) {
            case NON_PROJ: {
                switch (hyperParameter.parseType) {
                    case CRF:
                        parse = nonProjectiveSingleRootParse(inst, false, false);
                        break;
                    case CRF_PRIOR:
                        parse = nonProjectiveSingleRootParse(inst, true, false);
                        break;
                    case JOINT:
                        parse = nonProjectiveSingleRootParse(inst, false, true);
                        break;
                    case JOINT_PRIOR:
                        parse = nonProjectiveSingleRootParse(inst, true, true);
                        break;
                }
                break;
            }
            case PROJ: {
                switch (hyperParameter.parseType) {
                    case CRF:
                        parse = projectiveSingleRootParse(inst, false, false);
                        break;
                    case CRF_PRIOR:
                        parse = projectiveSingleRootParse(inst, true, false);
                        break;
                    case JOINT:
                        parse = projectiveSingleRootParse(inst, false, true);
                        break;
                    case JOINT_PRIOR:
                        parse = projectiveSingleRootParse(inst, true, true);
                        break;
                }
                break;
            }
        }

        int n = inst.length;
        int[] parseArray = new int[n];
        parseArray[0] = -1;
        for (int child = 1; child < n; child++) {
            parseArray[child] = parse.get(child);
        }
        return parseArray;
    }

    public Map<Integer, Integer> singleRootParse(DepInstance inst) {
        HyperParameter hyperParameter = HyperParameter.getInstance();

        if (hyperParameter.modelType == HyperParameter.ModelType.NON_PROJ) {
            return nonProjectiveSingleRootParse(inst, true, true);
        } else {
            return projectiveSingleRootParse(inst, true, true);
        }
    }

    public String singleRootParseString(DepInstance inst) {
        int n = inst.length - 1;
        Map<Integer, Integer> bestParse = singleRootParse(inst);

        String parseString = "";
        for (int child = 1; child <= n; child++) {
            parseString += bestParse.get(child).toString() + "\t";
        }

        return parseString;
    }

    public int[] singleRootParseArray(DepInstance inst) {
        Map<Integer, Integer> bestParse = singleRootParse(inst);

        int n = inst.length;
        int[] parseArray = new int[n];
        parseArray[0] = -1;
        for (int child = 1; child < n; child++) {
            parseArray[child] = bestParse.get(child);
        }
        return parseArray;
    }


}


