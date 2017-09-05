package edu.shanghaitech.nlp.algorithm.mst;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/*
 * Chu-Liu Edmond Algorithm for Dense Graph.
 * Time complexity: O(E * V) = O(V^3)
 */
public class ChuLiuEdmond {
    private ChuLiuEdmond() {
        throw new UnsupportedOperationException();
    }

    /**
     * Get minimum arborescence.
     *
     * @param root   :   the index of root of the arborescence with minimum weight.
     * @param weight : Weight matrix whose value at [i][j] is the minimum
     *               weight of parallel edges whose source is i and destination is j.
     *               If the edge doesn't exist, set the it's value to Infinity.
     * @return : TODO.
     */
    public static Map<Integer, Integer> getMinimumArborescence(int root, double[][] weight) {
        return partialSolution(root, weight);
    }


    /**
     * Get maximum arborescence.
     *
     * @param root   :   the index of root of the arborescence with maximum weight.
     * @param weight : Weight matrix whose value at [i][j] is the minimum
     *               weight of parallel edges whose source is i and destination is j.
     *               If the edge doesn't exist, set the it's value to -Infinity.
     * @return : TODO.
     */
    public static Map<Integer, Integer> getMaximumArborescence(int root, double[][] weight) {
        int n = weight.length;
        double[][] negativeWeight = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                negativeWeight[i][j] = -weight[i][j];
            }
        }

        return getMinimumArborescence(root, negativeWeight);
    }

    private static Map<Integer, Integer> partialSolution(int root, double[][] W) {
        /*
         * the `key` in the HashMap `edges` is the destination of an edge,
         * and the corresponding `value` is the source of the edge.
         */
        Map<Integer, Integer> edges = new HashMap<>();

        int n = W.length; // the number of nodes in graph.
        for (int i = 0; i < n; i++) {
            if (i != root) { // skip root
                int u = findTheLowestIncomingEdge(i, W);

                assert u != -1;
                edges.put(i, u);
            }
        }

        /*
         *  Find the cycle in the lowest incoming edges set.
         *  If a cycle is found, `cycleHead` is the index of node in the cycle.
         *  Otherwise, `cycleHead` is set to -1.
         */
        int cycleHead = -1;
        boolean findCycle = false;

        for (int i = 0; i < n && !findCycle; i++) {
            int v = i;
            Map<Integer, Integer> visited = new HashMap<>();

            while (edges.getOrDefault(v, -1) != -1 && visited.getOrDefault(v, -1) != i) {
                visited.put(v, i);
                v = edges.get(v);
            }

            if (v != root) {
                findCycle = true;
                cycleHead = v;
            }
        }

        Map<Integer, Integer> ret = null;

        if (!findCycle) {
            ret = edges;
        } else {
            Set<Integer> cycle = new HashSet<>();
            cycle.add(cycleHead);
            for (int i = edges.get(cycleHead); i != cycleHead; i = edges.get(i)) {
                cycle.add(i);
            }

            /*
             * The 0-th node in the new graph represents the "contracted" cycle.
             * The `nodeMap` is use for recording correspond relationship between
             * the new index and old index of the nodes which are not in the `cycle` in the original graph.
             * The `key` is new index and the `value` is original index.
             */
            Map<Integer, Integer> nodeMap = new HashMap<>();

            int newRootIndex = -1;
            int cnt = 1;
            for (int i = 0; i < n; i++) {
                if (!cycle.contains(i)) {
                    if (i == root) {
                        newRootIndex = cnt;
                    }
                    nodeMap.put(cnt, i);
                    cnt += 1;
                }
            }

            /*
             * TODO
             */
            Map<Integer, Integer> VcIncomingEdgeTo = new HashMap<>();


            /*
             * TODO.
             */
            Map<Integer, Integer> VcOutcomingEdgeFrom = new HashMap<>();

            double[][] newW = new double[cnt][cnt];
            for (int i = 0; i < cnt; i++) {
                newW[i][0] = Double.POSITIVE_INFINITY;
                newW[0][i] = Double.POSITIVE_INFINITY;
            }

            for (int i = 0; i < cnt; i++) {
                for (int j = 0; j < cnt; j++) {
                    if (i == j) {
                        newW[i][j] = Double.POSITIVE_INFINITY;
                    } else {
                        if (i == 0 || j == 0) {
                            if (i == 0) {
                                int y = nodeMap.get(j);
                                for (int c : cycle) {
                                    double newVal = W[c][y];
                                    if (newVal < newW[i][j]) {
                                        newW[i][j] = newVal;
                                        VcOutcomingEdgeFrom.put(y, c);
                                    }
                                }
                            }

                            if (j == 0) {
                                int x = nodeMap.get(i);
                                for (int c : cycle) {
                                    double newVal = W[x][c] - W[edges.get(c)][c];
                                    if (newVal < newW[i][j]) {
                                        newW[i][j] = newVal;
                                        VcIncomingEdgeTo.put(x, c);
                                    }
                                }
                            }
                        } else {
                            int x = nodeMap.get(i);
                            int y = nodeMap.get(j);
                            newW[i][j] = W[x][y];
                        }
                    }
                }
            }

            Map<Integer, Integer> retEdges = partialSolution(newRootIndex, newW);

            /*
             * Find the edge which we need to remove to break the cycle.
             */
            int newEdgeSrc = nodeMap.get(retEdges.get(0));
            int newEdgeDest = VcIncomingEdgeTo.get(newEdgeSrc);

            Map<Integer, Integer> result = new HashMap<>();
            for (int j : retEdges.keySet()) {
                if (j != 0) {
                    int x;
                    int y = nodeMap.get(j);
                    int i = retEdges.get(j);

                    if (i == 0) {
                        x = VcOutcomingEdgeFrom.get(y);
                    } else {
                        x = nodeMap.get(i);
                    }
                    result.put(y, x);
                } else {
                    for (int dest : cycle) {
                        if (dest != newEdgeDest) {
                            int src = edges.get(dest);
                            result.put(dest, src);
                        }
                    }
                    result.put(newEdgeDest, newEdgeSrc);
                }
            }
            ret = result;
        }
        return ret;
    }

    private static int findTheLowestIncomingEdge(int v, double[][] W) {
        int n = W.length;

        int u = -1;
        double minValue = Double.POSITIVE_INFINITY;

        for (int i = 0; i < n; i++) {
            if (i != v) { // skip self cycle
                if (W[i][v] < minValue) {
                    minValue = W[i][v];
                    u = i;
                }
            }
        }
        return u;
    }
}
