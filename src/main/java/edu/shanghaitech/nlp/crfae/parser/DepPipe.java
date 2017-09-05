package edu.shanghaitech.nlp.crfae.parser;

import java.io.*;
import java.util.*;
import java.util.stream.Stream;

public class DepPipe implements Serializable {
    public final static String ROOT_WORD = "<root>";
    public final static String ROOT_POS = "<root-POS>";

    public Alphabet featAlphabet;   // HashMap for features.
    public Alphabet wordAlphabet;   // HashMap for the corpus' words.
    public Alphabet posAlphabet;    // HashMap for pos tags.
    public Map<String, Integer> wordCount;

    public DepPipe() {
        featAlphabet = new Alphabet();
        wordAlphabet = new Alphabet();
        posAlphabet = new Alphabet();
        wordCount = new HashMap<>();
    }

    public String[][] getLines(BufferedReader in) throws IOException {
        String wordLine = in.readLine();
        String posLine = in.readLine();
        String uposLine = in.readLine();
        String depsLine = in.readLine();
        String emptyLine = in.readLine(); // blank line

        if (wordLine == null) return null;

        String[] toks = wordLine.split("\t");
        String[] pos = posLine.split("\t");
        String[] upos = uposLine.split("\t");
        String[] deps = depsLine.split("\t");

        String[] toks_new = new String[toks.length + 1];
        String[] pos_new = new String[pos.length + 1];
        String[] upos_new = new String[upos.length + 1];
        String[] deps_new = new String[deps.length + 1];

        toks_new[0] = DepPipe.ROOT_WORD;
        pos_new[0] = DepPipe.ROOT_POS;
        upos_new[0] = DepPipe.ROOT_POS;
        deps_new[0] = "-1";

        for (int i = 0; i < toks.length; i++) {
            toks_new[i + 1] = normalize(toks[i]);
            pos_new[i + 1] = pos[i];
            upos_new[i + 1] = upos[i];
            deps_new[i + 1] = deps[i];
        }
        toks = toks_new;
        pos = pos_new;
        upos = upos_new;
        deps = deps_new;

        String[][] result = new String[4][];
        result[0] = toks;
        result[1] = pos;
        result[2] = upos;
        result[3] = deps;
        return result;
    }

    public DepInstance[] createInstances(String file) throws IOException {
        BufferedReader in = new BufferedReader(
                new InputStreamReader(new FileInputStream(file), "UTF8")
        );
        List<DepInstance> lt = new LinkedList<>();

        String[][] lines = getLines(in);
        while (lines != null) {
            String[] toks = lines[0];
            String[] pos = lines[1];
            String[] upos = lines[2];
            String[] deps = lines[3];

            int[] ideps = Stream.of(deps).mapToInt(Integer::parseInt).toArray();

            DepInstance inst = new DepInstance(toks, pos, upos, ideps);
            lt.add(inst);

            lines = getLines(in);
        }

        in.close();

        return lt.toArray(new DepInstance[lt.size()]);
    }

    public void createAlphabet(DepInstance[] il) {
        System.out.print("Creating Alphabet ... ");

        for (DepInstance inst : il) {
            String[] words = inst.words;
            String[] pos = inst.pos;
            this.fillAlphabets(words, pos);
        }

        wordAlphabet = new Alphabet();
        int wordThreshold = HyperParameter.getInstance().wordThreshold;
        for (DepInstance inst : il) {
            for (int i = 0; i < inst.length; i++) {
                if (wordCount.getOrDefault(inst.words[i], 0) > wordThreshold) {
                    wordAlphabet.lookupIndex(inst.words[i]);
                } else {
                    wordAlphabet.lookupIndex(inst.pos[i]);
                }
            }
        }

        closeAlphabets();
        System.out.println("Done.");
    }

    private void fillAlphabets(String[] words, String[] pos) {
        for (String word : words) {
            wordAlphabet.lookupIndex(word);
            wordCount.put(word, wordCount.getOrDefault(word, 0) + 1);
        }
        for (String tag : pos) {
            posAlphabet.lookupIndex(tag);
        }
        this.fillFeatAlphabet(words, pos);
    }

    private void fillFeatAlphabet(String[] toks, String[] pos) {
        String[] posA = new String[pos.length];
        for (int i = 0; i < pos.length; i++) {
            posA[i] = pos[i].substring(0, 1);
        }

        for (int w1 = 0; w1 < toks.length; w1++) {
            for (int w2 = w1 + 1; w2 < toks.length; w2++) {
                for (int ph = 0; ph < 2; ph++) {
                    boolean attR = ph == 0 ? true : false;

                    int childInt = attR ? w2 : w1;
                    int parInt = attR ? w1 : w2;

                    createFeatureVector(toks, pos, posA, w1, w2, attR, new FeatureVector(-1, -1.0, null));
                }
            }
        }
    }

    private void closeAlphabets() {
        featAlphabet.stopGrowth();
        wordAlphabet.stopGrowth();
        posAlphabet.stopGrowth();
    }

    private String normalize(String s) {
        if (s.matches("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")) {
            return "<num>";
        } else if (s.matches("(https?:\\/\\/(?:www\\.|(?!www))[^\\s\\.]+\\.[^\\s]{2,}|www\\.[^\\s]+\\.[^\\s]{2,})")) {
            return "<url>";
        }
        return s;
    }


    private FeatureVector createFeatureVector(String[] toks,
                                              String[] pos,
                                              String[] posA,
                                              int small,
                                              int large,
                                              boolean attR,
                                              FeatureVector fv) {


        String att;
        if (attR)
            att = "RA";
        else
            att = "LA";

        int dist = Math.abs(large - small);
        String distBool = "0";
        if (dist > 1)
            distBool = "1";
        if (dist > 2)
            distBool = "2";
        if (dist > 3)
            distBool = "3";
        if (dist > 4)
            distBool = "4";
        if (dist > 5)
            distBool = "5";
        if (dist > 10)
            distBool = "10";


        String attDist = "&" + att + "&" + distBool;


        String pLeft = small > 0 ? pos[small - 1] : "STR";
        String pRight = large < pos.length - 1 ? pos[large + 1] : "END";
        String pLeftRight = small < large - 1 ? pos[small + 1] : "MID";
        String pRightLeft = large > small + 1 ? pos[large - 1] : "MID";


        // unigram
        fv = add("PD=" + pos[small] + attDist, 1.0, fv);
        fv = add("PD=" + pos[large] + attDist, 1.0, fv);
        // bigram
        fv = add("PPD=" + pos[small] + pos[large] + attDist, 1.0, fv);
        // trigram
        fv = add("PPPD=" + pos[small] + pLeft + pos[large] + attDist, 1.0, fv);
        fv = add("PPPD=" + pos[small] + pLeftRight + pos[large] + attDist, 1.0, fv);
        fv = add("PPPD=" + pos[small] + pos[large] + pRightLeft + attDist, 1.0, fv);
        fv = add("PPPD=" + pos[small] + pos[large] + pRight + attDist, 1.0, fv);

        return fv;
    }

    private FeatureVector add(String feat, double val, FeatureVector fv) {
        int num = featAlphabet.lookupIndex(feat);
        if (num >= 0)
            return new FeatureVector(num, val, fv);
        return fv;
    }

    public FeatureVector featureOf(DepInstance inst, Integer i, Integer j) {
        String[] posA = Arrays.stream(inst.pos).
                map(x -> x.substring(0, 1)).
                toArray(String[]::new);
        int w1 = i < j ? i : j;
        int w2 = i < j ? j : i;
        boolean attR = i < j;

        return createFeatureVector(
                inst.words, inst.pos, posA, w1, w2, attR, new FeatureVector(-1, -1.0, null)
        );
    }

    public FeatureVector[][] getFeatureMatrix(DepInstance inst) {
        int n = inst.length;
        String[] toks = inst.words;
        String[] pos = inst.pos;

        FeatureVector[][] fvs = new FeatureVector[n][n];

        String[] posA = new String[n];
        for (int i = 0; i < n; i++) {
            posA[i] = pos[i].substring(0, 1);
        }

        // Get production crap.
        for (int w1 = 0; w1 < n; w1++) {
            for (int w2 = w1 + 1; w2 < n; w2++) {
                for (int ph = 0; ph < 2; ph++) {
                    boolean attR = ph == 0 ? true : false;

                    int parInt = attR ? w1 : w2;
                    int childInt = attR ? w2 : w1;

                    FeatureVector prodFV = createFeatureVector(toks, pos, posA, w1, w2, attR,
                            new FeatureVector(-1, -1.0, null));

                    fvs[parInt][childInt] = prodFV;
                }
            }
        }
        return fvs;
    }

    // Serializable
    private void writeObject(ObjectOutputStream out) throws IOException {
        out.writeObject(featAlphabet);
        out.writeObject(wordAlphabet);
        out.writeObject(posAlphabet);
        out.writeObject(wordCount);
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        featAlphabet = (Alphabet) in.readObject();
        wordAlphabet = (Alphabet) in.readObject();
        posAlphabet = (Alphabet) in.readObject();
        wordCount = (HashMap<String, Integer>) in.readObject();
    }
}
