package edu.shanghaitech.nlp.crfae.parser;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import org.pmw.tinylog.Configurator;
import org.pmw.tinylog.Level;
import org.pmw.tinylog.Logger;
import org.pmw.tinylog.writers.FileWriter;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import edu.shanghaitech.nlp.crfae.parser.HyperParameter.*;

public class DepParserLauncher {
    public static String trainfile = null;
    public static String testfile = null;
    public static String valfile = null;

    public static String savedModel = null;

    private DepPipe pipe;
    private Parameters params;

    private final static double THRESHOLD = 0.001;

    public DepParserLauncher(DepPipe pipe) {
        this.pipe = pipe;
        params = new Parameters(pipe);
    }

    private void evaluatePerformance(String type, String goldfile, String maxSentenceSize) {
        try {
            int maxSenSize = maxSentenceSize.equals("all") ? 1000000 : Integer.parseInt(maxSentenceSize);
            int[][] predictions = getParses(goldfile, maxSenSize);
            double[] accuracy = DepEvaluator.evaluate(goldfile, predictions, maxSenSize);
            Logger.info("Acc of " + type + "-" + maxSentenceSize + ": " + goldfile);
            Logger.info("Unlabeled Accuracy: " + accuracy[0]);
            Logger.info("Unlabeled Complete Correct: " + accuracy[1] + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void train(DepInstance[] il) {
        System.out.println("========================");
        System.out.println("About to train.");
        System.out.println("Training set size: " + il.length);
        System.out.println();

        params.kmInit(il);

        boolean trainingDone = false;
        for (int i = 0; !trainingDone; i++) {
            System.out.println("========================");
            System.out.println("Iteration: " + i);
            System.out.println("========================");
            System.out.print("Processed: \n");

            Logger.info("Iteration: " + i);

            long start = System.currentTimeMillis();
            /// Training
            UnsupervisedTrainer trainer = new UnsupervisedTrainer(il, pipe, params);

            double objectiveBefore = trainer.objectiveFunction.valueAt(il, params);
            params = trainer.iteration();
            double objectiveAfter = trainer.objectiveFunction.valueAt(il, params);
            System.out.format("Objective: %f -> %f\n", objectiveBefore, objectiveAfter);
            Logger.info("Objective: {} -> {}", objectiveBefore, objectiveAfter);

            trainingDone = ((objectiveBefore - objectiveAfter) / objectiveBefore) < THRESHOLD;

            long end = System.currentTimeMillis();
            System.out.println("Training iteration took: " + (end - start) + " ms");
            Logger.info("Training iteration took: " + (end - start) + " ms \n");

        }
    }

    public int[][] getParses(String inputFile, int maxSentSize) throws IOException {
        DepInstance[] il = pipe.createInstances(inputFile);
        List<int[]> ret = new ArrayList<>();

        DepParser parser = new DepParser(params);
        for (int i = 0; i < il.length; i++) {
            // Because of DepInstance add Root in the Sentence.
            if (il[i].length - 1 <= maxSentSize) {
                ret.add(parser.testTimeSingleRootParseArray(il[i]));
            }
        }

        return ret.toArray(new int[ret.size()][]);
    }

    static void processArguments(String[] args) {
        ArgumentParser parser = ArgumentParsers.newArgumentParser("CRFAE").defaultHelp(true)
                .description("CRF AutoEncoder for unsupervised dependency parsing.");

        /* Data */
        parser.addArgument("--train-file");
        parser.addArgument("--val-file");
        parser.addArgument("--test-file");
        parser.addArgument("--exec-dir").setDefault("exec").help("The directory output");

        parser.addArgument("--model");

        /* Model Hyper Parameter */
        parser.addArgument("--model-type").choices("projective", "non-projective")
                .setDefault("projective").help("Dependency model: projective or non-projective");
        parser.addArgument("--training-type").choices("hard", "soft")
                .setDefault("hard").help("training type: hard means viterbi training.");
        parser.addArgument("--km-type").choices("decoder", "joint")
                .setDefault("decoder").help("km initialization type");
        parser.addArgument("--reg-type").choices("L1", "L2")
                .setDefault("L1").help("regularization type.");
        parser.addArgument("--parse-type").choices("crf", "crf-prior", "joint", "joint-prior")
                .setDefault("joint").help("determine the distribution used in test-time parsing.");

        /* The Hyper Parameters which we tune.*/
        parser.addArgument("--batch-size").type(Integer.class).setDefault(200).help("");
        parser.addArgument("--gd-num-passes").type(Integer.class).setDefault(2).help("");
        parser.addArgument("--em-num-passes").type(Integer.class).setDefault(2).help("");
        parser.addArgument("--init-rate").type(Double.class).setDefault(0.01).help("learning rate");
        parser.addArgument("--lambda").type(Double.class).setDefault(0.).help("regularization power");
        parser.addArgument("--prior-weight").setDefault(0.).type(Double.class);

        parser.addArgument("--rules-type").choices("wsj", "ud").
                setDefault("wsj").dest("rules_type").help("determine the linguist rule used for prior.");

        Namespace ns = null;
        try {
            ns = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }

        trainfile = ns.getString("train_file");
        valfile = ns.getString("val_file");
        testfile = ns.getString("test_file");
        savedModel = ns.getString("model");

        HyperParameter hyperParam = HyperParameter.getInstance();
        hyperParam.modelType = (
                ns.getString("model_type").equals("projective") ? ModelType.PROJ : ModelType.NON_PROJ
        );
        hyperParam.regType = (
                ns.getString("reg_type").equals("L1") ? RegType.L1 : RegType.L2
        );
        hyperParam.trainingType = (
                ns.getString("training_type").equals("hard") ? TrainingType.HARD : TrainingType.SOFT
        );
        hyperParam.kmType = (
                ns.getString("km_type").equals("joint") ? KMType.JOINT : KMType.DECODER
        );


        // test-time parse type
        if (ns.get("parse_type").equals("crf")) {
            hyperParam.parseType = ParseType.CRF;
        } else if (ns.get("parse_type").equals("joint")) {
            hyperParam.parseType = ParseType.JOINT;
        } else if (ns.get("parse_type").equals("crf-prior")) {
            hyperParam.parseType = ParseType.CRF_PRIOR;
        } else if (ns.get("parse_type").equals("joint-prior")) {
            hyperParam.parseType = ParseType.JOINT_PRIOR;
        }

        if (ns.get("rules_type").equals("wsj")) {
            hyperParam.rulesType = RulesType.WSJ;
        } else if (ns.get("rules_type").equals("ud")) {
            hyperParam.rulesType = RulesType.UD;
        }

        hyperParam.gdNumPasses = ns.getInt("gd_num_passes");
        hyperParam.emNumPasses = ns.getInt("em_num_passes");
        hyperParam.batchSize = ns.getInt("batch_size");
        hyperParam.initRate = ns.getDouble("init_rate");
        hyperParam.lambda = ns.getDouble("lambda");
        hyperParam.priorWeight = ns.getDouble("prior_weight");

        G.ns = ns;
    }

    public void saveModel() throws IOException {
        FileOutputStream fout = new FileOutputStream(
                G.outputDir + "/param.model");
        ObjectOutputStream oos = new ObjectOutputStream(fout);
        oos.writeObject(params);
        fout.close();
    }

    public void loadModel(String name) throws IOException, ClassNotFoundException {
        // TODO.
        FileInputStream fin = new FileInputStream(name);
        ObjectInputStream ois = new ObjectInputStream(fin);
        params = (Parameters) ois.readObject();
        pipe = params.pipe;
        fin.close();
    }

    /////////////////////////////////////////////////////
    // RUNNING THE PARSER
    ////////////////////////////////////////////////////
    public static void main(String[] args) throws Exception {
        // Get current Timestamp and Setup logger.
        long timeInNanos = System.nanoTime();
        String timeStamp = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").format(new Date());
        timeStamp += "." + ((int) (timeInNanos % 1000000000 / 100));


        // Program start.
        processArguments(args);
        G.modelInstanceName = timeStamp;
        G.outputDir = G.ns.getString("exec_dir") + "/" + timeStamp;

        // Create Directory
        Path dir = Paths.get(G.outputDir);
        Files.createDirectories(dir);

        System.out.println("------\nFLAGS\n------");
        System.out.println("train-file: " + trainfile);
        System.out.println("dev-file: " + valfile);
        System.out.println("test-file: " + testfile);
        System.out.println("hyper-parameter: " + HyperParameter.getInstance());
        System.out.println("------\n");


        Configurator.defaultConfig().writer(new FileWriter(G.outputDir + "/result.log"))
                .level(Level.INFO).formatPattern("{level}\t{date}\t\t{message}").activate();

        Logger.info(HyperParameter.getInstance());
        Logger.info(trainfile + "\n");

        DepPipe pipe = new DepPipe();
        DepParserLauncher dp = null;
        if (savedModel == null) {
            DepInstance[] trainingData = pipe.createInstances(trainfile);
            pipe.createAlphabet(trainingData);

            dp = new DepParserLauncher(pipe);

            int numFeats = pipe.featAlphabet.size();
            int numReconsParentDim = dp.params.reconsParentAlphabet.size();
            int numReconsChildDim = dp.params.reconsChildAlphabet.size();
            System.out.println("Num Feats: " + numFeats);
            System.out.println("Size of ReconsParent: " + numReconsParentDim);
            System.out.println("Size of ReconsChild: " + numReconsChildDim);

            // Training...
            dp.train(trainingData);
        } else {
            dp = new DepParserLauncher(pipe);
            dp.loadModel(savedModel);
        }

        // check dev accuracy.
        if (valfile != null) {
            dp.evaluatePerformance("Val", valfile, "10");
        }

        // check test accuracy.
        if (testfile != null) {
            dp.evaluatePerformance("Test", testfile, "10");
            dp.evaluatePerformance("Test", testfile, "all");
        }

        dp.saveModel();
    }
}

