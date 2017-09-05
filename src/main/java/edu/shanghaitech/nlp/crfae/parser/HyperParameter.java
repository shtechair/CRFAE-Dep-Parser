package edu.shanghaitech.nlp.crfae.parser;


public class HyperParameter {
    enum ParseType {
        CRF, CRF_PRIOR, JOINT, JOINT_PRIOR;

        @Override
        public String toString() {
            String ret = "";
            switch (this) {
                case CRF:
                    ret = "Crf";
                    break;
                case CRF_PRIOR:
                    ret = "CrfPiror";
                    break;
                case JOINT:
                    ret = "Joint";
                    break;
                case JOINT_PRIOR:
                    ret = "JointPrior";
                    break;
            }
            return ret;
        }
    }

    enum ModelType {
        PROJ, NON_PROJ;

        @Override
        public String toString() {
            return this.equals(NON_PROJ) ? "NonProj" : "Proj";
        }
    }

    enum TrainingType {
        SOFT, HARD;

        @Override
        public String toString() {
            return this.equals(SOFT) ? "Soft" : "Hard";
        }
    }

    enum RegType {
        L1, L2;

        @Override
        public String toString() {
            return this.equals(L1) ? "L1" : "L2";
        }
    }

    enum RulesType {
        WSJ, UD;

        @Override
        public String toString() {
            return this.equals(WSJ) ? "WSJ" : "UD";
        }
    }

    enum KMType{
        DECODER, JOINT;
        public String toString() {
            return this.equals(DECODER) ? "Decoder" : "Joint";
        }
    }

    public ModelType modelType = ModelType.NON_PROJ;
    public TrainingType trainingType = TrainingType.SOFT;
    public RegType regType = RegType.L1;
    public ParseType parseType = ParseType.JOINT;
    public RulesType rulesType = RulesType.WSJ;
    public KMType kmType = KMType.DECODER;

    public double initRate = 0.1;
    public double lambda = 0.1;
    public int batchSize = 100;
    public int gdNumPasses = 2;
    public int emNumPasses = 2;
    public double priorWeight = 0.;
    public double smoothingPower = 1e-8;

    public boolean decoderDist = false;
    public boolean decoderDir = false;
    public int wordThreshold = 20000000;

    private HyperParameter() {
        if (INSTANCE != null) {
            throw new IllegalAccessError();
        }
    }

    private static final HyperParameter INSTANCE = new HyperParameter();

    public static HyperParameter getInstance() {
        return INSTANCE;
    }

    @Override
    public String toString() {
        return "HyperParameter{" +
                "modelType=" + modelType +
                ", trainingType=" + trainingType +
                ", regType=" + regType +
                ", parseType=" + parseType +
                ", rulesType=" + rulesType +
                ", kmType=" + kmType +
                ", initRate=" + initRate +
                ", lambda=" + lambda +
                ", batchSize=" + batchSize +
                ", gdNumPasses=" + gdNumPasses +
                ", emNumPasses=" + emNumPasses +
                ", priorWeight=" + priorWeight +
                ", smoothingPower=" + smoothingPower +
                ", decoderDist=" + decoderDist +
                ", decoderDir=" + decoderDir +
                ", wordThreshold=" + wordThreshold +
                '}';
    }
}
