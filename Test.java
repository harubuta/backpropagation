package backPropagation;

public class Test {

	public static void main(String[] args) {
		//
		NeuralNetwork neuralnet = new NeuralNetwork(2, 4, 1);
        int maxRuns = 500000;
        double minErrorCondition = 0.001;
        neuralnet.run(maxRuns, minErrorCondition);
	}

}
