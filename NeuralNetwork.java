package backPropagation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;


/**
 * 誤差逆伝播
 * 課題1
 *
 */
public class NeuralNetwork {

     Random rand = new Random();
     ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
     ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
     ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
     Neuron bias = new Neuron();
     int randomWeightMultiplier = 1;
     int[] layers;

     double epsilon = 0.00000000001;

     double learningRate = 0.9f;
     double momentum = 0.7f;

    // 入力値
    final int inputs[][] = { { 1, 1 }, { 1, 0 }, { 0, 1 }, { 0, 0 } };

    // 出力結果 XOR
    final double expectedOutputs[][] = { { 0 }, { 1 }, { 1 }, { 0 } };
//    // 出力結果 OR
//    final double expectedOutputs[][] = { { 1 }, { 1 }, { 1 }, { 0 } };
//    // 出力結果 AND
//    final double expectedOutputs[][] = { { 1 }, { 0 }, { 0 }, { 0 } };
//    // 出力結果  NAND
//    final double expectedOutputs[][] = { { 1 }, { 1 }, { 1 }, { 0 } };

    double resultOutputs[][] = { { -1 }, { -1 }, { -1 }, { -1 } };
    double output[];

    final HashMap<String, Double> weightUpdate = new HashMap<String, Double>();


    public NeuralNetwork(int input, int hidden, int output) {
        this.layers = new int[] { input, hidden, output };

        //ニューロンの作成
        for (int i = 0; i < layers.length; i++) {
            //入力層の作成
        	if (i == 0) {
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    inputLayer.add(neuron);
                }
             // 中間層の作成
            } else if (i == 1) {
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    neuron.addInConnectionsS(inputLayer);
                    neuron.addBiasConnection(bias);
                    hiddenLayer.add(neuron);
                }
            }
        	// 出力層の作成
            else if (i == 2) {
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    neuron.addInConnectionsS(hiddenLayer);
                    neuron.addBiasConnection(bias);
                    outputLayer.add(neuron);
                }
            } else {
                System.out.println("!Error NeuralNetwork init");
            }
        }

        // 重みの初期値を乱数を用いて学習※w=0だと誤差がゼロになり学習が進まないため
        for (Neuron neuron : hiddenLayer) {
            ArrayList<Connection> connections = neuron.getAllInConnections();
            for (Connection conn : connections) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }
        for (Neuron neuron : outputLayer) {
            ArrayList<Connection> connections = neuron.getAllInConnections();
            for (Connection conn : connections) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }

        // カウンターの初期化
        Neuron.counter = 0;
        Connection.counter = 0;

    }
    // 乱数
    double getRandom() {
        return randomWeightMultiplier * (rand.nextDouble() * 2 - 1); // [-1;1[
    }

    /**
     *
     * @param inputs
     *
     */
    public void setInput(int inputs[]) {
        for (int i = 0; i < inputLayer.size(); i++) {
            inputLayer.get(i).setOutput(inputs[i]);
        }
    }

    public double[] getOutput() {
        double[] outputs = new double[outputLayer.size()];
        for (int i = 0; i < outputLayer.size(); i++)
            outputs[i] = outputLayer.get(i).getOutput();
        return outputs;
    }

    /**
     * 入力層から出力層まで順伝播
     */
    public void feedForward() {
        for (Neuron n : hiddenLayer)
            n.calculateOutput();
        for (Neuron n : outputLayer)
            n.calculateOutput();
    }

    /**
     * 出力層から入力層まで逆向きに伝播する
     *
     * @param expectedOutput
     *
     */
    public void backpropagation(double expectedOutput[]) {

        for (int i = 0; i < expectedOutput.length; i++) {
            double d = expectedOutput[i];
            if (d < 0 || d > 1) {
                if (d < 0)
                    expectedOutput[i] = 0 + epsilon;
                else
                    expectedOutput[i] = 1 - epsilon;
            }
        }

        int i = 0;
        for (Neuron n : outputLayer) {
            ArrayList<Connection> connections = n.getAllInConnections();
            for (Connection con : connections) {
                double yk = n.getOutput();
                double yi = con.leftNeuron.getOutput();
                double desiredOutput = expectedOutput[i];
                //結合係数の更新式
                double partialDerivative = -yk * (1 - yk) * yi
                        * (desiredOutput - yk);
                double deltaWeight = -learningRate * partialDerivative;
                double newWeight = con.getWeight() + deltaWeight;
                //更新結果のセット
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
            }
            i++;
        }

        // 中間層の重みの計算
        for (Neuron n : hiddenLayer) {
            ArrayList<Connection> connections = n.getAllInConnections();
            for (Connection con : connections) {
                double yj = n.getOutput();
                double yi = con.leftNeuron.getOutput();
                double sumKoutputs = 0;
                int j = 0;
                for (Neuron outNeu : outputLayer) {
                    double wjk = outNeu.getConnection(n.id).getWeight();
                    double desiredOutput = (double) expectedOutput[j];
                    double yk = outNeu.getOutput();
                    j++;
                    sumKoutputs = sumKoutputs
                            + (-(desiredOutput - yk) * yk * (1 - yk) * wjk);
                }

                double partialDerivative = yj * (1 - yj) * yi * sumKoutputs;
                double deltaWeight = -learningRate * partialDerivative;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
            }
        }
    }
    //最大ステップ数orエラーに達するまで繰り返す
    void run(int maxSteps, double minError) {
        int i;

        double error = 1;
        for (i = 0; i < maxSteps && error > minError; i++) {
            error = 0;
            for (int p = 0; p < inputs.length; p++) {
                setInput(inputs[p]);

                feedForward();

                output = getOutput();
                resultOutputs[p] = output;

                //２乗誤差の計算
                for (int j = 0; j < expectedOutputs[p].length; j++) {

                    double err = Math.pow(output[j] - expectedOutputs[p][j], 2);
                    error += err;
                }

                backpropagation(expectedOutputs[p]);
            }
        }
        printResult();

        System.out.println("二乗誤差の総和 = " + error);


    }

    void printResult()
    {
        System.out.println("入力値　　　　　　期待結果　　　　　　　実績値");
        for (int p = 0; p < inputs.length; p++) {
            System.out.print("INPUTS: ");
            for (int x = 0; x < layers[0]; x++) {
                System.out.print(inputs[p][x] + " ");
            }

            System.out.print("EXPECTED: ");
            for (int x = 0; x < layers[2]; x++) {
                System.out.print(expectedOutputs[p][x] + " ");
            }

            System.out.print("ACTUAL: ");
            for (int x = 0; x < layers[2]; x++) {
                System.out.print(resultOutputs[p][x] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

}