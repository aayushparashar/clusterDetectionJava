package com.neuralnetwork.backprop;

public class App {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		float[][] input = new float[][] { new float[] { 0, 0 }, new float[] { 0, 1 }, new float[] { 1, 0 },
				new float[] { 1, 1 } };
		float[][] output = new float[][] { new float[] { 0 }, new float[] { 1 }, new float[] { 1 }, new float[] { 0 } };
		NeuralNetwork nn = new NeuralNetwork(2, 3, 1);
		for (int i = 0; i < Constants.ITERATIONS; i++) {
			for (int j = 0; j < input.length; j++) {
				nn.train(input[j], output[j], Constants.learningRate, Constants.momentum);
			}
			System.out.println("Iteration no: " + (i + 1));
			System.out.println();
			for (int j = 0; j < input.length; j++) {
				System.out.printf("%.1f, %.1f -> %.3f\n", input[j][0], input[j][1], nn.run(input[j])[0]);
			}
			System.out.println();
		}
	}

}
