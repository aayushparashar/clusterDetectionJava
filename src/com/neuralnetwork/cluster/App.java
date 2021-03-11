package com.neuralnetwork.cluster;

public class App {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		float[][] input = new float[][] { new float[] { 0.1f, 0.2f }, new float[] { 0.3f, 0.2f },
				new float[] { 0.15f, 0.58f }, new float[] { 0.45f, 0.7f }, new float[] { 0.4f, 0.9f },

				// GREEN CIRCLES 2 -> (0,1,0)
				new float[] { 0.4f, 1.2f }, new float[] { 0.45f, 0.95f }, new float[] { 0.42f, 1f },
				new float[] { 0.5f, 1.1f }, new float[] { 0.52f, 1.45f },

				// BLUE CIRCLES 3 -> (0,0,1)
				new float[] { 0.6f, 0.2f }, new float[] { 0.75f, 0.7f }, new float[] { 0.9f, 0.34f },
				new float[] { 0.85f, 0.76f }, new float[] { 0.8f, 0.34f } };
		float[][] output = new float[][] { new float[] { 1, 0, 0 }, new float[] { 1, 0, 0 }, new float[] { 1, 0, 0 },
				new float[] { 1, 0, 0 }, new float[] { 1, 0, 0 }, new float[] { 0, 1, 0 }, new float[] { 0, 1, 0 },
				new float[] { 0, 1, 0 }, new float[] { 0, 1, 0 }, new float[] { 0, 1, 0 }, new float[] { 0, 0, 1 },
				new float[] { 0, 0, 1 }, new float[] { 0, 0, 1 }, new float[] { 0, 0, 1 }, new float[] { 0, 0, 1 } };
		NeuralNetwork nn = new NeuralNetwork(2, 4, 3);
		for (int i = 0; i < Constants.ITERATIONS; i++) {
			for (int j = 0; j < input.length; j++) {
				nn.train(input[j], output[j], Constants.learningRate, Constants.momentum);
			}
		}
		float[] result = nn.run(new float[] { 0.61f, 2.12f });
		System.out.print(result[0] + " : " + result[1] + " : " + result[2]);
	}

}
