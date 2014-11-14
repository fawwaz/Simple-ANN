
public class MainMlp {

	/**
	 * @Author Fawwaz
	 * The Re-documented ANN usage of xxx's java code
	 */
	public static void main(String[] args) {
		
		/* Construct The Multi Layer Perceptron
		 * There is 3 ways to construct it.
		 * 1. Passing Integer representing the total layer of the MLP (including input and output layer)
		 * 		Using this method, you need to call setInputType & setOutputType Manually
		 * 2. Passing String representing the filename that contain MLP configuration in such a format
		 * 		Using this method, you dont need to call the rest of function configuring the Mlp topology
		 * 3. Passing 3 Integer
		 * 		- First Number representing total layer of the MLP
		 * 		- Second Number representing the activation function of input layer, if you want to use sigmoid function just pass 1 if you want to use Step activation function just pass 0
		 * 		- Third Number representing the activation function of output layer. Its behave like the second parameter
		 *  	Using This method you DON'T need to call setinputType & setoutput type 
		 * */
		Mlp mlp = new Mlp(3);  	// Create a 3 layer Mlp, Input layer, One Hidden Layer, and Output Layer
		mlp.setInputType(0); 	// I am using Sigmoid activation function for the input layer
		mlp.setOutputType(0);	// I am using Sigmoid activation function for the output layer
		
		mlp.setLearningRate(0.2);	// Learning Rate 0.2
		mlp.setMomentumFactor(0.3);	// Momentum factor 0.3, let it be empty if you dont want to use momentum
		
		mlp.setLayerSize(0, 3);		// Set the input layer (always be zero) containing 3 Neuron
		mlp.setLayerSize(1, 4);		// Set the hidden layer containing 4 neuron too
		mlp.setLayerSize(2, 1);		// Set the output layer containing 1 Neuro		
		mlp.connectLayers(); 		// You should call this function manually after you configure the number of Neuron inside every layer in the Mlp

		
		/* The Training Set.
		 * See, that the size of attribute should match the size of first layer [mlp.setLayersize(0,XX)]; 
		 * The Dataset I use in this case is just to show the and function of 3 operand with 8 tuple
		 * */
		double[][] trainingset	= {	{ 0, 0, 0},
									{ 0, 0, 1},
									{ 0, 1, 0},
									{ 0, 1, 1},
									{ 1, 0, 0},
									{ 1, 0, 1},
									{ 1, 1, 0},
									{ 1, 1, 1}};
		
		double[][] desiredoutput = {{0},
									{0},
									{0},
									{0},
									{0},
									{0},
									{0},
									{1},};
		
		// Training 
		int epoch = 1000000;
		for (int i = 0; i < epoch; i++) {
			for (int j = 0; j < trainingset.length; j++) {
				mlp.setInput(trainingset[j]);
				mlp.setDesiredOutput(desiredoutput[j]);
				mlp.learn();
			}
		}
		
		// Classifying
		double[] testSet = {0,0,1};
		mlp.setInput(testSet);
		mlp.run();
		double[] output  = mlp.getOutput();
		
		// Showing all the 
		for (int i = 0; i < output.length; i++) {
			System.out.println(output[i]);
		}
	}

}
