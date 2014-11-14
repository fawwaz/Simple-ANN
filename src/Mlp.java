/***************************************************************************
 *
 *  This is an implementation of the multilayer perceptron neural network
 *  written by Gabor Takacs <gtakacs@sze.hu>.
 *
 ***************************************************************************/

import java.io.*;

public class Mlp {
    // Mlp.Neuron
    private class Neuron {
        public double bias;
        public double oldBias;
        public double[] weights;
        public double[] oldWeights;
        public double output;
        public double delta;
    }
    
    // output type constants
    public static int SIGMOID_OUTPUT = 0;
    public static int LINEAR_OUTPUT = 1;
    
    // private data members
    private int _inputType = 0;
    private int _outputType = SIGMOID_OUTPUT;
    private double _learningRate = 0.1;
    private double _momentumFactor = 0.8;
    private Neuron[][] _neurons;
    private double[] _input;
    private double[] _output;
    private double[] _desiredOutput;
    
    // get-set functions
    public int getInputType() { return _inputType; }
    public void setInputType(int type) { _inputType = type; }
    public int getOutputType() { return _outputType; }
    public void setOutputType(int outputType) { _outputType = outputType; }
    public double getLearningRate() { return _learningRate; }
    public void setLearningRate(double learningRate) { _learningRate = learningRate; }
    public double getMomentumFactor() { return _momentumFactor; }
    public void setMomentumFactor(double momentumFactor) { _momentumFactor = momentumFactor; }
    public int getNLayers() { return _neurons.length; }
    public void setNLayers(int nLayers) { _neurons = new Neuron[nLayers][]; }
    public int getLayerSize(int layerIndex) { return _neurons[layerIndex].length; }
    public void setLayerSize(int layerIndex, int layerSize) { _neurons[layerIndex] = new Neuron[layerSize]; }
    public double[] getInput() { return _input; }
    public void setInput(double[] input) { _input = input; }
    public double[] getDesiredOutput() { return _desiredOutput; }
    public void setDesiredOutput(double[] desiredOutput) { _desiredOutput = desiredOutput; }
    public double[] getOutput() { return _output; }     // read-only attribute
    
    /** The logistic sigmoid function. */
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    /** Constructor. */
    public Mlp(int nLayers) {
        setNLayers(nLayers);
    }
    
    /** Constructor. */
    public Mlp(int nLayers, int inputType, int outputType) {
        setNLayers(nLayers);
        _inputType = inputType;
        _outputType = outputType;
    }
    
    /** Constructor. */
    public Mlp(String fileName) {
        loadFromFile(fileName);
    }
    
    /** Connect layers. Must be called after the layer sizes has been set. */
    public void connectLayers() {
        _input = new double[_neurons[0].length];
        _output = new double[_neurons[_neurons.length - 1].length];
        _desiredOutput = new double[_neurons[_neurons.length - 1].length];
        
        for (int l = 0; l < _neurons.length; l++) {
            for (int i = 0; i < _neurons[l].length; i++) {
                _neurons[l][i] = new Neuron();
                if (l > 0) {
                    _neurons[l][i].weights = new double[_neurons[l - 1].length];
                    _neurons[l][i].oldWeights = new double[_neurons[l - 1].length];
                }
            }
        }
    }
    
    /** Perform one learning step with the backpropagation algorithm using momentum method. */
    public void learn() {
        int last = _neurons.length - 1;
        for (int l = last; l >= 1; l--) {
            for (int i = 0; i < _neurons[l].length; i++) {
                Neuron ni = _neurons[l][i];
                
                // compute delta
                double ei = 0;
                if (l == last) {
                    double derivative = 1;
                    if (_outputType == SIGMOID_OUTPUT) derivative = ni.output * (1 - ni.output);
                    
                    ei = _desiredOutput[i] - ni.output;
                    ni.delta = ei * derivative;
                }
                else {
                    for (int k = 0; k < _neurons[l + 1].length; k++)
                        ei += _neurons[l + 1][k].oldWeights[i] * _neurons[l + 1][k].delta;
                    
                    ni.delta = ei * ni.output * (1 - ni.output);
                }

                // modify bias weight
                double momentum = _momentumFactor * (ni.bias - ni.oldBias);
                ni.oldBias = ni.bias;
                double deltaW = _learningRate * ni.delta * ni.bias;
                ni.bias += deltaW + momentum;
                
                // modify other weights
                for (int j = 0; j < _neurons[l - 1].length; j++) {
                    momentum = _momentumFactor * (ni.weights[j] - ni.oldWeights[j]);
                    ni.oldWeights[j] = ni.weights[j];
                    deltaW = _learningRate * ni.delta * _neurons[l - 1][j].output;
                    ni.weights[j] += deltaW + momentum;
                }
            }
        }
    }
    
    /** Update the activation of input neurons from the input array. */
    private void updateInputNeurons() {
        for (int i = 0; i < _neurons[0].length; i++)
            _neurons[0][i].output = _input[i];
    }
    
    /** Update output array from the activation of output neurons. */
    private void updateOutputArray() {
        int last = _neurons.length - 1;
        for (int i = 0; i < _neurons[last].length; i++)
            _output[i] = _neurons[last][i].output;
    }
    
    /** Propagate activation through the network. */
    public void run() {
        updateInputNeurons();
        
        // for hidden layers        
        int last = _neurons.length - 1;
        for (int l = 1; l < last; l++) {
            for (int i = 0; i < _neurons[l].length; i++) {
                _neurons[l][i].output = _neurons[l][i].bias;
                for (int j = 0; j < _neurons[l - 1].length; j++)
                    _neurons[l][i].output += _neurons[l - 1][j].output * _neurons[l][i].weights[j];
                    
                _neurons[l][i].output = sigmoid(_neurons[l][i].output);
            }
        }
        
        // output layer
        for (int i = 0; i < _neurons[last].length; i++) {
            _neurons[last][i].output = _neurons[last][i].bias;
            for (int j = 0; j < _neurons[last - 1].length; j++)
                _neurons[last][i].output += _neurons[last - 1][j].output * _neurons[last][i].weights[j];
                
            if (_outputType == SIGMOID_OUTPUT)
                _neurons[last][i].output = sigmoid(_neurons[last][i].output);
        }

        updateOutputArray();
    }
    
    /** Randomize weights. Should be called before the learning process. */
    public void randomizeWeights() {
        for (int l = 1; l < _neurons.length; l++) {
            for (int i = 0; i < _neurons[l].length; i++) {
                _neurons[l][i].bias = Math.random() * 0.2 - 0.1;
                _neurons[l][i].oldBias = _neurons[l][i].bias;
            
                for (int j = 0; j < _neurons[l - 1].length; j++) {
                    _neurons[l][i].weights[j] = Math.random() * 0.2 - 0.1;
                    _neurons[l][i].oldWeights[j] = _neurons[l][i].weights[j];
                }
            }
        }
    }
    
    /** Save neural network to file. */
    public void saveToFile(String fileName) {
        try {
            FileOutputStream fileStream = new FileOutputStream(fileName);
            DataOutputStream dataStream = new DataOutputStream(fileStream);

            dataStream.writeInt(_inputType);
            dataStream.writeInt(_outputType);
            dataStream.writeDouble(_learningRate);
            dataStream.writeDouble(_momentumFactor);
            
            // write out structure
            dataStream.writeInt(_neurons.length);           // number of layers
            for (int l = 0; l < _neurons.length; l++)
                dataStream.writeInt(_neurons[l].length);    // layer size
            
            // write out weights
            for (int l = 1; l < _neurons.length; l++) {
                for (int i = 0; i < _neurons[l].length; i++) {
                    dataStream.writeDouble(_neurons[l][i].bias);
                    for (int j = 0; j < _neurons[l - 1].length; j++)
                        dataStream.writeDouble(_neurons[l][i].weights[j]);
                }
            }
            
            fileStream.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    /** Load neural network from file. */
    public void loadFromFile(String fileName) {
        try {
            FileInputStream fileStream = new FileInputStream(fileName);
            DataInputStream dataStream = new DataInputStream(fileStream);

            _inputType = dataStream.readInt();
            _outputType = dataStream.readInt();
            _learningRate = dataStream.readDouble();
            _momentumFactor = dataStream.readDouble();
            
            // read network structure
            _neurons = new Neuron[dataStream.readInt()][];      // number of layers
            for (int l = 0; l < _neurons.length; l++) {
                _neurons[l] = new Neuron[dataStream.readInt()]; // layer size
                for (int i = 0; i < _neurons[l].length; i++)
                    _neurons[l][i] = new Neuron();
            }
            
            connectLayers();
            
            // read weights
            for (int l = 1; l < _neurons.length; l++) {
                for (int i = 0; i < _neurons[l].length; i++) {
                    _neurons[l][i].bias = dataStream.readDouble();
                    _neurons[l][i].oldBias = _neurons[l][i].bias;
                    for (int j = 0; j < _neurons[l - 1].length; j++) {
                        _neurons[l][i].weights[j] = dataStream.readDouble();
                        _neurons[l][i].oldWeights[j] = _neurons[l][i].weights[j];
                    }
                }
            }
            
            fileStream.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
}
