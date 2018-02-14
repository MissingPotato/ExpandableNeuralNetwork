using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class V05Tester : MonoBehaviour {

	[Header("Initial Config")]

	[Tooltip("the amount of layers that are going to be on the Neural Network")]
	[Range(3, 50)]
	public int layerCount = 3;
	
	[Tooltip("The amount of Neurons that are going to be on each hidden layer")]
	[Range(1, 50)]
	public int neuronCount = 1;
	
	[Range(1, 50)]
	public int inputsCount = 2;

	[Tooltip("The amount of outputs that are going to be avaible")]
	[Range(1, 50)]
	public int outPutCount = 1;

	[Tooltip("Higher = Faster | Inprecise     Lower = Slower | Precise")]
	[Range(0f, 1f)]
	public float learningRate = 0.1f;

	[Space]
	[Header("Debugging")]

	[Tooltip("The target value that the neural network should achieve, for debugging only")]
	[Range(0f, 1f)]
	public float targetValue = 0.5f;

	[Tooltip("This is for debugging only")]
	public bool debugMODE;

	// --------------------------------------- :: Private Variables :: ---------------------------------------

	NNC neuralNetwork;

	float[] testInputs;

	float[] prediction;

	// --------------------------------------- :: Active Functions :: ---------------------------------------

	void Awake()
	{
		neuralNetwork = new NNC ( inputsCount, layerCount, neuronCount, outPutCount ); // Initializes the neural network

		if ( debugMODE )
			neuralNetwork.DebugInfo();

		testInputs = new float[inputsCount];

		for (int i = 0; i < inputsCount; i++)
			testInputs[i] = Random.Range(-1f, 1f);

		

	}

	void Update()
	{
		if ( Input.GetKey ( KeyCode.Return ) ){
			prediction = neuralNetwork.Predict( testInputs );
			for (int i = 0; i < outPutCount; i++)
				Debug.Log("NeuralNetwork Prediction["+ i + "]: " + prediction[i] + "\n SigmoidPrediction: " + neuralNetwork.Sigmoid (prediction[i]) );

			for (int i = 0; i < inputsCount; i++){
			
					testInputs[i] = Random.Range(-1f, 1f);
					Debug.Log("Input[" + i + "]: " + testInputs[i]);
			
			}
		}	
	}


}


public class NNC {


	// --------------------------------------- :: VALUES :: ---------------------------------------

	// Values
	public Layer[] layers; // These include the input and output nodes.
	// the layers countain nodes inside of them, as well as weights so there's no need to externally
	// define these types.
	// If you are going to optimize this code, or make a new version, do not make classes or structs
	// use straight values, make a 3D array of weights instead of nesting them inside of the layers
	// class, as that way it's easier for the processor's cache memory to be fully utilized.

	// Counts
	public int inputsCount = 1; // The amount of inputs
	
	public int layerCount = 3; // The amount of layers we have
	public int neuronCount = 2; // The amount of neurons that are going to be on the hidden layer

	public int outPutsCount = 1; // The amount of the outputs on the last layer, which has 0 weights.

	// --------------------------------------- :: CONSTRUCTOR :: ---------------------------------------

	public NNC (int _inputsCount, int _layerCount, int _neuronCount, int _outPutsCount) {

		// Saving the inputs for later on on the loop functions ( makes things easier )
		inputsCount = _inputsCount;

		layerCount = _layerCount;
		neuronCount = _neuronCount;

		outPutsCount = _outPutsCount;

		// Initializing the layers

		layers = new Layer[layerCount];

		for ( int i = 0; i < layerCount; i++ ) // we set all the layers to have connections as if they are all the same size
			layers[i] = new Layer(neuronCount, neuronCount);
		layers[0] = new Layer ( inputsCount, neuronCount ); // Right here we make the first layer the input layer, with a size of the input and the correct amount of weights
		layers[layerCount - 1] = new Layer ( outPutsCount, 0); // We make the output layer ( the last layer ) have the correct amount of neurons, also 0 weights ( as they are the weights from infront of it )
		layers[layerCount - 2] = new Layer ( neuronCount, outPutsCount ); // We make the second to last layer which has the hidden layer's amount of neurons but with the appropiate weights corresponding to the outputs

	}

	// --------------------------------------- :: FUNCTIONS :: ---------------------------------------

	public float[] Predict (float[] oInputs) {


		//We take in the inputs
		for ( int i = 0; i < inputsCount; i++ )
			layers[0].neuron[i].deltaValue = oInputs[i];

		// Start predicting the first layer's values
		// i = which neuron      j = which input weight
		for ( int i = 0; i < inputsCount; i++ )
			for (int j = 0; j < neuronCount; j++){
				layers[0].neuron[i].value = Sigmoid ( layers[0].neuron[i].deltaValue ); // we calculate the delta value of the inputs
				layers[1].neuron[j].deltaValue += layers[0].neuron[i].value * layers[0].weight[i, j]; // We calculate the delta value of the neuron from the first hidden layer
			}

		for ( int i = 0; i < neuronCount; i++ )
			layers[1].neuron[i].value = Sigmoid ( layers[1].neuron[i].deltaValue );

		// Predicting the hidden layer's results, save the neuron's raw value in deltaValue and the sigmoid of the deltaValue inside of Value!!!!
		// i  =  layer 		j = Neuron		k = Weight
		for ( int i = 2; i < layerCount - 1; i++ ){
			for ( int j = 0; j < neuronCount; j++){
				// Predict the rest of the neurons from the hidden layer only.
				for (int k = 0; k < neuronCount; k++){
					layers[i].neuron[j].deltaValue += layers[i - 1].neuron[k].value * layers[i - 1].weight[k, j];
				}
				layers[i].neuron[j].value = Sigmoid ( layers[i].neuron[j].value );
			}
		}

		// Calculating the output nodes.
		// i = outputNode		j = WhichNeuron from the previous layer
		for ( int i = 0; i < outPutsCount; i++ ){
			for ( int j = 0; j < neuronCount; j++){
				layers[layerCount - 1].neuron[i].deltaValue = layers[layerCount - 2].neuron[j].value * layers[layerCount - 2].weight[j, i]; 
			}
			layers[layerCount - 1].neuron[i].value = Sigmoid ( layers[layerCount - 1].neuron[i].value );
		}
		

		return layers[layerCount - 1].ReturnOutPuts(); // return the array of outputs
	}

	public void Train (float _prediction, float _answer) {

		// Loop thru the whole network, just like in the prediction function
		// use back propagation to recalculate the weights
		// calculate them based on the DELTAVALUE not the value
		// If this works, well, the neural network will be flawless and it's going to be bomb asf boii

	}

	public void DebugInfo () {

		Debug.Log("::DEBUG COMMENCING:: \n [Layer][Node][Weight]");

		for (int i = 0; i < layers[0].neuron.Length; i++)
		{
			Debug.Log("InputNeuron[" + i + "]: " + layers[0].neuron[i].deltaValue);

			for (int j = 0; j < neuronCount; j++)
			{
				Debug.Log("InputWeight[0][" + i + "][" + j + "]: " + layers[1].weight[i,j]);
			}
		}

		for (int i = 1; i < layerCount - 2; i++)
		{
			for (int j = 0; j < neuronCount; j++)
			{
				for (int k = 0; k < neuronCount; k++)
				{
					Debug.Log("Weight[" + i + "][" + j + "][" + k + "]: " + layers[i].weight[j,k]);
				}
			}
		}

		for (int i = 0; i < neuronCount; i++)
		{
			for (int j = 0; j < outPutsCount; j++)
			{
				Debug.Log("OutPutWeight[" + layerCount + "][" + i + "][" + j + "]: " + layers[layerCount - 2].weight[i,j]);
			}
		}

	}

	// --------------------------------------- :: MINI - FUNCTIONS :: ---------------------------------------
	// the "Mini-Functions" category represents all the small functions that help the neural network, these need to
	// be changed to be compatible outside of Unity

	public float Sigmoid ( float _x ) {
		return  1 / ( 1 + Mathf.Exp(-_x) );
	}

	public float SigDer ( float _x ) {
		return Sigmoid ( _x ) * ( 1 - Sigmoid ( _x) );
	}

	// --------------------------------------- :: EXTRA CLASSES :: ---------------------------------------
	// The "Extra Classes" category contains all the sub-classes of the neural network, like the layers and
	// the neurons from these layers, weights are just float values so there's no need for them to have a class

	public class Layer {

		public Neuron[] neuron;
		public float [,] weight;

		/// <summary>
		/// The constructor that automates almost everything!
		/// </summary>
		/// <param name="_neuronAmount">The amount of neurons that are going to be on this particular layer</param>
		/// <param name="_neuronAmountInNextLayer">The amount of connections steming from each of the neurons</param>
		public Layer( int _neuronAmount, int _neuronAmountInNextLayer ){
			neuron = new Neuron[_neuronAmount]; // creating the neurons array
			weight = new float[_neuronAmount, _neuronAmountInNextLayer]; // creating the weights array
			
			for ( int i = 0; i < _neuronAmount; i++ )
				neuron[i] = new Neuron();

			System.Random rand = new System.Random();

			// Looping thru all the weights to randomize them
			for ( int i = 0; i < _neuronAmount; i++ )
				for ( int j = 0; j < _neuronAmountInNextLayer; j++ ){
					weight [ i, j ] = (float)rand.NextDouble(); // randomizing the weight with a random float from 0 to 1
					if ( rand.Next(0, 11) > 5 )
						weight [ i, j ] *= -1; // 50% chances of making the weight negative
				}
		}
	
		/// <summary>
		/// Returns all the outputs in a more friendly format
		/// </summary>
		/// <returns>The outputs, the sigmoid function was not applied!</returns>
		public float[] ReturnOutPuts () {
			
			float[] outPuts = new float[neuron.Length];
			
			for (int i = 0; i < neuron.Length; i++)
			{
				outPuts[i] = neuron[i].deltaValue;
			}

			return outPuts;
		}

	}

	public class Neuron {
		public float value = 0f; // The original value
		public float deltaValue = 0f; // The sigmoid'd value
	}

}