using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetworkB : MonoBehaviour {


	// --------------- Variables -----------------

	#region Variables
	
	[Header("Basic Options")]

	[Tooltip("The input size")]
	[Range(1, 20)]
	public int inputSize = 1;
	
	[Tooltip("The number of hidden layers that are going to be avaible, 0 = disabled")]
	[Range(0, 100)]
	public int hiddenLayers = 0;

	[Tooltip("These can't be disabled, they are how many nodes to be used on the hidden layers")]
	[Range(1, 100)]
	public int hiddenNodes = 1;

	[Tooltip("The learning rate of the neural network, smaller = more precise, bigger = faster but less precise")]
	[Range(0f, 1f)]
	public float learningRate = 0.1f;

/*	To do for version C ( More outputs )

	[Tooltip("The neural network's output")]
	[Range(1, 20)]
	public int outPutNodes = 1;

*/
	public NNB neuralNetwork;
	
	public float outPut;

	#endregion

	// --------------- Active Functions -----------------
	
	#region Active Functions
	
	void Awake () {
		NeuralDebug();		

	}
	
	void Update () {
		
		if ( Input.GetKeyDown ( KeyCode.Q ) )
			neuralNetwork.Train(1f, neuralNetwork.Predict());

	}
	
	#endregion
	

	public void NeuralDebug(){
		print("Debugging Commenced: ");
		neuralNetwork = new NNB ( inputSize, hiddenLayers, hiddenNodes, learningRate);
		
		outPut = neuralNetwork.outPutNode;
	}
	
}

public class NNB {

	// ----------------------------------- Counters -----------------------------------
	public int inputNodesCOUNT; // The amount of input nodes we have

	public int LayerCOUNT; // The amount of layers we have
	public int hiddenNodesCOUNT; // The amount of hidden nodes we have inside of each layer

	
	// ----------------------------------- Values -----------------------------------

	public float learningRate;

	public float[] inputNode;
	public float[,] inputWeights;

	public float[,] hiddenNode;
	public float [,] hiddenPred;
	public float[,,] weights;

	public float outPutNode; // The output

	public NNB ( int _inputNodesCount, int _LayersCount, int _hiddenNodesCount, float _learningRate ){

		// Setting the values for the variables.
		inputNodesCOUNT = _inputNodesCount;
		LayerCOUNT = _LayersCount;
		hiddenNodesCOUNT = _hiddenNodesCount;
		learningRate = _learningRate;

		// Construct the sizes

		inputNode = new float[inputNodesCOUNT]; // The input nodes are unique
		inputWeights = new float[inputNodesCOUNT, hiddenNodesCOUNT];

		hiddenNode = new float[LayerCOUNT, hiddenNodesCOUNT]; // The hidden nodes are sorted LAYER : NODE
		hiddenPred = new float[LayerCOUNT, hiddenNodesCOUNT]; // The predicted values by each of the nodes, these are used in the training of the Neural Network
		weights = new float [LayerCOUNT, hiddenNodesCOUNT, hiddenNodesCOUNT];

		// Initializing weights with random values

		System.Random rand = new System.Random();

		for (int i = 0; i < LayerCOUNT; i++)
		{
			for (int j = 0; j < hiddenNodesCOUNT; j++)
			{
				for (int k = 0; k < hiddenNodesCOUNT; k++)
				{
					weights [ i , j , k ] = (float)rand.NextDouble();
				}
			}
		}

		// Initialize the values of the nodes randomly :: DEBUG ::
		// :: DEBUG ::

		rand = new System.Random();

		for (int i = 0; i < inputNodesCOUNT; i++)
		{
			inputNode[i] = (float)rand.NextDouble();
			Debug.Log("Input Node[" + i + "]: " + inputNode[i]);
		}

		Debug.Log("Finished random input initialization!");

		Debug.Log("Pred: " + Sigmoid( Predict() ) );
		Debug.Log("RawPred: " + Predict() );
		Debug.Log("This prediction is not based on any kind of target!");

		// :: DEBUG ::
	}

	/// <summary>
	/// This function will show the Neural Network's prediction, returns a nromal value not SIGMOID
	/// </summary>
	/// <returns>Prediction ( Not Rounded / Not Sigmoid'd )</returns>
	public float Predict () {

		// We pass the input to the first hidden layer (if there is one)
		
		if ( LayerCOUNT < 1 ) // Check if we can predict using hidden layers
			return PredictSimple(); // This will be ran instead of this function because there isn't a hidden layer


		// Calculate the first hidden layer

		// i = Node
		for ( int i = 0; i < inputNodesCOUNT; i++){ // we are looping thru all the hidden nodes from the first layer
			
			// j = Weight
			for ( int j = 0; j < inputNodesCOUNT; j++) // Looping all the weights
				hiddenNode[0, i] += inputNode[i] * inputWeights[i, j]; // Calculate the prediction
		
			hiddenPred[0, i] = hiddenNode[0, i];
			hiddenNode[0, i] = Sigmoid(hiddenNode[0, i]); // We set the hidden node
		}

		// Calculate the rest of the hidden layer nodes values

		// i = Layer   j = Node   k = weight   m = second node
		for ( int i = 1; i < LayerCOUNT; i++ ) // Looping the layers, we start from 1 because of the input layer from above ( already calculated 
			for ( int j = 0; j < hiddenNodesCOUNT; j++ ) {
				for (int m = 0; m < hiddenNodesCOUNT; m++) // Loop thru previous nodes
					for (int k = 0; k < hiddenNodesCOUNT; k++) // Loop thru weights
						hiddenNode[i, j] += hiddenNode[i - 1, m] * weights [ i - 1, m, k ];
				
				hiddenPred[i, j] = hiddenNode[i, j];
				hiddenNode[i, j] = Sigmoid(hiddenNode[i , j]);
			}

		// Calculate the final prediction, this is the output

		for (int i = 0; i < hiddenNodesCOUNT; i++)
		{
			outPutNode += hiddenNode[LayerCOUNT - 1, i] * weights [ LayerCOUNT - 1, i, i];
		}

		return outPutNode; // Placeholder value

	}

	/// <summary>
	/// The training function that will change all the weights according to the answer!
	/// </summary>
	/// <param name="Answer">The answer that the weights will adjust to, they will automatically start backtracing from the front all the way to the back nodes!</param>
	public void Train(float _answer, float _prediction){

		float prediction = _prediction; // We save the prediction just in case

		float cost = 2 * ( Sigmoid ( prediction ) - _answer ); // Calculating the cost for the last layer

		for (int i = 0; i < hiddenNodesCOUNT; i++) // Looping thru the nodes from the last layer
		{
			weights [ LayerCOUNT - 1, i, i] = weights[ LayerCOUNT - 1, i, i] - learningRate * ( cost * SigDer ( _answer ) * weights[ LayerCOUNT - 1, i, i] ) ; // ReCalculating the last layer of weights
		}

		// We start backtracing, calculating the second to last, the third to last, ( and so on ) layer's weights

		for ( int i = 1; i < LayerCOUNT; i++ ) // Looping the layers, we start from 1 because of the input layer from above ( already calculated 
			for ( int j = 0; j < hiddenNodesCOUNT; j++ ) {
				for (int m = 0; m < hiddenNodesCOUNT; m++) // Loop thru previous nodes
					for (int k = 0; k < hiddenNodesCOUNT; k++) // Loop thru weights
						hiddenNode[i, j] += hiddenNode[i - 1, m] * weights [ i - 1, m, k ];
				
				hiddenPred[i, j] = hiddenNode[i, j];
				hiddenNode[i, j] = Sigmoid(hiddenNode[i , j]);
			}


		for ( int i = LayerCOUNT; i > 1; i-- )
			for ( int j = hiddenNodesCOUNT; j >= 0; j-- )
				for ( int m = hiddenNodesCOUNT; m >= 0; m-- )
					for ( int k = 0; k < hiddenNodesCOUNT; k++ )
						// weights[i, j, k] = 


	}

    private float PredictSimple()
    {
        throw new NotImplementedException();
    }

	public float SigDer( float _x ){
		return Sigmoid(_x) * ( 1 - Sigmoid(_x) );
	}

	public float Sigmoid( float _x ){
		return 1 / ( 1 + Mathf.Exp(-_x) );
	}

}