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

	[Tooltip("The NeuralNetwork's target value")]
	[Range(0f, 1f)]
	public float targetValue;

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
		
		if ( Input.GetKey ( KeyCode.Q ) ){
			float pred = neuralNetwork.Predict();
			neuralNetwork.Train(targetValue, pred);
			float newPred = neuralNetwork.Predict();
			Debug.Log("OldPred: " + pred + "   " + neuralNetwork.Sigmoid(pred) + "\n NewPred: " + newPred  + "   " + neuralNetwork.Sigmoid( newPred ) );

			outPut = neuralNetwork.Sigmoid( newPred );

		}
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

					if ( rand.Next(0, 11) > 5)
						weights [ i , j , k ] *= -1;

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

		// i = input node
		for ( int i = 0; i < hiddenNodesCOUNT; i++){ // we are looping thru all the hidden nodes from the first layer
			
			// j = hidden node
			for ( int j = 0; j < inputNodesCOUNT; j++) // Looping thru the hidden nodes
				hiddenNode[0, i] += inputNode[j] * inputWeights[j, i]; // Calculate the prediction
		
			hiddenPred[0, i] = hiddenNode[0, i];
			hiddenNode[0, i] = Sigmoid(hiddenNode[0, i]); // We set the hidden node
		}

		// Calculate the rest of the hidden layer nodes values
		// i = Layer   j = Node   k = weight 
		for ( int i = 1; i < LayerCOUNT; i++ ) // Looping the layers, we start from 1 because of the input layer from above ( already calculated 
			for ( int j = 0; j < hiddenNodesCOUNT; j++ ) {
				for (int k = 0; k < hiddenNodesCOUNT; k++) // Loop thru weights
					hiddenNode[i, j] += hiddenNode[i - 1, k] * weights [ i - 1, k, j ];
			
				hiddenPred[i, j] = hiddenNode[i, j];
				hiddenNode[i, j] = Sigmoid(hiddenNode[i , j]);
			}

		// Calculate the final prediction, this is the output

		for (int i = 0; i < hiddenNodesCOUNT; i++)
		{
			outPutNode += hiddenNode[LayerCOUNT - 1, i] * weights [ LayerCOUNT - 1, i, 0];
		}

		return outPutNode;
	}

	/// <summary>
	/// The training function that will change all the weights according to the answer!
	/// </summary>
	/// <param name="Answer">The answer that the weights will adjust to, they will automatically start backtracing from the front all the way to the back nodes!</param>
	public void Train(float _answer, float _prediction){

		float prediction = _prediction; // We save the prediction just in case
		float cost = 2 * ( Sigmoid ( prediction ) - _answer ); // Calculating the cost for the last layer

		// Last layer's outputs corrected.
		for (int i = 0; i < hiddenNodesCOUNT; i++) // Looping thru the nodes from the last layer
		{
			weights [ LayerCOUNT - 1, i, 0] = weights[ LayerCOUNT - 1, i, 0] - learningRate * ( cost * SigDer ( _answer ) * weights[ LayerCOUNT - 1, i, 0] ) ; // ReCalculating the last layer of weights
		}

		// We start from beginning to end to correct ( well, from the first hidden layer, we deal with the input's layers after.)

		if ( LayerCOUNT > 1 ){ //

		for ( int i = 1; i < LayerCOUNT - 1; i++ ) // Looping the layers, we start from the last one because we want to backtrace our information and update the weights!
			for ( int j = 0; j < hiddenNodesCOUNT; j++ ){ // The node
				for (int m = 0; m < hiddenNodesCOUNT; m++) // Loop thru previous nodes
					for ( int k = 0; k < hiddenNodesCOUNT - 1; k++ ){ // The weight
						
						cost = 2 * (  hiddenPred[i, k] - _answer ); // Recalculating the cost
					
						weights [ i, j, k] = weights [ i, j, k ] - learningRate * ( cost * SigDer ( hiddenPred[i + 1, k] ) * weights [ i, j, k ] );

					}
			}
		
		} // 
	

		// We start fixing the input layer's weights

		// i = input node     j = input node's weight
		for (int i = 0; i < inputNodesCOUNT; i++){ // The amount of input nodes
			for (int j = 0; j < hiddenNodesCOUNT; j++){ // The amount of hidden nodes
				
				cost = 2 * ( hiddenPred [0 , j] - _answer ); // Recalculating the cost
				
				inputWeights[i, j] = inputWeights[i, j] - learningRate * ( cost * SigDer (hiddenPred [0 , j]) * inputWeights[i, j] );
			}

		}


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