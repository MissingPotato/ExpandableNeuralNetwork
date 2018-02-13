using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StagingNeuralNetworkA : MonoBehaviour {


	// --------------- Variables -----------------

	#region Variables
	
	[Range(1, 100)]
	public int inputSize;

	[Range(1, 50)]
	public int hiddenLayerSize;

	[Range(0f, 1f)]
	public float learningRate;

	[Space]

	[Range(1, 500)]
	public int trainAmount;

	NNA neuralNetowrk;

	#endregion

	// --------------- Active Functions -----------------
	
	#region Active Functions
	
	void Awake () {
		neuralNetowrk = new NNA(inputSize, hiddenLayerSize, learningRate);
	}//												     LearningRate
	
	void Update () {
		
		if ( Input.GetKeyDown(KeyCode.Return))
			neuralNetowrk.Train( 1f , trainAmount );
			//				Target	 TrainAmount
	}
	
	#endregion
	
	// --------------- Sleeper Functions -----------------
	
	#region Sleeper Functions
	
	public void InitializeNetwork() {

		

	}
	
	#endregion
	
	
}

public class NNA {

	#region Variables
	
	public float LearningRate;

	public float[] inputNodes;
	public float[,] hiddenNodes;

	public int hiddenlayers;

	public float[,,] weights;
	public float[] outPutWeights;

	#endregion

	public NNA(int _inputSize, int _hiddenlayerSize, float _learningRate){
																							// I  w  H  w  O

		inputNodes = new float[_inputSize]; // Create the input layers						// 0  -  0  -  o
		hiddenNodes = new float[_hiddenlayerSize, _inputSize]; // Create the hidden layers  // 0  -  0  / 
		hiddenlayers = _hiddenlayerSize;

		weights = new float[_hiddenlayerSize, _inputSize, _inputSize];
		outPutWeights = new float[_inputSize];

		LearningRate = _learningRate;

		System.Random rand = new System.Random(System.DateTime.Now.Millisecond);

		Debug.Log("Randomized Weights: \n");

		// Randomize weights
		for (int i = 0; i < _hiddenlayerSize; i++)
			for (int j = 0; j < _inputSize; j++)
				for (int k = 0; k < _inputSize; k++){
					weights[i,j,k] = (float)rand.NextDouble();
					// Debug.LogAssertion("Weight[" + i + "][" + j + "][" + k + "]: " + weights [ i , j , k ] );
				}

		for (int i = 0; i < _inputSize; i++)
			inputNodes[i] = (float)rand.NextDouble();


		for (int i = 0; i < _inputSize; i++)
			outPutWeights[i] = (float)rand.NextDouble();

		Debug.Log("Finished Initialization... \n");
		Debug.Log("\n");

	}

	#region Functions
		
		/// <summary>
		/// The neural network's prediction, doesn't change any variables, just predicts
		/// </summary>
		/// <returns>The prediction without sigmoid, you have to do that manually</returns>
		public float Predict(){

			float pred = 0;
			float tempPred;

			// Calculating the first forward feed, the first hidden layer fed.
			for (int i = 0; i < inputNodes.Length; i++){ // I'm pretty god damn sure this works, so don't worry
				tempPred = 0;
				for (int j = 0; j < inputNodes.Length; j++)
					tempPred += inputNodes[i] * weights[0, i, j]; 

				tempPred = Sigmoid(tempPred);
				hiddenNodes[0, i] = tempPred;
			}
			
			// Feeding the rest of the hidde layer
			tempPred = 0;
			for (int i = 1; i < hiddenlayers; i++){ // Which layer are we on?
				for (int j = 0; j < inputNodes.Length; j++){ // What node are we on?
					tempPred = 0;

					for (int k = 0; k < inputNodes.Length; k++) // Which weight are we on?
						tempPred += hiddenNodes[ i - 1 , j ] * weights[ i , j , k ];

					tempPred = Sigmoid(tempPred);

					hiddenNodes[i,j] = tempPred;
				}
			}
			
			for (int i = 0; i < inputNodes.Length; i++){ // Which node from the last hidden layer are we on?
				pred += hiddenNodes[ hiddenlayers - 1 , i ] * outPutWeights[i];
				
			}

			// pred = Sigmoid(pred);
			return pred;
		}

		public void CorrectWeights(float _pred, float _answer){

			float cost = 2 * ( Sigmoid(_pred) - _answer );

			for (int i = 0; i < hiddenlayers; i++)
			for (int j = 0; j < inputNodes.Length; j++)
				for (int k = 0; k < inputNodes.Length; k++){
					
					weights[i,j,k] = weights[ i , j , k ] - LearningRate * ( cost * Sigder(_pred) * weights[ i , j , k ] );

					// Debug.LogAssertion("Corrected Weight[" + i + "][" + j + "][" + k + "]: " + weights [ i , j , k ] );
				}

		}

		/// <summary>
		/// Calculates the derivate of sigmoid
		/// </summary>
		/// <param name="_x">power of the Exponent</param>
		/// <returns>returns anything between 0 and 1</returns>
		public float Sigder(float _x){
			return Sigmoid(_x) * ( 1 - Sigmoid(_x) );
		}

		/// <summary>
		/// Calculates the sigmoid
		/// </summary>
		/// <param name="_x">Exponent's power</param>
		/// <returns>Retruns from anything from 0 to 1</returns>
		public float Sigmoid(float _x){
			return ( 1 / ( 1 + Mathf.Exp(-_x) ) );
		}

		public void Train(float _answer, int _trainAmount){
			float pred = Predict();

			Debug.Log("Pred: " + Sigmoid(pred) + "\n NakedPred: " + pred);
			Debug.Log("Training...");

			for (int i = 0; i < _trainAmount; i++)
				CorrectWeights(pred, _answer);

			Debug.Log("Training done!");
		}


	#endregion


}