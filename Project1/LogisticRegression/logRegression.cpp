//============================================================================
// Name        : logRegression.cpp
// Author      : Kathryn Kingsley KLK170230
// Version     : 1.0
// Copyright   : Spring 2021
// Description : Logistic regression in C++.
//============================================================================

#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <string>
#include <C:\Eigen\Dense>

using Eigen::MatrixXd;
using namespace std;

//for me to follow my code
bool DEBUG= false;

//Function signatures
MatrixXd sigmoid(MatrixXd X);
MatrixXd gradientDescent(MatrixXd weights, MatrixXd labels, MatrixXd data);
MatrixXd prediction(MatrixXd weights, MatrixXd data);
void calculations(MatrixXd predictions, MatrixXd data);

int main(int argc, char** argv) {

	// ***VARIABLES***

	// create input file stream and open
	fstream infile;
	infile.open("titanic_project.csv");

	// create string line to hold each row for parsing
	string line;
	string pclassS, survivedS, xS, sexS, ageS;

	// int to count number of objects in file
	int numObjects = 0;

	// Matrices
	MatrixXd dataFrame(1046,2);
	MatrixXd trainData(900,2);   //matrix of d doubles
	MatrixXd trainLabels(900,1);
	MatrixXd testLabels(146,1);
	MatrixXd predictions(146,1);
	MatrixXd testData(146,2);	// matrix of d doubles
	MatrixXd weights(2,1);
	MatrixXd updatedWeights(2,1);
	weights<< 1 , 1;

	// see if file was opened, return error if not
	if(!infile.is_open()){
		cout<<"Error opening file"<<endl;
		return 1;
	}

	//grab first line with column names
	getline(infile, line);

	// WHILE LOOP TO PUT ITEMS IN TEST OR TRAIN
	while(infile.good()){

		getline(infile, xS, ',');  //grab x

		getline(infile, pclassS, ','); //grab pclass

		getline(infile, survivedS, ','); //grab survived

		getline(infile, sexS, ','); //grab sex

		getline(infile, ageS, '\n'); //grab sex

		dataFrame.row(numObjects)<< stoi(pclassS), stoi(survivedS);
		// first 900 in train
		if(numObjects<=899){
			trainData.row(numObjects)<<  1, stoi(pclassS);
			trainLabels.row(numObjects)<< stoi(survivedS);
		}
		else if (numObjects>899){
			int row = numObjects-900;
			testData.row(row)<< 1, stoi(pclassS);
			testLabels.row(row)<< stoi(survivedS);
		}

		numObjects++;  //increment num objects for next row
	}

	//close file- done with it
	infile.close();

		/*if(DEBUG){
			cout<<trainData.size()/2<<endl;
			cout<<testData.size()/2<<endl;
			cout<<trainData<<endl;
			cout<<testData<<endl;
			cout<<trainLabels<<endl;
			cout<<testLabels<<endl;
		}*/
	auto start = std::chrono::high_resolution_clock::now();
	updatedWeights = gradientDescent(weights, trainLabels,trainData);
	auto stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_sec = stop-start;
	predictions = prediction(updatedWeights, testData);

		if(DEBUG){
			 //cout<<predictions<<endl;
		}

	cout<<"The time taken was: "<< elapsed_sec.count()<<endl;
	calculations(predictions, testLabels);

	return 0;
}

//GRADIENT DESCENT OPTIMIZATION METHOD
MatrixXd gradientDescent(MatrixXd weights, MatrixXd labels, MatrixXd data){
	 double learningRate = 0.001;
	 MatrixXd probVector;
	 MatrixXd error;

	 for(int i=0; i<=50000; i++){
		 probVector = sigmoid(data*weights);
		 	 //if(DEBUG) {cout<<probVector<<endl;}
		 error = labels - probVector;
		 	 //if(DEBUG) {cout<<error<<endl;}
		 weights = weights + learningRate * (data.transpose()*error);

	 }

	 cout<<weights<<endl;

	 return weights;
}

// SIGMOID FUNCTION
MatrixXd sigmoid(MatrixXd X){
	 //if(DEBUG){ cout<<X<<endl; }

	 return 1/(1+(-X).array().exp());

}

//PREDICTION FUNCTION

MatrixXd prediction(MatrixXd weights, MatrixXd data){
	MatrixXd predicted = data*weights; // data is 146x2, weights is 2x1
	MatrixXd probs(predicted.size(),1); // 146 rows and 1 column
	probs = (predicted.array().exp())/(1+(predicted.array().exp()));
	MatrixXd predictions(probs.size(),1);   //will be 146x 1

			if(DEBUG){
				//cout<<predicted.size()<<endl;
				//cout<<probs.size()<<endl;
				//cout<<probs<<endl;
			}
	for(int i; i< probs.size(); i++){
		float x = probs(i, 0);
		if(x>0.5){
			predictions.row(i)<< 1;
		}
		else if (x<=0.5){
			predictions.row(i)<< 0;
		}
			if(DEBUG){
				//cout<<x<<endl;
				//cout<<predictions(i,0)<<endl;
			}
	}

	return predictions;


}

//CALCULATIONS FUNCTION
void calculations(MatrixXd predictions, MatrixXd data){
	// variables
		float tp=0, fp=0, fn=0, tn=0, pred=0, act=0;
		float accuracy, sensitivity, specificity;
		/*cout<<predictions<<endl;
		cout<<data<<endl;
		cout<<predictions.size()<<endl;
		cout<<data.size()<<endl;*/

		for(int i =0; i< predictions.size(); i++){
			pred = predictions(i,0);
			act = data(i,0);
			//cout<<pred<<endl;
			//cout<<act<<endl;

			if(pred == 1){
				if(act == 1){
					tp+=1;
				}
				else{
					fp+=1;
				}
			}
			if(pred == 0){
				if(act == 1){
					fn+=1;
				}
				else{
					tn+=1;
				}
			}
		}

		if(DEBUG){
			cout<<tp<<" "<<tn<<" "<<fp<<" "<<fn<<endl;
		}

		accuracy = (tp+tn)/(tp+tn+fp+fn);
		sensitivity = tp/(tp+fn);
		specificity = tn/(tn+fp);

		cout<<"The accuracy is: "<<accuracy<<endl;
		cout<<"The sensitivity is: "<<sensitivity<<endl;
		cout<<"The specificity is: "<<specificity<<endl;

		return;
}



