//============================================================================
// Name        : naiveBayes.cpp
// Author      : Kathryn Kingsley KLK170230
// Version     : 1.0
// Copyright   : Spring 2021
// Description : Naive Bayes in C++.
//============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <math.h>
#include <iomanip>
#include <string>
#include <C:\Eigen\Dense>

using Eigen::MatrixXd;
using namespace std;

//function signatures
MatrixXd getLH(MatrixXd X, string type, int length, float survived);
MatrixXd mean( MatrixXd X,int lived);
MatrixXd variances( MatrixXd X,MatrixXd means,int lived);
float calc_ageLH(float age, float mean, float var);
MatrixXd toPreds(MatrixXd rawProbs);
void calcs(MatrixXd predictions, MatrixXd data);
MatrixXd calcRawProbs(MatrixXd pclass, MatrixXd sex, MatrixXd age, MatrixXd LHclass,
		MatrixXd LHsex, MatrixXd prior, MatrixXd mean, MatrixXd var);

//for me to follow my code
bool DEBUG= true;

int main() {

	// ***VARIABLES***

	// create input file stream and open
	fstream infile;
	infile.open("titanic_project.csv");

	// create string line to hold each row for parsing
	string line;
	string pclassS, survivedS, xS, sexS, ageS;

	// int to count number of objects in file
	int numObjects = 0;
	float survivedTrain=0;
	float notSurvivedTrain=0;

	//Matrices
	MatrixXd trainData(900,4);  //training set
	MatrixXd testData(146,4);  //testing set
	MatrixXd apriori(1,2);  //priors
	//MatrixXd pclass(2,3);  //class
	MatrixXd pclassTest(146,1);
	MatrixXd pclassLH(2,3);  //pclass likelihoods
	MatrixXd sexTrain(1,2);
	MatrixXd sexTest(146,1);
	MatrixXd sexLH(2,2);    //sex likelihoods
	MatrixXd ageTrain(900,1);
	MatrixXd ageTest(146,1);
	MatrixXd ageLH(2,2);
	MatrixXd trainLabels(900,1);
	MatrixXd testLabels(146,1);
	MatrixXd predictions(2,2);  //to store confusion matrix
	MatrixXd testPreds(146,1);   //0 or 1 predictions on test data

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

		getline(infile, survivedS, ','); //grab survived and check for apriori

		getline(infile, sexS, ','); //grab sex

		getline(infile, ageS, '\n'); //grab age
			//age.row(numObjects)<<stoi(ageS);

		//TRAIN DATA
		if(numObjects<=899){
				trainData.row(numObjects)<<  stoi(survivedS), stoi(pclassS), stoi(sexS), stoi(ageS);
				if(survivedS=="0"){
							notSurvivedTrain+=1;
				}
				if(survivedS=="1"){
							survivedTrain+=1;
				}
				trainLabels.row(numObjects)<< stoi(survivedS);
		}
		else if (numObjects>899){
				int row = numObjects-900;
				testData.row(row)<< stoi(survivedS), stoi(pclassS), stoi(sexS), stoi(ageS);
				testLabels.row(row)<< stoi(survivedS);
				pclassTest.row(row)<< stoi(pclassS);
				sexTest.row(row)<< stoi(sexS);
				ageTest.row(row)<< stoi(ageS);

		}

		numObjects++;  //increment num objects for next row
	}

	//get apriori
	//START CLOCK
	auto start = std::chrono::high_resolution_clock::now();
	apriori<<(notSurvivedTrain/900),(survivedTrain/900);
			if(DEBUG){cout<<apriori<<endl;}

	//get class likelihood
	pclassLH=getLH(trainData, "class", 900, survivedTrain);
				if(DEBUG){cout<<pclassLH<<endl;}

	//get sex likelihood
	sexLH=getLH(trainData, "sex", 900, survivedTrain);
			if(DEBUG){cout<<sexLH<<endl;}

		//get mean and variance for ages
	MatrixXd ageMeans=mean(trainData, survivedTrain);
	MatrixXd ageVars=variances(trainData, ageMeans, survivedTrain);
	//STOP CLOCK
	auto stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_sec = stop-start;

	//get the raw probabilities using the model
	MatrixXd testProbs = calcRawProbs(pclassTest, sexTest, ageTest, pclassLH, sexLH, apriori, ageMeans, ageVars);
			//cout<<testProbs<<endl;

	//create predicted survival/perished matrix
	testPreds = toPreds(testProbs);

	//create confusion matrix
	calcs(testPreds, testLabels);

	//output metrics
	cout<<"The time taken was: "<< elapsed_sec.count()<<endl;
	return 0;
} //end of main

// Prediction receives a 146x2 matrix of raw probabilities and returns a matrix of classified observations
MatrixXd toPreds(MatrixXd x){
	MatrixXd testPreds(146,1);
	float die=0, live=0;
	int size = x.size()/2;
	//cout<<size<<endl;
	for(int i =0; i<size; i++){
		die= x(i,0);
		live = x(i,1);
		if(live>die){
			testPreds.row(i)<<1;
		}
		if(die>live){
			testPreds.row(i)<<0;
		}
		//cout<<x.row(i)<<endl;
		//cout<<testPreds.row(i)<<endl;

	}

	return testPreds;
}

// Receives a matrix, type, length, and number of survived for train data and returns a matrix
// of likelihoods
MatrixXd getLH(MatrixXd X, string type, int length, float survived){
	MatrixXd likelihood;
	float wL=0, mL=0, wD=0, mD=0;
	float c1L=0, c2L=0, c3L=0, c1D=0, c2D=0, c3D=0;
	float died=length-survived;

	//sex likelihood
	if(type=="sex"){
		likelihood.resize(2,2);
		for(int i=0; i<length; i++){
			float lived = X(i,0);
			float gender = X(i,2);
			if(gender ==0){
				if(lived==0){
					wD+=1;
				}
				if(lived==1){
					wL+=1;
				}
			}
			if(gender ==1){
				if(lived==0){
					mD+=1;
				}
				if(lived==1){
					mL+=1;
				}
			}
		}
		likelihood << (wD/died),(mD/died),(wL/survived),(mL/survived);
	}
	if(type=="class"){
		likelihood.resize(2,3);
		for(int i=0; i<length; i++){
			float lived = X(i,0);
			float c = X(i,1);
			if(c ==1){
				if(lived==0){
					c1D+=1;
				}
				if(lived==1){
					c1L+=1;
				}
			}
			if(c ==2){
				if(lived==0){
					c2D+=1;
				}
				if(lived==1){
					c2L+=1;
				}
			}
			if(c ==3){
				if(lived==0){
					c3D+=1;
				}
				if(lived==1){
					c3L+=1;
				}
			}

		}
		likelihood << (c1D/died),(c2D/died),(c3D/died),(c1L/survived),(c2L/survived),(c3L/survived);
	}

	return likelihood;
} //end of getLH

// Receives matrices for pclass, sex, age, likelihoods for class and sex, the priors, age mean, and age
// variances, and returns a matrix of raw probabilities for test data.
MatrixXd calcRawProbs(MatrixXd pclass, MatrixXd sex, MatrixXd age, MatrixXd LHclass,
						MatrixXd LHsex, MatrixXd prior, MatrixXd mean, MatrixXd var){
	//variables
	float numS=0, numP=0, denom =0;
	int size = pclass.size();
	MatrixXd probs (size,2);
	for (int i =0; i<size; i++){
		//cout<<LHclass(0, pclass(i,0)-1)<<endl;
		//cout<<LHsex(0, sex(i,0))<<endl;
		//cout<<prior(0,0)<<endl;
		//cout<<setprecision(9)<<calc_ageLH(age(i,0),mean(0,0),var(0,0))<<endl;
		//cout<<age(i,0)<<endl;
		//cout<<mean(0,0)<<endl;
		//cout<<var(0,0)<<endl;
		numS=LHclass(1, pclass(i,0)-1)*LHsex(1,sex(i,0))*prior(0,1)*calc_ageLH(age(i,0),mean(0,1),var(0,1));
		numP=LHclass(0, pclass(i,0)-1)*LHsex(0,sex(i,0))*prior(0,0)*calc_ageLH(age(i,0),mean(0,0),var(0,0));
		denom= numS+numP;
		//cout<<numS<<endl;
		//cout<<numP<<endl;
		//cout<<denom<<endl;
		probs.row(i)<<(numP/denom),(numS/denom);
		//cout<<setprecision(5)<<probs.row(i)<<endl;

	}
	return probs;
}//end of calcRawProbs


// Takes age, mean, and variance of 1 observation and returns the likelihood percentages
// Uses MLE
float  calc_ageLH(float age, float mean, float var){
	//cout<<age<<endl;
	//cout<<mean<<endl;
	//cout<<var<<endl;
	float power= pow((age-mean),2);
	float pt2 = -(power/(2*var));

	return (1/sqrt(2*M_PI*var))*(exp(pt2));
}


// Receives a 900x1 matrix and number of survived and returns a 1x2 matrix
// of means for perished and survive
MatrixXd mean( MatrixXd x,int lived)
{
	MatrixXd theMeans(1,2);
	int size = x.size()/4;
	float died =size -lived;
	float ageL=0, ageD=0;
	for(int i =0; i< size; i++){
		float lived = x(i,0);
		float a = x(i,3);
		if(lived ==0){
			ageD+=a;
		}
		if(lived ==1){
			ageL+=a;
		}
	}
	theMeans << (float)(ageD/died),(float)(ageL/lived);
			//if(DEBUG){cout<<theMeans<<endl;};
	return theMeans;
}//end of mean

// Variances takes a 900x1 matrix of ages, means, and number of survivors and returns
// a 1x2 matrix of variances for perished and survived
MatrixXd variances( MatrixXd x, MatrixXd means, int lived){

	//variables
	MatrixXd theVars(1,2);
	int size = x.size()/4;
	float died =size -lived;
	float meanL= means(0,1);
	float meanD= means(0,0);
	float sqL=0, sqD=0;

	for(int i =0; i< size; i++){
		float lived = x(i,0);
		float a = x(i,3);
		if(lived ==0){
			sqD+=(a-meanD)*(a-meanD);
		}
		if(lived ==1){
			sqL+=(a-meanL)*(a-meanL);
		}
	}
	theVars << (float)(sqD/died),(float)(sqL/lived);
			//if(DEBUG){cout<<theVars<<endl;}
	return theVars;
}//end of variances

//CALCULATIONS FUNCTION
void calcs(MatrixXd predictions, MatrixXd data){
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
			//cout<<tp<<" "<<tn<<" "<<fp<<" "<<fn<<endl;
		}

		accuracy = (tp+tn)/(tp+tn+fp+fn);
		sensitivity = tp/(tp+fn);
		specificity = tn/(tn+fp);

		cout<<"The accuracy is: "<<accuracy<<endl;
		cout<<"The sensitivity is: "<<sensitivity<<endl;
		cout<<"The specificity is: "<<specificity<<endl;

		return;
}

