//============================================================================
// Name        : main.cpp
// Author      : Kathryn Kingsley
// Version     : v0.1
// Copyright   : ML4375 S2021 KLK170230
// Description : main function for homework1. To avoid making it more than one
//				 file, as the instructions request, I've laid this .cpp out
//				 like this: preprocessor directives, function signatures,
//				 main(), and at the bottom, definitions of other functions
//============================================================================
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <math.h>
using namespace std;

bool DEBUG=false;  //global for testing/code tracing

//Function signatures
float theSum( vector<float> v);
float mean( vector<float> v);
float median( vector<float> v);
void range( vector<float> v);
float covariance( vector<float> x, vector<float> y);
float correlation( vector<float> x, vector<float> y);

//main
int main() {
	//open file for reading and ensure it opened
	ifstream inFile("Boston.csv");
	if(!inFile){
			cout<<"Couldn't open file"<<endl;
	}

	//declare variables
	vector<float> avgRooms;
	vector<float> medValue;
	float value;
	string line;//variable for each line

	//throw away first line, since I know what's in the file
	getline(inFile,line);

	//while loop to read file line by line
	while (getline(inFile, line))
	{

		stringstream ss(line);  //stream to break up specific line

		//push first value onto avgRooms, skip the comma, push second value into medValues
		ss>>value;
		avgRooms.push_back(value);

		if(ss.peek() == ','){ ss.ignore();}  //if next item is a ',', skip it

		//push second value onto medValue
		ss>>value;
		medValue.push_back(value);

	}

	//close file once done
	inFile.close();

	//call your functions for both rm and medv
	//SUM
	value= theSum(avgRooms);
	cout<<"The sum is: "<<value<<endl;
	value=theSum(medValue);
	cout<<"The sum is: "<<value<<endl;

	//MEAN
	value= mean(avgRooms);
	cout<<"The mean is: "<<value<<endl;
	value= mean(medValue);
	cout<<"The mean is: "<<value<<endl;

	//MEDIAN
	value= median(avgRooms);
	cout<<"The median is: "<<value<<endl;
	value= median(medValue);
	cout<<"The median is: "<<value<<endl;

	//RANGE
	range(avgRooms);
	range(medValue);

	//COVARIANCE
	//should be 4.493446 according to R
	value= covariance(avgRooms,medValue);
	cout<<"The covariance is: "<<value<<endl;

	//CORRELATION
	//Should be 0.6953599 according to R
	value= correlation(avgRooms,medValue);
	cout<<"The correlation is: "<<value<<endl;

	return 0;
}


//FUNCTION DEFINITIONS
//function that adds the sum of all items in vector
float theSum( vector<float> v)
{
	float theSum=0;

	 for (auto const &i: v) {
	        theSum += i;
	    }

	return theSum;
}

//function that finds the average
float mean( vector<float> v)
{
	float mean=0;
	float sum= theSum(v);
	mean=(sum/v.size());
	return mean;
}

//function that sorts the vector, then finds middle value
float median( vector<float> v)
{
	//sort the vector
	sort(v.begin(),v.end());
	int num=v.size()/2;

	return 	v.at(num);
}

//function that sorts the vector, then finds smallest and largest values
void range( vector<float> v)
{
	//sort the vector
	sort(v.begin(),v.end());
	int end = v.size()-1;
	float begin, last;
	begin = v.at(0);
	last=v.at(end);

	cout<<"The range is: ["<<begin<<","<<last<<"]"<<endl;

	return;
}

//should be 4.493446 according to R
float covariance( vector<float> x, vector<float> y)
{
	//variables
	float theCov=0;
	float xMean=mean(x);
	float yMean=mean(y);
	int n=x.size()-2; //-1 to account for index, -1 to account for (n-1)

	//make sure same size
	if(x.size()!=y.size())
	{
		cout<<"Error: vector size mismatch"<<endl;
		return theCov;
	}

	//loop for summation
	for(int i=0; i!=n; i++)
	{
		theCov+=((x.at(i)-xMean)*(y.at(i)-yMean));

	}

	//divide by n-1
	theCov/=n;
	//cout<<theCov<<endl;

	return theCov;
}


//Should be 0.6953599 according to R
float correlation( vector<float> x, vector<float> y)
{
	//Cov function checks size for us
	float theCov= covariance(x,y);
	float theCor=0;
	float sigX,sigY=0;

	//find sqaure root of the variance
	sigX=sqrt(covariance(x,x));
	sigY=sqrt(covariance(y,y));

	//divide the covariance by the variance
	theCor= theCov/(sigX*sigY);

	return theCor;
}
