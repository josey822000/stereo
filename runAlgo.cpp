#include "runAlgo.h"

#include "MRF/mrf.h"
#include "MRF/ICM.h"
#include "MRF/GCoptimization.h"
#include "MRF/MaxProdBP.h"
#include "MRF/TRW-S.h"
#include "MRF/BP-S.h"

#include "globalVar.h"
#include "ReadCameraParameter.h"
#include "ComputeMRF.h"
#include "ComputeEnergy.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <new>
#include <string>

using namespace std;

#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
using namespace cv;
extern int sizeX;
extern int sizeY;
extern string DIR_SEQ;
void getDisparities(MRF *mrf, int width, int height, IplImage* disp){
	//disp = cvCreateImage(cvSize(width, height),IPL_DEPTH_32F,1);
    int n = 0;
	int wStep = disp->widthStep / sizeof(uchar);
    for (int y = 0; y < height; y++) {
	//uchar *row = &disp.Pixel(0, y, 0);
		for (int x = 0; x < width; x++) {
			((uchar*)(disp->imageData+wStep * y))[x] = (dK(mrf->getLabel(n++))-D_MIN)/(D_MAX-D_MIN)*255;
	   // row[x] = mrf->getLabel(n++);
		}
    }
}

void outDisparities(MRF *mrf, int width, int height, std::string filename){
	//disp = cvCreateImage(cvSize(width, height),IPL_DEPTH_32F,1);
	std::ofstream out;
	out.open(filename.c_str());

    int n = 0;
	//int wStep = disp->widthStep / sizeof(uchar);
    for (int y = 0; y < height; y++) {
	//uchar *row = &disp.Pixel(0, y, 0);
		for (int x = 0; x < width; x++) {
			out << dK(mrf->getLabel(n++)) << "\t";
	   	// row[x] = mrf->getLabel(n++);
		}
    }
}

void setDisparities(IplImage* disp, MRF *mrf){
	int width = disp->width, height = disp->height; 
	int n = 0;
	for(int y = 0; y < height; y++)
		for(int x = 0;  x < width; x++){
			double d  = ((uchar*)(disp->imageData+disp->widthStep*y))[x]/255.0 * (D_MAX-D_MIN) + D_MIN;
			mrf->setLabel(n++, Kd(d));
			}
}

void setDisparities(std::string disp, MRF *mrf){
	//int width = disp->width, height = disp->height;
	int width = sizeX, height = sizeY;
	ifstream inFile;
	inFile.open(disp.c_str());
	
	int n = 0;
	for(int y = 0; y < height; y++)
		for(int x = 0;  x < width; x++){
			double d;
			inFile >> d;
			mrf->setLabel(n++, Kd(d));
			}
}


void runBP(EnergyFunction *energy, std::string filename, MRF *&mrf){

    //MRF* mrf;
    MRF::EnergyVal E;
    //double lowerBound;
    float t,tot_t;
    int iter;

	////////////////////////////////////////////////
	//          Belief Propagation                //
	////////////////////////////////////////////////
	
	printf("\n*******  Started MaxProd Belief Propagation *****\n");
    mrf = new MaxProdBP(sizeX,sizeY,numLabels,energy);
    mrf->initialize();
    mrf->clearAnswer();
	  
	//string disp = "File/_road/road0_50L_5_0.05_BP.txt";
	//setDisparities(disp, mrf);
	MRF::EnergyVal Ed, Es; //, Eold;
	Ed = mrf->dataEnergy();
	Es = mrf->smoothnessEnergy();

	E = Ed + Es; // mrf->totalEnergy()
	    
   printf("Energy at the Start= %g (%g,%g)\n", (double)E,(double)mrf->dataEnergy(), (double)mrf->smoothnessEnergy());

    tot_t = 0;
    for (iter=0; iter < 10; iter++) {
		mrf->optimize(1, t);
	
		E = mrf->totalEnergy();
		tot_t = tot_t + t ;

	}
	printf("energy = %g (%f secs) <%g, %g>\n", (double)E, tot_t, (double)mrf->dataEnergy(), (double)mrf->smoothnessEnergy());
		//
	IplImage* D_BP = cvCreateImage(cvSize(sizeX, sizeY),IPL_DEPTH_8U,1);
	getDisparities(mrf, sizeX, sizeY, D_BP);
	string name = filename + "_BP.txt";
	outDisparities(mrf, sizeX, sizeY, name);
	name.clear();
	name = filename + "_BP.png";
	if(!cvSaveImage(name.c_str() ,D_BP)) printf("Could not save: D_BP.png\n");
	//delete mrf;
}

void runBPS(EnergyFunction *energy, std::string filename, MRF *&mrf){
	
    //MRF* mrf;
    MRF::EnergyVal E;
    //double lowerBound;
    float t,tot_t;
    int iter;

	////////////////////////////////////////////////
	//                  BP-S                      //
	////////////////////////////////////////////////
	
    printf("\n*******Started BP-S *****\n");
    mrf = new BPS(sizeX,sizeY,numLabels,energy);

    // can disable caching of values of general smoothness function:
    //mrf->dontCacheSmoothnessCosts();
	
    mrf->initialize();
    mrf->clearAnswer();
  
    E = mrf->totalEnergy();
    printf("Energy at the Start= %g (%g,%g)\n", (float)E,
	   (float)mrf->dataEnergy(), (float)mrf->smoothnessEnergy());

    tot_t = 0;
    for (iter=0; iter<10; iter++) {
		mrf->optimize(10, t);
		E = mrf->totalEnergy();
		tot_t = tot_t + t ;
		printf("energy = %g (%f secs) <%g, %g>\n", (double)E, tot_t, (double)mrf->dataEnergy(), (double)mrf->smoothnessEnergy());
    }

	IplImage* D_BP =  cvCreateImage(cvSize(sizeX, sizeY),IPL_DEPTH_8U,1);
	string name = filename + "_BP-S.txt";
	getDisparities(mrf, sizeX, sizeY, D_BP);
	outDisparities(mrf, sizeX, sizeY, name);
	name.clear();
	name = filename + "_BP-S.png";
	if(!cvSaveImage(name.c_str(),D_BP)) printf("Could not save: D_BP-S.png\n");
    
	//delete mrf;
}

void runICM(EnergyFunction *energy, std::string filename, MRF *&mrf){

  	//MRF* mrf;
  	MRF::EnergyVal E;
  	//double lowerBound;
  	float t,tot_t;
  	int iter;


	cout <<"\n*******Started ICM *****\n";
        mrf = new ICM(sizeX,sizeY,numLabels,energy);
        mrf->initialize();
	mrf->clearAnswer();
	//IplImage *disp = cvLoadImage("File/road01_withSegPlane_20_500s.png");
	//if(!disp) cout << "cannot open ICM initial!" << endl;
	//setDisparities(disp,mrf);
	MRF::EnergyVal Ed, Es;	//, Eold;
	Ed = mrf->dataEnergy();
	Es = mrf->smoothnessEnergy();

    	E = Ed + Es;

	printf("Energy at the Start= %g (%g,%g)\n", (double)E, (double)mrf->dataEnergy(), (double)mrf->smoothnessEnergy());
	tot_t = 0;
	for (iter=0; iter<10; iter++) {
		mrf->optimize(5, t);

		E = mrf->totalEnergy();
		tot_t = tot_t + t ;
		printf("energy = %g (%f secs) \n", (double)E, tot_t);
	}

	//
	IplImage* D_ICM = cvCreateImage(cvSize(sizeX, sizeY), IPL_DEPTH_8U,1);
	getDisparities(mrf, sizeX, sizeY, D_ICM);
	string name = filename + "_ICM.txt";
	outDisparities(mrf, sizeX, sizeY, name);
	name.clear();
	name = filename + "_ICM.png";
	if(!cvSaveImage(name.c_str(), D_ICM)) printf("Could not save: D_ICM.png\n");
//	delete mrf;
}
void runExpansion(EnergyFunction *energy, std::string filename, MRF *&mrf){
  //	MRF* mrf;
  	MRF::EnergyVal E;
  	//double lowerBound;
  	float t,tot_t;
  	int iter;
	////////////////////////////////////////////////
	//          Graph-cuts expansion              //
	////////////////////////////////////////////////
	    printf("\n*******Started graph-cuts expansion *****\n");
	    mrf = new Expansion(sizeX,sizeY,numLabels,energy);
	    mrf->initialize();
	    mrf->clearAnswer();
	    
	    E = mrf->totalEnergy();
	   printf("Energy at the Start= %g (%g,%g)\n", (double)E, (double)mrf->dataEnergy(), (double)mrf->smoothnessEnergy());

#ifdef COUNT_TRUNCATIONS
	    truncCnt = totalCnt = 0;
#endif
	    tot_t = 0;
	    for (iter=0; iter<6; iter++) {
		mrf->optimize(1, t);

		E = mrf->totalEnergy();
		tot_t = tot_t + t ;
		printf("energy = %g (%f secs) <%g, %g>\n", (double)E, tot_t, (double)mrf->dataEnergy(), (double)mrf->smoothnessEnergy());
	    }
#ifdef COUNT_TRUNCATIONS
	    if (truncCnt > 0)
		printf("***WARNING: %d terms (%.2f%%) were truncated to ensure regularity\n", 
		       truncCnt, (float)(100.0 * truncCnt / totalCnt));
#endif
		
		//
	IplImage* D_EXP = cvCreateImage(cvSize(sizeX, sizeY), IPL_DEPTH_8U,1);
	getDisparities(mrf, sizeX, sizeY, D_EXP);
	string name = filename + "_EXP.txt";
	outDisparities(mrf, sizeX, sizeY, name);
	name.clear();
	name = filename + "_EXP.png";
	if(!cvSaveImage(name.c_str(), D_EXP)) printf("Could not save: D_EXP.png\n");
//	    delete mrf;
}

void runTRWS(EnergyFunction *energy, std::string filename, MRF *&mrf){
	  	//MRF* mrf;
  		MRF::EnergyVal E;
  		double lowerBound;
  		float t,tot_t;
  		int iter;
	////////////////////////////////////////////////
	//                   TRW-S                      //
	////////////////////////////////////////////////
	
	    printf("\n*******Started TRW-S *****\n");
	    mrf = new TRWS(sizeX,sizeY,numLabels,energy);

	    // can disable caching of values of general smoothness function:
	    //mrf->dontCacheSmoothnessCosts();

	    mrf->initialize();
	    mrf->clearAnswer();

	    
	    E = mrf->totalEnergy();
	    printf("Energy at the Start= %g (%g,%g)\n", (float)E,
		   (float)mrf->dataEnergy(), (float)mrf->smoothnessEnergy());

	    tot_t = 0;
	    for (iter=0; iter<10; iter++) {
		mrf->optimize(10, t);

		E = mrf->totalEnergy();
		lowerBound = mrf->lowerBound();
		tot_t = tot_t + t ;
		printf("energy = %g, lower bound = %f (%f secs)\n", (float)E, lowerBound, tot_t);
	    }

		//
		IplImage* D_TRWS = cvCreateImage(cvSize(sizeX, sizeY), IPL_DEPTH_8U,1);
		getDisparities(mrf, sizeX, sizeY, D_TRWS);
		string name = filename + "_TRWS.txt";
		outDisparities(mrf, sizeX, sizeY, name);
		name.clear();
		name = filename + "_TRWS.png";
		if(!cvSaveImage(name.c_str(), D_TRWS)) printf("Could not save: D_TRWS.png\n");
	    //delete mrf;
}

void runSwap(EnergyFunction *energy, std::string filename, MRF *&mrf){

		//MRF* mrf;
  		MRF::EnergyVal E;
		//double lowerBound;
  		float t,tot_t;
  		int iter;

	////////////////////////////////////////////////
	//          Graph-cuts swap                   //
	////////////////////////////////////////////////
	
	    printf("\n*******Started graph-cuts swap *****\n");
	    mrf = new Swap(sizeX,sizeY,numLabels,energy);
	    mrf->initialize();
	    mrf->clearAnswer();
	    
	    E = mrf->totalEnergy();
	    printf("Energy at the Start= %g (%g,%g)\n", (float)E,
		   (float)mrf->dataEnergy(), (float)mrf->smoothnessEnergy());

#ifdef COUNT_TRUNCATIONS
	    truncCnt = totalCnt = 0;
#endif
	    tot_t = 0;
	    for (iter=0; iter<8; iter++) {
		mrf->optimize(1, t);

		E = mrf->totalEnergy();
		tot_t = tot_t + t ;
		printf("energy = %g (%f secs)\n", (float)E, tot_t);
	    }
#ifdef COUNT_TRUNCATIONS
	    if (truncCnt > 0)
		printf("***WARNING: %d terms (%.2f%%) were truncated to ensure regularity\n", 
		       truncCnt, (float)(100.0 * truncCnt / totalCnt));
#endif

		//
		IplImage* D_Swap = cvCreateImage(cvSize(sizeX, sizeY), IPL_DEPTH_8U,1);
		getDisparities(mrf, sizeX, sizeY, D_Swap);
		string name = filename + "_Swap.txt";
		outDisparities(mrf, sizeX, sizeY, name);
		name.clear();
		name = filename + "_Swap.png";
		if(!cvSaveImage(name.c_str(), D_Swap)) printf("Could not save: D_Swap.png\n");
	  //  delete mrf;
	
}
