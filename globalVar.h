#ifndef GLOBALVAR_H
#define GLOBALVAR_H

#include <string>

//const static bool SEG = true;  // = runIni
//const static bool DIS = false; // = runF
const static bool FG = false;
//static std::string DIR_SEQ ;
const static int D_FRAME_START = 15;
const static int D_FRAME_END = 54;

const static int NO_NEIGH = 15;		// number of neighbor for video frame
const static double D_MAX = 0.01808;	// max value of disparity
const static double D_MIN = 0.0;	// min value of disparity
const static int LEVEL = 300;		// Level of disparity
const static int N = 5;			// number of neighbor for pv					
const static double RHO_D = 3.0;
const static int neighbor = 5;
const static double EPSILON = 20.0;	// 50
const static double ETA = 0.05*(D_MAX-D_MIN);
const static double OMEGA_S = 5.0 / (D_MAX - D_MIN);

/* for MRF */
const static int core = 10;
//int sizeX = 0;///8a08;	//576; //480;		// width of video
//int sizeY = 0;//315;	//324; //270;		// height of video
const static int numLabels = LEVEL+1;
const static double smoothMax_Final = 0.05*(D_MAX - D_MIN);
const static double smoothMax_init2 = 0.05*(D_MAX - D_MIN);	// edit josey original 0.05 , use partial of 0.05:1/300 = 0.16:1/100
const static double lambda_Final = 5.0/(D_MAX-D_MIN);
const static double lambda_init2 = 5.0/(D_MAX-D_MIN); // edit josey original 10.0, use partial of paper 5:300 = 1.66:100

const static bool aBP = true;
const static bool aBPS = false;
const static bool aICM = false;
const static bool aTRWS = false;
const static bool aSWAPS = false;
const static bool aEXP = false; 

//#define H_FILE   "hCue_road01_seg_test_50L.txt"
//#define OUT_FILE "road01_seg_test_10L_5_0.05"

#endif

