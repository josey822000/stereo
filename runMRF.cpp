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
#include <map>

using namespace std;

#define BOOST_FILESYSTEM_VERSION 3
#include "boost/filesystem.hpp"   // includes all needed Boost.Filesystem declarations
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
using namespace cv;

#ifdef COUNT_TRUNCATIONS
int truncCnt, totalCnt;
#endif
int sizeX = 0;
int sizeY = 0;
int NOW_FRAME =0;
int NO_FRAME = 0;
string DIR_SEQ;
int main(int argc, char **argv)
{
    cerr <<"usage: DIR now_frame no_frame"<<endl;
    assert(argc>=5);
    DIR_SEQ = argv[1];
    NOW_FRAME = atoi(argv[2]);
    NO_FRAME = atoi(argv[3]);
    int runIni = atoi(argv[4]);
    cerr<<"Dir:" << DIR_SEQ<< " now:" << NOW_FRAME << "num:" << NO_FRAME <<endl;
	BFS::path p( DIR_SEQ.c_str() );	// path of CameraParameter(camera.txt)
	ReadCameraParameter* CP = new ReadCameraParameter(p);

	string dir_vs = DIR_SEQ + "/seq";
	BFS::path p2( dir_vs.c_str() );	// path of Video Seq
    bool runF = runIni >0;
	ReadVideoSeq* VS = new ReadVideoSeq(p2,runF,!runF,FG);
	//ReadVideoSeq* VS = new ReadVideoSeq(p2,DIS,SEG,FG);
	VS->getImgSz(&sizeX,&sizeY);
    cerr << "ImgSz:"<< sizeX <<","<<sizeY<<endl;
	//int NOW_FRAME = 0, NO_NEIGH = 10;
	int noF = VS->no_frame - NOW_FRAME;

	size_t t1 = clock();
	if(runIni==0)		// ComputMRF.cpp
		Video_initial2(CP, VS, NOW_FRAME, NO_NEIGH, NO_FRAME, DIR_SEQ+"_init2_");
	size_t t2 = clock();
	if(runIni==1)
		VideoMRF_Final(CP, VS, NOW_FRAME, NO_NEIGH, NO_FRAME, DIR_SEQ+"_final_");
	size_t t3 = clock();
	
	cout << "Ed_init- " << (double) (t2-t1)/CLOCKS_PER_SEC << endl;
	cout << "Ed- " << (double)(t3-t2)/CLOCKS_PER_SEC << endl;

	delete CP;
	delete VS;
 
    return 0;
}


