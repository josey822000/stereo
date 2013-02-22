#ifndef SEGMENT_H
#define SEGMENT_H

#include "ReadCameraParameter.h"
#include "ReadVideoSeq.h"
#include "FrameData.h"
#include "globalVar.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <new>
#include <string>
#include <vector>
#include <map>

using namespace std;

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/cxcore.h"

std::map<int, std::vector<int> > computeSeg(int width, int height, std::string filename);	
void computeJacobian(std::vector<int> pixs, std::vector<FrameData*> _frames, CvMat* delta, CvMat* P_est, std::vector<CvMat*> As, std::vector<CvMat*> Bs);
void computeJacobian(std::vector<int> pixs, std::vector<FrameData*> _frames, CvMat* delta, CvMat* P_est,double* dCost_ori);
double Dplane(CvMat* pix, CvMat* planePara);
void WriteToFile(CvMat* M, std::string FileName);

#endif