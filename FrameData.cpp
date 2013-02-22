#include "FrameData.h"
#include <iostream>

CvMat* FrameData::getD(){ 
	if (! b_dis){
		std::cout << "No disparity in FrameData!" << endl;
		return 0;
		}
	else
		return _D;
}

CvMat* FrameData::getMask(){ 
	if( ! b_mask){
		std::cout << "No mask in FrameData!" << endl;
		return 0;
		}
	else
		return _mask;
}

FrameData::~FrameData(){
	cvReleaseImage(&_img);
	cvReleaseMat(&_K);
	cvReleaseMat(&_R);
	cvReleaseMat(&_T);
	cvReleaseMat(&_D);
	cvReleaseMat(&_mask);
}