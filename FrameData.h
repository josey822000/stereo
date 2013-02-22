#ifndef FRAMEDATA_H
#define FRAMEDATA_H

#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
using namespace cv;
using namespace std;

class FrameData{
public:
	FrameData(){}
	FrameData(IplImage* img, CvMat* k, CvMat* r, CvMat* t):_img(img),  _K(k), _R(r), _T(t), b_dis(false), b_mask(false){}
	void setD(CvMat* D){ b_dis = true; _D = D;}
	void setMask(CvMat* M){ b_mask = true; _mask = M;}
	IplImage* getImg() { return _img; }
	CvMat* getK(){ return _K;}
	CvMat* getR(){ return _R;}
	CvMat* getT(){ return _T;}
	CvMat* getD();
	CvMat* getMask();
    ~FrameData();

	bool b_dis;
	bool b_mask;

private:
	IplImage* _img;		// ori_img
	CvMat* _K;				// 3*3
	CvMat* _R;				// 3*3	
	CvMat* _T;				// 3*1
	CvMat* _D;				// disparity
	CvMat* _mask;	    // mask_img
};
#endif
