#include "ComputeEnergy.h"

#include "MRF/mrf.h"
#include "MRF/GCoptimization.h"
#include "MRF/MaxProdBP.h"
#include "MRF/BP-S.h"

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>

using namespace std;
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
using namespace cv;
extern int sizeX;
extern int sizeY;
extern string DIR_SEQ;
void GetFrame(ReadCameraParameter* cp, ReadVideoSeq* vs,  int now_frame, int Nframe, std::vector<FrameData*>* frames){ 
	// now_frame-Nframe/2 ~ now_frame+Nframe/2 <-- consider neighbor frames (number = Nframe)
	int start_frame = now_frame - Nframe;		
	int end_frame = now_frame + Nframe;
	if (start_frame < 0)
		start_frame = 0;
	if (end_frame >= vs->no_frame)
		end_frame = vs->no_frame - 1;
	//start_frame = now_frame;
	//end_frame = now_frame+1;

	cout << "from:" << start_frame << " to: " << end_frame << endl;
	// FrameData-frames   --> frames[0]:process frame, frame[1..X]: neighbor frames
	//std::vector<FrameData*> frames;
	FrameData* f = new FrameData(vs->getVideoFrame(now_frame), cp->getK(now_frame), cp->getR(now_frame), cp->getT(now_frame));

	if(vs->allF_dis[now_frame] == 1){
		f->setD(vs->getDisparity(now_frame));
	}
	if(vs->b_mask){
		f->b_mask = true;
		f->setMask(vs->getMask(now_frame));
	}
	frames->push_back(f);

	//delete f;
	for (int i = start_frame; i <= end_frame; i++){
		if( i != now_frame){
			// align ?? epipolar??  ReadCameraParameter.h
			FrameData* f = new FrameData(vs->getVideoFrame(i), cp->getK(i), cp->getR(i), cp->getT(i));
			if(vs->allF_dis[i] == 1) f->setD(vs->getDisparity(i));
			if(vs->b_mask){
				f->b_mask = true;
				f->setMask(vs->getMask(now_frame));
			}
			frames->push_back(f);
			//delete f;
		}
	}
}
void getConjugateX(CvMat* pix_h, CvMat* pix, double d, FrameData* f1, FrameData* f2){
	/* X(f2) = K(f2)tran_R(f2)R(f1)inv_K(f1)X(f1)+d(f1)K(f2)tran_R(f2)[T(f1)-T(f2)] */

	CvMat* R_tran = cvCreateMat(3,3,CV_64FC1);
	CvMat* K_inver = cvCreateMat(3,3,CV_64FC1);
	CvMat* tmp = cvCreateMat(3,3,CV_64FC1);
	CvMat* tmp2 = cvCreateMat(3,1,CV_64FC1);

	cvTranspose(f2->getR(), R_tran);		
	//R_tran = f2->getR();	

	cvInvert(f1->getK(), K_inver); 
	cvSub(f1->getT(), f2->getT(), tmp2); 
	cvMatMul(f2->getK(), R_tran, tmp);
	cvMatMul(tmp, tmp2, tmp2);
	cvMatMul(tmp, f1->getR(), tmp);
	cvMatMul(tmp, K_inver, tmp);
	cvMatMul(tmp, pix, pix_h);		//tmp

	cvConvertScale(tmp2, tmp2, d);
	cvAdd(pix_h, tmp2, pix_h);

	cvConvertScale(pix_h, pix_h, (double)(1.0/pix_h->data.db[2]));  // normalize

	cvReleaseMat(&R_tran);
	cvReleaseMat(&K_inver);
	cvReleaseMat(&tmp);
	cvReleaseMat(&tmp2);	
}

void GetAB(FrameData *f1, FrameData *f2, CvMat *A, CvMat *B){

	CvMat* R_tran = cvCreateMat(3,3,CV_64FC1);
	CvMat* K_inver = cvCreateMat(3,3,CV_64FC1);
	//	CvMat* A = cvCreateMat(3,3,CV_64FC1);
	//	CvMat* B = cvCreateMat(3,1,CV_64FC1);
	// Rt
	cvTranspose(f2->getR(), R_tran);		
	// K-1
	// check if the inverse is not exist
	assert(cvDet(f1->getK()) != 0);
	cvInvert(f1->getK(), K_inver); 
	// Tt-Tt'
	cvSub(f1->getT(), f2->getT(), B); 
	// KR
	cvMatMul(f2->getK(), R_tran, A);
	// formula(6)
	cvMatMul(A, B, B);
	// KRtR
	cvMatMul(A, f1->getR(), A);
	// KRtRK-1
	cvMatMul(A, K_inver, A);

	cvReleaseMat(&R_tran);
	cvReleaseMat(&K_inver);
}

void getConjugateX(CvMat* pix_h, CvMat* pix, double d, CvMat* A, CvMat* B){
	pix_h->data.db[0] = 
		A->data.db[0]*pix->data.db[0] + 
		A->data.db[1]*pix->data.db[1] +
		A->data.db[2]*pix->data.db[2];
	pix_h->data.db[1] = 
		A->data.db[3]*pix->data.db[0] + 
		A->data.db[4]*pix->data.db[1] +
		A->data.db[5]*pix->data.db[2];
	pix_h->data.db[2] = 
		A->data.db[6]*pix->data.db[0] + 
		A->data.db[7]*pix->data.db[1] +
		A->data.db[8]*pix->data.db[2];
	double z =  pix_h->data.db[2] + d*B->data.db[2];
	assert(z!=0.);
	pix_h->data.db[0] = (pix_h->data.db[0] + d*B->data.db[0])/z;
	pix_h->data.db[1] = (pix_h->data.db[1] + d*B->data.db[1])/z;
	pix_h->data.db[2] = 1;
}
double dK(int k){ 
	double dk = 0.0;
	dk = (double)(LEVEL-k)/(double)LEVEL*(double)D_MIN;
	dk += (double)k/(double)LEVEL*(double)D_MAX;
	return dk;
}
int Kd(double d){ 
	int k = 0;
	k = LEVEL*(d- D_MIN)/(D_MAX-D_MIN);
	return k;
}
double ColorDiff(int index,double x,double y, IplImage* img1, IplImage* img2){
	uchar b, g, r, b2, g2, r2;

	uchar* ptr = (uchar*)img1->imageData + index*3;
	b = *(img1->imageData+index*3);

	g = *(img1->imageData+index*3 +1);

	r = *(img1->imageData+index*3 +2);

	double x2 = min(max(x, 0.0), (double)img2->width-1);
	double y2 = min(max(y, 0.0), (double)img2->height-1);
	//double x2 = pix2->data.db[0], y2 = pix2->data.db[1];
	double cx2 = ceil(x2), cy2 = ceil(y2);
	double fx2 = floor(x2), fy2 = floor(y2);
	double dx2 = cx2-x2, dy2 = cy2-y2;

	b2 = ((uchar*)(img2->imageData+img2->widthStep *(int)fy2))[(int)fx2 * 3] * (dx2)*(dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)fy2))[(int)cx2 * 3] * (1-dx2)*(dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)cy2))[(int)fx2 * 3] * (dx2)*(1-dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)cy2))[(int)cx2 * 3] * (1-dx2)*(1-dy2);
	b2 /= 4.;

	g2= ((uchar*)(img2->imageData+img2->widthStep *(int)fy2))[(int)fx2 * 3+1] * (dx2)*(dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)fy2))[(int)cx2 * 3+1] * (1-dx2)*(dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)cy2))[(int)fx2 * 3+1] * (dx2)*(1-dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)cy2))[(int)cx2 * 3+1] * (1-dx2)*(1-dy2);
	g2 /= 4.;

	r2 =  ((uchar*)(img2->imageData+img2->widthStep *(int)fy2))[(int)fx2 * 3+2] * (dx2)*(dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)fy2))[(int)cx2 * 3+2] * (1-dx2)*(dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)cy2))[(int)fx2 * 3+2] * (dx2)*(1-dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)cy2))[(int)cx2 * 3+2] * (1-dx2)*(1-dy2);
	r2 /= 4.;

	double dist = sqrt((double)((b-b2)*(b-b2) + (g-g2)*(g-g2) + (r-r2)*(r-r2)));
	return dist;
}

double ColorDiff(CvMat* pix, CvMat* pix2, IplImage* img1, IplImage* img2){
	uchar b, g, r, b2, g2, r2;

	double x1 = min(max(pix->data.db[0], 0.0), (double)img1->width-1);
	double y1 = min(max(pix->data.db[1], 0.0), (double)img1->height-1);
	//double x1 = pix->data.db[0], y1 = pix->data.db[1];
	double cx1 = ceil(x1), cy1 = ceil(y1);
	double fx1 = floor(x1), fy1 = floor(y1);
	double dx1 = cx1-x1, dy1 = cy1-y1;

	b = ((uchar*)(img1->imageData+img1->widthStep *(int)fy1))[(int)fx1 * 3] * (dx1)*(dy1) +
		((uchar*)(img1->imageData+img1->widthStep *(int)fy1))[(int)cx1 * 3] * (1-dx1)*(dy1) +
		((uchar*)(img1->imageData+img1->widthStep *(int)cy1))[(int)fx1 * 3] * (dx1)*(1-dy1) +
		((uchar*)(img1->imageData+img1->widthStep *(int)cy1))[(int)cx1 * 3] * (1-dx1)*(1-dy1);
	b /= 4.;

	g = ((uchar*)(img1->imageData+img1->widthStep *(int)fy1))[(int)fx1 * 3+1] * (dx1)*(dy1) +
		((uchar*)(img1->imageData+img1->widthStep *(int)fy1))[(int)cx1 * 3+1] * (1-dx1)*(dy1) +
		((uchar*)(img1->imageData+img1->widthStep *(int)cy1))[(int)fx1 * 3+1] * (dx1)*(1-dy1) +
		((uchar*)(img1->imageData+img1->widthStep *(int)cy1))[(int)cx1 * 3+1] * (1-dx1)*(1-dy1);
	g /= 4.;

	r =  ((uchar*)(img1->imageData+img1->widthStep *(int)fy1))[(int)fx1 * 3+2] * (dx1)*(dy1) +
		((uchar*)(img1->imageData+img1->widthStep *(int)fy1))[(int)cx1 * 3+2] * (1-dx1)*(dy1) +
		((uchar*)(img1->imageData+img1->widthStep *(int)cy1))[(int)fx1 * 3+2] * (dx1)*(1-dy1) +
		((uchar*)(img1->imageData+img1->widthStep *(int)cy1))[(int)cx1 * 3+2] * (1-dx1)*(1-dy1);
	r /= 4.;

	double x2 = min(max(pix2->data.db[0], 0.0), (double)img2->width-1);
	double y2 = min(max(pix2->data.db[1], 0.0), (double)img2->height-1);
	//double x2 = pix2->data.db[0], y2 = pix2->data.db[1];
	double cx2 = ceil(x2), cy2 = ceil(y2);
	double fx2 = floor(x2), fy2 = floor(y2);
	double dx2 = cx2-x2, dy2 = cy2-y2;

	b2 = ((uchar*)(img2->imageData+img2->widthStep *(int)fy2))[(int)fx2 * 3] * (dx2)*(dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)fy2))[(int)cx2 * 3] * (1-dx2)*(dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)cy2))[(int)fx2 * 3] * (dx2)*(1-dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)cy2))[(int)cx2 * 3] * (1-dx2)*(1-dy2);
	b2 /= 4.;

	g2= ((uchar*)(img2->imageData+img2->widthStep *(int)fy2))[(int)fx2 * 3+1] * (dx2)*(dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)fy2))[(int)cx2 * 3+1] * (1-dx2)*(dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)cy2))[(int)fx2 * 3+1] * (dx2)*(1-dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)cy2))[(int)cx2 * 3+1] * (1-dx2)*(1-dy2);
	g2 /= 4.;

	r2 =  ((uchar*)(img2->imageData+img2->widthStep *(int)fy2))[(int)fx2 * 3+2] * (dx2)*(dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)fy2))[(int)cx2 * 3+2] * (1-dx2)*(dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)cy2))[(int)fx2 * 3+2] * (dx2)*(1-dy2) +
		((uchar*)(img2->imageData+img2->widthStep *(int)cy2))[(int)cx2 * 3+2] * (1-dx2)*(1-dy2);
	r2 /= 4.;

	double dist = sqrt((double)((b-b2)*(b-b2) + (g-g2)*(g-g2) + (r-r2)*(r-r2)));
	return dist;
}

double Pc(CvMat* pix, double d, FrameData* f1, FrameData* f2){
	//double d = dK(L);
	double sigma_c = 10.0;
	IplImage* img1 = f1->getImg();
	CvMat* new_pix = cvCreateMat(3,1,CV_64FC1);
	getConjugateX( new_pix, pix, d, f1, f2);
	// new_pix out of range! colordiff = infinite
	if (new_pix->data.db[0] < 0 || new_pix->data.db[0] >= img1->width || 
			new_pix->data.db[1] < 0 || new_pix->data.db[1] >= img1->height){
		cvReleaseMat(&new_pix);
		return 0.0;
	}
	else{
		double dist = ColorDiff(pix, new_pix, img1, f2->getImg());
		cvReleaseMat(&new_pix);
		return  (double)(sigma_c / (sigma_c + dist));
	}
}

CvMat* Pc(CvMat* pix, FrameData* f1, FrameData* f2, CvMat* A, CvMat* B){
	double sigma_c = 10.0;
	IplImage* img1 = f1->getImg();
	CvMat *PC = cvCreateMat(LEVEL+1, 1 ,CV_64FC1);
	CvMat* pix_h = cvCreateMat(3,1,CV_64FC1);
	cvMatMul(A, pix, pix_h);		//tmp

	CvMat* new_pix = cvCreateMat(3,1,CV_64FC1);
	CvMat* _tmp = cvCreateMat(3,1,CV_64FC1);
	double pc = 0.0;
	for(int K =0; K <= LEVEL; K++){
		cvConvertScale(B, _tmp, dK(K));
		cvAdd(pix_h, _tmp, new_pix);
		cvConvertScale(new_pix, new_pix, (double)(1.0/new_pix->data.db[2]));  // normalize

		// new_pix out of range!
		if (new_pix->data.db[0] < 0 || new_pix->data.db[0] >= img1->width || 
				new_pix->data.db[1] < 0 || new_pix->data.db[1] >= img1->height){
			//cout<<"\n**edited**\n";
			pc = 10001;		// edit by josey

		}
		else{
			double dist = ColorDiff(pix, new_pix, img1, f2->getImg());
			//cout <<"\n dist:"<< dist <<endl;
			pc =  sigma_c / (sigma_c + dist);
		}
		PC->data.db[K] = pc;
	}
	cvReleaseMat(&new_pix);
	cvReleaseMat(&_tmp);
	cvReleaseMat(&pix_h);

	double min_l, max_l;
	cvMinMaxLoc(PC, &min_l, &max_l);
	for(int K =0; K <= LEVEL; K++){
		if(PC->data.db[K] > 10000 ) PC->data.db[K] = min_l;
	}
	return PC;
}
/*
   double JoseyPc(int index,const std::vector<FrameData*>* _frames,idMap* IdxMap,const vector<CvMat*>* B, int level){
   double sigma_c = 10.0;
   double depth = dK(level);
   vector<double> frameCost;
   IplImage* img1 = (*_frames)[0]->getImg();
   CvMat* pix_h = cvCreateMat(3,1,CV_64FC1);
   CvMat* new_pix = cvCreateMat(3,1,CV_64FC1);
   CvMat* _tmp = cvCreateMat(3,1,CV_64FC1);
   double pc = 0;
   double Ax,Ay,Az;
   for(int i = 1; i != (int)(*_frames).size(); i++)		// for each frame get a cost
   {
   Ax = IdxMap->ptr[i]->data.db[index*3];
   Ay = IdxMap->ptr[i]->data.db[index*3+1];
   Az = IdxMap->ptr[i]->data.db[index*3+2];
   Az = Az+depth*(*B)[i-1]->data.db[2];
   Ax = (Ax+depth*(*B)[i-1]->data.db[0])/Az;
   Ay = (Ay+depth*(*B)[i-1]->data.db[1])/Az;
   cvConvertScale(new_pix, new_pix, (double)(1.0/new_pix->data.db[2]));  // homo coordinate
   assert(new_pix->data.db[2] != 0.);
// new_pix out of range!
if (Ax < 0 || Ax >= img1->width || 
Ay < 0 || Ay >= img1->height){
//frameCost.push_back(10000);
continue;					// don't use this frame
}
else{
double dist = ColorDiff(index,Ax,Ay, img1, (*_frames)[i]->getImg());
dist =  sigma_c / (sigma_c + dist);
frameCost.push_back(dist);
}
}
if(frameCost.size() < (*_frames).size()/2){	// enough few
if(frameCost.size() == 0)				// all is out of range
pc = -1;
else{
pc = std::accumulate(frameCost.begin(),frameCost.end(),0.);
pc = pc*(double)NO_NEIGH*2./(double)frameCost.size();
}
}
else{										// do sort
sort(frameCost.begin(),frameCost.end());
pc = std::accumulate(frameCost.begin()+frameCost.size()/2,frameCost.end(),0.);
pc = pc*(double)NO_NEIGH/(double)frameCost.size();
}
//	normalize


vector<double> _vtmp;
frameCost.clear();
frameCost.swap(_vtmp );
cvReleaseMat(&new_pix);
cvReleaseMat(&_tmp);
cvReleaseMat(&pix_h);
return pc;
}*/
double JoseyPc(CvMat* pix,const std::vector<FrameData*>* _frames,const vector<CvMat*>* A,const vector<CvMat*>* B, int level){
	double sigma_c = 10.0;
	double depth = dK(level);
	vector<double> frameCost;
	IplImage* img1 = (*_frames)[0]->getImg();
	CvMat* pix_h = cvCreateMat(3,1,CV_64FC1);
	CvMat* new_pix = cvCreateMat(3,1,CV_64FC1);
	CvMat* _tmp = cvCreateMat(3,1,CV_64FC1);
	double pc = 0;
	for(int i = 1; i != (int)(*_frames).size(); i++)		// for each frame get a cost
	{
		pix_h->data.db[0] = (*A)[i-1]->data.db[0]*pix->data.db[0] + 
			(*A)[i-1]->data.db[1]*pix->data.db[1] +
			(*A)[i-1]->data.db[2]*pix->data.db[2];
		pix_h->data.db[1] = (*A)[i-1]->data.db[3]*pix->data.db[0] + 
			(*A)[i-1]->data.db[4]*pix->data.db[1] +
			(*A)[i-1]->data.db[5]*pix->data.db[2];
		pix_h->data.db[2] = (*A)[i-1]->data.db[6]*pix->data.db[0] + 
			(*A)[i-1]->data.db[7]*pix->data.db[1] +
			(*A)[i-1]->data.db[8]*pix->data.db[2];

		double z =  pix_h->data.db[2] + depth*(*B)[i-1]->data.db[2];
		new_pix->data.db[0] = (pix_h->data.db[0] + depth*(*B)[i-1]->data.db[0])/z;

		new_pix->data.db[1] = (pix_h->data.db[1] + depth*(*B)[i-1]->data.db[1])/z;
		new_pix->data.db[2] = 1;
		assert(new_pix->data.db[2] != 0.);
		// new_pix out of range!
		if (new_pix->data.db[0] < 0 || new_pix->data.db[0] >= img1->width || 
				new_pix->data.db[1] < 0 || new_pix->data.db[1] >= img1->height){
			//frameCost.push_back(10000);
			continue;					// don't use this frame
		}
		else{
			double dist = ColorDiff(pix, new_pix, img1, (*_frames)[i]->getImg());
			dist =  sigma_c / (sigma_c + dist);
			frameCost.push_back(dist);
		}
	}
	if(frameCost.size() < (*_frames).size()/2){	// enough few
		if(frameCost.size() == 0)				// all is out of range
		{
			pc = -1;
		}
		else{
			pc = std::accumulate(frameCost.begin(),frameCost.end(),0.);
			pc = pc*(double)NO_NEIGH*2./(double)frameCost.size();
		}
	}
	else{										// do sort
		sort(frameCost.begin(),frameCost.end(),greater<double>());
		//partial_sort(frameCost.begin(),frameCost.begin()+frameCost.size()/2,frameCost.end(),greater<double>());
		pc = std::accumulate(frameCost.begin(),frameCost.begin()+frameCost.size()/2,0.);
		pc = pc*(double)NO_NEIGH*4/(double)frameCost.size();
	}
	
	//	normalize


	vector<double> _vtmp;
	frameCost.clear();
	frameCost.swap(_vtmp );
	cvReleaseMat(&new_pix);
	cvReleaseMat(&_tmp);
	cvReleaseMat(&pix_h);
	return pc;
}
CvMat* Pc(CvMat* pix, FrameData* f1, FrameData* f2, CvMat* A, CvMat* B, CvMat* cnt){
	double sigma_c = 10.0;
	IplImage* img1 = f1->getImg();
	CvMat *PC = cvCreateMat(LEVEL+1, 1 ,CV_64FC1);
	CvMat* pix_h = cvCreateMat(3,1,CV_64FC1);
	cvMatMul(A, pix, pix_h);		//tmp

	CvMat* new_pix = cvCreateMat(3,1,CV_64FC1);
	CvMat* _tmp = cvCreateMat(3,1,CV_64FC1);
	double pc = 0.0;
	for(int K =0; K <= LEVEL; K++){
		cvConvertScale(B, _tmp, dK(K));
		cvAdd(pix_h, _tmp, new_pix);
		cvConvertScale(new_pix, new_pix, (double)(1.0/new_pix->data.db[2]));  // normalize

		// new_pix out of range!
		if (new_pix->data.db[0] < 0 || new_pix->data.db[0] >= img1->width || 
				new_pix->data.db[1] < 0 || new_pix->data.db[1] >= img1->height){
			//cout<<"\n**edited**\n";
			pc = 10001;		// edit by josey

		}
		else{
			double dist = ColorDiff(pix, new_pix, img1, f2->getImg());
			//cout <<"\n dist:"<< dist <<endl;
			pc =  sigma_c / (sigma_c + dist);
			cnt->data.db[K] ++ ;	// edit by josey
		}
		PC->data.db[K] = pc;
	}
	cvReleaseMat(&new_pix);
	cvReleaseMat(&_tmp);
	cvReleaseMat(&pix_h);

	double min_l, max_l;
	cvMinMaxLoc(PC, &min_l, &max_l);
	for(int K =0; K <= LEVEL; K++){
		if(PC->data.db[K] > 10000 ) PC->data.db[K] = min_l;
	}
	return PC;
}

void GetPvMap (FrameData *f1, FrameData *f2, CvMat** Mx, CvMat** My){
	int width = f1->getImg()->width;
	int height = f1->getImg()->height;

	CvMat *A = cvCreateMat(3,3,CV_64FC1);
	CvMat *B = cvCreateMat(3,1,CV_64FC1);   // have counted twice
	GetAB(f2, f1, A, B);
	CvMat *D2 = cvCreateMat(height, width, CV_64FC1);
	D2 = f2->getD();

	CvMat *pix = cvCreateMat(3,1,CV_64FC1);
	CvMat *pix_h = cvCreateMat(3,1,CV_64FC1);
	int idx, xx, yy;
	for(int y = 0; y < height; y++){
		for(int x = 0; x < width; x++){
			pix->data.db[0] = x;
			pix->data.db[1] = y;
			pix->data.db[2] = 1.0;
			int nn =0;
			for (int i = -(int) neighbor/2; i <= (int) neighbor/2; ++i)
				for (int j = -(int) neighbor/2; j <= (int) neighbor/2; ++j){
					yy = min(max( 0, (y+i)), height-1);
					xx = min(max( 0, (x+j)), width-1);
					idx = yy*width+xx;
					getConjugateX(pix_h, pix, (double)D2->data.db[idx], A, B);
					Mx[y*width+x]->data.db[nn] = pix_h->data.db[0];
					My[y*width+x]->data.db[nn] = pix_h->data.db[1];		
					nn++;
				}
		}
	}
	cvReleaseMat(&pix);
	cvReleaseMat(&pix_h);

	cvReleaseMat(&A);
	cvReleaseMat(&B);
}

CvMat* Pv(CvMat* pix,  FrameData* f1, FrameData* f2){

	IplImage* img1 = f1->getImg();
	CvMat *PC = cvCreateMat(LEVEL+1, 1 ,CV_64FC1);
	CvMat* R_tran = cvCreateMat(3,3,CV_64FC1);
	CvMat* K_inver = cvCreateMat(3,3,CV_64FC1);
	CvMat* tmp = cvCreateMat(3,3,CV_64FC1);
	CvMat* tmp2 = cvCreateMat(3,1,CV_64FC1);
	CvMat* pix_h = cvCreateMat(3,1,CV_64FC1);

	cvTranspose(f2->getR(), R_tran);			
	cvInvert(f1->getK(), K_inver); 
	cvSub(f1->getT(), f2->getT(), tmp2); 
	cvMatMul(f2->getK(), R_tran, tmp);
	cvMatMul(tmp, tmp2, tmp2);
	cvMatMul(tmp, f1->getR(), tmp);
	cvMatMul(tmp, K_inver, tmp);
	cvMatMul(tmp, pix, pix_h);		//tmp

	CvMat *PVs = cvCreateMat(LEVEL+1, 1 ,CV_64FC1);

	CvMat* D2 = f2->getD();
	CvMat* A = cvCreateMat(3,3,CV_64FC1);
	CvMat* B = cvCreateMat(3,1,CV_64FC1);
	GetAB(f2, f1, A, B);
	double sum_t =0.0;
	double sum_t2 =0.0;
	for(int K =0; K <= LEVEL; K++){
		CvMat* new_pix = cvCreateMat(3,1,CV_64FC1);
		CvMat* _tmp = cvCreateMat(3,1,CV_64FC1);
		cvConvertScale(tmp2, _tmp, dK(K));
		cvAdd(pix_h, _tmp, new_pix);
		cvConvertScale(new_pix, new_pix, (double)(1.0/new_pix->data.db[2]));  // normalize
		cvReleaseMat(&_tmp);

		std::vector<double> PV;
		for (int i = -(int) neighbor/2; i <= (int) neighbor/2; ++i)
			for (int j = -(int) neighbor/2; j <= (int) neighbor/2; ++j){

				int y = min(max( 0, ((int) new_pix->data.db[1])+i), D2->rows-1);
				int x = min(max( 0, ((int) new_pix->data.db[0])+j), D2->cols-1);

				//double d2 = dK(D2->data.db[y*D2->cols+x]);
				double d2 = D2->data.db[y*D2->cols+x];

				CvMat* pix_H2 = cvCreateMat(3,1,CV_64FC1);		//  L(t'->t): x'
				getConjugateX(pix_H2, new_pix, d2, A, B);

				cvSub(pix, pix_H2, pix_H2);

				double dist = 	pix_H2->data.db[0]*pix_H2->data.db[0] +
					pix_H2->data.db[1]*pix_H2->data.db[1] +
					pix_H2->data.db[2]*pix_H2->data.db[2];
				cvReleaseMat(&pix_H2);
				double pv = exp((double)-1.0*dist/(2.0*RHO_D*RHO_D));
				PV.push_back(pv);
			}

		cvReleaseMat(&new_pix);
		PVs->data.db[K] = *max_element(PV.begin(), PV.end()); 
		std::vector<double>().swap(PV);
	}

	return PVs;
}
double JoseyPv(CvMat* pix,  const std::vector<FrameData*>* _frames,const vector<CvMat**>* Mx, const vector<CvMat**>* My, const vector<CvMat*>* A, const vector<CvMat*>* B,int level){
	double sigma_c = 10.0;
	double depth = dK(level);
	vector<double> frameCost;
	IplImage* img1 = (*_frames)[0]->getImg();
	CvMat* pix_h = cvCreateMat(3,1,CV_64FC1);
	CvMat* new_pix = cvCreateMat(3,1,CV_64FC1);
	CvMat* _tmp = cvCreateMat(3,1,CV_64FC1);
	double pcv = 0;
	double xDist,yDist;
	CvMat* _dist = cvCreateMat(neighbor*neighbor, 1, CV_64FC1);

	for(int i = 1; i != (int)(*_frames).size(); i++)		// for each frame get a cost
	{
		pix_h->data.db[0] = (*A)[i-1]->data.db[0]*pix->data.db[0] + 
			(*A)[i-1]->data.db[1]*pix->data.db[1] +
			(*A)[i-1]->data.db[2]*pix->data.db[2];
		pix_h->data.db[1] = (*A)[i-1]->data.db[3]*pix->data.db[0] + 
			(*A)[i-1]->data.db[4]*pix->data.db[1] +
			(*A)[i-1]->data.db[5]*pix->data.db[2];
		pix_h->data.db[2] = (*A)[i-1]->data.db[6]*pix->data.db[0] + 
			(*A)[i-1]->data.db[7]*pix->data.db[1] +
			(*A)[i-1]->data.db[8]*pix->data.db[2];
		//cvMatMul(A, pix, pix_h);		//tmp

		int indx;
		double z =  pix_h->data.db[2] + depth*(*B)[i-1]->data.db[2];
		new_pix->data.db[0] = (pix_h->data.db[0] + depth*(*B)[i-1]->data.db[0])/z;

		new_pix->data.db[1] = (pix_h->data.db[1] + depth*(*B)[i-1]->data.db[1])/z;
		new_pix->data.db[2] = 1;

		assert(new_pix->data.db[2] != 0.);

		if (new_pix->data.db[0] < 0 || new_pix->data.db[0] >= img1->width || 
				new_pix->data.db[1] < 0 || new_pix->data.db[1] >= img1->height){
			continue;
		}
		else{
			// Pc
			double pcDist = ColorDiff(pix, new_pix, img1, (*_frames)[i]->getImg());
			pcDist = sigma_c / (sigma_c + pcDist);
			// pv
			indx = floor(new_pix->data.db[1]) * img1->width + floor(new_pix->data.db[0]);	// pix index
			CvMat* _mx = ((*Mx)[i-1])[indx];
			CvMat* _my = ((*My)[i-1])[indx];
			for(int nei = 0; nei<neighbor*neighbor; nei++)
			{
				xDist = _mx->data.db[nei]-pix->data.db[0];
				xDist *= xDist;
				yDist = _my->data.db[nei]-pix->data.db[1];
				yDist *= yDist;
				_dist->data.db[nei] = xDist+yDist;			
			}
			double* minPvDist = std::min_element(_dist->data.db,_dist->data.db+neighbor*neighbor);
			/*
			   cvZero(_x);
			   cvAddS(_x, cvRealScalar(pix->data.db[0]), _x);
			   cvSub(_x, _mx, _x);
			   cvMul(_x, _x, _x);
			   cvZero(_y);
			   cvAddS(_y, cvRealScalar(pix->data.db[1]), _y);
			   cvSub(_y, _my, _y);
			   cvMul(_y, _y, _y);
			   cvAdd(_x, _y, _dist);
			   double _max, _min;
			   cvMinMaxLoc(_dist, &_min, &_max);
			   */
			double pvDist =  exp((double)-1.0*(*minPvDist)/(2.0*RHO_D*RHO_D));
			//cvReleaseMat(&tmp_B);
			frameCost.push_back(pcDist*pvDist);
		}
	}
	if(frameCost.size() < (*_frames).size()/2){	// enough few
		if(frameCost.size() == 0)				// all is out of range
			pcv = -1;
		else{
			pcv = std::accumulate(frameCost.begin(),frameCost.end(),0.);
			pcv = pcv*(double)NO_NEIGH*2./(double)frameCost.size();
		}
	}
	else{										// do sort
		sort(frameCost.begin(),frameCost.end(),greater<double>());
		pcv = std::accumulate(frameCost.begin(),frameCost.begin()+frameCost.size()/2,0.);
		pcv = pcv*(double)NO_NEIGH/(double)frameCost.size();
	}
	vector<double> _vtmp;
	frameCost.clear();
	frameCost.swap(_vtmp );
	cvReleaseMat(&pix_h);
	cvReleaseMat(&_tmp);
	cvReleaseMat(&new_pix);
	cvReleaseMat(&_dist);
	return pcv;
}


CvMat* Pv(CvMat* pix,  FrameData* f1, FrameData* f2, CvMat** Mx, CvMat** My, CvMat* A, CvMat* B){

	IplImage* img1 = f1->getImg();

	//	CvMat* R_tran = cvCreateMat(3,3,CV_64FC1);
	//	CvMat* K_inver = cvCreateMat(3,3,CV_64FC1);
	//	CvMat* tmp_A = cvCreateMat(3,3,CV_64FC1);
	//	CvMat* tmp_B = cvCreateMat(3,1,CV_64FC1);
	CvMat* pix_h = cvCreateMat(3,1,CV_64FC1);
	//
	//	cvTranspose(f2->getR(), R_tran);			
	//	cvInvert(f1->getK(), K_inver); 
	//	cvSub(f1->getT(), f2->getT(), tmp_B); 
	//	cvMatMul(f2->getK(), R_tran, tmp_A);
	//	cvMatMul(tmp_A, tmp_B, tmp_B);
	//	cvMatMul(tmp_A, f1->getR(), tmp_A);
	//	cvMatMul(tmp_A, K_inver, tmp_A);
	cvMatMul(A, pix, pix_h);		//tmp
	//	cvReleaseMat(&R_tran);
	//	cvReleaseMat(&K_inver);
	//	cvReleaseMat(&tmp_A);

	CvMat* _tmp = cvCreateMat(3,1,CV_64FC1);
	CvMat* new_pix = cvCreateMat(3,1,CV_64FC1);

	int indx;
	CvMat *PVs = cvCreateMat(LEVEL+1, 1 ,CV_64FC1);

	CvMat* _x = cvCreateMat(neighbor*neighbor, 1, CV_64FC1);
	CvMat* _y = cvCreateMat(neighbor*neighbor, 1, CV_64FC1);
	CvMat* _dist = cvCreateMat(neighbor*neighbor, 1, CV_64FC1);

	for(int K =0; K <= LEVEL; K++){
		double kk = dK(K);
		cvConvertScale(B, _tmp, kk);
		cvAdd(pix_h, _tmp, new_pix);
		cvConvertScale(new_pix, new_pix, (double)(1.0/new_pix->data.db[2]));  // normalize

		int _xx = min(max((int)new_pix->data.db[0], 0), img1->width-1);
		int _yy = min(max((int)new_pix->data.db[1], 0), img1->height-1);
		indx = _yy * img1->width + _xx;	// pix index

		CvMat* _mx = Mx[indx];
		CvMat* _my = My[indx];

		cvZero(_x);
		cvAddS(_x, cvRealScalar(pix->data.db[0]), _x);
		cvSub(_x, _mx, _x);
		cvMul(_x, _x, _x);

		cvZero(_y);
		cvAddS(_y, cvRealScalar(pix->data.db[1]), _y);
		cvSub(_y, _my, _y);
		cvMul(_y, _y, _y);

		cvAdd(_x, _y, _dist);
		double _max, _min;
		cvMinMaxLoc(_dist, &_min, &_max);
		_min = exp((double)-1.0*_min/(2.0*RHO_D*RHO_D));
		PVs->data.db[K] = _min;
	}
	//cvReleaseMat(&tmp_B);
	cvReleaseMat(&pix_h);
	cvReleaseMat(&_tmp);
	cvReleaseMat(&new_pix);
	cvReleaseMat(&_x);
	cvReleaseMat(&_y);
	cvReleaseMat(&_dist);

	return PVs;
}

CvMat* Pcv(CvMat* pix,  FrameData* f1, FrameData* f2){

	double sigma_c = 10.0;
	IplImage* img1 = f1->getImg();
	CvMat *PC = cvCreateMat(LEVEL+1, 1 ,CV_64FC1);
	CvMat *PVs = cvCreateMat(LEVEL+1, 1 ,CV_64FC1);
	CvMat *PCV = cvCreateMat(LEVEL+1, 1 ,CV_64FC1);
	CvMat* R_tran = cvCreateMat(3,3,CV_64FC1);
	CvMat* K_inver = cvCreateMat(3,3,CV_64FC1);
	CvMat* tmp = cvCreateMat(3,3,CV_64FC1);
	CvMat* tmp2 = cvCreateMat(3,1,CV_64FC1);
	CvMat* pix_h = cvCreateMat(3,1,CV_64FC1);

	cvTranspose(f2->getR(), R_tran);			
	cvInvert(f1->getK(), K_inver); 
	cvSub(f1->getT(), f2->getT(), tmp2); 
	cvMatMul(f2->getK(), R_tran, tmp);
	cvMatMul(tmp, tmp2, tmp2);
	cvMatMul(tmp, f1->getR(), tmp);
	cvMatMul(tmp, K_inver, tmp);
	cvMatMul(tmp, pix, pix_h);		//tmp

	CvMat* D2 = f2->getD();
	CvMat* A = cvCreateMat(3,3,CV_64FC1);
	CvMat* B = cvCreateMat(3,1,CV_64FC1);
	GetAB(f2, f1, A, B);
	for(int K =0; K <= LEVEL; K++){
		CvMat* new_pix = cvCreateMat(3,1,CV_64FC1);
		CvMat* _tmp = cvCreateMat(3,1,CV_64FC1);
		cvConvertScale(tmp2, _tmp, dK(K));
		cvAdd(pix_h, _tmp, new_pix);
		cvConvertScale(new_pix, new_pix, (double)(1.0/new_pix->data.db[2]));  // normalize
		cvReleaseMat(&_tmp);

		double pc = 0.0;
		if (new_pix->data.db[0] >= 0 && new_pix->data.db[0] < img1->width && 
				new_pix->data.db[1] >= 0 && new_pix->data.db[1] < img1->height){
			// pix_H  not out of ramge!
			double dist = ColorDiff(pix, new_pix, img1, f2->getImg());
			pc =  (double)(sigma_c / (sigma_c + dist));
		}
		else
			pc = 10000;
		PC->data.db[K] = pc;

		std::vector<double> PV;

		for (int i = -(int) neighbor/2; i <= (int) neighbor/2; ++i){
			for (int j = -(int) neighbor/2; j <= (int) neighbor/2; ++j){

				int y = min(max( 0, ((int) new_pix->data.db[1])+i), D2->rows-1);
				int x = min(max( 0, ((int) new_pix->data.db[0])+j), D2->cols-1);

				//double d2 = dK(D2->data.db[y*D2->cols+x]);
				double d2 = D2->data.db[y*D2->cols+x];

				CvMat* pix_H2 = cvCreateMat(3,1,CV_64FC1);		//  L(t'->t): x'
				getConjugateX(pix_H2, new_pix, d2, A, B);

				cvSub(pix, pix_H2, pix_H2);

				double dist = 	pix_H2->data.db[0]*pix_H2->data.db[0] +
					pix_H2->data.db[1]*pix_H2->data.db[1] +
					pix_H2->data.db[2]*pix_H2->data.db[2];
				cvReleaseMat(&pix_H2);

				double pv = exp((double)-1.0*dist/(2.0*RHO_D*RHO_D));

				PV.push_back(pv);
			}
		}
		cvReleaseMat(&new_pix);
		PVs->data.db[K] = *max_element(PV.begin(), PV.end()); 
		std::vector<double>().swap(PV);
	}

	double min_l, max_l;
	cvMinMaxLoc(PC, &min_l, &max_l);
	for(int K = 0; K <= LEVEL; K++)
		if(PC->data.db[K] == 10000) PC->data.db[K] = min_l;
	cvMul(PC,PVs,PCV);

	cvReleaseMat(&PC);
	cvReleaseMat(&PVs);
	cvReleaseMat(&R_tran);
	cvReleaseMat(&K_inver);
	cvReleaseMat(&tmp);
	cvReleaseMat(&tmp2);
	cvReleaseMat(&pix_h);

	return PCV;
}

double PCV(CvMat* pix, double d1, FrameData* f1, FrameData* f2){
	double sigma_c = 10.0, pc = 0.0;
	std::vector<double> PV;
	IplImage* img1 = f1->getImg();
	CvMat* pix_H = cvCreateMat(3,1,CV_64FC1);		//  L(t->t'): x_t'
	getConjugateX(pix_H, pix, d1, f1, f2);

	if (pix_H->data.db[0] >= 0 && pix_H->data.db[0] < img1->width && 
			pix_H->data.db[1] >= 0 && pix_H->data.db[1] < img1->height){
		// pix_H  not out of ramge!
		double dist = ColorDiff(pix, pix_H, img1, f2->getImg());
		pc =  (double)(sigma_c / (sigma_c + dist));
	}

	for (int i = -(int) neighbor/2; i <= (int) neighbor/2; ++i)
		for (int j = -(int) neighbor/2; j <= (int) neighbor/2; ++j){
			CvMat* D2 = f2->getD();
			int y = min(max( 0, ((int) pix_H->data.db[1])+i), D2->rows-1);
			int x = min(max( 0, ((int) pix_H->data.db[0])+j), D2->cols-1);

			double d2 = D2->data.db[y*D2->cols+x];

			CvMat* pix_H2 = cvCreateMat(3,1,CV_64FC1);		//  L(t'->t): x'
			getConjugateX(pix_H2, pix_H, d2, f2, f1);

			cvSub(pix, pix_H2, pix_H2);
			double dist = 	pix_H2->data.db[0]*pix_H2->data.db[0] +
				pix_H2->data.db[1]*pix_H2->data.db[1] +
				pix_H2->data.db[2]*pix_H2->data.db[2];
			cvReleaseMat(&pix_H2);
			double pv = exp((double)-1.0*dist/(2.0*RHO_D*RHO_D));
			PV.push_back(pv);
		}
	cvReleaseMat(&pix_H);
	double Max_pv= *max_element(PV.begin(), PV.end()); 
	std::vector<double>().swap(PV);
	return (double)Max_pv*pc;
}
bool compOp(const pair<double,int> p1,const pair<double,int> p2)
{
	return p1.first<p2.first;
}
/*
   void JoseyLikelihood_init(int y,int x,std::vector<FrameData*>* _frames, idMap* IdxMap, vector<CvMat*>* Bs,double* dCost){
   double val = 0;
   double minL,maxL,minP,maxP;
   int width = ((*_frames)[0]->getImg())->width;
   int index = y*width+x;
   double* pos = dCost+index*(LEVEL+1);

   for(int i = 0; i <=LEVEL; i++)		// find better frame for each level
   {
   val = JoseyPc(index,_frames,IdxMap, Bs, i);
//cerr<< "val:" << val <<endl;
pos[i] = val;
//cerr<< "pcs:" << pcs->data.db[i] << endl;
}
double* maxPtr = max_element(pos,pos+LEVEL+1);
if(*maxPtr<0)
for(int i=0;i <=LEVEL;i++)
pos[i] = 1;
else
for(int i = 0; i <=LEVEL; i++)
{
if(pos[i]<0)
pos[i] = 0; // or 1??
else
pos[i] = 1 - pos[i]/(*maxPtr);
}
}*/

CvMat* JoseyLikelihood_init(CvMat* pix, std::vector<FrameData*>* _frames, vector<CvMat*>* As, vector<CvMat*>* Bs){
	CvMat *pcs = cvCreateMat(LEVEL+1, 1, CV_64FC1);
	cvZero(pcs);
	double val = 0;
	for(int i = 0; i <=LEVEL; i++)		// find better frame for each level
	{
		val = JoseyPc(pix, _frames, As, Bs, i);
		pcs->data.db[i] = val;
	}
		
	double* maxL = std::max_element(pcs->data.db,pcs->data.db+LEVEL+1);
	/* normalize */
	if(*maxL < 0.0){
		cerr<< "pix with no data cost"<<endl;
		cvZero(pcs);
		cvAddS(pcs,cvRealScalar(1.0), pcs);
	}
	else{	
		// formula(3)
		std::replace(pcs->data.db,pcs->data.db+LEVEL+1,-1.,*maxL);
		CvMat *tmp = cvCreateMat(LEVEL+1,1,CV_64FC1);
		cvZero(tmp);
		cvAddWeighted(pcs, -1.0*(1.0/ *maxL), tmp, 0, 1, pcs);
	}
	return pcs;
}
CvMat* Likelihood_init(CvMat* pix, std::vector<FrameData*> _frames, vector<CvMat*>As, vector<CvMat*>Bs){
	CvMat *pcs = cvCreateMat(LEVEL+1, 1, CV_64FC1);
	CvMat *pc = cvCreateMat(LEVEL+1, 1, CV_64FC1);
	CvMat *cnt = cvCreateMat(LEVEL+1, 1, CV_64FC1);
	CvMat *tmp = cvCreateMat(LEVEL+1, 1, CV_64FC1);
	cvZero(pcs);
	cvZero(cnt);
	cvZero(tmp);
	size_t t = clock();
	for(int i = 1; i != (int) _frames.size(); ++i){
		pc = Pc(pix,  _frames[0], _frames[i], As[i-1], Bs[i-1], cnt);  //edit by josey
		cvAdd(pcs, pc, pcs);
	}
	cvAddWeighted(cnt,1,tmp,0,0.000001,cnt);
	cvDiv(pcs,cnt,pcs);
	cvConvertScale(pcs,pcs,(double)NO_NEIGH*2.);
	size_t tt = clock();
	//cout << "like_init " << (double)(tt-t)/CLOCKS_PER_SEC << endl;
	cvReleaseMat(&cnt);
	cvReleaseMat(&tmp);
	cvReleaseMat(&pc);
	return pcs;
}
CvMat* JoseyLikelihood(CvMat* pix, std::vector<FrameData*>* _frames,vector<CvMat**>* MapXs,vector<CvMat**>* MapYs, vector<CvMat*>* As, vector<CvMat*>* Bs){
	CvMat *pcs = cvCreateMat(LEVEL+1, 1, CV_64FC1);
	cvZero(pcs);
	double val =0;
	for(int i = 0; i <=LEVEL; i++)		// find better frame for each level
	{
		val = JoseyPv(pix, _frames,MapXs,MapYs,As,Bs,i);
		pcs->data.db[i] = val;
	}
	double* maxL = std::max_element(pcs->data.db,pcs->data.db+LEVEL+1);	
	/* normalize */
	if(*maxL < 0.0){
		cerr<< "pix with no data cost"<<endl;
		cvZero(pcs);
		cvAddS(pcs,cvRealScalar(1.0), pcs);
	}
	else{	
		// formula(3)
		std::replace(pcs->data.db,pcs->data.db+LEVEL+1,-1.,*maxL);
		CvMat *tmp = cvCreateMat(LEVEL+1,1,CV_64FC1);
		cvZero(tmp);
		cvAddWeighted(pcs, -1.0*(1.0/ *maxL), tmp, 0, 1, pcs);
	}

	return pcs;
}
CvMat* Likelihood(CvMat* pix, std::vector<FrameData*> _frames,vector<CvMat**>MapXs,vector<CvMat**>MapYs, vector<CvMat*>As, vector<CvMat*>Bs){
	CvMat *pcs = cvCreateMat(LEVEL+1, 1, CV_64FC1);
	cvZero(pcs);
	int width = _frames[0]->getImg()->width;
	int height = _frames[0]->getImg()->height;
	for(int i = 1; i != (int)_frames.size(); ++i){
		CvMat *pc = Pc(pix, _frames[0], _frames[i], As[i-1], Bs[i-1]);
		CvMat *pv = Pv(pix, _frames[0], _frames[i], MapXs[i-1], MapYs[i-1], As[i-1], Bs[i-1]);
		cvMul(pc, pv, pc);
		cvAdd(pcs, pc, pcs);
		cvReleaseMat(&pc);
		cvReleaseMat(&pv);
	}
	return pcs;
}

/* Foreground ---> only use pv */
CvMat* Likelihood_FG(CvMat* pix, std::vector<FrameData*> _frames,vector<CvMat**>MapXs,vector<CvMat**>MapYs, vector<CvMat*>As, vector<CvMat*>Bs){
	CvMat *pvs = cvCreateMat(LEVEL+1, 1, CV_64FC1);
	cvZero(pvs);
	int width = _frames[0]->getImg()->width;
	int height = _frames[0]->getImg()->height;
	for(int i = 1; i != (int)_frames.size(); ++i){
		CvMat *pv = Pv(pix, _frames[0], _frames[i], MapXs[i-1], MapYs[i-1], As[i-1], Bs[i-1]);
		cvAdd(pvs, pv, pvs);
		cvReleaseMat(&pv);
	}
	return pvs;
}
double Likelihood_init(CvMat *pix, double d, std::vector<FrameData*> _frames){
	double pcs = 0.0;
	//size_t t = clock();
	for(int i = 1; i != (int)_frames.size(); ++i){
		pcs += Pc(pix, d, _frames[0], _frames[i]);		
	}
	return pcs;
}
/*
   void Ed_init(int y,int x, std::vector<FrameData*>* _frames,idMap* IdxMap, vector<CvMat*>* Bs,double* dCost){


   JoseyLikelihood_init(y,x,_frames,IdxMap,Bs,dCost);		//edit by josey select frames
   }*/
void* multi_Ed(void *arg){
	int BlockPix = sizeX*sizeY/core+1;
	int progress =0;
	threadArg* myArg= (threadArg*)arg;
	CvMat* ED = cvCreateMat(LEVEL+1,1,CV_64FC1);
	CvMat* pix = cvCreateMat(3,1,CV_64FC1);
	CvMat *tmp = cvCreateMat(LEVEL+1,1,CV_64FC1);
	cvZero(tmp);
	pix->data.db[2] = 1;
	int startPix = myArg->threadId*BlockPix;
	for(int i = startPix;i<(myArg->threadId+1)*BlockPix && i<sizeX*sizeY;i++)
	{

		pix->data.db[0] = i%sizeX;
		pix->data.db[1] = i/sizeX;
		ED = JoseyLikelihood(pix,myArg->_frames, myArg->MapXs, myArg->MapYs, myArg->As, myArg->Bs);
		/*
		double min_l, max_l;
		cvMinMaxLoc(ED, &min_l, &max_l);

		if(max_l == 0.0){
			cvZero(ED);
			cvAddS(ED,cvRealScalar(1.0), ED);
		}
		else{
			CvMat *tmp = cvCreateMat(LEVEL+1,1,CV_64FC1);
			cvZero(tmp);
			cvAddWeighted(ED, -1.0*(1.0/max_l), tmp, 0, 1, ED);
			//	ED->data.db[k] = 1.0- ((double)ED->data.db[k]/max_l);
		}*/

		copy(ED->data.db,ED->data.db+LEVEL+1,myArg->dCost + i * (LEVEL+1));
	}
	cvReleaseMat(&ED);
	cvReleaseMat(&pix);
	cvReleaseMat(&tmp);
}


void *multi_Ed_init(void *arg)
{
	int BlockPix = sizeX*sizeY/core+1;
	int progress =0;
	threadArg* myArg= (threadArg*)arg;
	CvMat* ED = cvCreateMat(LEVEL+1,1,CV_64FC1);
	CvMat* pix = cvCreateMat(3,1,CV_64FC1);
	CvMat *tmp = cvCreateMat(LEVEL+1,1,CV_64FC1);
	cvZero(tmp);
	pix->data.db[2] = 1;
	int startPix = myArg->threadId*BlockPix;
	for(int i = startPix;i<(myArg->threadId+1)*BlockPix && i<sizeX*sizeY;i++)
	{
		pix->data.db[0] = i%sizeX;
		pix->data.db[1] = i/sizeX;
		ED = JoseyLikelihood_init(pix, myArg->_frames, myArg->As, myArg->Bs);		//edit by josey select frames
		/*
		double min_l, max_l;
		cvMinMaxLoc(ED, &min_l, &max_l);
		// normalize /
		if(max_l <= 0.0){
			cerr<< "pix with no data cost"<<endl;
			cvZero(ED);
			cvAddS(ED,cvRealScalar(1.0), ED);
		}
		else{	
			// formula(3)
			cvAddWeighted(ED, -1.0*(1.0/max_l), tmp, 0, 1, ED);
		}*/
		copy(ED->data.db,ED->data.db+LEVEL+1,myArg->dCost + i * (LEVEL+1));
	}
	cvReleaseMat(&ED);
	cvReleaseMat(&pix);
	cvReleaseMat(&tmp);
}

CvMat* Ed_init(CvMat *pix, std::vector<FrameData*>* _frames, vector<CvMat*>* As, vector<CvMat*>* Bs){
	CvMat* ED = cvCreateMat(LEVEL+1,1,CV_64FC1);
	ED = JoseyLikelihood_init(pix,  _frames, As, Bs);		//edit by josey select frames
	double min_l, max_l;
	cvMinMaxLoc(ED, &min_l, &max_l);
	if(max_l == 0.0){
		cerr<< "pix with no data cost"<<endl;
		cvZero(ED);
		cvAddS(ED,cvRealScalar(1.0), ED);
	}
	else{	
		// formula(3)
		CvMat *tmp = cvCreateMat(LEVEL+1,1,CV_64FC1);
		cvZero(tmp);
		cvAddWeighted(ED, -1.0*(1.0/max_l), tmp, 0, 1, ED);
		cvReleaseMat(&tmp);
		//	ED->data.db[k] = 1.0- ((double)ED->data.db[k]/max_l);
	}
	return ED;
}

CvMat* Ed(CvMat *pix, std::vector<FrameData*> _frames,vector<CvMat**>MapXs,vector<CvMat**>MapYs, vector<CvMat*>As, vector<CvMat*>Bs){
	CvMat *ED = Likelihood(pix,  _frames, MapXs, MapYs, As, Bs);
	double min_l, max_l;
	cvMinMaxLoc(ED, &min_l, &max_l);

	/* normalize */
	if(max_l == 0.0){
		cvZero(ED);
		cvAddS(ED,cvRealScalar(1.0), ED);
	}
	else{
		CvMat *tmp = cvCreateMat(LEVEL+1,1,CV_64FC1);
		cvZero(tmp);
		cvAddWeighted(ED, -1.0*(1.0/max_l), tmp, 0, 1, ED);
		//	ED->data.db[k] = 1.0- ((double)ED->data.db[k]/max_l);
	}
	return ED;
}

/* Foreground --> only using pv */
CvMat* Ed_FG(CvMat *pix, std::vector<FrameData*> _frames,vector<CvMat**>MapXs,vector<CvMat**>MapYs, vector<CvMat*>As, vector<CvMat*>Bs){
	CvMat *ED = Likelihood_FG(pix,  _frames, MapXs, MapYs, As, Bs);
	double min_l, max_l;
	cvMinMaxLoc(ED, &min_l, &max_l);

	/* normalize */
	if(max_l == 0.0){
		cvZero(ED);
		cvAddS(ED,cvRealScalar(1.0), ED);
	}
	else{
		CvMat *tmp = cvCreateMat(LEVEL+1,1,CV_64FC1);
		cvZero(tmp);
		cvAddWeighted(ED, -1.0*(1.0/max_l), tmp, 0, 1, ED);
		//	ED->data.db[k] = 1.0- ((double)ED->data.db[k]/max_l);
	}
	return ED;
}

//void Ed(CvMat *pix, std::vector<FrameData*> _frames, MRF::CostVal *dCost){
//	int width = _frames[0]->getImg()->width;
//	int n = pix->data.db[1]*width + pix->data.db[0];
//	CvMat *ED = cvCreateMat(LEVEL+1, 1, CV_64FC1);
//
//	for(int k = 0; k <= LEVEL; ++k){
//		double tmp = Likelihood(pix, dK(k), _frames);
//		dCost[n*(LEVEL+1)+k] = tmp;
//		ED->data.db[k] = tmp;
//
//	}
//	double min_l, max_l;
//	cvMinMaxLoc(ED, &min_l, &max_l);
//	cvReleaseMat(&ED);
//	/* normalize */
//	for(int k = 0; k <= LEVEL; ++k){
//		if(max_l == 0.0 )	dCost[n*(LEVEL+1)+k] = 1.0;
//		else 
//			dCost[n*(LEVEL+1)+k] = 1.0 - (double)(dCost[n*(LEVEL+1)+k]/max_l);
//	}
//}

double Ed_init(CvMat* pix, double d, std::vector<FrameData*> _frames){
	std::vector<double> likelihood;
	for(int k = 0; k <= LEVEL; ++k){
		double tmp = 	Likelihood_init(pix, dK(k), _frames);
		likelihood.push_back(tmp);
	}
	double max_l = *max_element(likelihood.begin(), likelihood.end());
	double _likelihood = Likelihood_init(pix, d, _frames);
	//double _likelihood = Likelihood_init(pix, dK(Kd(d)), _frames);
	return (max_l == 0.0) ? (1.0) : (1.0-(double)_likelihood/max_l);
}

double u_lambda(CvMat* pix, IplImage* img){
	double dist = 0.0;
	//IplImage* img = _frames[frame]->getImg();
	//int wStep = img->widthStep;

	CvMat *new_pix = cvCreateMat(3,1,CV_64FC1);
	new_pix->data.db[2] = 1.0; 
	// (x,y) (x,y-1)
	new_pix->data.db[1] = min(max( 0.0, ( pix->data.db[1])-1.), (double)img->height-1);
	new_pix->data.db[0] = pix->data.db[0];
	double diff = ColorDiff(pix, new_pix, img, img);
	dist += (double) (1.0/ (diff+EPSILON));
	// (x,y) (x,y+1)
	new_pix->data.db[1] = min(max( 0.0, ( pix->data.db[1])+1.), (double)img->height-1);
	diff = ColorDiff(pix, new_pix, img, img);
	dist += (double) (1.0/ (diff+EPSILON));
	// (x,y) (x+1,y)
	new_pix->data.db[1] = pix->data.db[1];
	new_pix->data.db[0] = min(max( 0.0, ( pix->data.db[0])+1.), (double)img->width-1);
	diff = ColorDiff(pix, new_pix, img, img);
	dist += (double) (1.0/ (diff+EPSILON));
	// (x,y) (x-1,y)
	new_pix->data.db[0] = min(max( 0.0, ( pix->data.db[0])-1.), (double)img->width-1);
	diff = ColorDiff(pix, new_pix, img, img);
	dist += (double) (1.0/ (diff+EPSILON));
	// |N(x)| = 4
	return (double) (4.0 / dist);
}

double lambda(CvMat* pix1, CvMat* pix2, IplImage* img){
	double lambda = 0.0;
	//double OMEGA_S = 5.0 / (D_MAX - D_MIN);
	//IplImage* img = _frames[frame]->getImg();
	//int wStep = img->widthStep;
	double color_diff = ColorDiff(pix1, pix2, img, img);

	lambda = (double) (u_lambda(pix1, img)/(color_diff+EPSILON));
	//cout << "lambda: " << lambda << endl;	//
	return lambda;
}

double rho(double d1, double d2){
	//double d1 = dK(L1);
	//double d2 = dK(L2);
	double d = fabs(d1-d2);
	return min( d, ETA);
}

double rho(CvMat* pix1, CvMat* pix2, CvMat* D){
	//double d1 = dK(D->data.db[D->cols*(int)pix1->data.db[1]+(int)pix1->data.db[0]]);
	//double d2 = dK(D->data.db[D->cols*(int)pix2->data.db[1]+(int)pix2->data.db[0]]);
	double d1 = D->data.db[D->cols*(int)pix1->data.db[1]+(int)pix1->data.db[0]];
	double d2 = D->data.db[D->cols*(int)pix2->data.db[1]+(int)pix2->data.db[0]];
	double d = fabs(d1 - d2);
	return min(d, ETA);
}

double Es(CvMat* pix1, CvMat* pix2, IplImage* img, CvMat* D){
	//double d1 = D[frame]->data.db[D[frame]->cols*(int)pix1->data.db[1]+(int)pix1->data.db[0]];
	//double d2 = D[frame]->data.db[D[frame]->cols*(int)pix2->data.db[1]+(int)pix2->data.db[0]];
	//return lambda(pix1, pix2, frame)*rho(d1,d2);
	return OMEGA_S*lambda(pix1, pix2, img)*rho(pix1, pix2, D);
}

double Es(CvMat* pix1, CvMat* pix2, IplImage* img, double d1, double d2){
	//double d1 = D[frame]->data.db[D[frame]->cols*(int)pix1->data.db[1]+(int)pix1->data.db[0]];
	//double d2 = D[frame]->data.db[D[frame]->cols*(int)pix2->data.db[1]+(int)pix2->data.db[0]];
	//return lambda(pix1, pix2, frame)*rho(d1,d2);
	return OMEGA_S*lambda(pix1, pix2, img)*rho(d1, d2);
}
double Es(CvMat* pix, IplImage* img){
	double es= 0.0;
	CvMat *pix2 = cvCreateMat(3,1,CV_64FC1);
	pix2->data.db[2] = 1.0;
	// (x,y) (x,y-1)
	pix2->data.db[1] = min(max( 0.0, ( pix->data.db[1])-1.0), (double)img->height-1);
	pix2->data.db[0] = min(max( 0.0, ( pix->data.db[0])), (double)img->width-1);
	es += OMEGA_S*lambda(pix, pix2, img);
	// (x,y) (x,y+1)
	pix2->data.db[1] = min(max( 0.0, ( pix->data.db[1])+1.0), (double)img->height-1);
	es += OMEGA_S*lambda(pix, pix2, img);
	// (x,y) (x+1,y)
	pix2->data.db[1] = min(max( 0.0, ( pix->data.db[1])), (double)img->height-1);
	pix2->data.db[0] = min(max( 0.0, ( pix->data.db[0])+1.0), (double)img->width-1);
	es += OMEGA_S*lambda(pix, pix2, img);
	// (x,y) (x-1,y)
	pix2->data.db[0] = min(max( 0.0, ( pix->data.db[0])-1.0), (double)img->width-1);
	es += OMEGA_S*lambda(pix, pix2, img);

	return es;
}
