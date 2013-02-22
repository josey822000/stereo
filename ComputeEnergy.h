#ifndef COMPUTEENERGY_H
#define COMPUTEENERGY_H

#include "ReadCameraParameter.h"
#include "ReadVideoSeq.h"
#include "FrameData.h"
#include "globalVar.h"
#include "FileIO.h"

#include "MRF/mrf.h"
#include "MRF/GCoptimization.h"
#include "MRF/MaxProdBP.h"
#include "MRF/BP-S.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <new>
#include <string>
#include <vector>
#include <map>
class threadArg{
    public:
       	int threadId;
       	double* dCost;
       	vector<CvMat*>* As;
       	vector<CvMat*>* Bs;
       	vector<CvMat**>* MapXs;
       	vector<CvMat**>* MapYs;
		vector<FrameData*>* _frames;
};
class idMap{
    public:
    idMap(int s, int w, int h)
    {
        sz = s;
        ptr = new CvMat*[sz];
        for(int i = 0;i<s;i++)
            ptr[i] = cvCreateMat(3,h*w,CV_64FC1);
        for(int y = 0;y<h;y++)
            for(int x=0;x<w;x++)
            {
                ptr[0]->data.db[(y*w+x)] = x;
                ptr[0]->data.db[(y*w+x)+ h*w] = y;
                ptr[0]->data.db[(y*w+x)+ h*w*2] = 1;
            }
    }
    void CalcTrans(vector<CvMat*>* A)
    {
        //cvMat* tmpF0 = cvCreate(ptr[0].size(),CV_64FC1);
        //cvMat* tmpFi = cvCreate(ptr[0].size(),CV_64FC1);
        //cvConvert(ptr[0],tmpF0);
        for(int i=1;i<sz;i++)
        {
            //cvMatMul((*A)[i-1],ptr[0],ptr[i]);

            //cvMat* tmpMap = cvCreate(ptr[0].size(),CV_64FC1);
            cvMatMul((*A)[i-1],ptr[0],ptr[i]);
            //cvGEMM((*A)[i-1],tmpF0,1,tmp,dx,tmpMap);
            //cvReleaseMat(&tmp);
            //cvConvertPointsHomogeneous(tmpMap,tmpFi);
            //cvConvert(tmpFi,ptr[i]);
            //cvReleaseMat(&tmpMap);
        }
        //cvReleaseMat(&tmpF0);
        //cvReleaseMat(&tmpFi);
   
    }
    /*
    void CalcTrans(vector<CvMat*>* A,vector<CvMat*>* B,int level)
    {
        double dx = dK(level);
        cvMat* tmpF0 = cvCreate(ptr[0].size(),CV_64FC1);
        cvMat* tmpFi = cvCreate(ptr[0].size(),CV_64FC1);
        cvConvert(ptr[0],tmpF0);
        for(int i=1;i<sz;i++)
        {
            //cvMatMul((*A)[i-1],ptr[0],ptr[i]);

            cvMat* tmpMap = cvCreate(ptr[0].size(),CV_64FC1);
            CvMat* tmp = cvCreate(ptr[0].size(),CV_64FC1);
            cvRepeat((*B)[i-1],tmp);
            cvGEMM((*A)[i-1],tmpF0,1,tmp,dx,tmpMap);
            cvReleaseMat(&tmp);
            cvConvertPointsHomogeneous(tmpMap,tmpFi);
            cvConvert(tmpFi,ptr[i]);
            cvReleaseMat(&tmpMap);
        }
        cvReleaseMat(&tmpF0);
        cvReleaseMat(&tmpFi);
    }*/
    ~idMap()
    {
        for(int i=0;i<sz;i++)
            cvReleaseMat(&(ptr[i]));
        delete [] ptr;
    }
    CvMat** ptr;
    int sz;
};


void GetAB(FrameData *f1, FrameData *f2, CvMat *A, CvMat *B);
void GetPvMap (FrameData *f1, FrameData *f2, CvMat **Mx, CvMat **My);
void GetFrame(ReadCameraParameter* CP, ReadVideoSeq* VS, int now_frame, int Nframe, std::vector<FrameData*>* frames);
double dK(int k);					// get the disparity of level k
int Kd(double d);
//void getConjugateX(CvMat* pix_h, CvMat* pix, double d, CvMat* K, CvMat* R, CvMat* T, CvMat* K2, CvMat* R2, CvMat* T2);
void getConjugateX(CvMat* pix_h, CvMat* pix, double d, CvMat* A, CvMat* B);
void getConjugateX(CvMat* pix_h, CvMat* pix, double d, FrameData* f1, FrameData* f2);
	/* data term */
CvMat* Pc(CvMat* pix, FrameData* f1, FrameData* f2, CvMat* A, CvMat* B);
CvMat* Pv(CvMat* pix,  FrameData* f1, FrameData* f2);
CvMat* Pv(CvMat* pix,  FrameData* f1, FrameData* f2, CvMat** Mx, CvMat** My, CvMat* A, CvMat* B);
CvMat* Pcv(CvMat* pix, FrameData* f1, FrameData* f2);
double Pc(CvMat* pix, double d, FrameData* f1, FrameData* f2);	// color similarity term
double ColorDiff(CvMat* pix, CvMat* pix2, IplImage* img1, IplImage* img2);
//double Pv(CvMat* pix, double d1, FrameData* f1, FrameData* f2);		// geometry  coherence term
double PCV(CvMat* pix, double d1, FrameData* f1, FrameData* f2);

CvMat* Likelihood_init(CvMat* pix,  std::vector<FrameData*> _frames, std::vector<CvMat*>As, std::vector<CvMat*>Bs);
// Josey vvvvv
CvMat* JoseyLikelihood_init(CvMat* pix,  std::vector<FrameData*>* _frames, std::vector<CvMat*>* As, std::vector<CvMat*>* Bs);
CvMat* JoseyLikelihood(CvMat* pix, std::vector<FrameData*>* _frames, std::vector<CvMat**>* MapXs,std::vector<CvMat**>* MapYs, std::vector<CvMat*>* As, std::vector<CvMat*>* Bs);
void *multi_Ed_init(void *arg);
void *multi_Ed(void *arg);
// Josey ^^^^^^
CvMat* Likelihood(CvMat* pix, std::vector<FrameData*> _frames, std::vector<CvMat**>MapXs,std::vector<CvMat**>MapYs, std::vector<CvMat*>As, std::vector<CvMat*>Bs);

CvMat* Likelihood_FG(CvMat* pix, std::vector<FrameData*> _frames, std::vector<CvMat**>MapXs,std::vector<CvMat**>MapYs, std::vector<CvMat*>As, std::vector<CvMat*>Bs);
double Likelihood_init(CvMat* pix, double d, std::vector<FrameData*> _frames);		// for data term 
	
double Ed_init(CvMat* pix, double d, std::vector<FrameData*> _frames);	// data term

void Ed_init(int y,int x, std::vector<FrameData*>* _frames,idMap* IdxMap, vector<CvMat*>* Bs,double* dCost);  //edit by josey

//void Ed_init(CvMat *pix, std::vector<FrameData*> _frames, MRF::CostVal *dCost);
CvMat* Ed_init(CvMat *pix, std::vector<FrameData*>* _frames, std::vector<CvMat*>* As, std::vector<CvMat*>* Bs);
CvMat* Ed(CvMat *pix, std::vector<FrameData*> _frames, std::vector<CvMat**>MapXs, std::vector<CvMat**>MapYs, std::vector<CvMat*>As, std::vector<CvMat*>Bs);
CvMat* Ed_FG(CvMat *pix, std::vector<FrameData*> _frames, std::vector<CvMat**>MapXs, std::vector<CvMat**>MapYs, std::vector<CvMat*>As, std::vector<CvMat*>Bs);
//void Ed(CvMat *pix, std::vector<FrameData*> _frames, MRF::CostVal *dCost);

/* smoothness term */
double u_lambda(CvMat* pix, IplImage* img);		// normalization fator
double lambda(CvMat* pix1, CvMat* pix2, IplImage* img);	// smoothness weight
double rho(double d1, double d2);				// function
double rho(CvMat* pix1, CvMat* pix2, CvMat* D);	
double Es(CvMat* pix1, CvMat* pix2, IplImage* img, CvMat* D);	// smoothness term
double Es(CvMat* pix1, CvMat* pix2, IplImage* img, double d1, double d2);
double Es(CvMat* pix, IplImage* img);
//void File2Mat(CvMat* m, std::string file);

	/* sum of term for a frame */
	//MRF::CostVal* dCost_init(int frame);
	//void dCost_init(int frame);
	//MRF::CostVal dCost(int frame, int pix, int L);
	//MRF::CostVal sCost(int frame, int pix1, int pix2);
	//EnergyFunction* generateE(int frame);
	//double E();

#endif
