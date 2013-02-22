#include "ComputeMRF.h"
#include "ReadCameraParameter.h"
#include "ReadVideoSeq.h"
#include "FrameData.h"
#include "ComputeEnergy.h"
#include "Segment.h"
#include "FileIO.h"
#include "runAlgo.h"

#include "MRF/ICM.h"
#include "MRF/TRW-S.h"
#include "MRF/mrf.h"
#include "MRF/GCoptimization.h"
#include "MRF/MaxProdBP.h"
#include "MRF/BP-S.h"

#include <fstream>
#include <vector>
#include <map>
#include <ctime>
#include <string>
#include <iostream>
#include <algorithm>

using namespace BFS;
using namespace std;

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/cxcore.h"

using namespace cv;
extern int sizeX;
extern int sizeY;
extern string DIR_SEQ;
void ComputeMRF(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, MRF::CostVal *&dCost, MRF::CostVal *&hCue, MRF::CostVal *&vCue){

	size_t t1 = clock();

	/* save data of frames -> frames */
	std::vector<FrameData*> frames;
	/* ComputeEnergy.cpp */
	GetFrame(cp, vs, now_frame, Nframe, &frames);

	IplImage* img = frames[0]->getImg();
	int height = img->height;
	int width = img->width;

	vector<CvMat*> As, Bs;
	for(int i = 1; i != (int)frames.size(); ++i){
		/* compute A B for conjugate */
		CvMat* a = cvCreateMat(3, 3, CV_64FC1);
		CvMat* b = cvCreateMat(3, 1, CV_64FC1);
		/* ComputeEnergy.cpp */
		GetAB(frames[0], frames[i], a, b);
		As.push_back(a);
		Bs.push_back(b);
	}
	/* smoothness */
	// MRF::CostVal smoothMax = eta =0.05*(D_MAX-D_MIN)
	// lambda = 5/ (D_MAX-D_MIN)
	hCue = new MRF::CostVal[width*height];
	vCue = new MRF::CostVal[width*height];
	dCost = new MRF::CostVal[width*height*(LEVEL+1)];

	int n = 0;
	std::cout << "\nCompute hCue & vCue..." << std::endl;
	CvMat* pix = cvCreateMat(3,1,CV_64FC1);		// (x, y)
	CvMat* pix2 = cvCreateMat(3,1,CV_64FC1);	// (x+1, y) , (x, y+1)
	pix->data.db[2] = 1.;
	pix2->data.db[2] = 1.;
	int indx;
	for(int y = 0; y < height; ++y){
		for(int x = 0; x < width; ++x){
			pix->data.db[0] = x;
			pix->data.db[1] = y;		
			indx = y*width+x;
			if( frames[0]->b_mask && (frames[0]->getMask()->data.db[indx] != 0)){
				// fg
				for(int k = 0; k <= LEVEL; k++)
					dCost[(indx)*(LEVEL+1)+k] = 1.0;   // ??
			}
			else{
				/* Compute data cost */
				/* ComputeEnergy.cpp */
				CvMat *ED = Ed_init(pix, &frames, &As, &Bs);
				assert(ED != NULL);
				for(int k = 0; k <= LEVEL; k++)
					dCost[(y*width+x)*(LEVEL+1)+k] = ED->data.db[k];
				cvReleaseMat(&ED);
			}
			/* smoothness */
			pix2->data.db[0] = x+ (x < (width-1)); // (x+1, y)
			pix2->data.db[1] = y;
			hCue[n] = lambda(pix, pix2, img);	

			pix2->data.db[0] = x;						  // (x, y+1)
			pix2->data.db[1] = y+ ( y < (height-1));		
			vCue[n] = lambda(pix, pix2, img);

			n++;
			//	if (n% 5000 == 0)	cout << n/5000 << "\t"; 
		}
	}
	cvReleaseMat(&pix);
	cvReleaseMat(&pix2);

	size_t t2 = clock();
	cout << "\nMRF_time = " << (double)(t2-t1)/CLOCKS_PER_SEC << std::endl;
}


void ComputeMRF_Seg(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, MRF::CostVal *&dCost, MRF::CostVal *&hCue, MRF::CostVal *&vCue){

	size_t t1 = clock();

	/* save data of frames -> frames */
	std::vector<FrameData*> frames;
	GetFrame(cp, vs, now_frame, Nframe, &frames);

	IplImage* img = frames[0]->getImg();
	int height = img->height;
	int width = img->width;

	vector<CvMat*> As, Bs;
	for(int i = 1; i != (int)frames.size(); ++i){
		/* compute A B for conjugate */
		CvMat* a = cvCreateMat(3, 3, CV_64FC1);
		CvMat* b = cvCreateMat(3, 1, CV_64FC1);
		GetAB(frames[0], frames[i], a, b);
		As.push_back(a);
		Bs.push_back(b);
	}	

	// lambda = 5/ (D_MAX-D_MIN)
	hCue = new MRF::CostVal[width*height];
	vCue = new MRF::CostVal[width*height];
	int n = 0;
	std::cout << "\nCompute hCue & vCue..." << std::endl;
	CvMat* pix = cvCreateMat(3,1,CV_64FC1);		// (x, y)
	CvMat* pix2 = cvCreateMat(3,1,CV_64FC1);	// neighbor
	pix->data.db[2] = 1.;
	pix2->data.db[2] = 1.;

	for(int y = 0; y < height; ++y){
		for(int x = 0; x < width; ++x){
			pix->data.db[0] = x;	
			pix->data.db[1] = y;

			pix2->data.db[0] = x+ (x < (width-1));		// (x+1, y)
			pix2->data.db[1] = y;
			hCue[n] = lambda(pix, pix2, img);	
			pix2->data.db[0] = x;		// (x, y+1)
			pix2->data.db[1] = y+ ( y < (height-1));		
			vCue[n] = lambda(pix, pix2, img);
			n++;
			//	if (n% 5000 == 0)	cout << n/5000 << "\t"; 
		}
	}
	cvReleaseMat(&pix);
	cvReleaseMat(&pix2);


	/*  dCost */
	//MRF::CostVal* dCost_init = NULL; 
	dCost = new MRF::CostVal[width*height*(LEVEL+1)];
	//dCost_ori = new MRF::CostVal[width*height*(LEVEL+1)];
	std::cout << "\nCompute dCost..." << std::endl;

	//n = 0;
	map<int, vector<int> > segment = computeSeg(img->width, img->height, vs->getSeg(now_frame));
	map<int, vector<int> >::iterator it = segment.begin();

	CvMat *sum_ED = cvCreateMat(LEVEL+1,1,CV_64FC1);
	CvMat* pix_seg = cvCreateMat(3,1,CV_64FC1);
	pix_seg->data.db[2] = 1.;
	for(; it!= segment.end(); ++it){
		//cout << "seg no:" << (*it).first << "..." << segment.size() << "\t";
		vector<int>::iterator it_vec;
		n = 0;
		cvZero(sum_ED);

		for(it_vec = (*it).second.begin();it_vec!=(*it).second.end(); ++ it_vec){
			int y = (*it_vec)/(img->width), x = (*it_vec)% (img->width);
			pix_seg->data.db[0] = x;
			pix_seg->data.db[1] = y;
			CvMat *ED = cvCreateMat(LEVEL+1,1,CV_64FC1);
			if( frames[0]->b_mask && (frames[0]->getMask()->data.db[y*width+x] != 0)){
				/* fg region */
				cvZero(ED);
				cvAddS(ED, cvRealScalar(1.0), ED); // ??
			}
			else{
				// ComputeEnergy.cpp
				ED = Ed_init(pix_seg, &frames, &As, &Bs);
			}
			cvAdd(sum_ED, ED, sum_ED);
			cvReleaseMat(&ED);
		}
		cvConvertScale(sum_ED, sum_ED, 1.0/(*it).second.size());	// josey ??
		for(it_vec = (*it).second.begin(); it_vec != (*it).second.end(); ++ it_vec){
			for(int L = 0; L <=LEVEL; ++L)
				dCost[(*it_vec)*(LEVEL+1)+L] = sum_ED->data.db[L];
		}
	}
	cvReleaseMat(&pix_seg);
	cvReleaseMat(&sum_ED);
	size_t t2 = clock();
	cout << "\nSeg time = " << (double)(t2-t1)/CLOCKS_PER_SEC << std::endl;
}
/*
 *	draw Epipolar on now_frame
 * 
 * */
void getEp(vector<CvMat*>* A,vector<CvMat*>* B,vector<FrameData*>* _frames,int now_frame){
	CvMat* pix_h = cvCreateMat(3,1,CV_64FC1);
	CvMat* pix = cvCreateMat(3,1,CV_64FC1);
	pix->data.db[0] = 896;
	pix->data.db[1] = 363;
	pix->data.db[2] = 1;

	CvMat* new_pix = cvCreateMat(3,1,CV_64FC1);
	CvMat* _tmp = cvCreateMat(3,1,CV_64FC1);
	double pc = 0;
	for(int i = 1; i != (int)(*_frames).size(); i++)		// for each frame get a cost
	{
		CvMat* imgi = cvCreateMat(sizeY,sizeX,CV_8UC3);
		cvConvert((*_frames)[i]->getImg(),imgi);
		pix_h->data.db[0] = (*A)[i-1]->data.db[0]*pix->data.db[0] + 
			(*A)[i-1]->data.db[1]*pix->data.db[1] +
			(*A)[i-1]->data.db[2]*pix->data.db[2];
		pix_h->data.db[1] = (*A)[i-1]->data.db[3]*pix->data.db[0] + 
			(*A)[i-1]->data.db[4]*pix->data.db[1] +
			(*A)[i-1]->data.db[5]*pix->data.db[2];
		pix_h->data.db[2] = (*A)[i-1]->data.db[6]*pix->data.db[0] + 
			(*A)[i-1]->data.db[7]*pix->data.db[1] +
			(*A)[i-1]->data.db[8]*pix->data.db[2];
		for(int j = 0;j<=LEVEL; j++){

			double depth = dK(j);
			double z =  pix_h->data.db[2] + depth*(*B)[i-1]->data.db[2];
			new_pix->data.db[0] = (pix_h->data.db[0] + depth*(*B)[i-1]->data.db[0])/z;

			new_pix->data.db[1] = (pix_h->data.db[1] + depth*(*B)[i-1]->data.db[1])/z;
			new_pix->data.db[2] = 1;
			assert(new_pix->data.db[2] != 0.);
			// new_pix out of range!
			cerr<<new_pix->data.db[0]<<","<<new_pix->data.db[1]<<endl;
			if (new_pix->data.db[0] < 0 || new_pix->data.db[0] >= sizeX || 
					new_pix->data.db[1] < 0 || new_pix->data.db[1] >= sizeY){
				//frameCost.push_back(10000);
				continue;					// don't use this frame
			}
			else{
				imgi->data.ptr[((int)new_pix->data.db[0]+(int)new_pix->data.db[1]*sizeX)*3] = 255;
				imgi->data.ptr[((int)new_pix->data.db[0]+(int)new_pix->data.db[1]*sizeX)*3+1] = 0;
				imgi->data.ptr[((int)new_pix->data.db[0]+(int)new_pix->data.db[1]*sizeX)*3+2] = 0;
			}
		}
		int t = (i<=15?now_frame-16+i:now_frame-15+i);
		string tmp = "/tmp2/josey/" + int2str(now_frame) + '_' + int2str(t)  + ".jpg";
		cerr<<"write:"<<tmp<<endl;
		cvSaveImage(tmp.c_str(),imgi);
	}
}
void multi_ComputeMRF_Seg(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, MRF::CostVal *&dCost, MRF::CostVal *&hCue, MRF::CostVal *&vCue, double*&dCost_ori){

	size_t t1 = clock();

	/* save data of frames -> frames */
	std::vector<FrameData*> frames;
	GetFrame(cp, vs, now_frame, Nframe, &frames);

	size_t t2 = clock();
	cout << "\n[get frames] = " << (double)(t2-t1)/CLOCKS_PER_SEC << std::endl;
	IplImage* img = frames[0]->getImg();
	int height = img->height;
	int width = img->width;

	/* smoothness */
	// MRF::CostVal smoothMax = eta =0.05*(D_MAX-D_MIN)
	// lambda = 5/ (D_MAX-D_MIN)
	hCue = new MRF::CostVal[width*height];
	vCue = new MRF::CostVal[width*height];
	int n = 0;
	std::cout << "\nCompute hCue & vCue..." << std::endl;
	CvMat* pix = cvCreateMat(3,1,CV_64FC1);		// (x, y)
	CvMat* pix2 = cvCreateMat(3,1,CV_64FC1);	// (x+1, y), (x, y+1)
	pix->data.db[2] = 1.;
	pix2->data.db[2] = 1.;

	for(int y = 0; y < height; ++y){
		for(int x = 0; x < width; ++x){
				pix->data.db[0] = x;
				pix->data.db[1] = y;
				pix2->data.db[0] = x+ (x < (width-1));
				pix2->data.db[1] = y;
				hCue[n] = lambda(pix, pix2, img);	

				pix2->data.db[0] = x;		// (x, y+1)
				pix2->data.db[1] = y+ ( y < (height-1));		
				vCue[n] = lambda(pix, pix2, img);
				n++;
		}
	}

	size_t t3 = clock();
	cout << "\n[lambda] = " << (double)(t3-t2)/CLOCKS_PER_SEC << std::endl;
	cvReleaseMat(&pix);
	cvReleaseMat(&pix2);


	/*  dCost */
	//MRF::CostVal* dCost_init = NULL; 
	dCost = new MRF::CostVal[width*height*(LEVEL+1)];
	dCost_ori = new double[width*height*(LEVEL+1)];
	std::cout << "\nCompute dCost..." << std::endl;
	vector<CvMat*> As, Bs;
	for(int i = 1; i != (int)frames.size(); ++i){
		/* compute A B for conjugate */
		CvMat* a = cvCreateMat(3, 3, CV_64FC1);
		CvMat* b = cvCreateMat(3, 1, CV_64FC1);
		GetAB(frames[0], frames[i], a, b);
		As.push_back(a);
		Bs.push_back(b);
	}


	time_t t4;
	time(&t4);
	cout << "\n[GetAB] = " << (double)(t4-t3)/CLOCKS_PER_SEC << std::endl;
	//n = 0;

	getEp(&As,&Bs,&frames,now_frame);
	time_t t5;
	time(&t5);
	cout << "\n[map] = " << (double)(t5-t4)/CLOCKS_PER_SEC << std::endl;
	pthread_t threadid[core];
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	threadArg* ArgArr = new threadArg[core];
	void* status;
	cerr<< "thread_create: " <<endl;
	for(int t=0;t<core;t++)
	{
		ArgArr[t].threadId=t;
		ArgArr[t].dCost = dCost;
		ArgArr[t].As = &As;
		ArgArr[t].Bs = &Bs;
		ArgArr[t]._frames = &frames;
		int a = pthread_create(threadid+t,&attr,&multi_Ed_init,(void *)(ArgArr+t));
		cerr << t <<" ";
		//Ed_init(pix_seg, &frames, &As, &Bs);
	}
	for(int t=0;t<core;t++)
		pthread_join(threadid[t],&status);
	cerr << "\nmulti_thread ED finish\n" ;
	copy(dCost,dCost + width*height*(LEVEL+1),dCost_ori);
	for(int i = 0; i < (int)As.size(); ++i){
		cvReleaseMat( &As[i] );
		cvReleaseMat( &Bs[i] );
	}
	vector<CvMat*>_cvtmp;
	As.clear();
	Bs.clear();
	As.swap(_cvtmp);
	Bs.swap(_cvtmp);
	vector<FrameData*> _tmp;
	frames.clear();
	frames.swap(_tmp);
	time_t t6;
	time(&t6); 
	cout << "\n[Data Cost] = " << difftime(t6,t5) << std::endl;
}

void ComputeMRF_Seg(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, MRF::CostVal *&dCost, MRF::CostVal *&hCue, MRF::CostVal *&vCue, double*&dCost_ori){

	size_t t1 = clock();

	/* save data of frames -> frames */
	std::vector<FrameData*> frames;
	GetFrame(cp, vs, now_frame, Nframe, &frames);

	size_t t2 = clock();
	cout << "\n[get frames] = " << (double)(t2-t1)/CLOCKS_PER_SEC << std::endl;
	IplImage* img = frames[0]->getImg();
	int height = img->height;
	int width = img->width;

	/* smoothness */
	// MRF::CostVal smoothMax = eta =0.05*(D_MAX-D_MIN)
	// lambda = 5/ (D_MAX-D_MIN)
	hCue = new MRF::CostVal[width*height];
	vCue = new MRF::CostVal[width*height];
	int n = 0;
	std::cout << "\nCompute hCue & vCue..." << std::endl;
	CvMat* pix = cvCreateMat(3,1,CV_64FC1);		// (x, y)
	CvMat* pix2 = cvCreateMat(3,1,CV_64FC1);	// (x+1, y), (x, y+1)
	pix->data.db[2] = 1.;
	pix2->data.db[2] = 1.;

	for(int y = 0; y < height; ++y){
		for(int x = 0; x < width; ++x){
			if( frames[0]->b_mask && (frames[0]->getMask()->data.db[y*width+x] != 0)){
				cout<< "use b_mask" << endl;
				hCue[n] = 1.0;  //fg
				vCue[n] = 1.0;	//fg
			}
			else{
				pix->data.db[0] = x;
				pix->data.db[1] = y;
				pix2->data.db[0] = x+ (x < (width-1));
				pix2->data.db[1] = y;
				hCue[n] = lambda(pix, pix2, img);	

				pix2->data.db[0] = x;		// (x, y+1)
				pix2->data.db[1] = y+ ( y < (height-1));		
				vCue[n] = lambda(pix, pix2, img);
			}
			n++;
		}
	}

	size_t t3 = clock();
	cout << "\n[lambda] = " << (double)(t3-t2)/CLOCKS_PER_SEC << std::endl;
	cvReleaseMat(&pix);
	cvReleaseMat(&pix2);


	/*  dCost */
	//MRF::CostVal* dCost_init = NULL; 
	dCost = new MRF::CostVal[width*height*(LEVEL+1)];
	dCost_ori = new double[width*height*(LEVEL+1)];
	std::cout << "\nCompute dCost..." << std::endl;
	vector<CvMat*> As, Bs;
	for(int i = 1; i != (int)frames.size(); ++i){
		/* compute A B for conjugate */
		CvMat* a = cvCreateMat(3, 3, CV_64FC1);
		CvMat* b = cvCreateMat(3, 1, CV_64FC1);
		GetAB(frames[0], frames[i], a, b);
		As.push_back(a);
		Bs.push_back(b);
	}


	size_t t4 = clock();
	cout << "\n[GetAB] = " << (double)(t4-t3)/CLOCKS_PER_SEC << std::endl;
	//n = 0;
	map<int, vector<int> > segment = computeSeg(img->width, img->height, vs->getSeg(now_frame));
	map<int, vector<int> >::iterator it = segment.begin();


	size_t t5 = clock();
	cout << "\n[map] = " << (double)(t5-t4)/CLOCKS_PER_SEC << std::endl;
	CvMat *sum_ED = cvCreateMat(LEVEL+1,1,CV_64FC1);
	CvMat* pix_seg = cvCreateMat(3,1,CV_64FC1);
	pix_seg->data.db[2] = 1.;
	for(int y=0;y<height;y++)
	{

		size_t tt = clock();
		cout << "[" << y<<"] : " << (double)(tt-t5)/CLOCKS_PER_SEC << std::endl;
		for(int x=0;x<width;x++)
		{
			n = 0;
			pix_seg->data.db[0] = x;
			pix_seg->data.db[1] = y;
			CvMat *ED = cvCreateMat(LEVEL+1,1,CV_64FC1);
			if( frames[0]->b_mask && (frames[0]->getMask()->data.db[y*width+x] != 0)){
				// fg
				cout<< " use fg " <<endl;
				cvZero(ED);
				cvAddS(ED, cvRealScalar(1.0), ED); // ??
			}
			else{
				ED = Ed_init(pix_seg, &frames, &As, &Bs);
				//cerr << "loc:" << b.x <<" " <<b.y<<endl;
			}
			for(int L = 0; L <=LEVEL; ++L)
			{
				dCost_ori[(y*width+x)*(LEVEL+1)+L] = ED->data.db[L];
				dCost[(y*width+x)*(LEVEL+1)+L] = ED->data.db[L];		// edit by josey
			}
			cvReleaseMat(&ED);
		}
	}
	cvReleaseMat(&pix_seg);
	for(int i = 0; i < (int)As.size(); ++i){
		cvReleaseMat( &As[i] );
		cvReleaseMat( &Bs[i] );
	}
	size_t t6 = clock();
	cout << "\n[t6] = " << (double)(t6-t5)/CLOCKS_PER_SEC << std::endl;
}
/* use: Video_initial2 */
void ComputeMRF_SegPlane(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, std::string segFile,std::string outDFile, double* dCost_ori){

	size_t t1 = clock();

	if(vs->allF_dis[now_frame]==0){
		cerr << "No disparity for segPlane!" << endl;
		exit(0);
	}
	else{
		std::vector<FrameData*> frames;
		GetFrame(cp, vs, now_frame, Nframe, &frames);

		IplImage* img = frames[0]->getImg();
		CvMat* _D = frames[0]->getD();

		map<int, vector<int> > segment = computeSeg(img->width, img->height, segFile);
		map<int, vector<int> >::iterator it = segment.begin();

		vector<int>::iterator it_vec;
		CvMat* new_D = cvCreateMat(img->height, img->width, CV_64FC1);	// new disparity

		CvMat* P = cvCreateMat(3,1,CV_64FC1);	// plane parameter
		P->data.db[0] = 0.;
		P->data.db[1] = 0.;
		CvMat* pix_seg = cvCreateMat(3,1,CV_64FC1);
		pix_seg->data.db[2] = 1.;
		/////////////////////////////////////
		if(frames[0]->b_mask){	// fg 
			CvMat* segMask = cvCreateMat(img->height, img->width, CV_64FC1);// new disparity
			cvZero(segMask);
			CvMat* fgMask = frames[0]->getMask();

			for(; it!= segment.end(); ++it){	// for each segment
				it_vec =(*it).second.begin();
				double mean_d = 0.0;
				for( it_vec; it_vec != (*it).second.end(); it_vec++){
					int y = (*it_vec)/(img->width), x = (*it_vec)% (img->width);
					mean_d += _D->data.db[y*_D->cols+x];
					segMask->data.db[(*it_vec)] = 1.0; 
				}
				cvMul(segMask, fgMask, segMask);
				if ( cvCountNonZero(segMask) != 0){ // do not change dispairty
					for( it_vec = (*it).second.begin(); it_vec != (*it).second.end(); ++ it_vec)
						new_D->data.db[(*it_vec)] = _D->data.db[(*it_vec)]; 
				}
				else{
					mean_d /= (*it).second.size();
					CvMat* new_P = cvCreateMat(3,1,CV_64FC1);	// new plane parameter
					P->data.db[2] = mean_d;
					computeJacobian((*it).second, frames,  P, new_P, dCost_ori);
					//cout << "new	" <<  new_P->data.db[0] << "\t" <<  new_P->data.db[1] <<  "\t" << new_P->data.db[2] << endl;

					for( it_vec = (*it).second.begin(); it_vec != (*it).second.end(); ++ it_vec){
						int y = (*it_vec)/(img->width), x = (*it_vec) % (img->width);
						pix_seg->data.db[0] = x;
						pix_seg->data.db[1] = y;
						double newD = min(max(cvDotProduct(pix_seg, new_P), D_MIN), D_MAX); 
						new_D->data.db[y*new_D->cols+x] = newD; //Dplane(pix_seg, new_P);
					}
					cvReleaseMat(&new_P);
				} // end  change
			}// end segment
		}
		else{
			for(; it!= segment.end(); ++it){	// for each segment
				//cout << "seg no:" << (*it).first << "..." << segment.size() << "\t";
				it_vec =(*it).second.begin();
				double mean_d = 0.0;
				for( it_vec; it_vec != (*it).second.end(); it_vec++){
					int y = (*it_vec)/(img->width), x = (*it_vec)% (img->width);
					mean_d += _D->data.db[y*_D->cols+x];

				}

				mean_d /= (*it).second.size();
				CvMat* new_P = cvCreateMat(3,1,CV_64FC1);	// new plane parameter
				P->data.db[2] = mean_d;
				computeJacobian((*it).second, frames,  P, new_P, dCost_ori);
				//cout << "new	" <<  new_P->data.db[0] << "\t" <<  new_P->data.db[1] <<  "\t" << new_P->data.db[2] << endl;

				for( it_vec = (*it).second.begin(); it_vec != (*it).second.end(); ++ it_vec){
					int y = (*it_vec)/(img->width), x = (*it_vec) % (img->width);
					pix_seg->data.db[0] = x;
					pix_seg->data.db[1] = y;
					double newD = min(max(cvDotProduct(pix_seg, new_P), D_MIN), D_MAX); 
					new_D->data.db[y*new_D->cols+x] = newD; //Dplane(pix_seg, new_P);
				}
				cvReleaseMat(&new_P);
			}
		}
		cvReleaseMat(&P);
		cvReleaseMat(&pix_seg);
		//vs->setDisparity(new_D, now_frame);
		WriteToFile(new_D, outDFile);
		cerr<< outDFile;
		outDFile.replace(outDFile.end()-3, outDFile.end(),"png");
		// cout << "!!!!!" << outDFile << endl;
		WriteDisparityMap(new_D, outDFile);
		cvReleaseMat(&new_D);
	}
	size_t t2 = clock();
	cout << "\nSegPlane time = " << (double)(t2-t1)/CLOCKS_PER_SEC << std::endl;
}



void ComputeMRF_SegPlane(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, std::string segFile,std::string outDFile){

	size_t t1 = clock();

	if(! vs->b_dis){
		cout << "No disparity for segPlane!" << endl;
		exit(0);
	}
	else{
		std::vector<FrameData*> frames;
		GetFrame(cp, vs, now_frame, Nframe, &frames);

		vector<CvMat*> As, Bs;
		for(int i = 1; i != (int)frames.size(); ++i){
			/* compute A B for conjugate */
			CvMat* a = cvCreateMat(3, 3, CV_64FC1);
			CvMat* b = cvCreateMat(3, 1, CV_64FC1);
			GetAB(frames[0], frames[i], a, b);
			As.push_back(a);
			Bs.push_back(b);
		}

		IplImage* img = frames[0]->getImg();

		map<int, vector<int> > segment = computeSeg(img->width, img->height, segFile);
		map<int, vector<int> >::iterator it = segment.begin();
		CvMat* _D = cvCreateMat(img->height, img->width, CV_64FC1);
		_D = frames[0]->getD();
		CvMat* new_D = cvCreateMat(img->height, img->width, CV_64FC1);
		CvMat* P = cvCreateMat(3,1,CV_64FC1);	// plane parameter
		P->data.db[0] = 0.;
		P->data.db[1] = 0.;
		CvMat* pix_seg = cvCreateMat(3,1,CV_64FC1);
		pix_seg->data.db[2] = 1.;

		for(; it!= segment.end(); ++it){
			//cout << "seg no:" << (*it).first << "..." << segment.size() << "\t";
			vector<int>::iterator it_vec =(*it).second.begin();

			for(int i = 0; i< (int)(*it).second.size()/2; i++)
				it_vec++;
			int y = (*it_vec)/(img->width), x = (*it_vec)% (img->width);
			double d = _D->data.db[y*_D->cols+x];
			CvMat* new_P = cvCreateMat(3,1,CV_64FC1);	// new plane parameter
			P->data.db[2] = d;
			computeJacobian((*it).second, frames,  P, new_P, As, Bs);
			//cout << "new	" <<  new_P->data.db[0] << "\t" <<  new_P->data.db[1] <<  "\t" << new_P->data.db[2] << endl;

			for( it_vec = (*it).second.begin(); it_vec != (*it).second.end(); ++ it_vec){
				int y = (*it_vec)/(img->width), x = (*it_vec) % (img->width);
				pix_seg->data.db[0] = x;
				pix_seg->data.db[1] = y;
				new_D->data.db[y*new_D->cols+x] = Dplane(pix_seg, new_P);

			}
			cvReleaseMat(&new_P);
		}
		cvReleaseMat(&P);
		cvReleaseMat(&pix_seg);
		//vs->setDisparity(new_D, now_frame);
		WriteToFile(new_D, outDFile);
		outDFile.replace(outDFile.end()-3, outDFile.end(),"png");
		// cout << "!!!!!" << outDFile << endl;
		WriteDisparityMap(new_D, outDFile);

		cvReleaseMat(&new_D);
	}
	size_t t2 = clock();
	cout << "\nSegPlane time = " << (double)(t2-t1)/CLOCKS_PER_SEC << std::endl;
}

void multi_ComputeMRF_D(ReadCameraParameter* cp, ReadVideoSeq* vs,int now_frame, int Nframe, MRF::CostVal *&dCost, MRF::CostVal *&hCue, MRF::CostVal *&vCue){

	size_t t1 = clock();	

	if (! vs->b_dis){
		cout << "No initial disparity!" << endl;
		exit(0);
	}
	else{

		vector<FrameData*> frames;
		GetFrame(cp, vs, now_frame, Nframe, &frames);

		IplImage* img = frames[0]->getImg();
		int height = img->height;
		int width = img->width;

		/* compute pv-map */
		vector<CvMat**> MapXs, MapYs;
		vector<CvMat*> As, Bs;
		cerr << "[Get AB]" <<endl;
		for(int i = 1; i != (int)frames.size(); ++i){
			/* compute A B for conjugate */
			CvMat* a = cvCreateMat(3, 3, CV_64FC1);
			CvMat* b = cvCreateMat(3, 1, CV_64FC1);
			GetAB(frames[0], frames[i], a, b);
			As.push_back(a);
			Bs.push_back(b);
			/* end */
			CvMat** MapX = new CvMat* [width*height];
			CvMat** MapY = new CvMat* [width*height];
			for(int n = 0; n < width*height; n++){
				MapX[n] = cvCreateMat(neighbor*neighbor, 1, CV_64FC1);
				MapY[n] = cvCreateMat(neighbor*neighbor, 1, CV_64FC1);
			}

			GetPvMap ( frames[0], frames[i], MapX, MapY); // get Pv Map	ComputeEnergy
			MapXs.push_back(MapX);
			MapYs.push_back(MapY);	
		}

		/* smoothness */
		// MRF::CostVal smoothMax = eta =0.05*(D_MAX-D_MIN)
		// lambda = 5/ (D_MAX-D_MIN)
		hCue = new MRF::CostVal[width*height];
		vCue = new MRF::CostVal[width*height];
		dCost = new MRF::CostVal[width*height*(LEVEL+1)];

		int n = 0;
		std::cout << "\nCompute hCue & vCue..." << std::endl;
		CvMat* pix = cvCreateMat(3,1,CV_64FC1);		// (x, y)
		CvMat* pix2 = cvCreateMat(3,1,CV_64FC1);	// (x+1, y), (x, y+1)
		pix->data.db[2] = 1.;
		pix2->data.db[2] = 1.;
		CvMat *ED = NULL;
		for(int y = 0; y < height; ++y)
			for(int x = 0; x < width; ++x){
				/* smoothness term */
				if( frames[0]->b_mask && (frames[0]->getMask()->data.db[y*width+x] != 0)){
					hCue[n] = 1.0;	// fg
					vCue[n] = 1.0;	// fg
				}
				else{
					pix2->data.db[0] = x+ (x < (width-1));
					pix2->data.db[1] = y;
					hCue[n] = lambda(pix, pix2, img);	// (x+1, y)
					//hCueOut << hCue[n]<< "\t";
					pix2->data.db[0] = x;
					pix2->data.db[1] = y+ (y < (height-1));
					vCue[n] = lambda(pix, pix2, img);	// (x, y+1)
				}
				n++;
			}
		pthread_t threadid[core];
		pthread_attr_t attr;
		pthread_attr_init(&attr);
		threadArg* ArgArr = new threadArg[core];
		void* status;
		cerr<< "thread_create: " <<endl;
		for(int t=0;t<core;t++)
		{
			ArgArr[t].threadId=t;
			ArgArr[t].dCost = dCost;
			ArgArr[t].As = &As;
			ArgArr[t].Bs = &Bs;
			ArgArr[t].MapXs = &MapXs;

			ArgArr[t].MapYs = &MapYs;
			ArgArr[t]._frames = &frames;
			int a = pthread_create(threadid+t,&attr,&multi_Ed,(void *)(ArgArr+t));
			cerr << t <<" ";
			//Ed_init(pix_seg, &frames, &As, &Bs);
		}
		for(int t=0;t<core;t++)
		{
			cerr<<t<<"finish"<<endl;
			pthread_join(threadid[t],&status);
		}
		cerr << "\nmulti_thread ED finish\n" ;

		cvReleaseMat(&ED);

		for(int i = 0; i < (int)MapXs.size(); ++i){
			cvReleaseMat( &As[i] );
			cvReleaseMat( &Bs[i] );
			for(int n = 0; n < width*height; n++){
				cvReleaseMat( &((MapXs[i])[n]) );
				cvReleaseMat( &((MapYs[i])[n]) );
			}
			delete [](MapXs[i]);
			delete [](MapYs[i]);
		}
		//dCostOut.close();
		//hCueOut.close();
		//vCueOut.close();
	}

	size_t t2 = clock();
	std::cout << "\nD time = " << (double)(t2-t1)/CLOCKS_PER_SEC << std::endl;
}


void ComputeMRF_D(ReadCameraParameter* cp, ReadVideoSeq* vs,int now_frame, int Nframe, MRF::CostVal *&dCost, MRF::CostVal *&hCue, MRF::CostVal *&vCue){

	size_t t1 = clock();	

	if (! vs->b_dis){
		cout << "No initial disparity!" << endl;
		exit(0);
	}
	else{

		vector<FrameData*> frames;
		GetFrame(cp, vs, now_frame, Nframe, &frames);

		IplImage* img = frames[0]->getImg();
		int height = img->height;
		int width = img->width;

		/* compute pv-map */
		vector<CvMat**> MapXs, MapYs;
		vector<CvMat*> As, Bs;
		cerr << "[Get AB]" <<endl;
		for(int i = 1; i != (int)frames.size(); ++i){
			/* compute A B for conjugate */
			CvMat* a = cvCreateMat(3, 3, CV_64FC1);
			CvMat* b = cvCreateMat(3, 1, CV_64FC1);
			GetAB(frames[0], frames[i], a, b);
			As.push_back(a);
			Bs.push_back(b);
			/* end */
			CvMat** MapX = new CvMat* [width*height];
			CvMat** MapY = new CvMat* [width*height];
			for(int n = 0; n < width*height; n++){
				MapX[n] = cvCreateMat(neighbor*neighbor, 1, CV_64FC1);
				MapY[n] = cvCreateMat(neighbor*neighbor, 1, CV_64FC1);
			}

			GetPvMap ( frames[0], frames[i], MapX, MapY); // get Pv Map
			MapXs.push_back(MapX);
			MapYs.push_back(MapY);	
		}

		/* smoothness */
		// MRF::CostVal smoothMax = eta =0.05*(D_MAX-D_MIN)
		// lambda = 5/ (D_MAX-D_MIN)
		// hCue[width*height], vCue[width*height]
		hCue = new MRF::CostVal[width*height];
		vCue = new MRF::CostVal[width*height];
		dCost = new MRF::CostVal[width*height*(LEVEL+1)];

		int n = 0;
		//ofstream dCostOut, vCueOut, hCueOut;
		//dCostOut.open("dCostOut.txt");
		//vCueOut.open("vCueOut.txt");
		//hCueOut.open("hCueOut.txt");
		std::cout << "\nCompute hCue & vCue..." << std::endl;
		CvMat* pix = cvCreateMat(3,1,CV_64FC1);		// (x, y)
		CvMat* pix2 = cvCreateMat(3,1,CV_64FC1);	// (x+1, y), (x, y+1)
		pix->data.db[2] = 1.;
		pix2->data.db[2] = 1.;
		CvMat *ED = NULL;
		for(int y = 0; y < height; ++y)
			for(int x = 0; x < width; ++x){
				/* data term */
				pix->data.db[0] = x;
				pix->data.db[1] = y;
				if( frames[0]->b_mask && (frames[0]->getMask()->data.db[y*width+x] != 0)){
					/* fg */
					ED = Ed_FG(pix, frames, MapXs, MapYs, As, Bs);
					for(int k = 0; k <= LEVEL; k++)
						dCost[(y*width+x)*(LEVEL+1)+k] = ED->data.db[k];
					//dCost[(y*width+x)*(LEVEL+1)+k] = 1.0;   // ??
				}
				else{
					ED = Ed(pix, frames, MapXs, MapYs, As, Bs);
					for(int k = 0; k <= LEVEL; k++){
						dCost[(y*width+x)*(LEVEL+1)+k] = ED->data.db[k];
						//dCostOut << ED->data.db[k]<< "\t";
					}
					//dCostOut << "\n";
				}

				/* smoothness term */
				if( frames[0]->b_mask && (frames[0]->getMask()->data.db[y*width+x] != 0)){
					hCue[n] = 1.0;	// fg
					vCue[n] = 1.0;	// fg
				}
				else{
					pix2->data.db[0] = x+ (x < (width-1));
					pix2->data.db[1] = y;
					hCue[n] = lambda(pix, pix2, img);	// (x+1, y)
					//hCueOut << hCue[n]<< "\t";
					pix2->data.db[0] = x;
					pix2->data.db[1] = y+ (y < (height-1));
					vCue[n] = lambda(pix, pix2, img);	// (x, y+1)
				}
				//vCueOut << vCue[n] << "\t";

				//if (n% 5000 == 0)
				//	cout << n/5000 << "\t"; 

				n++;
			}

		cvReleaseMat(&pix);
		cvReleaseMat(&pix2);
		cvReleaseMat(&ED);

		for(int i = 0; i < (int)MapXs.size(); ++i){
			cvReleaseMat( &As[i] );
			cvReleaseMat( &Bs[i] );
			for(int n = 0; n < width*height; n++){
				cvReleaseMat( &((MapXs[i])[n]) );
				cvReleaseMat( &((MapYs[i])[n]) );
			}
			delete [](MapXs[i]);
			delete [](MapYs[i]);
		}
		//dCostOut.close();
		//hCueOut.close();
		//vCueOut.close();
	}

	size_t t2 = clock();
	std::cout << "\nD time = " << (double)(t2-t1)/CLOCKS_PER_SEC << std::endl;
}

/* video */

void VideoMRF(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, int seqSize, std::string name){
	for (int i = 0; i < seqSize; i++){
		int n = now_frame+i;	// nth frame

		MRF::CostVal *dCost = NULL, *hCue = NULL , *vCue = NULL;
		ComputeMRF(cp, vs, n, Nframe, dCost, hCue, vCue);

		/* MRF */
		//MRF* mrf;
		//EnergyFunction *energy;
		//MRF::EnergyVal E;
		//double lowerBound;
		//int iter;
		//	fprintf(stderr, "using truncated quadratic smoothness cost\n");
		DataCost *data         = new DataCost(dCost);
		double smoothMax = 0.05*(D_MAX-D_MIN);
		double lamda = 5.0/(D_MAX - D_MIN);
		SmoothnessCost *smooth = new SmoothnessCost(1, smoothMax, lamda,hCue,vCue);
		EnergyFunction *energy    = new EnergyFunction(data,smooth);
		string t_name;
		t_name = int2str(n);
		string OUT_FILE = name;
		OUT_FILE += t_name;

		MRF *mrf;
		if (aBP)		/* run BP */
			runBP(energy, OUT_FILE, mrf);

		if (aBPS)	/* run BP-S */
			runBPS(energy, OUT_FILE, mrf);

		if (aICM)	/* run ICM */
			runICM(energy, OUT_FILE, mrf);

		if (aTRWS)		/* run TRW-S */
			runTRWS(energy, OUT_FILE, mrf);

		if (aSWAPS)		/* run Swap */
			runSwap(energy, OUT_FILE, mrf);

		if(aEXP)	/* run expansion */
			runExpansion(energy, OUT_FILE, mrf);

	}
}


void VideoMRF_seg(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, int seqSize, std::string name){
	for (int i = 0; i < seqSize; i++){
		int n = now_frame+i;	// nth frame

		MRF::CostVal *dCost = NULL, *hCue = NULL , *vCue = NULL;
		ComputeMRF_Seg(cp, vs, n, Nframe, dCost, hCue, vCue);

		/* MRF */
		//MRF* mrf;
		//EnergyFunction *energy;
		//MRF::EnergyVal E;
		//double lowerBound;
		//int iter;
		//		fprintf(stderr, "using truncated quadratic smoothness cost\n");
		DataCost *data         = new DataCost(dCost);
		double smoothMax = 0.05*(D_MAX-D_MIN);
		double lamda = 5.0/(D_MAX - D_MIN);
		SmoothnessCost *smooth = new SmoothnessCost(1, smoothMax, lamda,hCue,vCue);
		EnergyFunction *energy    = new EnergyFunction(data,smooth);
		string t_name;
		t_name = int2str(n);
		string OUT_FILE = name;
		OUT_FILE += t_name;

		MRF *mrf;
		if (aBP)		/* run BP */
			runBP(energy, OUT_FILE, mrf);

		if (aBPS)	/* run BP-S */
			runBPS(energy, OUT_FILE, mrf);

		if (aICM)	/* run ICM */
			runICM(energy, OUT_FILE, mrf);

		if (aTRWS)		/* run TRW-S */
			runTRWS(energy, OUT_FILE, mrf);

		if (aSWAPS)		/* run Swap */
			runSwap(energy, OUT_FILE, mrf);

		if(aEXP)	/* run expansion */
			runExpansion(energy, OUT_FILE, mrf);

	}
}

void Video_SegPlane(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, int seqSize, std::string name){
	if (vs->b_seg && vs->b_dis){
		for (int i = 0; i < seqSize; i++){
			int n = now_frame+i;
			string num;
			num = int2str(n);	
			string outDFile = name;		// road_
			outDFile += num;				// 0
			outDFile += "_SegPlane.txt";			// .txt
			cout << outDFile << endl;
			string seg = vs->getSeg(n);
			ComputeMRF_SegPlane(cp, vs, n, Nframe, seg, outDFile);
		}
	}
	else if (! vs->b_seg)
		cout << "No segment file!" << endl;
	else
		cout << "No disparity file!" << endl;
}
// use in runF
void VideoMRF_Final(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, int seqSize, std::string name){
	if (vs->b_dis){
		for (int i = 0; i < seqSize; i++){
			int n = now_frame+i;
			fprintf(stderr, "process...%d\n", n);
			MRF::CostVal *dCost = NULL, *hCue = NULL , *vCue = NULL;
			multi_ComputeMRF_D(cp, vs, n, Nframe, dCost, hCue, vCue);
			//ComputeMRF_D(cp, vs, n, Nframe, dCost, hCue, vCue);

			/* MRF */
			DataCost *data         = new DataCost(dCost);
			SmoothnessCost *smooth = new SmoothnessCost(1, smoothMax_Final, lambda_Final,hCue,vCue);
			EnergyFunction *energy    = new EnergyFunction(data,smooth);
			string t_name;
			t_name = int2str(n);
			//string OUT_FILE = name +"Final_";
			string OUT_FILE = "/tmp2/josey/" + name;
			OUT_FILE += t_name;

			MRF *mrf;
			if (aBP)		/* run BP */
				runBP(energy, OUT_FILE, mrf);

			if (aBPS)	/* run BP-S */
				runBPS(energy, OUT_FILE, mrf);

			if (aICM)	/* run ICM */
				runICM(energy, OUT_FILE, mrf);

			if (aTRWS)		/* run TRW-S */
				runTRWS(energy, OUT_FILE, mrf);

			if (aSWAPS)		/* run Swap */
				runSwap(energy, OUT_FILE, mrf);

			if(aEXP)	/* run expansion */
				runExpansion(energy, OUT_FILE, mrf);

			/* update disparity */
			int count = 0;
			int height = vs->m_height; 
			int width = vs->m_width;
			CvMat *m = cvCreateMat(height, width, CV_64FC1);
			cvZero(m);
			//cout << "mrf " << dK(mrf->getLabel(count));
			for (int y = 0; y < height; y++) 	
				for (int x = 0; x < width; x++) 
					m->data.db[m->cols*y+x] = dK(mrf->getLabel(count++));
			vs->setDisparity(m, n);
		}
	}
	else
		cout << "No disparity file!" << endl;
}

void Video_initial(ReadCameraParameter* cp, ReadVideoSeq *vs, int now_frame, int Nframe, int seqSize, std::string name){
	for (int i = 0; i < seqSize; i++){
		size_t t1 = clock();
		int n = now_frame+i;	// nth frame

		MRF::CostVal *dCost = NULL, *hCue = NULL , *vCue = NULL;
		ComputeMRF_Seg(cp, vs, n, Nframe, dCost, hCue, vCue);
		size_t t2 = clock();
		/* MRF */
		//MRF* mrf;
		//EnergyFunction *energy;
		//MRF::EnergyVal E;
		//double lowerBound;
		//int iter;
		fprintf(stderr, "using truncated quadratic smoothness cost\n");
		DataCost *data         = new DataCost(dCost);
		double smoothMax = 0.05*(D_MAX-D_MIN);
		//double lamda = 5.0/(D_MAX - D_MIN);
		double lamda = 10.0/(D_MAX - D_MIN);
		SmoothnessCost *smooth = new SmoothnessCost(1, smoothMax, lamda,hCue,vCue);
		EnergyFunction *energy    = new EnergyFunction(data,smooth);
		string t_name;
		t_name = int2str(n);
		string OUT_FILE = name;		// no .txt
		OUT_FILE += t_name;

		MRF *mrf = NULL;
		if (aBP)		/* run BP */
			runBP(energy, OUT_FILE, mrf);

		if (aBPS)	/* run BP-S */
			runBPS(energy, OUT_FILE, mrf);

		if (aICM)	/* run ICM */
			runICM(energy, OUT_FILE, mrf);

		if (aTRWS)		/* run TRW-S */
			runTRWS(energy, OUT_FILE, mrf);

		if (aSWAPS)		/* run Swap */
			runSwap(energy, OUT_FILE, mrf);

		if(aEXP)	/* run expansion */
			runExpansion(energy, OUT_FILE, mrf);

		size_t t3 = clock();

		cout << "start SegPlane!" << endl;
		int count = 0;
		int height = vs->m_height; 
		int width = vs->m_width;
		CvMat *m = cvCreateMat(height, width, CV_64FC1);
		cvZero(m);
		for (int y = 0; y < height; y++) 	
			for (int x = 0; x < width; x++) 
				m->data.db[m->cols*y+x] = dK(mrf->getLabel(count++));
		vs->setDisparity(m, n);
		vs->b_dis = true;
		//string outDFile = "road_SegPlane.txt";
		string outDFile = name + "/seq/disparity/";
		outDFile += t_name;
		outDFile += ".txt";
		cout << outDFile << endl;
		size_t t4 = clock();
		ComputeMRF_SegPlane(cp, vs, n, Nframe, vs->getSeg(n), outDFile);
		size_t t5 = clock();
		cout << "Seg: " << (double)(t2-t1)/CLOCKS_PER_SEC << endl;
		cout << "MRF: " << (double)(t3-t2)/CLOCKS_PER_SEC << std::endl;
		cout << "SegPlane: " << (double)(t5-t4)/CLOCKS_PER_SEC << std::endl;
	}
}

void Video_initial2(ReadCameraParameter* cp, ReadVideoSeq *vs, int now_frame, int Nframe, int seqSize, std::string name){
	/*
now_frame : start frame
*/
	double seg_tot = 0.0, LM_tot = 0.0 , MRF_tot = 0.0;
	for (int i = 0; i < seqSize; i++)
	{
		int n = now_frame+i;	// nth frame
		fprintf(stderr,"process frame %d ...\n", n);
		time_t t1 = clock();
		MRF::CostVal *dCost = NULL, *hCue = NULL , *vCue = NULL;
		double *dCost_ori = NULL;
		// use multi thread
		multi_ComputeMRF_Seg(cp, vs, n, Nframe, dCost, hCue, vCue, dCost_ori);      // edit by Josey
		time_t t2 = clock();
		// test eff of smooth
		//memset(hCue,0,8*sizeX*sizeY);
		//memset(vCue,0,8*sizeX*sizeY);
		DataCost *data         = new DataCost(dCost);
		SmoothnessCost *smooth = new SmoothnessCost(1, smoothMax_init2, lambda_init2,hCue,vCue);
		EnergyFunction *energy    = new EnergyFunction(data,smooth);
		string t_name;
		t_name = int2str(n);
		string OUT_FILE = "/tmp2/josey/" + name;		// no .txt
		//string OUT_FILE =  name;		// no .txt
		OUT_FILE += t_name;

		MRF *mrf = NULL;
		if (aBP)		/* run BP */
			runBP(energy, OUT_FILE, mrf);

		if (aBPS)	/* run BP-S */
			runBPS(energy, OUT_FILE, mrf);

		if (aICM)	/* run ICM */
			runICM(energy, OUT_FILE, mrf);

		if (aTRWS)		/* run TRW-S */
			runTRWS(energy, OUT_FILE, mrf);

		if (aSWAPS)		/* run Swap */
			runSwap(energy, OUT_FILE, mrf);

		if (aEXP)	/* run expansion */
			runExpansion(energy, OUT_FILE, mrf);

		size_t t3 = clock();

		int count = 0;
		int height = vs->m_height; 
		int width = vs->m_width;
		CvMat *m = cvCreateMat(height, width, CV_64FC1);
		cvZero(m);
		// get disparity and set it
		for (int y = 0; y < height; y++) 	
			for (int x = 0; x < width; x++) 
				m->data.db[m->cols*y+x] = dK(mrf->getLabel(count++));
		vs->setDisparity(m, n);
		vs->allF_dis[n] = 1;
		string outDFile = "/tmp2/josey/";
		outDFile += t_name;
		outDFile += ".txt";
		cout << outDFile << endl;
		size_t t4 = clock();

		ComputeMRF_SegPlane(cp, vs, n, Nframe, vs->getSeg(n), outDFile, dCost_ori);
		size_t t5 = clock();
		//	cout <<"new: " << vs->getDisparity(n)->data.db[100] << endl;
		seg_tot += (double)(t2-t1)/CLOCKS_PER_SEC;
		MRF_tot += (double)(t3-t2)/CLOCKS_PER_SEC; 
		LM_tot += (double)(t5-t4)/CLOCKS_PER_SEC;
		cout << "Seg: " << seg_tot << endl;
		cout << "MRF: " << MRF_tot << std::endl;
		cout << "SegPlane: " << LM_tot << std::endl;
		delete [] dCost;
		delete [] dCost_ori;
		delete [] hCue;
		delete [] vCue;
		delete data;
		delete smooth;
		delete energy;
		// BP
		mrf->~MRF();
	}
	cout << "============ end =======================" << endl;
	cout << "Total_seg:" << seg_tot << "\t Total_MRF:" << MRF_tot << "\tTotal_LM:" << LM_tot << endl;
	vs->b_dis = true;
}
