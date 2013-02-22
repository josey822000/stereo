#include "Segment.h"
#include "ComputeEnergy.h"
#include "FileIO.h"
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <cmath>

using namespace std;
#include "opencv/cv.h"
#include "opencv/cxcore.h"
using namespace cv;

/////////////////////////////////////////////////
//													   //
// initial depth map with segmentation //
//													  //
////////////////////////////////////////////////

map<int, vector<int> > computeSeg(int width, int height, string filename){
	ifstream seg;		// segment file from edison
	seg.open(filename.c_str());
	int n = 0, label;
	map<int, vector<int> > count_seg;  // <labelValue, <number, n> >
	map<int, vector<int> >::iterator it;

	while( ! seg.eof()){
		seg >> label;		// read label
		it = count_seg.find(label);
		if( it == count_seg.end() ){	// not existed label
			vector<int> tmp;
			tmp.clear();
			tmp.push_back(n);
			count_seg.insert(make_pair(label, tmp));
		}
		else			// push the pix location
			(*it).second.push_back(n);
		++n;
	}
	seg.close();
	return count_seg;
}



double Dplane(CvMat* pix, CvMat* planePara){	// return disparity value
	return (min(max(pix->data.db[0]*planePara->data.db[0] + pix->data.db[1]*planePara->data.db[1] + planePara->data.db[2],D_MIN),D_MAX));
}

/*  pixs -> pixel of segment, delta-> planePara*/  // return delta_lm for new_planePara
void computeJacobian(vector<int> pixs, vector<FrameData*> _frames, CvMat* P, CvMat* P_est, double* dCost_ori){
	int n_iters = 10;				// times of iterator
	double _lamda = 0.01;
	int n_lamda = 10;
	bool updateJ = true;

	int Ndata= pixs.size();			// number of data
	int Nparams = 3;					// number of para

	CvMat *data=cvCreateMat(Ndata,1,CV_64FC1);
	//CvMat *P_est = cvCreateMat(Nparams,1,CV_64FC1);	// parameter of each iteration 
	CvMat *P_lm = cvCreateMat(Nparams,1,CV_64FC1);
	// new para of next iteration
	cvCopy(P, P_est);
	double  e, e_lm;			// error
	CvMat *d_lm = cvCreateMat(Ndata,1,CV_64FC1);  
	CvMat* dP = cvCreateMat(Nparams,1,CV_64FC1);  // delta P
	CvMat* J = cvCreateMat(Ndata, Nparams, CV_64FC1); // Jacobian matrix
	CvMat* H = cvCreateMat(Nparams,Nparams,CV_64FC1);

	IplImage* img1;
	img1 = _frames[0]->getImg();
	int width = img1->width;

	CvMat* pix = cvCreateMat(3, 1, CV_64FC1);
	pix->data.db[2] = 1.;
	for( int k = 0; k < n_iters; ++k){	//  for iterator
		CvMat *d=cvCreateMat(Ndata,1,CV_64FC1);		// diff
		vector<int>::iterator it = pixs.begin();
		if(updateJ){	/* compute the Jacobian */
			int n = 0;
			for(; it != pixs.end(); ++it){	// each pix
				int x = (int)(*it)%width;	// x of pix
				int y = (int)(*it)/width;	// y of pix
				pix->data.db[0] = x;
				pix->data.db[1] = y;
				double tmp_d = min(max(cvDotProduct(pix, P_est), D_MIN), D_MAX);
				int l = Kd(tmp_d);	// get the level of pix
				//(1)
				//d->data.db[d->cols*n] = Ed_init(pix, tmp_d, _frames);
				//(2)
				d->data.db[d->cols*n] = dCost_ori[(*it)*(LEVEL+1)+l];
				int l_idx = (*it)*(LEVEL+1)+l;	
				double PartialL;
				if(l>=0 && l<=LEVEL-1){
					PartialL = ((1-dCost_ori[l_idx+1])-(1-dCost_ori[l_idx])) / (dK(l+1) - dK(l));
				}else{
					PartialL = 0.0;
				}

				J->data.db[J->cols*n] = (double)x*PartialL;
				J->data.db[J->cols*n+1] = (double)y*PartialL;
				J->data.db[J->cols*n+2] = PartialL;
				n++;
			}
			cvMulTransposed(J,H,1);
			if (k == 0)
				e = cvDotProduct(d,d);
		}

		CvMat* H_lm = cvCreateMat(Nparams,Nparams,CV_64FC1);
		cvSetIdentity(H_lm);  // H_lm = I(3,3)
		cvGEMM(H_lm, H_lm, _lamda, H, 1.0, H_lm); // H_lm =H+ lamda*I(3)
		double chk = cvInvert(H_lm, H_lm); // H_lm = inv(H_lm)
		assert(chk !=0);
		cvGEMM(J, d, 1, 0, 0, dP, CV_GEMM_A_T); // dP = (J')*d
		cvMatMul(H_lm, dP, dP); // dP = (H_lm)*dP
		//cvConvertScale(dP, dP, -1);
		cvReleaseMat(&d);

		cvAdd(dP, P_est, P_lm);	// update a, b, c -> P_lm

		it = pixs.begin();
		int n = 0;
		for(; it != pixs.end(); ++it){	// each pix
			int x = (int)(*it)%width;	// x of pix
			int y = (int)(*it)/width;	// y of pix
			pix->data.db[0] = x;
			pix->data.db[1] = y;
			double tmp_d = min(max(cvDotProduct(pix, P_lm), D_MIN), D_MAX);
			int indx = (*it)*(LEVEL+1)+Kd(tmp_d);
			d_lm->data.db[n] =  dCost_ori[indx];
						//Ed_init(pix, tmp_d, _frames);
			n++;
		}
		e_lm = cvDotProduct(d_lm,d_lm);	// new error
		//	cout << "error:" << e << "\tnew error: " << e_lm << "\tlamda:" << _lamda << endl;
		/* If the total distance error of the updated parameters is less than the previous one
		   then makes the updated parameters to be the current parameters
		   and decreases the value of the damping factor  */

		if (e_lm < e){
			//if (delta_est == delta_lm)
			//	return delta_lm;
			//else{
			_lamda = _lamda /n_lamda;
			P_est->data.db[0] = P_lm->data.db[0];
			P_est->data.db[1] = P_lm->data.db[1];
			P_est->data.db[2] = P_lm->data.db[2];
			e = e_lm;
			updateJ = true;
			//			}
		}
		else{ // increases the value of the damping factor
			updateJ = false;
			_lamda = _lamda*n_lamda;
		}


		//cout << "fininsh..." << k << endl;
	}	// end for iter
	cvReleaseMat(&pix);
	cvReleaseMat(&H);
	cvReleaseMat(&d_lm);
	cvReleaseMat(&J);
	//cvReleaseMat(&P_est);
	cvReleaseMat(&P_lm);
	cvReleaseMat(&dP);
	cvReleaseMat(&data);
	//cvReleaseMat(&obs);
	//cvReleaseMat(&y_est);
	//return P_est;
}

void computeJacobian(vector<int> pixs, vector<FrameData*> _frames, CvMat* P, CvMat* P_est, vector<CvMat*> As, vector<CvMat*> Bs){
	
	int n_iters = 10;				// times of iterator
	double _lamda = 0.01;
	int n_lamda = 10;
	bool updateJ = true;

	int Ndata= pixs.size();			// number of data
	int Nparams = 3;					// number of para

	CvMat *data=cvCreateMat(Ndata,1,CV_64FC1);
  	//CvMat *P_est = cvCreateMat(Nparams,1,CV_64FC1);	// parameter of each iteration 
	CvMat *P_lm = cvCreateMat(Nparams,1,CV_64FC1);
	// new para of next iteration
	cvCopy(P, P_est);
	double  e, e_lm;			// error
	CvMat *d_lm = cvCreateMat(Ndata,1,CV_64FC1);  
	CvMat* dP = cvCreateMat(Nparams,1,CV_64FC1);  // delta P
	CvMat* J = cvCreateMat(Ndata, Nparams, CV_64FC1); // Jacobian matrix
	CvMat* H = cvCreateMat(Nparams,Nparams,CV_64FC1);
	
	IplImage* img1;
	img1 = _frames[0]->getImg();
	int width = img1->width;

	for( int k = 0; k < n_iters; ++k){	//  for iterator
		CvMat *d=cvCreateMat(Ndata,1,CV_64FC1);		// diff
		vector<int>::iterator it = pixs.begin();
		if(updateJ){	/* compute the Jacobian */
			int n = 0;
			for(; it != pixs.end(); ++it){	// each pix
				int x = (int)(*it)%width;	// x of pix
				int y = (int)(*it)/width;	// y of pix
				CvMat* pix = cvCreateMat(3, 1, CV_64FC1);
					pix->data.db[0] = x;
					pix->data.db[1] = y;
					pix->data.db[2] = 1.;
				double tmp_d = Dplane(pix, P_est);
				int l = Kd(tmp_d);	// get the level of pix
				d->data.db[d->cols*n] = Ed_init(pix, tmp_d, _frames);
				//////////////////////////////////////////////////	
				CvMat* ED = Ed_init(pix, &_frames, &As, &Bs);
				cout << "xy:" << x <<" "<< y; 
				cout << " Ed:(AB)"<< ED->data.db[l] << "\n";
				/////////////////////////////////////////////////

				double PartialL;
				CvMat *likelihood = Likelihood_init(pix, _frames, As, Bs);

				if( (l+1) > LEVEL && (l) < 0 ){
					PartialL = (likelihood->data.db[LEVEL] -
						likelihood->data.db[0]) / (dK(LEVEL) - dK(0));
				}
				else if ( (l+1) > LEVEL){
					if(dK(LEVEL) == dK(l))	PartialL = 0;
					else
						PartialL =  (likelihood->data.db[LEVEL] -likelihood->data.db[l]) / (dK(LEVEL) - dK(l));	
				}
				else{
					PartialL = (likelihood->data.db[l+1] -
						likelihood->data.db[l]) / (dK(l+1) - dK(l));
				}

				cvReleaseMat(&pix);
		
				J->data.db[J->cols*n] = (double)x*PartialL;
				J->data.db[J->cols*n+1] = (double)y*PartialL;
				J->data.db[J->cols*n+2] = PartialL;
				n++;
			}
			cvMulTransposed(J,H,1);
			if (k == 0)
				e = cvDotProduct(d,d);
		}
		
		CvMat* H_lm = cvCreateMat(Nparams,Nparams,CV_64FC1);
		cvSetIdentity(H_lm);  // H_lm = I(3,3)
		cvGEMM(H_lm, H_lm, _lamda, H, 1.0, H_lm); // H_lm =H+ lamda*I(3)
		cvInvert(H_lm, H_lm); // H_lm = inv(H_lm)
		cvGEMM(J, d, 1, 0, 0, dP, CV_GEMM_A_T); // dP = (J')*d
		cvMatMul(H_lm, dP, dP); // dP = (H_lm)*dP
		//cvConvertScale(dP, dP, -1);
		cvReleaseMat(&d);

		cvAdd(dP, P_est, P_lm);	// update a, b, c -> P_lm

		it = pixs.begin();
		int n = 0;
		for(; it != pixs.end(); ++it){	// each pix
					int x = (int)(*it)%width;	// x of pix
					int y = (int)(*it)/width;	// y of pix
					CvMat* pix = cvCreateMat(3, 1, CV_64FC1);
						pix->data.db[0] = x;
						pix->data.db[1] = y;
						pix->data.db[2] = 1.;
					double tmp_d = Dplane(pix, P_lm);
					d_lm->data.db[d_lm->cols*n] =  Ed_init(pix, tmp_d, _frames);
					n++;
					cvReleaseMat(&pix);
		}
		e_lm = cvDotProduct(d_lm,d_lm);	// new error
	//	cout << "error:" << e << "\tnew error: " << e_lm << "\tlamda:" << _lamda << endl;
		  /* If the total distance error of the updated parameters is less than the previous one
				then makes the updated parameters to be the current parameters
				and decreases the value of the damping factor  */

		if (e_lm < e){
			//if (delta_est == delta_lm)
			//	return delta_lm;
			//else{
				_lamda = _lamda /n_lamda;
				P_est->data.db[0] = P_lm->data.db[0];
				P_est->data.db[1] = P_lm->data.db[1];
				P_est->data.db[2] = P_lm->data.db[2];
				e = e_lm;
				updateJ = true;
//			}
		}
		else{ // increases the value of the damping factor
			updateJ = false;
			_lamda = _lamda*n_lamda;
		}

	//cout << "fininsh..." << k << endl;
	}	// end for iter
	cvReleaseMat(&H);
	cvReleaseMat(&d_lm);
	cvReleaseMat(&J);
	//cvReleaseMat(&P_est);
	cvReleaseMat(&P_lm);
	cvReleaseMat(&dP);
	cvReleaseMat(&data);
	//cvReleaseMat(&obs);
	//cvReleaseMat(&y_est);
	//return P_est;
}
