#include "ReadCameraParameter.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;
using namespace std;

ReadCameraParameter::ReadCameraParameter(boost::filesystem::path p){
	ifstream camera;
	std::string ignore;
	double tmp ;

	BFS::path txt = p/ "camera.txt";
	camera.open(txt.string().c_str());
	if( !camera.is_open() ) cout << "Cannot open cameraParameter!" << endl;
	else cout << "Success!!!!" << endl;
	if( !camera.eof()){
		for (int i = 0; i< 5; ++i) getline(camera, ignore, '\n');  // ignore words	
		camera >> no_frame;	// number of frames
		cout << "There are " << no_frame << " frames." << endl;
		getline(camera, ignore, '\n');  // ignore 

		for(int s = 0; s < no_frame; s++){ // read camera_para of frames
			
			/* read matrix for K */
			CvMat* k = cvCreateMat(3, 3, CV_64FC1);
			for(int i = 0; i < 9; i++){
					camera >> tmp;
					cvmSet(k, (int) (i/3), i%3, tmp);
					//cout << i << " " << tmp[i] << endl;
				}
			K.push_back(k);

			/* read matrix for R */
			CvMat* r = cvCreateMat(3, 3, CV_64FC1);
			for(int i = 0; i < 9; i++){
					camera >> tmp;
					cvmSet(r, (int) (i/3), i%3, tmp);
					//cout << i << " " << tmp[i] << endl;
				}
			R.push_back(r);
			
			/* read matrix for T */
			CvMat* t = cvCreateMat(3, 1, CV_64FC1);
			for(int i = 0; i < 3; i++){
					camera >> tmp;
					cvmSet(t, i%3, 0, tmp);
					//cout << i << " " << tmp[i] << endl;
				}
			T.push_back(t);
			
			for (int i =0; i< 3; ++i)
			getline(camera, ignore, '\n');  // ignore 
		}
		
	}
	camera.close();
}

CvMat* ReadCameraParameter::getHomoX(CvMat* pix, double d, int frame1, int frame2){
	/* X(f2) = K(f2)tran_R(f2)R(f1)inv_K(f1)X(f1)+d(f1)K(f2)tran_R(f2)[T(f1)-T(f2)] */
	
	CvMat* pix_h = cvCreateMat(3,1,CV_64FC1);
	CvMat* R_tran = cvCreateMat(3,3,CV_64FC1);
	CvMat* K_inver = cvCreateMat(3,3,CV_64FC1);
	CvMat* tmp = cvCreateMat(3,3,CV_64FC1);
	CvMat* tmp2 = cvCreateMat(3,1,CV_64FC1);

	cvTranspose(R[frame2], R_tran);
	cvInvert(K[frame1], K_inver); 
	cvSub(T[frame1], T[frame2], tmp2); 
	cvMatMul(K[frame2], R_tran, tmp);
	cvMatMul(tmp, tmp2, tmp2);
	cvMatMul(tmp, R[frame1], tmp);
	cvMatMul(tmp, K_inver, tmp);
	cvMatMul(tmp, pix, pix_h);		//tmp

	cvConvertScale(tmp2, tmp2, d);
	cvAdd(pix_h, tmp2, pix_h);

	cvConvertScale(pix_h, pix_h, 1/pix_h->data.db[2]);  // normalize

	cvReleaseMat(&R_tran);
	cvReleaseMat(&K_inver);
	cvReleaseMat(&tmp);
	cvReleaseMat(&tmp2);	
	
	return pix_h;
}

const ReadCameraParameter &ReadCameraParameter::operator =( const ReadCameraParameter &right){
	no_frame = right.no_frame;
	K = right.K;
	R = right.R;
	T = right.T;
	return *this;
}

ReadCameraParameter::~ReadCameraParameter(){
	std::vector<CvMat*>().swap(T);
	std::vector<CvMat*>().swap(R);
	std::vector<CvMat*>().swap(K);
}
