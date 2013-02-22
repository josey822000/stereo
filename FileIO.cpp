#include "FileIO.h"
#include "globalVar.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


using namespace std;
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
using namespace cv;

void WriteToFile(CvMat* M, string FileName){
	ofstream File;
	File.open(FileName.c_str());
	for(int j = 0; j < M->rows; ++j)
		for(int i = 0; i < M->cols; ++i){
			File << M->data.db[M->cols*j+i] << "\t";
		}
	File.close();
}
void WriteDisparityMap(CvMat* D, string MapName){
	CvMat* tmp = cvCreateMat(D->rows, D->cols, D->type);
	//double maxD, minD;		// find max and min element
	//cvMinMaxLoc(D, &minD, &maxD);
	double dist = 255/(D_MAX-D_MIN);
	cvSubS(D,cvRealScalar(D_MIN),tmp);
	cvConvertScale(tmp, tmp, dist);
	// save img
	cvSaveImage(MapName.c_str(), tmp);
}

void File2Mat(CvMat* m, std::string file){
	ifstream in;
	in.open(file.c_str());
	if( ! in) cout << "cannot open " << file << endl;
	else{
		for(int j = 0; j < m->height; ++j)
			for(int i = 0; i < m->width; ++i){
				double d;
				in >> d;
				m->data.db[m->cols*j+i] = d;
				}
		}
}

void saveImg(int width, int height, string inFile, string outFile){
	ifstream in;
	in.open(inFile.c_str());
	double disp;
	CvMat * D = cvCreateMat(height, width, CV_64FC1);
	for(int j = 0; j < height; ++j)
		for(int i = 0; i < width; ++i){
			in >> disp;
			D->data.db[D->cols*j+i] = disp;
		}
	WriteDisparityMap(D, outFile);
	cvReleaseMat(&D);
}

string int2str(int &i){
	string s;
	stringstream ss(s);
	ss << i;
	return ss.str();
}

string double2str(double &i){
	string s;
	stringstream ss(s);
	ss << i;
	return ss.str();
}
