#ifndef READCAMERAPARAMETER_H
#define READCAMERAPARAMETER_H

#define BOOST_FILESYSTEM_VERSION 3
#include "boost/filesystem.hpp"   // includes all needed Boost.Filesystem declarations
#include <vector>
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"

namespace BFS = boost::filesystem;
using namespace BFS;
using namespace cv;

class ReadCameraParameter{
public:
	ReadCameraParameter(){}
	ReadCameraParameter(BFS::path p);
	//void ReadFrames(BFS::path p);
	CvMat* getHomoX(CvMat* pix, double depth, int frame, int frame2);
	~ReadCameraParameter();
	int getNo_Frame(){ return no_frame; }
	const ReadCameraParameter &operator=( const ReadCameraParameter &);
	CvMat* getK(int frame){ return K[frame]; }
	CvMat* getR(int frame){ return R[frame]; }
	CvMat* getT(int frame){ return T[frame]; }
	//std::vector<IplImage*> m_videoSeq;
	//std::vector<IplImage*> getVideoSeq(){ return m_videoSeq; }
private:
	int no_frame;					// number of frames
	std::vector<CvMat*>K;	// 3*3	Intrinsic   Mat
	std::vector<CvMat*>R;	// 3*3  Rotational  Mat
	std::vector<CvMat*>T;	// 3*1  Translation Mat
};

#endif
