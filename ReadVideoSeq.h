#ifndef READVIDEOSEQ_H
#define READVIDEOSEQ_H

#define BOOST_FILESYSTEM_VERSION 3
#include "boost/filesystem.hpp"  // includes all needed Boost.Filesystem declarations
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include <vector>
#include "ReadFGMask.h"

namespace BFS = boost::filesystem;
using namespace BFS;
using namespace cv;

class ReadVideoSeq{
public:
	ReadVideoSeq(boost::filesystem::path p, bool disparity, bool seg, bool fg);
	void ReadVideoDisparity(boost::filesystem::path p);
	void ReadVideoSeg(boost::filesystem::path p);
	void ReadMask();
	const ReadVideoSeq &operator =( const ReadVideoSeq &right);
	~ReadVideoSeq();
	IplImage* getVideoFrame(int frame);
	std::string getSeg(int frame){return m_segment[frame];}
	CvMat* getDisparity(int frame){ return m_disparity[frame]; }
	CvMat* getMask(int frame){return m_mask[frame];}	
	void setDisparity(CvMat* m, int frame){  m_disparity[frame] = m; }
	void setMask(CvMat* m, int frame){ m_mask[frame] = m;}
	void getImgSz(int* x,int* y)
	{
		*y = m_videoSeq[0]->height;
		*x = m_videoSeq[0]->width;
	}
	int no_frame;
	int m_width;
	int m_height;
	bool b_dis;
	bool b_seg;
	bool b_fg;
	bool b_mask;

	ReadFGMask* FGMask;
	char* allF_dis;
private:
	std::vector<IplImage*> m_videoSeq;
	std::vector<CvMat*> m_disparity;
	std::vector<std::string> m_segment;
	std::vector<CvMat*> m_mask;
	/* for foreground Mask */

};
#endif
