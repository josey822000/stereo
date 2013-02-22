#ifndef READFGMASK_H
#define READFGMASK_H

#include <string>
#include <utility>
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"

class ReadFGMask{
public:
	ReadFGMask(std::string filename);
	~ReadFGMask();
	std::pair<cv::Point2d, cv::Point2d> GetPoints(int frame){ return m_rect[frame];}
private:
	//std::vector<CvMat*> m_mask;
	std::vector<std::pair<cv::Point2d, cv::Point2d> >m_rect;
};
#endif