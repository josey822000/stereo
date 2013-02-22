#include "ReadFGMask.h"

#include <string>
#include <fstream>
#include <iostream>

using namespace std;
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
using namespace cv;

ReadFGMask::ReadFGMask(string filename){
	m_rect.clear();
	ifstream in;
	in.open(filename.c_str());
	//vector<pair<Point2d, Point2d> > test;
	if(in.is_open()){
		cout << "Open " << filename << "!\n";
		while( ! in.eof()){
			Point2d p1, p2;
			double x1,y1,x2,y2;
			pair<Point2d, Point2d> tmp;
			in >> x1 >> y1 >> x2 >> y2;
			p1.x = x1;
			p1.y = y1;
			p2.x = x2;
			p2.y = y2;
			tmp = make_pair(p1, p2);
			m_rect.push_back(tmp);
		}
	}
	else
		std::cout << "Open FGMask file fail! " << std::endl;
}

ReadFGMask::~ReadFGMask(){
	vector<pair<Point2d, Point2d> >().swap(m_rect);
	//vector<CvMat*>().swap(m_mask);
}
