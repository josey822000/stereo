#include "ReadVideoSeq.h"
#include "ReadFGMask.h"
#include "globalVar.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>

using namespace std;

ReadVideoSeq::ReadVideoSeq(boost::filesystem::path p, bool disparity, bool seg, bool FG){
    if( exists( p ) && is_directory( p ) )
    {
        std::cout << p << " is a directory containing:" << std::endl;
        std::vector<BFS::path> v;
        copy( directory_iterator( p ), directory_iterator(), back_inserter( v ) );
        sort( v.begin(), v.end() );
        for( std::vector<path>::const_iterator it( v.begin() ); it != v.end(); ++it ){
            if ( ! is_directory( *it )){
                IplImage* tmp = 0;
                tmp= cvLoadImage((*it).string().c_str());
                if(!tmp) std::cout << "Could not load image file!" << std::endl;
                else{
                    //std::cout << "Open image!!" << std::endl;
                    //cout << " " << *it << endl;
                    m_videoSeq.push_back(tmp);
                }
            }
        }
        std::cout << "Finish loading frames! " << (int)(v.end()-v.begin()) << "files!" <<std::endl;
    } 
    // save data
    m_width = m_videoSeq[0]->width;
    m_height = m_videoSeq[0]->height;
    no_frame = m_videoSeq.size();
    BFS::path txt = p/ "disparity";
	allF_dis = new char[no_frame];
    if (disparity){	// load level and disparity value
		memset(allF_dis,1,no_frame);
        b_dis = true;
        ReadVideoDisparity(txt);
    }
    else{			// runF
		memset(allF_dis,0,no_frame);
        b_dis = false;
        CvMat* tmp = cvCreateMat(m_height, m_width, CV_64FC1);	// josey use reserve is better?
        cvZero(tmp);
        for(int i = 0; i < no_frame; ++i)
            m_disparity.push_back(tmp);
    }

    txt = p.parent_path()/ "seg";
    if (seg){	// load name of segment file
        b_seg = true;
        ReadVideoSeg(txt);
    }
    else		// runIni
        b_seg = false;
    if(FG){
        b_fg = true;
        b_mask = true;
        txt = p.parent_path()/"FGMask.txt";
        string MaskFile = txt.string();
        FGMask = new ReadFGMask(MaskFile);
        ReadMask();
    }
    else{		// 0 present not forground
        b_fg = false;
        CvMat* tmp = cvCreateMat(m_height, m_width, CV_64FC1);
        cvZero(tmp);
        //cvAddS(tmp,cvRealScalar(1),tmp);
        for(int i = 0; i < no_frame; ++i)
            m_mask.push_back(tmp);
    }

}

void ReadVideoSeq::ReadMask(){
    for(int i = 0; i < no_frame; i++){
        pair<Point2d, Point2d> ps;
        ps = FGMask->GetPoints(i);
        CvRect rect = cvRect(ps.first.x, ps.first.y, ps.second.x-ps.first.x, ps.second.y-ps.first.y);
        if( rect.width < 0){
            rect.x += rect.width;
            rect.width *= -1;
        }
        if(rect.height < 0){
            rect.y += rect.height;
            rect.height *= -1;
        }
        CvMat* tmp = cvCreateMat(m_height, m_width, CV_64FC1);
        cvZero(tmp);
        CvMat* submat = cvCreateMat(rect.height,rect.width,CV_64FC1);
        cvGetSubRect(tmp, submat, rect);
        cvZero(submat);
        cvAddS(submat, cvRealScalar(1.0), submat); // 1-> for fg; 0 -> for bg
        m_mask.push_back(tmp);
    }
}

void ReadVideoSeq::ReadVideoDisparity(boost::filesystem::path p)
{

    if( exists( p ) && is_directory( p ) )
    {
        CvMat* tmp = cvCreateMat(m_height, m_width, CV_64FC1);	// josey use reserve is better?
        cvZero(tmp);
        for(int i = 0; i <D_FRAME_START; ++i)
            m_disparity.push_back(tmp);


        std::cout << p << " is a directory containing:" << std::endl;
        std::vector<BFS::path> v;
        copy( directory_iterator( p ), directory_iterator(), back_inserter( v ) );
        sort( v.begin(), v.end() );
        cerr<< "disparity num: "<<v.size() << endl;
        for( std::vector<path>::const_iterator it( v.begin() ); it != v.end(); ++it ){
            if ( ! is_directory( *it )){
                ifstream file;
                file.open((*it).string().c_str());
                if(!file.is_open()) std::cout << "Could not load disparity file!" << std::endl;
                else{
                    CvMat* tmp = cvCreateMat(m_height, m_width,CV_64FC1);
                    for(int h = 0; h < m_height; ++h){
                        for(int w = 0; w < m_width; ++w){

                            double d;
                            file >> d;
                            tmp->data.db[tmp->cols*h+w] = d;
                        }
                    }
                    m_disparity.push_back(tmp);
                }
            }
        }
        for(int i = 0; i <no_frame - D_FRAME_END; ++i)
            m_disparity.push_back(tmp);

        std::cout << "Finish loading files of disparity!" << std::endl;
    }
    else
        std::cout<< "Can't open disparity" <<endl;
}

void ReadVideoSeq::ReadVideoSeg(boost::filesystem::path p){
    if( exists( p ) && is_directory( p ) )
    {
        std::cout << p << " is a directory containing:" << std::endl;
        std::vector<BFS::path> v;
        copy( directory_iterator( p ), directory_iterator(), back_inserter( v ) );
        sort( v.begin(), v.end() );
        for( std::vector<path>::const_iterator it( v.begin() ); it != v.end(); ++it ){
            if ( ! is_directory( *it )){
                m_segment.push_back((*it).string());
            }
        }
        std::cout << "Finish loading files of seg!" << (int)(v.end()-v.begin()) << "files!" <<std::endl;
    } 
}

const ReadVideoSeq &ReadVideoSeq::operator =( const ReadVideoSeq &right){

    m_videoSeq = right.m_videoSeq;
    no_frame = right.no_frame;
    m_width = right.m_width;
    m_height = right.m_height;

    if(right.b_dis)
        m_disparity = right.m_disparity;

    return *this;
}

IplImage* ReadVideoSeq::getVideoFrame(int frame){
    return m_videoSeq[frame];
}
ReadVideoSeq::~ReadVideoSeq(){
    std::vector<IplImage*>().swap(m_videoSeq);
    std::vector<CvMat*>().swap(m_disparity);
}
