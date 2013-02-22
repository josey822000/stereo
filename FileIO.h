#ifndef FILEIO_H
#define FILEIO_H

#include <fstream>
#include <string>
#include <sstream>

using namespace std;
#include <opencv/cv.h>
#include <opencv/cxcore.h>
using namespace cv;

void WriteToFile(CvMat* M, std::string FileName);
void WriteDisparityMap(CvMat* D, std::string MapName);
void File2Mat(CvMat* m, std::string file);
void saveImg(int width, int height, std::string inFile, std::string outFile);
std::string int2str(int &i);
std::string double2str(double &i);

#endif