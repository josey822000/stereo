#ifndef COMPUTEMRF_H
#define COMPUTEMRF_H

#include "MRF/mrf.h"
#include "MRF/GCoptimization.h"
#include "MRF/MaxProdBP.h"
#include "MRF/BP-S.h"

#include "ReadCameraParameter.h"
#include "ReadVideoSeq.h"
#include "FrameData.h"
#include <string>
#include <fstream>

void ComputeMRF(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, MRF::CostVal *&dCost, MRF::CostVal *&hCue, MRF::CostVal *&vCue);

void ComputeMRF_Seg(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, MRF::CostVal *&dCost, MRF::CostVal *&hCue, MRF::CostVal *&vCue);

void ComputeMRF_SegPlane(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, std::string segFile,std::string outDFile);

void ComputeMRF_D(ReadCameraParameter* cp, ReadVideoSeq* vs,int now_frame, int Nframe, MRF::CostVal *&dCost, MRF::CostVal *&hCue, MRF::CostVal *&vCue);

void ComputeMRF_Seg(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, MRF::CostVal *&dCost, MRF::CostVal *&hCue, MRF::CostVal *&vCue, double*&dCost_ori);

void multi_ComputeMRF_Seg(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, MRF::CostVal *&dCost, MRF::CostVal *&hCue, MRF::CostVal *&vCue, double*&dCost_ori);

void ComputeMRF_SegPlane(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, std::string segFile,std::string outDFile, double* dCost_ori);

 /* Process Video  */

void VideoMRF(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, int seqSize, std::string name);

void VideoMRF_seg(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, int seqSize, std::string name);

void Video_SegPlane(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, int seqSize, std::string name);

void VideoMRF_Final(ReadCameraParameter* cp, ReadVideoSeq* vs, int now_frame, int Nframe, int seqSize, std::string name);

void Video_initial(ReadCameraParameter* cp, ReadVideoSeq *vs, int now_frame, int Nframe, int seqSize, std::string name);

void Video_initial2(ReadCameraParameter* cp, ReadVideoSeq *vs, int now_frame, int Nframe, int seqSize, std::string name);
#endif
