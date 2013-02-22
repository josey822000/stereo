#ifndef RUNALGO_H
#define RUNALGO_H

#include "MRF/mrf.h"
#include "MRF/ICM.h"
#include "MRF/GCoptimization.h"
#include "MRF/MaxProdBP.h"
#include "MRF/TRW-S.h"
#include "MRF/BP-S.h"

#include <string>

void runBP(EnergyFunction *energy, std::string filename, MRF *&mrf);
void runBPS(EnergyFunction *energy, std::string filename, MRF *&mrf);
void runICM(EnergyFunction *energy, std::string filename, MRF *&mrf);
void runExpansion(EnergyFunction *energy, std::string filename, MRF *&mrf);
void runTRWS(EnergyFunction *energy, std::string filename, MRF *&mrf);
void runSwap(EnergyFunction *energy, std::string filename, MRF *&mrf);

#endif
