VERSION = VideoStereo

SRC =  MRF/mrf.cpp MRF/ICM.cpp MRF/GCoptimization.cpp \
       MRF/graph.cpp  MRF/maxflow.cpp \
       MRF/MaxProdBP.cpp MRF/LinkedBlockList.cpp MRF/regions-maxprod.cpp \
       MRF/TRW-S.cpp MRF/BP-S.cpp

LIBS = $(SUBLIBS) -L./../../OpenCV/lib -lopencv_core -lopencv_highgui -lboost_system -lboost_filesystem -L. -lMRF

SOURCES       = ComputeEnergy.cpp \
		FileIO.cpp \
		FrameData.cpp \
		ReadCameraParameter.cpp \
		ReadVideoSeq.cpp \
		runAlgo.cpp \
		Segment.cpp \
		ComputeMRF.cpp \
		ReadFGMask.cpp \
		runMRF.cpp
		
		

CC = g++

WARN   = -W -W
OPT ?= -O3
#CPPFLAGS = $(OPT) $(WARN)
CPPFLAGS = $(OPT) $(WARN) -DUSE_64_BIT_PTR_CAST

OBJ = $(SRC:.cpp=.o) $(SOURCES:.cpp=.o) 

all: libMRF.a VideoStereo

libMRF.a: $(OBJ)
	rm -f libMRF.a
	ar ruc libMRF.a $(OBJ)
	ranlib libMRF.a

VideoStereo: libMRF.a $(SOURCES)
	$(CC) -g -o VideoStereo $(SOURCES) $(LIBS) 

clean: 
	rm -f $(OBJ) core *.stackdump *.bak

allclean: clean
	rm -f libMRF.a VideoStereo VideoStereo.exe

depend:
	@makedepend -Y -- $(CPPFLAGS) -- $(SRC) 2>> /dev/null

# DO NOT DELETE THIS LINE -- make depend depends on it.

BP-S.o: MRF/BP-S.cpp MRF/BP-S.h \
		MRF/mrf.h \
		MRF/typeTruncatedQuadratic2D.h
	
GCoptimization.o: MRF/GCoptimization.cpp MRF/energy.h \
		MRF/graph.h \
		MRF/block.h \
		MRF/mrf.h \
		MRF/GCoptimization.h \
		MRF/LinkedBlockList.h
	
graph.o: MRF/graph.cpp MRF/graph.h \
		MRF/block.h \
		MRF/mrf.h
	
ICM.o: MRF/ICM.cpp MRF/ICM.h \
		MRF/mrf.h \
		MRF/LinkedBlockList.h
	
LinkedBlockList.o: MRF/LinkedBlockList.cpp MRF/LinkedBlockList.h
	
maxflow.o: MRF/maxflow.cpp MRF/graph.h \
		MRF/block.h \
		MRF/mrf.h
	
MaxProdBP.o: MRF/MaxProdBP.cpp MRF/MaxProdBP.h \
		MRF/mrf.h \
		MRF/LinkedBlockList.h \
		MRF/regions-new.h
	
mrf.o: MRF/mrf.cpp MRF/mrf.h
	
regions-maxprod.o: MRF/regions-maxprod.cpp MRF/MaxProdBP.h \
		MRF/mrf.h \
		MRF/LinkedBlockList.h \
		MRF/regions-new.h

TRW-S.o: MRF/TRW-S.cpp MRF/TRW-S.h \
		MRF/mrf.h \
		MRF/typeTruncatedQuadratic2D.h
	
ComputeEnergy.o: ComputeEnergy.cpp ComputeEnergy.h \
		ReadCameraParameter.h \
		ReadVideoSeq.h \
		FrameData.h \
		globalVar.h \
		FileIO.h \
		MRF/mrf.h \
		MRF/GCoptimization.h \
		MRF/LinkedBlockList.h \
		MRF/graph.h \
		MRF/block.h \
		MRF/energy.h \
		MRF/MaxProdBP.h \
		MRF/regions-new.h \
		MRF/BP-S.h

ComputeMRF.o: ComputeMRF.cpp ComputeMRF.h \
		ReadCameraParameter.h \
		ReadFGMask.h \
		ReadVideoSeq.h \
		FrameData.h \
		ComputeEnergy.h \
		globalVar.h \
		FileIO.h \
		Segment.h \
		runAlgo.h \
		MRF/ICM.h \
		MRF/TRW-S.h \
		MRF/mrf.h \
		MRF/GCoptimization.h \
		MRF/LinkedBlockList.h \
		MRF/graph.h \
		MRF/block.h \
		MRF/energy.h \
		MRF/MaxProdBP.h \
		MRF/regions-new.h \
		MRF/BP-S.h 

FileIO.o: FileIO.cpp FileIO.h \
		globalVar.h

FrameData.o: FrameData.cpp FrameData.h

ReadCameraParameter.o: ReadCameraParameter.cpp ReadCameraParameter.h

ReadFGMask.o: ReadFGMask.cpp ReadFGMask.h

ReadVideoSeq.o: ReadVideoSeq.cpp ReadVideoSeq.h \
		ReadFGMask.h

runAlgo.o: runAlgo.cpp runAlgo.h \
		MRF/mrf.h \
		MRF/ICM.h \
		MRF/LinkedBlockList.h \
		MRF/GCoptimization.h \
		MRF/graph.h \
		MRF/block.h \
		MRF/energy.h \
		MRF/MaxProdBP.h \
		MRF/regions-new.h \
		MRF/TRW-S.h \
		MRF/BP-S.h \
		globalVar.h \
		ReadCameraParameter.h \
		ComputeMRF.h \
		ReadVideoSeq.h \
		FrameData.h \
		ComputeEnergy.h \
		FileIO.h

Segment.o: Segment.cpp Segment.h \
		ReadCameraParameter.h \
		ReadVideoSeq.h \
		FrameData.h \
		globalVar.h \
		ComputeEnergy.h \
		FileIO.h \
		MRF/mrf.h \
		MRF/GCoptimization.h \
		MRF/LinkedBlockList.h \
		MRF/graph.h \
		MRF/block.h \
		MRF/energy.h \
		MRF/MaxProdBP.h \
		MRF/regions-new.h \
		MRF/BP-S.h

runMRF.o: runMRF.cpp \
	       	globalVar.h \
		ReadCameraParameter.h \
		ComputeMRF.h \
		ReadVideoSeq.h \
		FrameData.h \
		ComputeEnergy.h \
		FileIO.h\
	 	MRF/mrf.h \
		MRF/ICM.h \
		MRF/LinkedBlockList.h \
		MRF/GCoptimization.h \
		MRF/graph.h \
		MRF/block.h \
		MRF/energy.h \
		MRF/MaxProdBP.h \
		MRF/regions-new.h \
		MRF/TRW-S.h \
		MRF/BP-S.h



