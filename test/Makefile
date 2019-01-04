CXX ?= g++
CC      ?= gcc
CFLAGS  ?= -pedantic -Wall
CXXFLAGS  ?= -pedantic -Wall -ggdb
LDFLAGS ?= -L/usr/local/lib/

OBJ_ORB = orb.o
OBJ_ORB_OCL = orb_ocl.o ocl.o
OBJ_DISPARITY = disparity.o xunit_lib_tara.o

exec_prefix ?= /usr
bindir ?= $(exec_prefix)/bin

PROGRAMS ?= orb orb-ocl disparity

all: $(PROGRAMS)

orb: $(OBJ_ORB)
	${CXX} $(CXXFLAGS) -o orb $(OBJ_ORB) -lopencv_core -lopencv_features2d -lopencv_videoio -lopencv_highgui $(LDFLAGS)

orb-ocl: $(OBJ_ORB_OCL)
	${CXX} $(CXXFLAGS) -o orb-ocl $(OBJ_ORB_OCL) -lOpenCL -lopencv_core -lopencv_imgcodecs -lopencv_highgui $(LDFLAGS)

disparity: $(OBJ_DISPARITY)
	${CXX} $(CXXFLAGS) -o disparity $(OBJ_DISPARITY) -lOpenCL -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_calib3d -lopencv_videoio -lopencv_imgcodecs -lopencv_highgui -ludev $(LDFLAGS) -lopencv_ximgproc

install: all
	install -d $(DESTDIR)$(bindir)
	install -m 0755 orb orb-ocl $(DESTDIR)$(bindir)

clean: $(OBJ_ORB) $(OBJ_ORB_OCL) $(OBJ_DISPARITY)
	@echo "Clean object files"
	@rm -f $<
	@rm -f $(PROGRAMS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<
