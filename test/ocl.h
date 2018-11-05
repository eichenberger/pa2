#ifndef _OCL_H_
#define _OCL_H_

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/opencl.h>

#define ARRAY_SIZE 1000

void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3]);
bool CreateMemObjects(cl_context context, cl_mem memObjects[3], float * a, float * b);
cl_program CreateProgram(cl_context context, cl_device_id device, const char *fileName);
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id * device);
cl_context CreateContext();

#endif /* _OCL_H_ */
