#include <iostream>
#include <sstream>
#include <fstream>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


#include "ocl.h"

using namespace std;
using namespace cv;

int main(int argc, char ** argv) {
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = { 0, 0, 0 };
    cl_int errNum;

    Mat image;

    image = imread(argv[1], 0);   // Read the file
//    imshow("image", image);
//    waitKey(0);

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL) {
        cerr << "Failed to create OpenCL context." << endl;
        return 1;
    }
    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL) {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }
    // Create OpenCL program from HelloWorld.cl kernel source
    program = CreateProgram(context, device, "fast.cl");
    if (program == NULL) {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }
    // Create OpenCL kernel
    kernel = clCreateKernel(program, "locate_features", NULL);
    if (kernel == NULL) {
        cerr << "Failed to create kernel" << endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    cl_image_format cif;
    cif.image_channel_order = CL_R;
    cif.image_channel_data_type = CL_UNSIGNED_INT8;

    Size imSize = image.size();
    cout << "size: " << imSize.width << "x" << imSize.height <<endl;

    cl_image_desc cid;
    cid.image_type= CL_MEM_OBJECT_IMAGE2D;
    cid.image_width= imSize.width;
    cid.image_height= imSize.height;
    cid.image_depth= 1;
    cid.image_array_size= 0;
    cid.image_row_pitch= 0;
    cid.image_slice_pitch= 0;
    cid.num_mip_levels= 0;
    cid.num_samples= 0;
    cid.buffer= NULL;

    cl_mem cl_image = clCreateImage(context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    &cif,
                                    &cid,
                                    image.data,
                                    &errNum);
    if (errNum != CL_SUCCESS) {
        cerr << "Failed to create image: " << errNum << endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    cl_mem cl_score;
    cl_score= clCreateBuffer(context, CL_MEM_READ_WRITE,
                    imSize.width*imSize.height*sizeof(float), NULL, &errNum);
    if (errNum != CL_SUCCESS) {
        cerr << "Failed to create score buffer" << endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }
    // Set the kernel arguments (a, b, result)
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_image);
    float threshold = 31;
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_float), &threshold);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_score);
    if (errNum != CL_SUCCESS) {
        cerr << "Error setting kernel arguments." << endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

#define GROUP_SIZE 4
    size_t globalWorkSize[2] = {
        (size_t)imSize.width,
        (size_t)imSize.height
    };
    size_t localWorkSize[2] = {
        GROUP_SIZE,
        GROUP_SIZE
    };

    float *score;
    score = new float[imSize.width*imSize.height];
    double e1, e2, time = 0;
    #define REPEAT 5000
    for (int i = 0; i < REPEAT; i++) {
        e1 = getTickCount();
        // Queue the kernel up for execution across the array
        errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
            globalWorkSize, localWorkSize,
            0, NULL, NULL);
        if (errNum != CL_SUCCESS) {
            cerr << "Error queuing kernel for execution." << endl;
            Cleanup(context, commandQueue, program, kernel, memObjects);
            return 1;
        }

        e2 = getTickCount();
        time += (e2 - e1)/ getTickFrequency();
    }
    printf("exec time: %f", time/REPEAT);

    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(commandQueue,
                                 cl_score,
                                 CL_TRUE,
                                 0,
                                 sizeof(float)*imSize.width*imSize.height,
                                 score,
                                 0,
                                 NULL,
                                 NULL);
    if (errNum != CL_SUCCESS) {
        cerr << "Error reading result buffer: " << errNum << endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    cout << "Result:" << endl;
    for (int i = 0; i < imSize.width*imSize.height; i++) {
        if (score[i] > 0)
            cout <<  i%imSize.width << "x" << i/imSize.width << ": " << score[i] << endl;
    }
    cout << endl;

    delete score;

    cout << "Executed program successfully." << endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 0;
}
