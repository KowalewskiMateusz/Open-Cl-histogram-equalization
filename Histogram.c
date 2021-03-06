#define _CRT_SECURE_NO_DEPRECATE
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include <FreeImage.h>
#define BINS 256
#define DEVICE 0

cl_int status;
cl_uint numDevices;
cl_device_id *devices = NULL;

char buffer[100000];
cl_uint buf_uint;
cl_ulong buf_ulong;
size_t buf_sizet;

cl_int ciErr;
cl_int dimensions = 2;


int main(int argc, const char * argv[]) {
	
	FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, "Test_3.png", 0);
	FIBITMAP *imageBitmapGrey = FreeImage_ConvertToGreyscale(imageBitmap);
	cl_int width = FreeImage_GetWidth(imageBitmapGrey);
	cl_int height = FreeImage_GetHeight(imageBitmapGrey);
	cl_int pitch = FreeImage_GetPitch(imageBitmapGrey);

	cl_uchar  *imageIn = (cl_uchar *)malloc(height*width * sizeof(cl_uchar));
	cl_ulong *histogram = (cl_uint *)malloc(BINS * sizeof(cl_uint));

	FreeImage_ConvertToRawBits(imageIn, imageBitmapGrey, pitch, 8, 0xFF, 0xFF, 0xFF, TRUE);

	FreeImage_Unload(imageBitmapGrey);
	FreeImage_Unload(imageBitmap);

	const size_t szGlobalWorkSize[2] = { width,height };


	cl_uchar* srcA;
	cl_ulong* srcB;
	cl_ulong* srcC;
	cl_ulong* srcD;


	srcA = (void *)malloc(height * width * sizeof(cl_uchar));
	srcC = (void *)malloc(BINS * sizeof(cl_ulong));
	srcB = (void *)malloc(BINS * sizeof(cl_ulong));
	srcD = (void *)malloc(BINS * sizeof(cl_ulong));
	srcA = imageIn;

	for (int i = 0; i < BINS; i++)
	{
		srcB[i] = 0;
		srcC[i] = 0;
		srcD[i] = 0;
	}


	status = clGetDeviceIDs(
		NULL,
		CL_DEVICE_TYPE_ALL,
		0,
		NULL,
		&numDevices);
	if (status != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}


	devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

	status = clGetDeviceIDs(
		NULL,
		CL_DEVICE_TYPE_ALL,
		numDevices,
		devices,
		NULL);
	if (status != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}
	
	for (int i = 0; i < numDevices; i++)
	{
		clGetDeviceInfo(devices[i],
			CL_DEVICE_NAME,
			sizeof(buffer),
			buffer,
			NULL);


		clGetDeviceInfo(devices[i],
			CL_DEVICE_MAX_COMPUTE_UNITS,
			sizeof(buf_uint),
			&buf_uint,
			NULL);

		clGetDeviceInfo(devices[i],
			CL_DEVICE_MAX_WORK_GROUP_SIZE,
			sizeof(buf_sizet),
			&buf_sizet,
			NULL);


		clGetDeviceInfo(devices[i],
			CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
			sizeof(buf_uint),
			&buf_uint,
			NULL);

		size_t workitem_size[3];
		clGetDeviceInfo(devices[i],
			CL_DEVICE_MAX_WORK_ITEM_SIZES,
			sizeof(workitem_size),
			&workitem_size,
			NULL);

		clGetDeviceInfo(devices[i],
			CL_DEVICE_LOCAL_MEM_SIZE,
			sizeof(buf_ulong),
			&buf_ulong,
			NULL);
	}


	cl_context context = NULL;
	context = clCreateContext(
		NULL,
		numDevices,
		devices,
		NULL,
		NULL,
		&status);

	if (!context)
	{	
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}
	
	cl_command_queue cmdQueue;

	cmdQueue = clCreateCommandQueue(
		context,
		devices[DEVICE], 
		CL_QUEUE_PROFILING_ENABLE,
		&status);

	if (!cmdQueue)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}
	
	cl_mem bufferA;
	cl_mem bufferB;
	cl_mem bufferC;
	cl_mem bufferD;
	size_t datasize = sizeof(cl_int);
	size_t datasizeC = height * width * sizeof(cl_uchar);
	size_t hist = BINS * sizeof(cl_ulong);
	

	bufferA = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		datasizeC,
		NULL,
		&status);

	bufferB = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		hist,
		NULL,
		&status);

	bufferC = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		hist,
		NULL,
		&status);


	bufferD = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		hist,
		NULL,
		&status);



	
	FILE* programHandle;          
	size_t programSize;
	char *programBuffer;
	cl_program cpProgram;

	programHandle = fopen("/Users/kowal/source/repos/Histogram/Histogram/kernel.cl", "rb");
	fseek(programHandle, 0, SEEK_END);
	programSize = ftell(programHandle);
	rewind(programHandle);

	printf("Program size = %lu B \n", programSize);


	programBuffer = (char*)malloc(programSize + 1);

	programBuffer[programSize] = '\0'; // add null-termination
	fread(programBuffer, sizeof(char), programSize, programHandle);
	fclose(programHandle);


	cpProgram = clCreateProgramWithSource(
		context,
		1,
		(const char **)&programBuffer,
		&programSize,
		&ciErr);
	if (!cpProgram)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}
	free(programBuffer);

	ciErr = clBuildProgram(
		cpProgram,
		0,
		NULL,
		NULL,
		NULL,
		NULL);

	if (ciErr != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(cpProgram,
			devices[DEVICE],
			CL_PROGRAM_BUILD_LOG,
			sizeof(buffer),
			buffer,
			&len);
		printf("%s\n", buffer);
		exit(1);
	}


	cl_kernel ckKernel;
	ckKernel = clCreateKernel(
		cpProgram,
		"histogram",
		&ciErr);
	if (!ckKernel || ciErr != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	status = clEnqueueWriteBuffer(
		cmdQueue,
		bufferA,
		CL_FALSE,
		0,
		datasizeC,
		srcA,
		0,
		NULL,
		NULL);

	status = clEnqueueWriteBuffer(
		cmdQueue,
		bufferB,
		CL_FALSE,
		0,
		hist,
		srcB,
		0,
		NULL,
		NULL);

	
	ciErr |= clSetKernelArg(ckKernel,
		0,
		sizeof(cl_mem),
		(void*)&width);
	ciErr |= clSetKernelArg(ckKernel,
		1,
		sizeof(cl_mem),
		(void*)&height);
	ciErr |= clSetKernelArg(ckKernel,
		2,
		sizeof(cl_mem),
		(void*)&bufferA);
	ciErr |= clSetKernelArg(ckKernel,
		3,
		sizeof(cl_mem),
		(void*)&bufferB);

	ciErr = clEnqueueNDRangeKernel(
		cmdQueue,
		ckKernel,
		dimensions,
		NULL,
		szGlobalWorkSize,
		NULL,
		0,
		NULL,
		NULL);
	if (ciErr != CL_SUCCESS)
	{
		printf("Error launchung kernel1!\n");
	}

	clFinish(cmdQueue);

	ciErr = clEnqueueReadBuffer(
		cmdQueue,
		bufferB,
		CL_TRUE,
		0,
		hist,
		srcB,
		0,
		NULL,
		NULL);
	


	clFinish(cmdQueue);

	ckKernel = clCreateKernel(
		cpProgram,
		"cumulative",
		&ciErr);
	if (!ckKernel || ciErr != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	status = clEnqueueWriteBuffer(
		cmdQueue,
		bufferB,
		CL_FALSE,
		0,
		hist,
		srcB,
		0,
		NULL,
		NULL);

	status = clEnqueueWriteBuffer(
		cmdQueue,
		bufferC,
		CL_FALSE,
		0,
		hist,
		srcC,
		0,
		NULL,
		NULL);

	ciErr |= clSetKernelArg(ckKernel,
		0,
		sizeof(cl_mem),
		(void*)&width);
	ciErr |= clSetKernelArg(ckKernel,
		1,
		sizeof(cl_mem),
		(void*)&height);
	ciErr |= clSetKernelArg(ckKernel,
		2,
		sizeof(cl_mem),
		(void*)&bufferB);
	ciErr |= clSetKernelArg(ckKernel,
		3,
		sizeof(cl_mem),
		(void*)&bufferC);
	
	size_t szLocalWorkSize1 =  16 ;
	size_t szGlobalWorkSize1 =  BINS ;
	dimensions = 1;

	ciErr = clEnqueueNDRangeKernel(
		cmdQueue,
		ckKernel,
		dimensions,
		NULL,
		&szGlobalWorkSize1,
		&szLocalWorkSize1,
		0,
		NULL,
		NULL);

	if (ciErr != CL_SUCCESS)
	{
		printf("Error launchung kernel2!\n");
	}

	clFinish(cmdQueue);


	ciErr = clEnqueueReadBuffer(
		cmdQueue,
		bufferC,
		CL_TRUE,
		0,
		hist,
		srcC,
		0,
		NULL,
		NULL);

	clFinish(cmdQueue);

	
	
	ckKernel = clCreateKernel(
		cpProgram,
		"Equalize",
		&ciErr);

	if (!ckKernel || ciErr != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	status = clEnqueueWriteBuffer(
		cmdQueue,
		bufferA,
		CL_FALSE,
		0,
		hist,
		srcA,
		0,
		NULL,
		NULL);

	status = clEnqueueWriteBuffer(
		cmdQueue,
		bufferC,
		CL_FALSE,
		0,
		hist,
		srcC,
		0,
		NULL,
		NULL);

	status = clEnqueueWriteBuffer(
		cmdQueue,
		bufferD,
		CL_FALSE,
		0,
		hist,
		srcC,
		0,
		NULL,
		NULL);

	ciErr |= clSetKernelArg(ckKernel,
		0,
		sizeof(cl_mem),
		(void*)&width);
	ciErr |= clSetKernelArg(ckKernel,
		1,
		sizeof(cl_mem),
		(void*)&height);
	ciErr |= clSetKernelArg(ckKernel,
		2,
		sizeof(cl_mem),
		(void*)&bufferA);

	ciErr |= clSetKernelArg(ckKernel,
		3,
		sizeof(cl_mem),
		(void*)&bufferC);
	ciErr |= clSetKernelArg(ckKernel,
		4,
		sizeof(cl_mem),
		(void*)&bufferD);





	const size_t szLocalWorkSize2[2] = { 1,1 }; 
	const size_t szGlobalWorkSize2[2] = {height ,width};
	dimensions = 2;

	
	ciErr = clEnqueueNDRangeKernel(
		cmdQueue,
		ckKernel,
		dimensions,
		NULL,
		&szGlobalWorkSize2,
		&szLocalWorkSize2,
		0,
		NULL,
		NULL);

	if (ciErr != CL_SUCCESS)
	{
		printf("Error launchung kernel3!\n");
	}

	clFinish(cmdQueue);


	
	ciErr = clEnqueueReadBuffer(
		cmdQueue,
		bufferA,
		CL_TRUE,
		0,
		datasizeC,
		srcA,
		0,
		NULL,
		NULL);

	clFinish(cmdQueue);



		 FIBITMAP * imageOutBitmap = FreeImage_ConvertFromRawBits(srcA, width, height, width, 8, 0xFF, 0xFF,
			  0xFF, TRUE);
		
		FreeImage_Save(FIF_PNG, imageOutBitmap, " image_out .png", 0);



	free(srcA);
	free(srcB);
	free(srcC);
	free(srcD);
	free(histogram);
	free(devices);
	if (cpProgram) clReleaseProgram(cpProgram);
	if (cmdQueue) clReleaseCommandQueue(cmdQueue);
	if (context) clReleaseContext(context);
	if (bufferA) clReleaseMemObject(bufferA);
	if (bufferB) clReleaseMemObject(bufferB);
	if (bufferC) clReleaseMemObject(bufferC);
	if (bufferD) clReleaseMemObject(bufferD);


	return 0;
}