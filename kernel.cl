#define BINS 256
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
__kernel void histogram(int width,
	int height,
	__global unsigned char* imageIn,
	__global unsigned long* histogram)
{

	__local int histogramLocal[BINS];


	int G_row_ID = get_global_id(1);
	int G_col_ID = get_global_id(0);

	int global_row_size = get_global_size(1);
	int global_column_size = get_global_size(0);


	int L_row_ID = get_local_id(1);
	int L_col_ID = get_local_id(0);

	int local_row_size = get_local_size(1);
	int local_colum_size = get_local_size(0);


	int start_row = height / global_row_size * G_row_ID;			//calcute starting and ending row number for each work group
	int end_row = height / global_row_size * (G_row_ID + 1);

	int start_col = width / global_column_size * G_col_ID;
	int end_col = width / global_column_size * (G_col_ID + 1);

	int local_vector_ID = L_row_ID * local_row_size + L_col_ID; ;	//calculate the local_id in 1D

	for (int i = local_vector_ID; i < BINS; i += local_row_size * local_colum_size) //loop to put 0 in local histograms
		histogramLocal[i] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = start_row; i < end_row; i++)		//Nested loops that creates local cumulative histograms					
		for (int j = start_col; j < end_col; j++)
			atom_inc(&histogramLocal[imageIn[i*width + j]]);

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = local_vector_ID; i < BINS; i += local_row_size * local_colum_size)  //Adding local histograms to the global one
		atomic_add(histogram + i, histogramLocal[i]);
}



__kernel void cumulative(int width,
	int height,
	__global unsigned long* histogramIn,
	__global unsigned long* histogramOut)
{
	int id = get_global_id(0);
	for (int i = id; i < BINS; i++)							//every work item executes loop and add the number of previous elements that equals work item id
		atomic_add(&histogramOut[i], histogramIn[id]);
}





unsigned char scale(__global unsigned long *cdf, unsigned long min, int imageSize) {	//scale function same as sequential one

	float scale;
	scale = (float)(*cdf - min) / (float)(imageSize - min);
	scale = round(scale * (float)(BINS - 1));
	return (int)scale;
}


inline int findMin(__global unsigned long * min)
{
	int id = get_global_id(1);
	int size = get_global_size(1);

	for (int i = size / 2; i > 0;i /= 2)									//parallel reduction algorithm for finding minimum value
	{
		barrier(CLK_GLOBAL_MEM_FENCE);
		if (id < i && min[id] != 0 && min[id + i] != 0)
			min[id] = (min[id] < min[id + i]) ? min[id] : min[id + i];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (id == 0)
	{
		return min;
	}

}



__kernel void Equalize(int width, int heigth, __global unsigned char * image, __global unsigned long * cdf, __global unsigned long * min_cdf) {

	int g_row_id = get_global_id(0);
	int g_col_id = get_global_id(1);
	int imageSize = width * heigth;
	if (g_col_id == 0 && g_row_id < 256)		//only 256 work items should execute min function
		min_cdf = findMin(min_cdf);
	barrier(CLK_GLOBAL_MEM_FENCE);

	int min = min_cdf[0];
	image[g_row_id* width + g_col_id] = scale(&cdf[image[g_row_id* width + g_col_id]], min, imageSize);  //one work item assign one pixel value
}



