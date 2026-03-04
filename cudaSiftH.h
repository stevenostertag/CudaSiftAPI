#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudautils.h"
#include "cudaImage.h"

struct SiftDeviceContext;  // defined in cudaSiftD.h

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

int ExtractSiftLoop(SiftData *siftData, CudaImage *img, int numOctaves, float initBlur, float thresh, float lowestScale, float edgeLimit, float subsampling, float *memoryTmp, float *memorySub, SiftDeviceContext &ctx);
void ExtractSiftOctave(SiftData *siftData, CudaImage *img, int octave, float thresh, float lowestScale, float edgeLimit, float subsampling, float *memoryTmp, SiftDeviceContext &ctx);
double ScaleDown(CudaImage *res, CudaImage *src, float variance, SiftDeviceContext &ctx);
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage *src, SiftData *siftData, int octave, SiftDeviceContext &ctx);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData *siftData, float subsampling, int octave, SiftDeviceContext &ctx);
double LowPass(CudaImage *res, CudaImage *src, float scale, SiftDeviceContext &ctx);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage *baseImage, CudaImage *results, int octave, SiftDeviceContext &ctx);
double FindPointsMulti(CudaImage *sources, SiftData *siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave, SiftDeviceContext &ctx);

#endif
