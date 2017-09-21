#include <stdlib.h>

#include <stdio.h>

#include <cuda_runtime.h>

#include <time.h>



#define __DEBUG

#define TSCALE 1.0

#define VSQR 0.1

#define THREADS 512



#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )

#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)



/**************************************

* void __cudaSafeCall(cudaError err, const char *file, const int line)

* void __cudaCheckError(const char *file, const int line)

*

* These routines were taken from the GPU Computing SDK

* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"

**************************************/

inline void __cudaSafeCall( cudaError err, const char *file, const int line )

{

#ifdef __DEBUG



#pragma warning( push )

#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);

  do

  {

    if ( cudaSuccess != err )

    {

      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",

              file, line, cudaGetErrorString( err ) );

      exit( -1 );

    }

  } while ( 0 );

#pragma warning( pop )

#endif  // __DEBUG

  return;

}



inline void __cudaCheckError( const char *file, const int line )

{

#ifdef __DEBUG

#pragma warning( push )

#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);

  do

  {

    cudaError_t err = cudaGetLastError();

    if ( cudaSuccess != err )

    {

      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",

               file, line, cudaGetErrorString( err ) );

      exit( -1 );

    }

    // More careful checking. However, this will affect performance.

    // Comment if not needed.

    /*err = cudaThreadSynchronize();

    if( cudaSuccess != err )

    {

      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",

               file, line, cudaGetErrorString( err ) );

      exit( -1 );

    }*/

  } while ( 0 );

#pragma warning( pop )

#endif // __DEBUG

  return;

}





__global__ void evolve13ptcuda(double *un, double *uc, double *uo, double *pebbles, int *n, double *h, double *dt, double *t,int *nthreads,int *nblocks){

     __shared__ double suc[THREADS];

	 

	 int blocksRemainder=blockIdx.x%(*nblocks);

	 int blocksDiv=blockIdx.x/(*nblocks);

	 int threadsRemainder=threadIdx.x%(*nthreads);

	 int threadsDiv=threadIdx.x/(*nthreads);



	 

	 const unsigned int idx =(blocksDiv*((*n)*(*nthreads)))+((*nthreads)*blocksRemainder)+(*n)*threadsDiv+threadsRemainder;

	 const unsigned int lidx=(*nthreads)*threadsDiv+threadsRemainder;

	 suc[lidx]=uc[idx];

	 __syncthreads();



	 int i=lidx/(*nthreads);

	 int j=lidx%(*nthreads);

	 if(i == 0 || i == (*nthreads)-1 || j == 0 || j == (*nthreads)-1 || i == 1 || i == (*nthreads)-2 || j == 1 || j == (*nthreads)-2){

                i=idx/(*n);

	            j=idx%(*n);

				

				if(i == 0 || i == (*n)-1 || j == 0 || j == (*n)-1 || i == 1 || i == (*n)-2 || j == 1 || j == (*n)-2){

				un[idx] = 0.;

				}else{

				

				float f= -expf(-TSCALE * (*t)) * pebbles[idx];

                un[idx] = 2*uc[idx] - uo[idx] + VSQR * ((*dt) * (*dt)) * \

                          ( ( ( uc[idx-1] + uc[idx+1] + uc[idx+(*n)] + uc[idx-(*n)] ) + \

                            (0.25 * ( uc[idx-(*n)-1] + uc[idx-(*n)+1] + uc[idx+(*n)-1] + uc[idx+(*n)+1] ) ) + \

                            (0.125 * ( uc[idx-2] + uc[idx+2] + uc[idx - 2*(*n)] + uc[idx + 2*(*n)]) ) - \

                            (6 * uc[idx]) )/ ((*h) * (*h)) + f );

				}	

            } else {

               float f= -expf(-TSCALE * (*t)) * pebbles[idx];

                un[idx] = 2*suc[lidx] - uo[idx] + VSQR * ((*dt) * (*dt)) * \

                          ( ( ( suc[lidx-1] + suc[lidx+1] + suc[lidx+(*nthreads)] + suc[lidx-(*nthreads)] ) + \

                            (0.25 * ( suc[lidx-(*nthreads)-1] + suc[lidx-(*nthreads)+1] + suc[lidx+(*nthreads)-1] + suc[lidx+(*nthreads)+1] ) ) + \

                            (0.125 * ( suc[lidx-2] + suc[lidx+2] + suc[lidx - 2*(*nthreads)] + suc[lidx + 2*(*nthreads)]) ) - \

                            (6 * suc[lidx]) )/ ((*h) * (*h)) + f );

                          

            }

	 



       

}



/*int tpdt(double *t, double dt, double tf)

{

  if((*t) + dt > tf) return 0;

  (*t) = (*t) + dt;

  return 1;

}*/



void init_cpu_memory(double *u0,double *u1, double **uc,double **uo,int *nblocks,int nthreads,int n){

  

  *nblocks=(n-2)/nthreads;



  *uc = (double*)malloc(sizeof(double) * n * n);

  *uo = (double*)malloc(sizeof(double) * n * n);

  

  memcpy(*uo, u0, sizeof(double) * n * n);

  memcpy(*uc, u1, sizeof(double) * n * n);

  

  

}



void init_gpu_memory( double *pebbles, int n, double h, int nthreads,int nblocks, double dt, double **t_d, double **dt_d, double **un_d,  double **uc_d,  double **uo_d,  double **h_d,   double **pebbles_d,  int **nblocks_d,int **nthreads_d,int **n_d){

  

    cudaMalloc( (void **)un_d, sizeof(double) * n * n);

    cudaMalloc( (void **)uc_d, sizeof(double) * n * n);

    cudaMalloc( (void **)uo_d, sizeof(double) * n * n);

    cudaMalloc( (void **)t_d, sizeof(double) * 1);

    cudaMalloc( (void **)dt_d, sizeof(double) * 1);

    cudaMalloc( (void **)n_d, sizeof(int) * 1);

    cudaMalloc( (void **)h_d, sizeof(double) * 1);

    cudaMalloc( (void **)pebbles_d, sizeof(double) * n * n);

	cudaMalloc( (void **)nblocks_d, sizeof(int) * 1);

	cudaMalloc( (void **)nthreads_d, sizeof(int) * 1);



	

    

    

    cudaMemcpy(*n_d, &n, sizeof(int) * 1, cudaMemcpyHostToDevice);

    cudaMemcpy(*h_d, &h, sizeof(double) * 1, cudaMemcpyHostToDevice);

    cudaMemcpy(*pebbles_d, pebbles, sizeof(double) * n * n, cudaMemcpyHostToDevice);

	cudaMemcpy(*nthreads_d, &nthreads, sizeof(int) * 1, cudaMemcpyHostToDevice);

	cudaMemcpy(*nblocks_d, &nblocks, sizeof(int) * 1, cudaMemcpyHostToDevice);

	cudaMemcpy(*dt_d, &dt, sizeof(double) * 1, cudaMemcpyHostToDevice);



   

}



void run_gpu(double *u, double *uc, double *uo, double *un_d,double *uc_d,double *uo_d,double *pebbles_d, int *n_d, double *h_d,int *nthreads_d,int *nblocks_d,double *t_d,double *dt_d,int nthreads,int nblocks,int n,int t)

{

//	cudaEvent_t kstart, kstop;

//	float ktime;

    

	/* HW2: Define your local variables here */



    /* Set up device timers */  

//	CUDA_CALL(cudaSetDevice(0));

//	CUDA_CALL(cudaEventCreate(&kstart));

//	CUDA_CALL(cudaEventCreate(&kstop));



	/* HW2: Add CUDA kernel call preperation code here */

    

	cudaMemcpy(uc_d, uc, sizeof(double) * n * n, cudaMemcpyHostToDevice);

    cudaMemcpy(uo_d, uo, sizeof(double) * n * n, cudaMemcpyHostToDevice);

	cudaMemcpy(t_d, &t, sizeof(double) * 1, cudaMemcpyHostToDevice);

	

	



	/* Start GPU computation timer */

//	CUDA_CALL(cudaEventRecord(kstart, 0));



	/* HW2: Add main lake simulation loop here */

    

    evolve13ptcuda<<<nblocks*nblocks, nthreads*nthreads>>>(un_d, uc_d, uo_d, pebbles_d, n_d, h_d, dt_d, t_d,nthreads_d,nblocks_d);

    cudaMemcpy(u, un_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

	

    /* Stop GPU computation timer */

//	CUDA_CALL(cudaEventRecord(kstop, 0));

//	CUDA_CALL(cudaEventSynchronize(kstop));

//	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));

//	printf("GPU computation: %f msec\n", ktime);



	/* HW2: Add post CUDA kernel call processing and cleanup here */

	

	/* timer cleanup */

//	CUDA_CALL(cudaEventDestroy(kstart));

//	CUDA_CALL(cudaEventDestroy(kstop));

}
