#include <stdlib.h>

#include <stdio.h>

#include <string.h>

#include <math.h>

#include <sys/time.h>

#include <limits.h>

#include <time.h>

#include "mpi.h"



#define ROOT 0



#define _USE_MATH_DEFINES



#define XMIN 0.0

#define XMAX 1.0

#define YMIN 0.0

#define YMAX 1.0



#define MAX_PSZ 10

#define TSCALE 1.0

#define VSQR 0.1



void init(double *u, double *pebbles, int n);

void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);

void evolve13pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);

int tpdt(double *t, double dt, double end_time);

void print_heatmap(const char *filename, double *u, int n, double h);

void init_pebbles(double *p, int pn, int n);



int get_my_quadrant(int n, int rank, double *input, double *output);

int get_my_row(int n, int rank, double *input, double *result);

int get_my_column(int n, int rank, double *input, double *result);

int get_my_corner(int n, int rank, double *input, double *result);

//int merge_result(int src_rank, double *input, double *output);

int get_range(int rank, int n, int *x_min, int *x_max, int *y_min, int *y_max);

int set_my_column(int n, int rank, double *col_array, double *result);

int set_my_row(int n, int rank, double *row_array, double *result);

int set_my_corner(int n, int rank, double *corner_array, double *result);





void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);



extern void run_gpu(double *u, double *uc, double *uo, double *un_d,double *uc_d,double *uo_d,double *pebbles_d, int *n_d, double *h_d,int *nthreads_d,int *nblocks_d,double *t_d,double *dt_d,int nthreads,int nblocks,int n,int t);



extern void init_gpu_memory( double *pebbles, int n, double h, int nthreads,int nblocks, double dt, double **t_d, double **dt_d, double **un_d,  double **uc_d,  double **uo_d,  double **h_d,   double **pebbles_d,  int **nblocks_d,int **nthreads_d,int **n_d);



extern void init_cpu_memory(double *u0,double *u1, double **uc,double **uo,int *nblocks,int nthreads,int n);





int get_range(int rank, int n, int *x_min, int *x_max, int *y_min, int *y_max)

{

  if(rank == 0)

  {

    *x_min = 0;

    *x_max = n/2+1;

    *y_min = 0;

    *y_max = n/2+1;

  }

  else if(rank == 1)

  {

    *x_min = 0;

    *x_max = n/2+1;

    *y_min = n/2-2;

    *y_max = n-1;

  }

  else if(rank == 2)

  {

    *x_min = n/2-2;

    *x_max = n-1;

    *y_min = 0;

    *y_max = n/2+1;

  }

  else

  {

    *x_min = n/2-2;

    *x_max = n-1;

    *y_min = n/2-2;

    *y_max = n-1;

  }

  return 0;

}



int get_my_quadrant(int n, int rank, double *input, double *output)

{

  int start_row = 0;

  int end_row = 0;

  int start_index= 0;

  int row_size = (n/2);

  int x_min, x_max, y_min, y_max;

  int r = 0;



  double *optr = output;



  get_range(rank, n, &x_min, &x_max, &y_min, &y_max);



  start_row = x_min;

  end_row = x_max;



  for(r=start_row; r <= end_row; r++)

  {

    if(rank == 0 || rank == 2)

      start_index = r*n;

    else

      start_index = r*n + row_size-2;



    memcpy(optr, input+start_index, (row_size+2)*sizeof(double));

    optr += row_size;

    optr += 2;

  }

  return 0;

}



int get_my_corner(int n, int rank, double *input, double *result)

{

  int k=0, idx;

  int m = n/2-1;



  if(rank == 0)

  {

    idx = (n/2 + 2)*(m - 1) + m - 1;

    result[k++] = input[idx++];

    result[k++] = input[idx];

    idx = (n/2 + 2)*(m) + m - 1;

    result[k++] = input[idx++];

    result[k++] = input[idx];

  }

  else if(rank == 1)

  {

    idx = (n/2 + 2)*(m - 1) + 2;

    result[k++] = input[idx++];

    result[k++] = input[idx];

    idx = (n/2 + 2)*(m) + 2;

    result[k++] = input[idx++];

    result[k++] = input[idx];

  }

  else if(rank == 2)

  {

    idx = 2*(n/2 + 2) + (m-1);

    result[k++] = input[idx++];

    result[k++] = input[idx];

    idx = 3*(n/2 + 2) + (m-1);

    result[k++] = input[idx++];

    result[k++] = input[idx];

  }

  else if(rank == 3)

  {

    idx = 2*(n/2 + 2) + 2;

    result[k++] = input[idx++];

    result[k++] = input[idx];

    idx = 3*(n/2 + 2) + 2;

    result[k++] = input[idx++];

    result[k++] = input[idx];

  }

  return 0;

}



int get_my_column(int n, int rank, double *input, double *result)

{

  int i, k=0, idx;

  int m = n/2-1;

  

  if(rank == 0 || rank == 2)

  {

    for(i=0; i <= n/2-1; i++)

    {

      idx = i*(n/2 + 2) + (m - 1);

      result[k++] = input[idx];

    }

    for(i=0; i <= n/2-1; i++)

    {

      idx = i*(n/2 + 2) + m;

      result[k++] = input[idx];

    }

  }

  else if(rank == 1 || rank == 3)

  {

    for(i=0; i <= n/2-1; i++)

    {

      idx = i*(n/2 + 2) + 2;

      result[k++] = input[idx];

    }

    for(i=0; i <= n/2-1; i++)

    {

      idx = i*(n/2 + 2) + 3;

      result[k++] = input[idx];

    }

  }

  return 0;

}



int get_my_row(int n, int rank, double *input, double *result)

{

  int j, k=0, idx;

  int m = n/2 - 1;

  

  if(rank == 0 || rank == 1)

  {

    for(j = 0; j <= n/2-1; j++)

    {

      idx = (m - 1)*(n/2 + 2) + j;

      result[k++] = input[idx];

    }

    for(j = 0; j <= n/2-1; j++)

    {

       idx = (m)*(n/2 + 2) + j;

       result[k++] = input[idx];

    }

  }

  else if(rank == 2 || rank == 3)

  {

    for(j = 0; j <= n/2-1; j++)

    {

       idx = (n/2 + 2)*2 + j;

       result[k++] = input[idx];

    }

    for(j = 0; j <= n/2-1; j++)

    {

       idx = (n/2 + 2)*3 + j;

       result[k++] = input[idx];

    }

  }

  return 0;

}



int set_my_column(int n, int rank, double *col_array, double *result)

{

  int i, k=0, idx;

  int m = n/2+1;



  if(rank == 0 || rank == 2)

  {

    for(i=0; i < n/2; i++)

    {

      idx = i*(n/2+2) + (m-1);

      result[idx] = col_array[k++];

    }

    for(i=0; i < n/2; i++)

    {

      idx = i*(n/2+2) + m;

      result[idx] = col_array[k++];

    }

  }

  else if(rank == 1 || rank == 3)

  {

    for(i=0; i < n/2; i++)

    {

      idx = i*(n/2+2) + 0;

      result[idx] = col_array[k++];

    }

    for(i=0; i < n/2; i++)

    {

      idx = i*(n/2+2) + 1;

      result[idx] = col_array[k++];

    }

  }

  return 0;

}



int set_my_row(int n, int rank, double *row_array, double *result)

{

  int j, k=0, idx;

  int m = n/2 + 1;



  if(rank == 0 || rank == 1)

  {

    for(j = 0; j <= n/2-1; j++)

    {

      idx = (m - 1)*(n/2 + 2) + j;

      result[idx] = row_array[k++];

    }

    for(j = 0; j <= n/2-1; j++)

    {

       idx = (m)*(n/2 + 2) + j;

       result[idx] = row_array[k++];

    }

  }

  else if(rank == 2 || rank == 3)

  {

    for(j = 0; j <= n/2-1; j++)

    {

       idx = (n/2 + 2)*0 + j;

       result[idx] = row_array[k++];

    }

    for(j = 0; j <= n/2-1; j++)

    {

       idx = (n/2 + 2)*1 + j;

       result[idx] = row_array[k++];

    }

  }

  return 0;

}



int copy_to_orginal(int n, int rank, double *block_array, double *orginal_array)

{

  int i,j, idx,blockdx,offset;



  if(rank==1 || rank==3){

	  offset=n/2;

  }else{

	  offset=0;

  }

  if(rank == 0 || rank == 1)

  {

	  for(i=0;i<=n/2-1;i++){

		for(j = 0; j <= n/2-1; j++)

		{

			blockdx = i*(n/2 + 2) + j;

			idx=i*(n)+j;

			orginal_array[idx+offset] = block_array[blockdx];

		}

	  }

  }

  else if(rank == 2 || rank == 3)

  {

    for(i=n/2;i<=n-1;i++){

		for(j = 0; j <= n/2-1; j++)

		{

			blockdx = i*(n/2 + 2) + j;

			idx=i*(n)+j;

			orginal_array[idx+offset] = block_array[blockdx];

		}

	  }

  }

  return 0;

}



int set_my_corner(int n, int rank, double *corner_array, double *result)

{

  int k=0, idx;

  int m = n/2+1;



  if(rank == 0)

  {

    idx = (n/2 + 2)*(m - 1) + m - 1;

    result[idx++] = corner_array[k++];

    result[idx] = corner_array[k++];

    idx = (n/2 + 2)*(m) + m - 1;

    result[idx++] = corner_array[k++];

    result[idx] = corner_array[k++];

  }

  else if(rank == 1)

  {

    idx = (n/2 + 2)*(m - 1);

    result[idx++] = corner_array[k++];

    result[idx] = corner_array[k++];

    idx = (n/2 + 2)*(m);

    result[idx++] = corner_array[k++];

    result[idx] = corner_array[k++];

  }

  else if(rank == 2)

  {

    idx = 0*(n/2 + 2) + (m-1);

    result[idx++] = corner_array[k++];

    result[idx] = corner_array[k++];

    idx = 1*(n/2 + 2) + (m-1);

    result[idx++] = corner_array[k++];

    result[idx] = corner_array[k++];

  }

  else if(rank == 3)

  {

    idx = 0;

    result[idx++] = corner_array[k++];

    result[idx] = corner_array[k++];

    idx = 1*(n/2+2);

    result[idx++] = corner_array[k++];

    result[idx] = corner_array[k++];

  }

  return 0;

}



/*

int merge_result(int src_rank, double *input, double *output){

    int start_row = 0, 

        end_row = 0, 

        start_index = 0,

        bstart_index = 0,

        row_size = n/2 ;

        brow_size = (n/2) + 2,

        x_min,

        x_max,

        y_min,

        y_max,

        r;



    double *op = output;



    get_range(src_rank, n, &x_min, &x_max, &y_min, &y_max);

    start_row = x_min+2;

    end_row = x_max-2;

    bstart_row = 2;

    bend_row = (n/2) - 2;



    for(r = start_row; r <= end_row; r++){

        

        if(src_rank == 0 || src_rank == 2){

            start_index = r*n ;

        } else {

            start_index = r*n + row_size;

        }

    }



    return 0;

}

*/



int main(int argc, char *argv[])

{



  if(argc != 5)

  {

    printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);

    return 0;

  }



  int     npoints   = atoi(argv[1]);

  int     npebs     = atoi(argv[2]);

  double  end_time  = (double)atof(argv[3]);

  int     nthreads  = atoi(argv[4]);

  int 	  narea	    = npoints * npoints;



  double *u_i0, *u_i1;

  double *u_cpu, *pebs;

  double h;



  double elapsed_cpu,

         elapsed_gpu;



  struct timeval cpu_start,

                 cpu_end,

                 gpu_start,

                 gpu_end;



  int my_rank,

      num_proc,

      src,

      dest,

      tag, 

      msg_size,

      ret_val,

      iterator,

      transport_size,

      block_area;



  //int x_min,

    //  x_max,

      //y_min,

      //y_max;



  // MPI Arrays for boundary conditions

  double *get_column,

         *get_row,

         *get_corner,

         *send_column,

         *send_row,

         *send_corner,

         *u_block_i0,

         *u_block_i1,

         *pebs_block,

         *u_block_gpu,

		 *result,

         t,

         dt;



  // MESSAGE BUFFER

  //char *msg;



  MPI_Status status;



  // Start up MPI

  MPI_Init(&argc, &argv);

  printf("MPI Intialization done\n");



  // Number of processors

  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  printf("MPI Comm Size - %d\n", num_proc);



  // Get current process' rank

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  printf("My rank - %d", my_rank);



  // Barrier before start

  MPI_Barrier(MPI_COMM_WORLD);



  printf("Number of processors: %d\n", num_proc);

  

  

  if(my_rank==0){

  u_i0 = (double*)malloc(sizeof(double) * narea);

  u_i1 = (double*)malloc(sizeof(double) * narea);

  pebs = (double*)malloc(sizeof(double) * narea);

  u_cpu = (double*)malloc(sizeof(double) * narea);

  }

   result = (double*)malloc(sizeof(double) * narea);

   memset(result, 0., sizeof(double)*narea);



  



  // Do this for all ranks

  transport_size = npoints/2; // Given n is always divisible by 2

  get_column = (double*)malloc(sizeof(double) * 2 * transport_size);

  get_row = (double*)malloc(sizeof(double) * 2 * transport_size);

  get_corner = (double*)malloc(sizeof(double) * 2 * 2);

  send_column = (double*)malloc(sizeof(double) * 2 * transport_size);

  send_row = (double*)malloc(sizeof(double) * 2 * transport_size);

  send_corner = (double*)malloc(sizeof(double) * 2 * 2);



  // Initializing blocks of sub-matrices

  block_area = (npoints/2+2) * (npoints/2+2);

  u_block_i0 = (double*) malloc(sizeof(double) * block_area);

  u_block_i1 = (double*) malloc(sizeof(double) * block_area);

  pebs_block = (double*) malloc(sizeof(double) * block_area);

  u_block_gpu = (double*) malloc(sizeof(double) * block_area);





  printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints, npoints, end_time, nthreads);



  h = (XMAX - XMIN)/npoints;



  if(my_rank==0){

  init_pebbles(pebs, npebs, npoints);

  init(u_i0, pebs, npoints);

  init(u_i1, pebs, npoints);

  }

  



 





  int tagOffsetu_0=100;

  int tagOffsetu_1=200;

   if(my_rank == ROOT){

    for(iterator = 1; iterator <  num_proc; ++iterator){

         get_my_quadrant(npoints, iterator, pebs, pebs_block);

         ret_val = MPI_Send(pebs_block, block_area, MPI_DOUBLE, iterator, iterator, MPI_COMM_WORLD);

		 get_my_quadrant(npoints, iterator, u_i0, u_block_i0);

         ret_val = MPI_Send(u_block_i0, block_area, MPI_DOUBLE, iterator, iterator+tagOffsetu_0, MPI_COMM_WORLD);

		 get_my_quadrant(npoints, iterator, u_i1, u_block_i1);

         ret_val = MPI_Send(u_block_i1, block_area, MPI_DOUBLE, iterator, iterator+tagOffsetu_1, MPI_COMM_WORLD);

    }

	get_my_quadrant(npoints, my_rank, u_i0, u_block_i0);

	get_my_quadrant(npoints, my_rank, u_i1, u_block_i1);

	get_my_quadrant(npoints, my_rank, pebs, pebs_block);

	

 

  } else {

    msg_size = block_area;

    src = ROOT;

    tag = my_rank;

  //  ret_val = MPI_Recv(&u_block_i0, msg_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

  //  ret_val = MPI_Recv(&u_block_i1, msg_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

    ret_val = MPI_Recv(pebs_block, msg_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

	ret_val = MPI_Recv(u_block_i0, msg_size, MPI_DOUBLE, src, tag+tagOffsetu_0, MPI_COMM_WORLD, &status);

    ret_val = MPI_Recv(u_block_i1, msg_size, MPI_DOUBLE, src, tag+tagOffsetu_1, MPI_COMM_WORLD, &status);



  //  ret_val = MPI_Recv(&u_block_gpu, msg_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

  }

  

    printf("I got my quandrant %d\n"+my_rank);



  if(my_rank == ROOT){

    print_heatmap("lake_i.dat", u_i0, npoints, h);

  } else {

    // TODO PRINT RESPECTIVE HEATMAP HERE

  }



  if(my_rank == ROOT){

      gettimeofday(&cpu_start, NULL);

      run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);

      gettimeofday(&cpu_end, NULL);



      elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(

                      cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));

      printf("CPU took %f seconds (Performed only on RANK 0)\n", elapsed_cpu);

  }

  

  t = 0.;

  dt = h / 2.;

  

  double *uc,*uo,*temp;

  int nblocks;

  double *t_d,*dt_d,*un_d,*uc_d, *uo_d, *h_d,*pebbles_d;

  int *nblocks_d,*nthreads_d,*n_d;

  

tag=1;   

  init_cpu_memory(u_block_i0,u_block_i1, &uc,&uo,&nblocks,nthreads,(npoints/2)+2);

  init_gpu_memory(pebs_block, (npoints/2)+2, h, nthreads,nblocks, dt, &t_d, &dt_d, &un_d, &uc_d, &uo_d,&h_d,&pebbles_d,&nblocks_d,&nthreads_d,&n_d);



  if(my_rank == ROOT){

    gettimeofday(&gpu_start, NULL);

  }



  while(1){

      // Need to modify this after abhinand's change

	  run_gpu(u_block_gpu, uc, uo, un_d,uc_d,uo_d,pebbles_d, n_d, h_d,nthreads_d,nblocks_d,t_d,dt_d,nthreads,nblocks,(npoints/2)+2,t);

          

          printf("Run GPU completed\n");

      if(my_rank == 0){

            src = 1, dest = 1; // Get & Send Row

            ret_val = get_my_column(npoints, my_rank, u_block_gpu, send_column);

            printf("Rank %d: Got column to send\n", my_rank);

            ret_val = MPI_Recv(get_column, transport_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

            printf("Rank %d: Received column from %d\n", my_rank, src);

            ret_val = MPI_Send(send_column, transport_size, MPI_DOUBLE, dest, tag,MPI_COMM_WORLD);

            printf("Rank %d: Sent column to %d\n", my_rank, dest);

            ret_val = set_my_column(npoints, my_rank, get_column, u_block_gpu);

            printf("Rank %d: Set column\n", my_rank);            



            src = 2, dest = 2; // Get & Send column

            ret_val = get_my_row(npoints, my_rank, u_block_gpu, send_row);

            printf("Rank %d: Got row to send\n", my_rank);

            ret_val = MPI_Recv(get_row, transport_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

            printf("Rank %d: Received row from %d\n", my_rank, src);

            ret_val = MPI_Send(send_row, transport_size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

            printf("Rank %d: Sent row to %d\n", my_rank, dest);

            ret_val = set_my_row(npoints,my_rank, get_row, u_block_gpu);

            printf("Rank %d: Set row\n", my_rank);



            src = 3, dest = 3; // Get & Send corner

            ret_val = get_my_corner(npoints, my_rank, u_block_gpu, send_corner);

            printf("Rank %d: Got corner to send\n", my_rank);



            ret_val = MPI_Recv(get_corner, 2 * 2, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

            printf("Rank %d: Received corner from %d\n", my_rank, src);

            ret_val = MPI_Send(send_corner, 2 * 2, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

            printf("Rank %d: Sent corner to %d\n", my_rank, dest);

            ret_val = set_my_corner(npoints, my_rank, get_corner, u_block_gpu);

            printf("Rank %d: Set corner\n", my_rank);



      } else if(my_rank == 1){

            src = 0, dest = 0;

            ret_val = get_my_column(npoints, my_rank, u_block_gpu, send_column);

            printf("Rank %d: Got column to send\n", my_rank);

            ret_val = MPI_Send(send_column, transport_size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

            printf("Rank %d: Sent column to %d\n", my_rank, dest);

            ret_val = MPI_Recv(get_column, transport_size, MPI_DOUBLE, src,tag, MPI_COMM_WORLD, &status);

            printf("Rank %d: Received column from %d\n", my_rank, src);

            ret_val = set_my_column(npoints, my_rank, get_column, u_block_gpu);

            printf("Rank %d: Set column\n", my_rank);



            src = 3, dest = 3;

            ret_val = get_my_row(npoints, my_rank, u_block_gpu, send_row);

            printf("Rank %d: Got row to send\n", my_rank);

            ret_val = MPI_Send(send_row, transport_size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

            printf("Rank %d: Sent row to %d\n", my_rank, dest);

            ret_val = MPI_Recv(get_row, transport_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

            printf("Rank %d: Received row from %d\n", my_rank, src);

            ret_val = set_my_row(npoints, my_rank, get_row,u_block_gpu);

            printf("Rank %d: Set row\n", my_rank);



            src = 2, dest = 2;

            ret_val = get_my_corner(npoints, my_rank, u_block_gpu, send_corner);

            printf("Rank %d: Got corner to send\n", my_rank);

            ret_val = MPI_Send(send_corner, 2 * 2, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

            printf("Rank %d: Sent corner to %d\n", my_rank, dest);

            ret_val = MPI_Recv(get_corner, 2 * 2, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

            printf("Rank %d: Received corner from %d\n", my_rank, src);

            ret_val = set_my_corner(npoints, my_rank, get_corner,u_block_gpu);

            printf("Rank %d: Set corner\n", my_rank);

      

      } else if(my_rank == 2){

            src = 0, dest = 0;

            ret_val = get_my_row(npoints, my_rank, u_block_gpu, send_row);

            printf("Rank %d: Got row to send\n", my_rank);

            ret_val = MPI_Send(send_row, transport_size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

            printf("Rank %d: Sent row to %d\n", my_rank, dest);

            ret_val = MPI_Recv(get_row, transport_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

            printf("Rank %d: Received row from %d\n", my_rank, src);

            ret_val = set_my_row(npoints, my_rank, get_row, u_block_gpu);

            printf("Rank %d: Set row\n", my_rank);



            src = 3, dest = 3;

            ret_val = get_my_column(npoints, my_rank, u_block_gpu, send_column);

            printf("Rank %d: Got column to send\n", my_rank);

            ret_val = MPI_Recv(get_column, transport_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

            printf("Rank %d: Received column from %d\n", my_rank, src);

            ret_val = MPI_Send(send_column, transport_size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

            printf("Rank %d: Sent column to %d\n", my_rank, dest);

            ret_val = set_my_column(npoints, my_rank, get_column,u_block_gpu);

            printf("Rank %d: Set row\n", my_rank);



            src = 1, dest = 1;

            ret_val = get_my_corner(npoints, my_rank, u_block_gpu, send_corner);

            printf("Rank %d: Got corner to send\n", my_rank);

            ret_val = MPI_Recv(get_corner, 2 * 2, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

            printf("Rank %d: Received corner from %d\n", my_rank, src);

            ret_val = MPI_Send(send_corner, 2 * 2, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

            printf("Rank %d: Sent corner to %d\n", my_rank, dest);

            ret_val = set_my_corner(npoints, my_rank,get_corner, u_block_gpu);

            printf("Rank %d: Set corner\n", my_rank);

          

      } else if(my_rank == 3){

              src = 1, dest = 1;

              ret_val = get_my_row(npoints, my_rank, u_block_gpu, send_row);

              printf("Rank %d: Got row to send\n", my_rank);

              ret_val = MPI_Recv(get_row, transport_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

              printf("Rank %d: Received row from %d\n", my_rank, src);

              ret_val = MPI_Send(send_row, transport_size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

              printf("Rank %d: Sent row to %d\n", my_rank, dest);

              ret_val = set_my_row(npoints, my_rank, get_row,u_block_gpu);

              printf("Rank %d: Set row\n", my_rank);



              src = 2, dest = 2;

              ret_val = get_my_column(npoints, my_rank, u_block_gpu, send_column);

              printf("Rank %d: Got column to send\n", my_rank);

              ret_val = MPI_Send(send_column, transport_size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

              printf("Rank %d: Sent column to %d\n", my_rank, dest);

              ret_val = MPI_Recv(get_column, transport_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

              printf("Rank %d: Received column from %d\n", my_rank, src);

              ret_val = set_my_column(npoints, my_rank,get_column, u_block_gpu);

              printf("Rank %d: Set column\n", my_rank);



              src = 0, dest = 0;

              ret_val = get_my_corner(npoints, my_rank, u_block_gpu, send_corner);

              printf("Rank %d: Got corner to send\n", my_rank);

              ret_val = MPI_Send(send_corner, 2 * 2, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

              printf("Rank %d: Sent corner to %d\n", my_rank, dest);

              ret_val = MPI_Recv(get_corner, 2 * 2, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

              printf("Rank %d: Received corner from %d\n", my_rank, src);

              

              ret_val = set_my_corner(npoints, my_rank,get_corner, u_block_gpu);

              printf("Rank %d: Set corner\n", my_rank);

      }

	  temp=uo;

	  uo=uc;

	  uc=u_block_gpu;

	  u_block_gpu=temp;

      

      if(!tpdt(&t, dt, end_time)) break;

      printf("Time %f\n", t);

  }



  if(my_rank == ROOT){

     gettimeofday(&gpu_end, NULL);

     elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(

                  gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));

     printf("GPU took %f seconds in rank%d\n", elapsed_gpu, my_rank);

  }



  //print heatmap

  char s[32];

  sprintf(s, "lake_f_%d.dat", my_rank);

  copy_to_orginal(npoints, my_rank, uc, result);

  print_heatmap(s, result, npoints, h);

  // AFTER COMPLETION i.e. END TIME REACHED

  // TRANSFER BLOCKS FROM NON-ZERO RANK to ROOT

  //if(my_rank == ROOT){

   // print_heatmap("lake_f.dat", u_cpu, npoints, h);

    // Copy values from u_block_gpu to u_gpu for RANK 0

    // Only then proceed!!



    //for(iterator = 1; iterator < num_proc; i++){

      //  ret_val = MPI_Recv(&u_block_gpu, msg_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

        // Based on where you receive from add it 

        // to final output: "u_gpu"

        

        //ret_val = merge_result(iterator, u_block_gpu, u_gpu);

    //}

  //} else {

    // TODO PRINT RESPECTIVE HEATMAP HERE

    // SEND Calculated values to ROOT/ RANK 0

    //ret_val = MPI_Send(&u_block_gpu, msg_size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD);

  //}



  // MEMORY CLEANUP

  free(get_column);

  free(get_row);

  free(send_column);

  free(send_row);

  free(get_corner);

  free(send_corner);

  if(my_rank==0){

  free(u_i0);

  free(u_i1);

  free(pebs);

  free(u_cpu);

  }

    free(u_block_i0);

     free(u_block_i1);

     free(pebs_block);

     free(u_block_gpu);

  //free(u_gpu);



  // Shut down MPI

  MPI_Finalize();



  return 0;

}



void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)

{

  double *un, *uc, *uo;

  double t, dt;



  un = (double*)malloc(sizeof(double) * n * n);

  uc = (double*)malloc(sizeof(double) * n * n);

  uo = (double*)malloc(sizeof(double) * n * n);



  memcpy(uo, u0, sizeof(double) * n * n);

  memcpy(uc, u1, sizeof(double) * n * n);



  t = 0.;

  dt = h / 2.;



  while(1)

  {

    evolve13pt(un, uc, uo, pebbles, n, h, dt, t);



    memcpy(uo, uc, sizeof(double) * n * n);

    memcpy(uc, un, sizeof(double) * n * n);



    if(!tpdt(&t,dt,end_time)) break;

  }

  

  memcpy(u, un, sizeof(double) * n * n);

}



void init_pebbles(double *p, int pn, int n)

{

  int i, j, k, idx;

  int sz;



  srand( time(NULL) );

  memset(p, 0, sizeof(double) * n * n);



  for( k = 0; k < pn ; k++ )

  {

    i = rand() % (n - 4) + 2;

    j = rand() % (n - 4) + 2;

    sz = rand() % MAX_PSZ;

    idx = j + i * n;

    p[idx] = (double) sz;

  }

}



double f(double p, double t)

{

  return -expf(-TSCALE * t) * p;

}



int tpdt(double *t, double dt, double tf)

{

  if((*t) + dt > tf) return 0;

  (*t) = (*t) + dt;

  return 1;

}



void init(double *u, double *pebbles, int n)

{

  int i, j, idx;



  for(i = 0; i < n ; i++)

  {

    for(j = 0; j < n ; j++)

    {

      idx = j + i * n;

      u[idx] = f(pebbles[idx], 0.0);

    }

  }

}



void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)

{

  int i, j, idx;



  for( i = 0; i < n; i++)

  {

    for( j = 0; j < n; j++)

    {

      idx = j + i * n;



      if( i == 0 || i == n - 1 || j == 0 || j == n - 1)

      {

        un[idx] = 0.;

      }

      else

      {

        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + 

                    uc[idx + n] + uc[idx - n] - 4 * uc[idx])/(h * h) + f(pebbles[idx],t));

      }

    }

  }

}



void evolve13pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t){

    int i, j, idx;



    for(i = 0; i < n; i++){

        for(j = 0; j < n; j++){

            idx = j + i * n;

            

            if(i == 0 || i == n-1 || j == 0 || j == n-1 || i == 1 || i == n-2 || j == 1 || j == n-2){

                un[idx] = 0.;

            } else {

                un[idx] = 2*uc[idx] - uo[idx] + VSQR * (dt * dt) * \

                          ( ( ( uc[idx-1] + uc[idx+1] + uc[idx+n] + uc[idx-n] ) + \

                            (0.25 * ( uc[idx-n-1] + uc[idx-n+1] + uc[idx+n-1] + uc[idx+n+1] ) ) + \

                            (0.125 * ( uc[idx-2] + uc[idx+2] + uc[idx - 2*n] + uc[idx + 2*n]) ) - \

                            (6 * uc[idx]) )/ (h * h) + f(pebbles[idx], t) );

                          

            }

        }

    }

}



void print_heatmap(const char *filename, double *u, int n, double h)

{

  int i, j, idx;



  FILE *fp = fopen(filename, "w");  



  for( i = 0; i < n; i++ )

  {

    for( j = 0; j < n; j++ )

    {

      idx = j + i * n;

      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);

    }

  }

  

  fclose(fp);

} 
