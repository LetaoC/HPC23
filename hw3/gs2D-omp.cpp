#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_num_threads() { return 1;}
#endif

#include <stdio.h>
#include <math.h>
#include "utils.h"



using namespace std;

double cal_norm(double* u, double* f, long N){

  double* tmp = (double*) malloc(N*N*sizeof(double));
  double h = 1/(double(N) + 1);

  for (long i = 0; i < N*N; i++){
    if (i < N){
      if (i == 0) tmp[i] = (4*u[i] - u[i+1] - u[i+N])/(h*h);
      else if (i == N-1) tmp[i] = (4*u[i] - u[i-1] - u[i+N])/(h*h);
      else tmp[i] = (4*u[i] - u[i-1] - u[i+1] - u[i+N])/(h*h);
    }
    else if (i >= N*(N-1)){
      if(i==N*(N-1)) tmp[i] = (4*u[i] - u[i+1] - u[i-N])/(h*h);
      else if (i == N*N-1) tmp[i] = (4*u[i] - u[i-1] - u[i-N])/(h*h);
      else tmp[i] = (4*u[i] - u[i-1] - u[i+1] - u[i-N])/(h*h);
    }
    else{
      if(i%N==0) tmp[i] = (4*u[i]- u[i-N] - u[i+1] - u[i+N])/(h*h);
      else if (i%N==N-1) tmp[i] = (4*u[i] - u[i-N] - u[i-1] - u[i+N])/(h*h);
      else tmp[i] = (4*u[i] - u[i-N] - u[i-1] - u[i +1] - u[i+N])/(h*h);
    }
  }
  double sum = 0;
  for (long i = 0; i < N; i++){
    sum += (f[i] - tmp[i])*(f[i] - tmp[i]);
  }
  free(tmp);
  return sqrt(sum);
}

void GS2D(long N, int THREADS){
  //initialization
  double h = 1/(double(N)+1);

  double *u0 = (double*) malloc(N*N*sizeof(double));
  double *u = (double*) malloc(N*N*sizeof(double));
  double *f = (double*) malloc(N*N*sizeof(double));
  for (long i = 0; i < N*N; i++) {
    u[i] = 0;
    u0[i] = 0;
    f[i] = 1;
  }
  long double res = sqrt(N);
  long double update_res = res;
  long iter = 0;
  bool omp = false;
  while (iter < 5000 && res/update_res < 10e4){
#pragma omp parallel num_threads(THREADS)
    {
      omp = true;
            #pragma omp for
      for (long i = 0; i < N*N; i+=2){
	if (i < N){
	  if (i == 0) u[i] = (h*h*f[i] + u0[i+1] + u0[i+N])/4;
	  else if (i == N-1) u[i] = (h*h*f[i] + u0[i-1] + u0[i+N])/4;
	  else u[i] = (h*h*f[i] + u0[i-1] + u0[i+1] + u0[i+N])/4;
	}
	else if (i >= N*(N-1)){
	  if(i==N*(N-1)) u[i] = (h*h*f[i] + u0[i+1] + u0[i-N])/4;
	  else if (i == N*N-1) u[i] = (h*h*f[i] + u0[i-1] + u0[i-N])/4;
	  else u[i] = (h*h*f[i] + u0[i-1] + u0[i+1] + u0[i-N])/4;
	}
	else{
	  if(i%N==0) u[i] = (h*h*f[i]+u0[i-N] + u0[i+1] +u0[i+N])/4;
	  else if (i%N==N-1) u[i] = (h*h*f[i] + u0[i-N] + u0[i-1] + u0[i+N])/4;
	  else u[i] = (h*h*f[i] + u0[i-N] + u0[i-1] + u0[i +1] + u0[i+N])/4;
	}
      }

            #pragma omp for
      for (long i = 1; i < N*N; i +=2){
	if (i < N){
	  if (i == 0) u[i] = (h*h*f[i] + u[i+1] + u[i+N])/4;
	  else if (i == N-1) u[i] = (h*h*f[i] + u[i-1] + u[i+N])/4;
	  else u[i] = (h*h*f[i] + u[i-1] + u[i+1] + u[i+N])/4;
	}
	else if (i >= N*(N-1)){
	  if(i==N*(N-1)) u[i] = (h*h*f[i] + u[i+1] + u[i-N])/4;
	  else if (i == N*N-1) u[i] = (h*h*f[i] + u[i-1] + u[i-N])/4;
	  else u[i] = (h*h*f[i] + u[i-1] + u[i+1] + u[i-N])/4;
	}
	else{
	  if(i%N==0) u[i] = (h*h*f[i]+u[i-N] + u[i+1] +u[i+N])/4;
	  else if (i%N==N-1) u[i] = (h*h*f[i] + u[i-N] + u[i-1] + u[i+N])/4;
	  else u[i] = (h*h*f[i] + u[i-N] + u[i-1] + u[i +1] + u[i+N])/4;
	}

      }
    }
    if (not omp){
      for (long i = 0; i < N*N; i+=2){
	if (i < N){
	  if (i == 0) u[i] = (h*h*f[i] + u0[i+1] + u0[i+N])/4;
	  else if (i == N-1) u[i] = (h*h*f[i] + u0[i-1] + u0[i+N])/4;
	  else u[i] = (h*h*f[i] + u0[i-1] + u0[i+1] + u0[i+N])/4;
	}
	else if (i >= N*(N-1)){
	  if(i==N*(N-1)) u[i] = (h*h*f[i] + u0[i+1] + u0[i-N])/4;
	  else if (i == N*N-1) u[i] = (h*h*f[i] + u0[i-1] + u0[i-N])/4;
	  else u[i] = (h*h*f[i] + u0[i-1] + u0[i+1] + u0[i-N])/4;
	}
	else{
	  if(i%N==0) u[i] = (h*h*f[i]+u0[i-N] + u0[i+1] +u0[i+N])/4;
	  else if (i%N==N-1) u[i] = (h*h*f[i] + u0[i-N] + u0[i-1] + u0[i+N])/4;
	  else u[i] = (h*h*f[i] + u0[i-N] + u0[i-1] + u0[i +1] + u0[i+N])/4;
	}
      }
      for (long i = 1; i < N*N; i +=2){
	if (i < N){
	  if (i == 0) u[i] = (h*h*f[i] + u[i+1] + u[i+N])/4;
	  else if (i == N-1) u[i] = (h*h*f[i] + u[i-1] + u[i+N])/4;
	  else u[i] = (h*h*f[i] + u[i-1] + u[i+1] + u[i+N])/4;
	}
	else if (i >= N*(N-1)){
	  if(i==N*(N-1)) u[i] = (h*h*f[i] + u[i+1] + u[i-N])/4;
	  else if (i == N*N-1) u[i] = (h*h*f[i] + u[i-1] + u[i-N])/4;
	  else u[i] = (h*h*f[i] + u[i-1] + u[i+1] + u[i-N])/4;
	}
	else{
	  if(i%N==0) u[i] = (h*h*f[i]+u[i-N] + u[i+1] +u[i+N])/4;
	  else if (i%N==N-1) u[i] = (h*h*f[i] + u[i-N] + u[i-1] + u[i+N])/4;
	  else u[i] = (h*h*f[i] + u[i-N] + u[i-1] + u[i +1] + u[i+N])/4;
	}

      }

    }
    update_res = cal_norm(u, f, N);
    printf("The residual when N = %ld at iteration %ld is %Lf\n", N, iter, update_res);
    u0 = u;
    iter++;
  }
  free(u);
  free(f);


}

int main(){
  Timer t;
  long N1 = 7;
  long N2 = 15;
  long N3 = 35;
  long N4 = 49;
  int THREADS = 4;
  t.tic();
  GS2D(N3, THREADS);
  double time = t.toc();
  printf("execution time is %f s", time);
  return 0;
}
