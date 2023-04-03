#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    //printf("pre_sum %ld is %ld and A added is %ld and pre prefix is %ld\n",i,prefix_sum[i],A[i-1],prefix_sum[i-1]);
    //printf("pre_sum %ld is %ld\n",i, prefix_sum[i]);
  }
  // printf("scan_seq\n");
  //for (long i = 0; i< n; i++) printf("%ld\n",prefix_sum[i]);
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  int p = omp_get_num_threads();
  int t = omp_get_thread_num();
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
  if (n==0) return;
#pragma omp parallel
  {
    p = omp_get_num_threads();
  }
  printf("number of threads is %d\n",p);
  long* tmp = (long*) malloc ((p-1)*sizeof(long));
#pragma omp parallel
  {
    #pragma omp for
    for (long i = 1; i< n; i++){
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
      t = omp_get_thread_num();
      if(t<p-1) tmp[t] = prefix_sum[i];
    }
    //printf("tmp\n");
    //for (long i = 0; i < p-1; i++) printf("%ld\n", tmp[i]);
  }
  //printf("tmp\n");
  //for (long i = 0; i < p-1; i++) printf("%ld\n", tmp[i]);
  for (long i = 1; i < p-1; i++) tmp[i] = tmp[i]+ tmp[i-1];
  //printf("new tmp\n");
  //for (long i = 0; i < p-1; i++) printf("%ld\n", tmp[i]);
#pragma omp parallel
  {
  #pragma omp for
  for (long i = 1; i < n; i++){
    t = omp_get_thread_num();
    if (t>0) {
      //printf("thre %d pre_sum %ld is %ld add %ld is %ld\n",t,i,prefix_sum[i],tmp[t-1],prefix_sum[i]+tmp[t-1]);
      prefix_sum[i] += tmp[t-1];
    }
   }
  }
  //printf("scan_omp\n");                                                                                                                                                                         
  //for (long i = 0; i< n; i++) printf("%ld\n",prefix_sum[i]);
  free(tmp);
}

int main() {
  long N = 100000000;
  //long N = 10;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;
  
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);
  //printf("org A\n");
  //for (int i=0;i<N;i++) printf("%ld\n",A[i]);
  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
