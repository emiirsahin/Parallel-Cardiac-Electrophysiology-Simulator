/*
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * and reimplementation by Scott B. Baden, UCSD
 *
 * Modified and  restructured by Didem Unat, Koc University
 *
 */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
using namespace std;

// Utilities
//

// Timer
// Make successive calls and take a difference to get the elapsed time.
static const double kMicro = 1.0e-6;
double getTime()
{
  struct timeval TV;
  struct timezone TZ;

  const int RC = gettimeofday(&TV, &TZ);
  if (RC == -1)
  {
    cerr << "ERROR: Bad call to gettimeofday" << endl;
    return (-1);
  }

  return (((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec));

} // end getTime()

// Allocate a 2D array
double **alloc2D(int m, int n)
{
  double **E;
  int nx = n, ny = m;
  E = (double **)malloc(sizeof(double *) * ny + sizeof(double) * nx * ny);
  assert(E);
  int j;
  for (j = 0; j < ny; j++)
    E[j] = (double *)(E + ny) + j * nx;
  return (E);
}

// Reports statistics about the computation
// These values should not vary (except to within roundoff)
// when we use different numbers of  processes to solve the problem
double stats(double **E, int m, int n, double *_mx)
{
  double mx = -1;
  double l2norm = 0;
  int i, j;
  for (j = 1; j <= m; j++)
    for (i = 1; i <= n; i++)
    {
      l2norm += E[j][i] * E[j][i];
      if (E[j][i] > mx)
        mx = E[j][i];
    }
  *_mx = mx;
  l2norm /= (double)((m) * (n));
  l2norm = sqrt(l2norm);
  return l2norm;
}

// External functions
extern "C"
{
  void splot(double **E, double T, int niter, int m, int n);
}
void cmdLine(int argc, char *argv[], double &T, int &n, int &px, int &py, int &plot_freq, int &no_comm, int &num_threads);

void simulate(double *new_data, double *local_data, double *r_local_data,
              double *bottom_row, double *top_row, int rank, int *groups, int size,
              const double alpha, const int n, const int m, const double kk,
              const double dt, const double a, const double epsilon,
              const double M1, const double M2, const double b)
{
  /*
   * Copy data from boundary of the computational box
   * to the padding region, set up for differencing
   * on the boundary of the computational box
   * Using mirror boundaries
   */

  if (groups[rank] > 1)
  {
    if (rank == size - 1)
    {
#pragma omp parallel for
      for (int i = 0; i < n + 2; i++)
      {
        bottom_row[i] = local_data[(groups[rank] - 2) * (n + 2) + i];
      }
    }

    if (rank == 0)
    {
#pragma omp parallel for
      for (int i = 0; i < n + 2; i++)
      {
        top_row[i] = local_data[(n + 2) + i];
      }
    }
  }

#pragma omp parallel for
  for (int j = 0; j < groups[rank]; j++)
    local_data[0 + j * (n + 2)] = local_data[2 + j * (n + 2)];

#pragma omp parallel for
  for (int j = 0; j < groups[rank]; j++)
    local_data[n + 1 + j * (n + 2)] = local_data[n - 1 + j * (n + 2)];

  for (int i = 0; i < groups[rank]; i++)
  {
    #pragma omp parallel for
    for (int j = 1; j <= n; j++)
    {
      if (groups[rank] > 1)
      {
        if (i == 0)
        {
          new_data[j] = local_data[j] + alpha * (local_data[j + 1] + local_data[j - 1] - 4 * local_data[j] + local_data[j + n + 2] + top_row[j]);
        }
        if (i == groups[rank] - 1)
        {
          new_data[j + (n + 2) * i] = local_data[j + (n + 2) * i] + alpha * (local_data[j + (n + 2) * i + 1] + local_data[j + (n + 2) * i - 1] - 4 * local_data[j + (n + 2) * i] + bottom_row[j] + local_data[j + (n + 2) * (i - 1)]);
        }
        if (i > 0 && i < groups[rank] - 1)
        {
          new_data[j + (n + 2) * i] = local_data[j + (n + 2) * i] + alpha * (local_data[j + (n + 2) * i + 1] + local_data[j + (n + 2) * i - 1] - 4 * local_data[j + (n + 2) * i] + local_data[j + (n + 2) * (i + 1)] + local_data[j + (n + 2) * (i - 1)]);
        }
      }
      else
      {
        new_data[j] = local_data[j] + alpha * (local_data[j + 1] + local_data[j - 1] - 4 * local_data[j] + bottom_row[j] + top_row[j]);
      }
    }
  }
  /*
   * Solve the ODE, advancing excitation and recovery to the
   *     next timtestep
   */
  #pragma omp parallel for
  for (int i = 0; i < groups[rank]; i++)
  {
    #pragma omp parallel for
    for (int j = 1; j <= n; j++)
    {
      new_data[j + (n + 2) * i] = new_data[j + (n + 2) * i] - dt * (kk * new_data[j + (n + 2) * i] * (new_data[j + (n + 2) * i] - a) * (new_data[j + (n + 2) * i] - 1) + new_data[j + (n + 2) * i] * r_local_data[j + (n + 2) * i]);
    }
  }

  for (int i = 0; i < groups[rank]; i++)
  {
    for (int j = 1; j <= n; j++)
    {
      r_local_data[j + (n + 2) * i] = r_local_data[j + (n + 2) * i] + dt * (epsilon + M1 * r_local_data[j + (n + 2) * i] / (new_data[j + (n + 2) * i] + M2)) * (-r_local_data[j + (n + 2) * i] - kk * new_data[j + (n + 2) * i] * (new_data[j + (n + 2) * i] - b - 1));
    }
  }
}

// Main program
int main(int argc, char **argv)
{

  int rank, size, ierr;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  /*
   *  Solution arrays
   *   E is the "Excitation" variable, a voltage
   *   R is the "Recovery" variable
   *   E_prev is the Excitation variable for the previous timestep,
   *      and is used in time integration
   */
  double **E, **R, **E_prev;

  // Various constants - these definitions shouldn't change
  const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;

  double T = 1000.0;
  int m = 200, n = 200;
  int plot_freq = 0;
  int px = 1, py = 1;
  int no_comm = 0;
  int num_threads = 1;

  cmdLine(argc, argv, T, n, px, py, plot_freq, no_comm, num_threads);
  m = n;
  // Allocate contiguous memory for solution arrays
  // The computational box is defined on [1:m+1,1:n+1]
  // We pad the arrays in order to facilitate differencing on the
  // boundaries of the computation box

  if (rank == 0)
  {
    E = alloc2D(m + 2, n + 2);
    E_prev = alloc2D(m + 2, n + 2);
    R = alloc2D(m + 2, n + 2);

    // Initialization
    for (int j = 1; j <= m; j++)
      for (int i = 1; i <= n; i++)
        E_prev[j][i] = R[j][i] = 0;

    for (int j = 1; j <= m; j++)
      for (int i = n / 2 + 1; i <= n; i++)
        E_prev[j][i] = 1.0;

    for (int j = m / 2 + 1; j <= m; j++)
      for (int i = 1; i <= n; i++)
        R[j][i] = 1.0;
  }

  double dx = 1.0 / n;

  // For time integration, these values shouldn't change
  double rp = kk * (b + 1) * (b + 1) / 4;
  double dte = (dx * dx) / (d * 4 + ((dx * dx)) * (rp + kk));
  double dtr = 1 / (epsilon + ((M1 / M2) * rp));
  double dt = (dte < dtr) ? 0.95 * dte : 0.95 * dtr;
  double alpha = d * dt / (dx * dx);

  if (rank == 0)
  {
    cout << "Grid Size       : " << n << endl;
    cout << "Duration of Sim : " << T << endl;
    cout << "Time step dt    : " << dt << endl;
    cout << "Process geometry: " << px << " x " << py << endl;
    if (no_comm)
      cout << "Communication   : DISABLED" << endl;

    cout << endl;
  }

  // Start the timer
  double t0 = getTime();

  // Simulated time is different from the integer timestep number
  // Simulated time
  double t = 0.0;
  // Integer timestep number
  int niter = 0;

  int rows_per_process = m / size;
  int remainder = m % size;
  int groups[size];

  for (int i = 0; i < size; i++)
  {
    groups[i] = rows_per_process;
  }
  for (int i = 0; i < remainder; i++)
  {
    groups[i]++;
  }

  double *sub_data = (double *)malloc(groups[rank] * (n + 2) * sizeof(double));
  double *new_data = (double *)malloc(groups[rank] * (n + 2) * sizeof(double));
  double *top_row = (double *)malloc((n + 2) * sizeof(double));
  double *bottom_row = (double *)malloc((n + 2) * sizeof(double));
  double *r_sub_data = (double *)malloc(groups[rank] * (n + 2) * sizeof(double));

  MPI_Status status[2];
  MPI_Request request[2];
  MPI_Request requestGather[size * 2];
  MPI_Status statusGather[size * 2];

  for (int j = 0; j < groups[rank]; j++)
    for (int i = 0; i <= n + 1; i++)
      new_data[(n + 2) * j + i] = 0;

  while (t < T)
  {
    t += dt;
    niter++;

    if (rank == 0)
    {
      MPI_Isend(&E_prev[1][0], groups[rank] * (n + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request[0]);
      // MPI_Wait(&request[0], &status[0]);
      for (int i = 1; i < size; i++)
      {
        int passAmount = 0;
        for (int j = 0; j < i; j++)
        {
          passAmount += groups[j];
        }
        MPI_Isend(&E_prev[1 + passAmount][0], groups[i] * (n + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request[0]);
        // MPI_Wait(&request[0], &status[0]);
      }
    }

    MPI_Irecv(sub_data, groups[rank] * (n + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Wait(&request[0], &status[0]);

    if (rank == 0)
    {
      MPI_Isend(&R[1][0], groups[rank] * (n + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request[1]);
      // MPI_Wait(&request[1], &status[1]);
      for (int i = 1; i < size; i++)
      {
        int passAmount = 0;
        for (int j = 0; j < i; j++)
        {
          passAmount += groups[j];
        }
        MPI_Isend(&R[1 + passAmount][0], groups[i] * (n + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request[1]);
        // MPI_Wait(&request[1], &status[1]);
      }
    }

    MPI_Irecv(r_sub_data, groups[rank] * (n + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Wait(&request[1], &status[1]);

    if (rank < size - 1)
    {
      MPI_Isend(&sub_data[(n + 2) * (groups[rank] - 1)], n + 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request[0]);
      // MPI_Wait(&request[0], &status[0]);
      MPI_Irecv(bottom_row, n + 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Wait(&request[0], &status[0]);
    }

    if (rank > 0)
    {
      MPI_Isend(&sub_data[0], n + 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request[0]);
      // MPI_Wait(&request[0], &status[0]);
      MPI_Irecv(top_row, n + 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Wait(&request[0], &status[0]);
    }

    if (rank == 1 && groups[0] == 1)
    {
      MPI_Isend(&sub_data[0], n + 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request[0]);
      // MPI_Wait(&request[0], &status[0]);
    }

    if (rank == size - 2 && groups[size] == 1)
    {
      MPI_Isend(&sub_data[(groups[rank] - 1) * (n + 2)], n + 2, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, &request[0]);
      // MPI_Wait(&request[0], &status[0]);
    }

    if (rank == 0 && groups[0] == 1)
    {
      MPI_Irecv(top_row, n + 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Wait(&request[0], &status[0]);
    }

    if (rank == size - 1 && groups[size] == 1)
    {
      MPI_Irecv(bottom_row, n + 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Wait(&request[0], &status[0]);
    }

    simulate(new_data, sub_data, r_sub_data, bottom_row, top_row, rank, groups, size, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);

    MPI_Isend(&new_data[0], groups[rank] * (n + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requestGather[rank]);
    // MPI_Wait(&requestGather[rank], &statusGather[rank]);

    if (rank == 0)
    {
      MPI_Irecv(E[1], groups[0] * (n + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requestGather[0]);
      // MPI_Wait(&requestGather[0], &statusGather[0]);

      int accumulate = 0;
      for (int i = 1; i < size; i++)
      {
        accumulate += groups[i - 1];
        MPI_Irecv(E[1 + accumulate], groups[i] * (n + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requestGather[i]);
        // MPI_Wait(&requestGather[i], &statusGather[i]);
      }
    }

    MPI_Isend(&r_sub_data[0], groups[rank] * (n + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requestGather[rank + size]);
    // MPI_Wait(&requestGather[rank + size], &statusGather[rank + size]);

    if (rank == 0)
    {
      MPI_Irecv(R[1], groups[0] * (n + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requestGather[size]);
      // MPI_Wait(&requestGather[size], &statusGather[size]);

      int accumulate = 0;
      for (int i = 1; i < size; i++)
      {
        accumulate += groups[i - 1];
        MPI_Irecv(R[1 + accumulate], groups[i] * (n + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requestGather[i + size]);
        // MPI_Wait(&requestGather[i + size], &statusGather[i + size]);
      }
    }

    if (rank == 0)
    {
      MPI_Waitall(size * 2, requestGather, statusGather);
    }

    // swap current E with previous E
    if (rank == 0)
    {
      double **tmp = E;
      E = E_prev;
      E_prev = tmp;
    }

    if (rank == 0)
    {
      if (plot_freq)
      {
        int k = (int)(t / plot_freq);
        if ((t - k * plot_freq) < dt)
        {
          splot(E, t, niter, m + 2, n + 2);
        }
      }
    }

  } // end of while loop

  if (rank == 0)
  {
    double time_elapsed = getTime() - t0;

    double Gflops = (double)(niter * (1E-9 * n * n) * 28.0) / time_elapsed;
    double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0)) / time_elapsed;

    cout << "Number of Iterations        : " << niter << endl;
    cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
    cout << "Sustained Gflops Rate       : " << Gflops << endl;
    cout << "Sustained Bandwidth (GB/sec): " << BW << endl
         << endl;

    double mx;
    double l2norm = stats(E_prev, m, n, &mx);
    cout << "Max: " << mx << " L2norm: " << l2norm << endl;

    if (plot_freq)
    {
      cout << "\n\nEnter any input to close the program and the plot..." << endl;
      getchar();
    }
  }

  free(sub_data);
  free(top_row);
  free(bottom_row);
  free(new_data);
  free(r_sub_data);
  if (rank == 0)
  {
    free(E);
    free(E_prev);
    free(R);
  }
  MPI_Finalize();

  return 0;
}
