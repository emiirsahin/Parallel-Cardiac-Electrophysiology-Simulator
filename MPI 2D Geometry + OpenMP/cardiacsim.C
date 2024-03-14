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
              double *top_row, double *bottom_row, double *left_col,
              double *right_col, int rank, int *groupsXN, int *groupsYN, int size,
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

  for (int j = 0; j < groupsYN[rank]; j++)
  {
    for (int i = 0; i < groupsXN[rank]; i++)
    {
      if (j == 0)
      {
        if (i == 0)
        {
          new_data[0] = local_data[0] + alpha * (local_data[1] + left_col[0] - 4 * local_data[0] + top_row[0] + local_data[groupsXN[rank]]);
        }
        if (i == (groupsXN[rank] - 1))
        {
          new_data[i] = local_data[i] + alpha * (right_col[0] + local_data[i - 1] - 4 * local_data[i] + top_row[i] + local_data[(2 * (i + 1)) - 1]);
        }
        if (i > 0 && i < (groupsXN[rank] - 1))
        {
          new_data[i] = local_data[i] + alpha * (local_data[i + 1] + local_data[i - 1] - 4 * local_data[i] + top_row[i] + local_data[groupsXN[rank] + i]);
        }
      }
      if (j != 0 && j != (groupsYN[rank] - 1) && i == 0)
      {
        new_data[j * groupsXN[rank]] = local_data[j * groupsXN[rank]] + alpha * (local_data[j * groupsXN[rank] + 1] + left_col[j] - 4 * local_data[j * groupsXN[rank]] + local_data[groupsXN[rank] * (j - 1)] + local_data[groupsXN[rank] * (j + 1)]);
      }
      if (j != 0 && j != (groupsYN[rank] - 1) && i == (groupsXN[rank] - 1))
      {
        new_data[j * groupsXN[rank] + groupsXN[rank] - 1] = local_data[j * groupsXN[rank] + groupsXN[rank] - 1] + alpha * (right_col[j] + local_data[j * groupsXN[rank] + (groupsXN[rank] - 2)] - 4 * local_data[j * groupsXN[rank] + groupsXN[rank] - 1] + local_data[j * groupsXN[rank] - 1] + local_data[(j + 2) * groupsXN[rank] - 1]);
      }
      if (j != 0 && j != (groupsYN[rank] - 1) && i != 0 && i != (groupsXN[rank] - 1))
      {
        new_data[j * groupsXN[rank] + i] = local_data[j * groupsXN[rank] + i] + alpha * (local_data[j * groupsXN[rank] + i + 1] + local_data[j * groupsXN[rank] + i - 1] - 4 * local_data[j * groupsXN[rank] + i] + local_data[(j - 1) * groupsXN[rank] + i] + local_data[(j + 1) * groupsXN[rank] + i]);
      }
      if (j == (groupsYN[rank] - 1))
      {
        if (i == 0)
        {
          new_data[j * groupsXN[rank]] = local_data[j * groupsXN[rank]] + alpha * (local_data[j * groupsXN[rank] + 1] + left_col[j] - 4 * local_data[j * groupsXN[rank]] + local_data[(j - 1) * groupsXN[rank]] + bottom_row[0]);
        }
        if (i == (groupsXN[rank] - 1))
        {
          new_data[(j + 1) * groupsXN[rank] - 1] = local_data[(j + 1) * groupsXN[rank] - 1] + alpha * (right_col[j] + local_data[(j + 1) * groupsXN[rank] - 2] - 4 * local_data[(j + 1) * groupsXN[rank] - 1] + local_data[j * groupsXN[rank] - 1] + bottom_row[i]);
        }
        if (i > 0 && i < (groupsXN[rank] - 1))
        {
          new_data[j * groupsXN[rank] + i] = local_data[j * groupsXN[rank] + i] + alpha * (local_data[j * groupsXN[rank] + i + 1] + local_data[j * groupsXN[rank] + i - 1] - 4 * local_data[j * groupsXN[rank] + i] + local_data[(j - 1) * groupsXN[rank] + i] + bottom_row[i]);
        }
      }
    }
  }

  for (int j = 0; j < groupsYN[rank]; j++)
  {
    for (int i = 0; i < groupsXN[rank]; i++)
    {
      new_data[j * groupsXN[rank] + i] = new_data[j * groupsXN[rank] + i] - dt * (kk * new_data[j * groupsXN[rank] + i] * (new_data[j * groupsXN[rank] + i] - a) * (new_data[j * groupsXN[rank] + i] - 1) + new_data[j * groupsXN[rank] + i] * r_local_data[j * groupsXN[rank] + i]);
    }
  }

  for (int j = 0; j < groupsYN[rank]; j++)
  {
    for (int i = 0; i < groupsXN[rank]; i++)
    {
      r_local_data[j * groupsXN[rank] + i] = r_local_data[j * groupsXN[rank] + i] + dt * (epsilon + M1 * r_local_data[j * groupsXN[rank] + i] / (new_data[j * groupsXN[rank] + i] + M2)) * (-r_local_data[j * groupsXN[rank] + i] - kk * new_data[j * groupsXN[rank] + i] * (new_data[j * groupsXN[rank] + i] - b - 1));
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

  if (px * py != size)
  {
    printf("Error in processor geometry");
    return 0;
  }

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

  int rows_per_process = m / py;
  int remainderY = m % py;
  int groupsY[py];

  for (int i = 0; i < py; i++)
  {
    groupsY[i] = rows_per_process;
  }

  for (int i = 0; i < remainderY; i++)
  {
    groupsY[i]++;
  }

  int groupsYN[py * px];

  for (int i = 0; i < py; i++)
  {
    for (int j = 0; j < px; j++)
    {
      groupsYN[j + i * px] = groupsY[i];
    }
  }

  int cols_per_process = n / px;
  int remainderX = n % px;
  int groupsX[px];

  for (int i = 0; i < px; i++)
  {
    groupsX[i] = cols_per_process;
  }
  for (int i = 0; i < remainderX; i++)
  {
    groupsX[i]++;
  }

  int groupsXN[py * px];

  for (int i = 0; i < py; i++)
  {
    for (int j = 0; j < px; j++)
    {
      groupsXN[j + i * px] = groupsX[j];
    }
  }

  double *sub_data = (double *)malloc(groupsYN[rank] * groupsXN[rank] * sizeof(double));
  double *new_data = (double *)malloc(groupsYN[rank] * groupsXN[rank] * sizeof(double));

  double *top_row = (double *)malloc(groupsXN[rank] * sizeof(double));
  double *bottom_row = (double *)malloc(groupsXN[rank] * sizeof(double));
  double *left_col = (double *)malloc(groupsYN[rank] * sizeof(double));
  double *right_col = (double *)malloc(groupsYN[rank] * sizeof(double));

  double *r_sub_data = (double *)malloc(groupsYN[rank] * groupsXN[rank] * sizeof(double));

  MPI_Status status[2];
  MPI_Request request[2];
  MPI_Request requestGather[size * 2];
  MPI_Status statusGather[size * 2];

  for (int j = 0; j < groupsYN[rank]; j++)
    for (int i = 0; i < groupsXN[rank]; i++)
      new_data[i + j * groupsXN[rank]] = 0;

  bool checker = true;

  while (t < T)
  {

    t += dt;
    niter++;

    if (rank == 0)
    {
      for (int l = 0; l < py; l++)
      {
        int passAmountY = 0;
        for (int j = 0; j < l; j++)
        {
          passAmountY += groupsYN[j * px];
        }
        for (int k = 0; k < px; k++)
        {
          double *send_data = (double *)malloc(groupsYN[k + l * px] * groupsXN[k + l * px] * sizeof(double));
          int passAmountX = 0;
          for (int j = 0; j < k; j++)
          {
            passAmountX += groupsXN[j + l * px];
          }

          for (int i = 1; i <= groupsYN[k + l * px]; i++)
          {
            for (int j = 1; j <= groupsXN[k + l * px]; j++)
            {
              send_data[(j - 1) + (i - 1) * groupsXN[k + l * px]] = E_prev[i + passAmountY][j + passAmountX];
            }
          }
          MPI_Isend(&send_data[0], groupsYN[k + l * px] * groupsXN[k + l * px], MPI_DOUBLE, k + l * px, 0, MPI_COMM_WORLD, &request[0]);
          free(send_data);
        }
      }
    }

    MPI_Irecv(sub_data, groupsYN[rank] * groupsXN[rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request[0]);

    if (rank == 0)
    {
      for (int l = 0; l < py; l++)
      {
        int passAmountY = 0;
        for (int j = 0; j < l; j++)
        {
          passAmountY += groupsYN[j];
        }
        for (int k = 0; k < px; k++)
        {
          double *send_data = (double *)malloc(groupsYN[k + l * px] * groupsXN[k + l * px] * sizeof(double));
          int passAmountX = 0;
          for (int j = 0; j < k; j++)
          {
            passAmountX += groupsXN[j + l * px];
          }

          for (int i = 1; i <= groupsYN[k + l * px]; i++)
          {
            for (int j = 1; j <= groupsXN[k + l * px]; j++)
            {
              send_data[(j - 1) + (i - 1) * groupsXN[k + l * px]] = R[i + passAmountY][j + passAmountX];
            }
          }
          MPI_Isend(&send_data[0], groupsYN[k + l * px] * groupsXN[k + l * px], MPI_DOUBLE, k + l * px, 0, MPI_COMM_WORLD, &request[1]);
          free(send_data);
        }
      }
    }

    MPI_Irecv(r_sub_data, groupsYN[rank] * groupsXN[rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Wait(&request[0], &status[0]);
    MPI_Wait(&request[1], &status[1]);

    // Start of top-bottom corner cases
    if ((rank - px) < px && rank >= px && groupsYN[rank - px] == 1)
    {
      MPI_Isend(&sub_data[0], groupsXN[rank - px], MPI_DOUBLE, rank - px, 0, MPI_COMM_WORLD, &request[0]);
    }

    if (rank < px && groupsYN[rank] == 1)
    {
      MPI_Irecv(top_row, groupsXN[rank], MPI_DOUBLE, rank + px, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Wait(&request[0], &status[0]);
    }

    if ((rank + px) >= (px * (py - 1)) && rank < (px * (py - 1)) && groupsYN[rank + px] == 1)
    {
      MPI_Isend(&sub_data[groupsXN[rank] * (groupsYN[rank] - 1)], groupsXN[rank + px], MPI_DOUBLE, rank + px, 0, MPI_COMM_WORLD, &request[0]);
    }

    if (rank >= (px * (py - 1)) && groupsYN[rank] == 1)
    {
      MPI_Irecv(bottom_row, groupsXN[rank], MPI_DOUBLE, rank - px, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Wait(&request[0], &status[0]);
    }
    // End of top-bottom corner cases

    // Start of left-right corner cases
    if ((rank - 1) % px == 0 && groupsXN[rank - 1] == 1)
    {
      double *send_left_col = (double *)malloc(groupsYN[rank] * sizeof(double));
      for (int i = 0; i < groupsYN[rank]; i++)
      {
        send_left_col[i] = sub_data[i];
      }
      MPI_Isend(&send_left_col[0], groupsYN[rank - 1], MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request[0]);
      free(send_left_col);
    }

    if (rank % px == 0 && groupsXN[rank] == 1)
    {
      MPI_Irecv(left_col, groupsYN[rank], MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Wait(&request[0], &status[0]);
    }

    if ((rank + 2) % px == 0 && groupsXN[rank + 1] == 1)
    {
      double *send_right_col = (double *)malloc(groupsYN[rank] * sizeof(double));
      for (int i = 0; i < groupsYN[rank]; i++)
      {
        send_right_col[i] = sub_data[(groupsXN[rank] - 1) + (i * groupsXN[rank])];
      }
      MPI_Isend(&send_right_col[0], groupsYN[rank + 1], MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request[0]);
      free(send_right_col);
    }

    if ((rank + 1) % px == 0 && groupsXN[rank] == 1)
    {
      MPI_Irecv(right_col, groupsYN[rank], MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Wait(&request[0], &status[0]);
    }
    // End of left-right corner cases

    if (rank < px && groupsYN[rank] > 1)
    {
      for (int i = 0; i < groupsXN[rank]; i++)
      {
        top_row[i] = sub_data[groupsXN[rank] + i];
      }
    }

    if (rank >= (px * (py - 1)) && groupsYN[rank] > 1)
    {
      for (int i = 0; i < groupsXN[rank]; i++)
      {
        bottom_row[i] = sub_data[(groupsYN[rank] - 2) * groupsXN[rank] + i];
      }
    }

    if (rank % px == 0 && groupsXN[rank] > 1)
    {
      for (int i = 0; i < groupsYN[rank]; i++)
      {
        left_col[i] = sub_data[groupsXN[rank] * i + 1];
      }
    }

    if ((rank + 1) % px == 0 && groupsXN[rank] > 1)
    {
      for (int i = 0; i < groupsYN[rank]; i++)
      {
        right_col[i] = sub_data[groupsXN[rank] * (i + 1) - 2];
      }
    }

    // CORNER CASES COMPLETE

    if (rank >= px)
    {
      MPI_Isend(&sub_data[0], groupsXN[rank], MPI_DOUBLE, rank - px, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Irecv(top_row, groupsXN[rank], MPI_DOUBLE, rank - px, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Wait(&request[0], &status[0]);
    }

    if (rank < (px * (py - 1)))
    {
      MPI_Isend(&sub_data[groupsXN[rank] * (groupsYN[rank] - 1)], groupsXN[rank], MPI_DOUBLE, rank + px, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Irecv(bottom_row, groupsXN[rank], MPI_DOUBLE, rank + px, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Wait(&request[0], &status[0]);
    }

    if ((rank % px) > 0)
    {
      double *send_left_col = (double *)malloc(groupsYN[rank] * sizeof(double));
      for (int i = 0; i < groupsYN[rank]; i++)
      {
        send_left_col[i] = sub_data[i * groupsXN[rank]];
      }
      MPI_Isend(&send_left_col[0], groupsYN[rank - 1], MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request[0]);
      free(send_left_col);
      MPI_Irecv(left_col, groupsYN[rank], MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Wait(&request[0], &status[0]);
    }

    if ((rank % px) < (px - 1))
    {
      double *send_right_col = (double *)malloc(groupsYN[rank] * sizeof(double));
      for (int i = 0; i < groupsYN[rank]; i++)
      {
        send_right_col[i] = sub_data[(groupsXN[rank] - 1) + (i * groupsXN[rank])];
      }
      MPI_Isend(&send_right_col[0], groupsYN[rank + 1], MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request[0]);
      free(send_right_col);
      MPI_Irecv(right_col, groupsYN[rank], MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Wait(&request[0], &status[0]);
    }

    simulate(new_data, sub_data, r_sub_data, top_row, bottom_row, left_col, right_col, rank, groupsXN, groupsYN, size, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);

    // SIMULATION SUCCESSFUL

    MPI_Isend(&new_data[0], groupsYN[rank] * groupsXN[rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requestGather[rank]);
    MPI_Isend(&r_sub_data[0], groupsYN[rank] * groupsXN[rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requestGather[rank + size]);

    if (rank == 0)
    {
      for (int u = 0; u < py; u++)
      {
        for (int v = 0; v < px; v++)
        {
          int accumulate = 0;
          for (int q = 0; q < v; q++)
          {
            accumulate += groupsXN[q + u * px];
          }
          double *recv_data = (double *)malloc(groupsYN[v + u * px] * groupsXN[v + u * px] * sizeof(double));
          MPI_Irecv(recv_data, groupsYN[v + u * px] * groupsXN[v + u * px], MPI_DOUBLE, v + u * px, 0, MPI_COMM_WORLD, &requestGather[v + u * px]);
          MPI_Wait(&requestGather[v + u * px], &statusGather[v + u * px]);
          /*for (int k = 0; k < groupsYN[v + u * px]; k++)
          {
            for (int l = 0; l < groupsXN[v + u * px]; l++)
            {
              E[k + 1][(l + 1) + accumulate] = recv_data[l + k * groupsXN[v + u * px]];
            }
          } RECEIVING IS INCOMPLETE*/
          printf("FROM RANK %d\n", v + u * px);
          for(int i = 0; i < groupsYN[v + u * px]; i++) {
            for(int j = 0; j < groupsXN[v + u * px]; j++) {
              printf("%.2f ", recv_data[j + i * groupsXN[v + u * px]]);
            }
            printf("\n");
          }
          free(recv_data);
        }
      }
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
  free(left_col);
  free(right_col);
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
