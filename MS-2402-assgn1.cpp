// parallel addition of n numbers

#include<mpi.h>
#include<iostream>
#include<vector>

using namespace std;
  
int main(int argc, char* argv[])
{
    MPI_Init(&argc,&argv);

    int rank, size;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    if (rank == 0) {
        cout << "Enter the value of n: ";
        cin >> n;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int part = n / size;
    int remainder = n % size;

    if(rank == 0)
    {
        vector<int> A;
        for(int i = 1; i <= n; i++){
            A.push_back(i);
        }
    
        for(int p = 1; p < size; p++)
        {
            MPI_Send(A.data() + p*part, part, MPI_INT, p, 1, MPI_COMM_WORLD);
        }

        int lsum = 0;
        for(int i = 0; i < part; i++)
        {
            lsum = lsum + A[i];
        }

        for(int i = n - remainder; i < n; i++)
        {
            if (remainder > 0) lsum += A[i];
        }

        cout << "\nRank " << rank << endl;
        cout << "Local Sum : " << lsum << endl;

        int FinalSum = lsum;
        for(int p = 1; p < size; p++) 
        {
            int tempsum;
            MPI_Recv(&tempsum, 1, MPI_INT, p, 2, MPI_COMM_WORLD, &status);
            FinalSum += tempsum;
        }
        cout << "Final Sum: " << FinalSum << endl;
    }

    if(rank != 0)
    {
        vector<int> B(part);
        MPI_Recv(B.data(), part, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

        int lsum = 0;
        for(int i = 0; i < part; i++)
            lsum = lsum + B[i];
        
        cout << "\nRank " << rank << endl;
        cout << "Local Sum : " << lsum << endl;

        MPI_Send(&lsum, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}


// mpic++ MS-2402-assgn1.cpp -o Myexe
// mpirun -oversubscribe -n 4 Myexe