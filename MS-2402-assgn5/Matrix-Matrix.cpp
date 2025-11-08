#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M = 0, N = 0, P = 0;  
    vector<double> A, B, C;

    if (rank == 0) {
        ifstream fileA("inputA.txt"), fileB("inputB.txt");
        if (!fileA.is_open() || !fileB.is_open()) {
            cerr << "Error: Cannot open inputA.txt or inputB.txt\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        vector<vector<double>> tempA;
        string line;
        while (getline(fileA, line)) {
            if (line.empty()) continue;
            stringstream ss(line);
            vector<double> row;
            double val;
            while (ss >> val) row.push_back(val);
            if (!row.empty()) tempA.push_back(row);
        }
        fileA.close();

        M = tempA.size();
        N = tempA[0].size();
        A.resize(M * N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A[i * N + j] = tempA[i][j];

        vector<vector<double>> tempB;
        while (getline(fileB, line)) {
            if (line.empty()) continue;
            stringstream ss(line);
            vector<double> row;
            double val;
            while (ss >> val) row.push_back(val);
            if (!row.empty()) tempB.push_back(row);
        }
        fileB.close();

        int rowsB = tempB.size();
        int colsB = tempB[0].size();
        if (rowsB != N) {
            cerr << "Error: Matrix A columns (" << N << ") do not match Matrix B rows (" << rowsB << ")\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        P = colsB;
        B.resize(N * P);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < P; j++)
                B[i * P + j] = tempB[i][j];

        cout << "Detected Matrix A: " << M << "x" << N << "\n";
        cout << "Detected Matrix B: " << N << "x" << P << "\n";
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) B.resize(N * P);
    MPI_Bcast(B.data(), N * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int baseRows = M / size;
    int extra = M % size;
    int localRows = baseRows + (rank < extra ? 1 : 0);

    vector<int> sendCounts(size), displs(size);
    if (rank == 0) {
        for (int i = 0; i < size; i++)
            sendCounts[i] = (baseRows + (i < extra ? 1 : 0)) * N;

        displs[0] = 0;
        for (int i = 1; i < size; i++)
            displs[i] = displs[i - 1] + sendCounts[i - 1];
    }

    vector<double> localA(localRows * N);
    MPI_Scatterv(A.data(), sendCounts.data(), displs.data(), MPI_DOUBLE,
                 localA.data(), localRows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> localC(localRows * P, 0.0);
    for (int i = 0; i < localRows; i++)
        for (int j = 0; j < P; j++)
            for (int k = 0; k < N; k++)
                localC[i * P + j] += localA[i * N + k] * B[k * P + j];

    vector<int> recvCounts(size), recvDispls(size);
    if (rank == 0) {
        for (int i = 0; i < size; i++)
            recvCounts[i] = (baseRows + (i < extra ? 1 : 0)) * P;

        recvDispls[0] = 0;
        for (int i = 1; i < size; i++)
            recvDispls[i] = recvDispls[i - 1] + recvCounts[i - 1];

        C.resize(M * P);
    }

    MPI_Gatherv(localC.data(), localRows * P, MPI_DOUBLE,
                C.data(), recvCounts.data(), recvDispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        ofstream fout("output.txt");
        cout << "\nResultant Matrix C (" << M << "x" << P << "):\n";
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < P; j++) {
                cout << C[i * P + j] << " ";
                fout << C[i * P + j] << " ";
            }
            cout << "\n";
            fout << "\n";
        }
        fout.close();
        cout << "\nResult written to outputC.txt\n";
    }

    MPI_Finalize();
    return 0;
}
