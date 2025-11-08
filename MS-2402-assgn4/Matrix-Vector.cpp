#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;

// Function to read matrix A and vector B on rank 0
void readInputData(int& rows, int& cols, vector<double>& matrixA, vector<double>& vectorB, int rank) {
    if (rank == 0) {
        ifstream fileA("inputA.txt");
        ifstream fileB("inputB.txt");

        if (!fileA.is_open() || !fileB.is_open()) {
            cerr << "[Rank 0] ERROR: Unable to open inputA.txt or inputB.txt\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read matrix A and detect dimensions
        string line;
        vector<vector<double>> tempMatrix;
        while (getline(fileA, line)) {
            if (line.empty()) continue;
            stringstream ss(line);
            vector<double> row;
            double value;
            while (ss >> value)
                row.push_back(value);
            if (!row.empty())
                tempMatrix.push_back(row);
        }
        fileA.close();

        if (tempMatrix.empty()) {
            cerr << "[Rank 0] ERROR: inputA.txt is empty\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        rows = tempMatrix.size();
        cols = tempMatrix[0].size();
        matrixA.resize(rows * cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrixA[i * cols + j] = tempMatrix[i][j];

        // Read vector B
        double val;
        while (fileB >> val)
            vectorB.push_back(val);
        fileB.close();

        if ((int)vectorB.size() != cols) {
            cerr << "[Rank 0] ERROR: Vector B size (" << vectorB.size() << ") does not match matrix A columns (" << cols << ")\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        cout << "[Rank 0] Successfully read matrix A (" << rows << "x" << cols << ") and vector B (" << vectorB.size() << " elements)\n\n";
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M = 0, N = 0;
    vector<double> matrixA;
    vector<double> vectorB;
    vector<double> localMatrixSegment;
    vector<double> localResult;
    vector<double> resultVector;

    // Read input data on rank 0
    readInputData(M, N, matrixA, vectorB, rank);

    // Broadcast matrix dimensions to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast vector B to all processes
    if (rank != 0) vectorB.resize(N);
    MPI_Bcast(vectorB.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Determine rows handled by each process
    int baseRowCount = M / size;
    int remainderRows = M % size;
    int localRowCount = baseRowCount + (rank < remainderRows ? 1 : 0);

    vector<int> sendCounts(size), displacements(size);
    if (rank == 0) {
        for (int i = 0; i < size; i++)
            sendCounts[i] = (baseRowCount + (i < remainderRows ? 1 : 0)) * N;

        displacements[0] = 0;
        for (int i = 1; i < size; i++)
            displacements[i] = displacements[i - 1] + sendCounts[i - 1];
    }

    localMatrixSegment.resize(localRowCount * N);
    MPI_Scatterv(matrixA.data(), sendCounts.data(), displacements.data(), MPI_DOUBLE,
                 localMatrixSegment.data(), localRowCount * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Compute local matrix-vector multiplication
    localResult.resize(localRowCount);
    for (int i = 0; i < localRowCount; i++) {
        localResult[i] = 0.0;
        for (int j = 0; j < N; j++)
            localResult[i] += localMatrixSegment[i * N + j] * vectorB[j];
    }

    // Gather the partial results back to rank 0
    vector<int> recvCounts(size), recvDispls(size);
    if (rank == 0) {
        for (int i = 0; i < size; i++)
            recvCounts[i] = baseRowCount + (i < remainderRows ? 1 : 0);

        recvDispls[0] = 0;
        for (int i = 1; i < size; i++)
            recvDispls[i] = recvDispls[i - 1] + recvCounts[i - 1];

        resultVector.resize(M);
    }

    MPI_Gatherv(localResult.data(), localRowCount, MPI_DOUBLE,
                resultVector.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Display the final result on rank 0
    if (rank == 0) {
        cout << "[Rank 0] Computed Result Vector C (" << M << "x1):\n";
        for (double val : resultVector)
            cout << val << " ";
        cout << "\n";

        ofstream fout("outputC.txt");
        for (double val : resultVector)
            fout << val << " ";
        fout.close();

        cout << "[Rank 0] Result vector saved to outputC.txt\n";
    }

    MPI_Finalize();
    return 0;
}
