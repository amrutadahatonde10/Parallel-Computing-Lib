#include <fstream>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// Swap function for integers
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Partition function for quicksort
int partition(vector<int>& data, int start, int end) {
    int pivot = data[start];
    int i = start + 1;
    for (int j = start + 1; j <= end; j++) {
        if (data[j] <= pivot) {
            swap(&data[i], &data[j]);
            i++;
        }
    }
    swap(&data[start], &data[i - 1]);
    return i - 1;
}

// QuickSort function
void quickSort(vector<int>& data, int start, int end) {
    if (start < end) {
        int p = partition(data, start, end);
        quickSort(data, start, p - 1);
        quickSort(data, p + 1, end);
    }
}

// Function to read integers from a file and return as a vector
vector<int> readInputFromFile(const string& filename) {
    ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        cerr << "Unable to open file: " << filename << "\n";
        exit(1);
    }

    vector<int> numbers;
    int number;
    while (inputFile >> number) {   // Read until end of file
        numbers.push_back(number);
    }
    inputFile.close();
    return numbers;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int processRank, totalProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);

    // Ensure the program runs with exactly 4 processes
    if (totalProcesses != 4) {
        if (processRank == 0)
            cerr << "Please run the program with exactly 4 processes\n";
        MPI_Finalize();
        return 0;
    }

    vector<int> data;
    int totalElements;

    // Only rank 0 reads the input file
    if (processRank == 0) {
        data = readInputFromFile("input.txt");
        totalElements = data.size();

        cout << "Initial data (" << totalElements << " elements): ";
        for (int num : data) cout << num << " ";
        cout << "\n\n";
    }

    // Broadcast the total number of elements to all processes
    MPI_Bcast(&totalElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the counts and displacements for scattering data
    int baseCount = totalElements / totalProcesses;
    int counts[4], displacements[4];
    for (int i = 0; i < totalProcesses; i++)
        counts[i] = baseCount + (i < (totalElements % totalProcesses) ? 1 : 0);

    displacements[0] = 0;
    for (int i = 1; i < totalProcesses; i++)
        displacements[i] = displacements[i - 1] + counts[i - 1];

    vector<int> localData(counts[processRank]);
    MPI_Scatterv(processRank == 0 ? data.data() : nullptr, counts, displacements, MPI_INT,
                 localData.data(), counts[processRank], MPI_INT, 0, MPI_COMM_WORLD);

    // Perform quicksort on the local data
    quickSort(localData, 0, localData.size() - 1);

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Rank " << processRank << " sorted local data: ";
    for (int num : localData) cout << num << " ";
    cout << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform parallel quicksort rounds (2 rounds for this example)
    int rounds = 2;
    for (int round = 0; round < rounds; round++) {
        int groupSize = 1 << (round + 1);  // Doubling the group size each round
        int groupRank = processRank / groupSize;
        int partnerRank;

        // Determine the partner rank for data exchange
        if ((processRank % groupSize) < (groupSize / 2))
            partnerRank = processRank + (groupSize / 2);
        else
            partnerRank = processRank - (groupSize / 2);

        int groupLeaderRank = groupRank * groupSize;
        int pivotValue = 0;

        // Only the group leader computes the pivot and broadcasts it
        if (processRank == groupLeaderRank && !localData.empty())
            pivotValue = localData[localData.size() / 2];

        MPI_Bcast(&pivotValue, 1, MPI_INT, groupLeaderRank, MPI_COMM_WORLD);

        vector<int> lowerPart, upperPart;
        // Split data into lower and upper parts based on pivot
        for (int val : localData) {
            if (val <= pivotValue)
                lowerPart.push_back(val);
            else
                upperPart.push_back(val);
        }

        // Decide which part to keep and which part to send
        bool keepLower = ((processRank % groupSize) < (groupSize / 2));
        vector<int>& partToSend = keepLower ? upperPart : lowerPart;
        vector<int>& partToKeep = keepLower ? lowerPart : upperPart;

        // Exchange data between processes
        int sendSize = partToSend.size(), recvSize;
        MPI_Sendrecv(&sendSize, 1, MPI_INT, partnerRank, 0,
                     &recvSize, 1, MPI_INT, partnerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        vector<int> receivedData(recvSize);
        if (sendSize > 0 || recvSize > 0) {
            MPI_Sendrecv(partToSend.data(), sendSize, MPI_INT, partnerRank, 1,
                         receivedData.data(), recvSize, MPI_INT, partnerRank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Merge the received data with the part to keep
        partToKeep.insert(partToKeep.end(), receivedData.begin(), receivedData.end());
        localData = partToKeep;

        // Perform quicksort again on the merged data
        quickSort(localData, 0, localData.size() - 1);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Rank " << processRank << " final data after merging: ";
    for (int num : localData) cout << num << " ";
    cout << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather the final sorted data from all processes
    int localFinalCount = localData.size();
    int finalCounts[4];
    MPI_Gather(&localFinalCount, 1, MPI_INT, finalCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int finalDisplacements[4];
    if (processRank == 0) {
        finalDisplacements[0] = 0;
        for (int i = 1; i < 4; i++)
            finalDisplacements[i] = finalDisplacements[i - 1] + finalCounts[i - 1];
        data.resize(totalElements);
    }

    MPI_Gatherv(localData.data(), localFinalCount, MPI_INT,
                data.data(), finalCounts, finalDisplacements, MPI_INT, 0, MPI_COMM_WORLD);

    // Only rank 0 writes the final sorted data to a file
    if (processRank == 0) {
        cout << "\nFinal sorted data: ";
        for (int num : data) cout << num << " ";
        cout << "\n";

        ofstream outputFile("output.txt");
        for (int num : data) outputFile << num << " ";
        outputFile << "\n";
        outputFile.close();

        cout << "\nFinal sorted data written to output.txt\n";
    }

    MPI_Finalize();
    return 0;
}
