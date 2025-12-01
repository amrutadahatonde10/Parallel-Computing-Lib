#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = -1, world_size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const char* INFILE  = "input.txt";
    const char* OUTFILE = "output.txt";

    int n = 0;
    std::vector<int> A, B;

    if (world_rank == 0) {
        std::ifstream fin(INFILE);
        if (!fin) {
            std::cerr << "[0] Failed to open " << INFILE << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!(fin >> n)) {
            std::cerr << "[0] Failed to read n\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (n <= 0) {
            std::cerr << "[0] Invalid n\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        A.assign(n*n, 0);
        B.assign(n*n, 0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                fin >> A[i*n + j];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                fin >> B[i*n + j];
        fin.close();
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int p = world_size;
    int q = static_cast<int>(std::round(std::sqrt((double)p)));
    if (q * q != p) {
        if (world_rank == 0) std::cerr << "[0] Number of processes p=" << p << " is not a perfect square\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (n % q != 0) {
        if (world_rank == 0) std::cerr << "[0] Matrix size n=" << n << " is not divisible by q=" << q << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int block = n / q;

    MPI_Comm cart = MPI_COMM_NULL;
    int dims[2] = { q, q };
    int periods[2] = { 1, 1 };   
    int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart);
    if (cart == MPI_COMM_NULL) {
        if (world_rank == 0) std::cerr << "[0] MPI_Cart_create failed\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = -1;
    MPI_Comm_rank(cart, &rank);
    int coords[2] = {0,0};
    MPI_Cart_coords(cart, rank, 2, coords); 

    if (rank == 0) {
        std::cout << "[rank 0] Running Cannon with p=" << p << ", q=" << q
                  << " (grid " << q << "x" << q << "), block=" << block << " ("
                  << block << "x" << block << ")\n";
    }

    std::vector<int> A_blk(block * block);
    std::vector<int> B_blk(block * block);
    std::vector<int> C_blk(block * block, 0);

    auto pack_block = [&](const std::vector<int>& M, int r0, int c0, int *dest) {
        for (int i = 0; i < block; ++i)
            memcpy(dest + i*block, &M[(r0 + i)*n + c0], block * sizeof(int));
    };
    auto unpack_block = [&](std::vector<int>& M, int r0, int c0, const int *src) {
        for (int i = 0; i < block; ++i)
            memcpy(&M[(r0 + i)*n + c0], src + i*block, block * sizeof(int));
    };

    if (rank == 0) {
        for (int dest = 0; dest < p; ++dest) {
            int dcoords[2];
            MPI_Cart_coords(cart, dest, 2, dcoords);
            int r0 = dcoords[0] * block;
            int c0 = dcoords[1] * block;
            std::vector<int> tmpA(block*block), tmpB(block*block);
            pack_block(A, r0, c0, tmpA.data());
            pack_block(B, r0, c0, tmpB.data());
            if (dest == 0) {
                std::copy(tmpA.begin(), tmpA.end(), A_blk.begin());
                std::copy(tmpB.begin(), tmpB.end(), B_blk.begin());
            } else {
                MPI_Send(tmpA.data(), block*block, MPI_INT, dest, 11, cart);
                MPI_Send(tmpB.data(), block*block, MPI_INT, dest, 12, cart);
            }
        }
    } else {
        MPI_Status st;
        MPI_Recv(A_blk.data(), block*block, MPI_INT, 0, 11, cart, &st);
        MPI_Recv(B_blk.data(), block*block, MPI_INT, 0, 12, cart, &st);
    }

    int left, right, up, down;
    MPI_Cart_shift(cart, 1, -1, &left, &right); 
    MPI_Cart_shift(cart, 0, -1, &up, &down);    

    for (int s = 0; s < coords[0]; ++s) {
        MPI_Sendrecv_replace(A_blk.data(), block*block, MPI_INT,
                             left, 100,  
                             right, 100, 
                             cart, MPI_STATUS_IGNORE);
    }

    for (int s = 0; s < coords[1]; ++s) {
        MPI_Sendrecv_replace(B_blk.data(), block*block, MPI_INT,
                             up, 200,
                             down, 200,
                             cart, MPI_STATUS_IGNORE);
    }

    for (int step = 0; step < q; ++step) {
        for (int i = 0; i < block; ++i) {
            for (int j = 0; j < block; ++j) {
                int sum = 0;
                for (int k = 0; k < block; ++k)
                    sum += A_blk[i*block + k] * B_blk[k*block + j];
                C_blk[i*block + j] += sum;
            }
        }

        MPI_Sendrecv_replace(A_blk.data(), block*block, MPI_INT,
                             left, 3000 + step,
                             right, 3000 + step,
                             cart, MPI_STATUS_IGNORE);

        MPI_Sendrecv_replace(B_blk.data(), block*block, MPI_INT,
                             up, 4000 + step,
                             down, 4000 + step,
                             cart, MPI_STATUS_IGNORE);
    }

    if (rank == 0) {
        std::vector<int> C(n*n, 0);
        int my_r0 = coords[0]*block, my_c0 = coords[1]*block;
        unpack_block(C, my_r0, my_c0, C_blk.data());

        for (int src = 1; src < p; ++src) {
            std::vector<int> tmp(block*block);
            MPI_Status st;
            MPI_Recv(tmp.data(), block*block, MPI_INT, src, 5000, cart, &st);
            int scoords[2]; MPI_Cart_coords(cart, src, 2, scoords);
            unpack_block(C, scoords[0]*block, scoords[1]*block, tmp.data());
        }

        std::ofstream fout(OUTFILE);
        if (!fout) {
            std::cerr << "[0] Failed to open " << OUTFILE << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fout << n << "\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fout << C[i*n + j] << (j+1==n ? "" : " ");
            }
            fout << "\n";
        }
        fout.close();
        std::cout << "[0] Done. Result written to " << OUTFILE << "\n";
    } else {
        MPI_Send(C_blk.data(), block*block, MPI_INT, 0, 5000, cart);
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
