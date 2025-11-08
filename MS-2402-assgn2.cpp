//Numerical Integration - Simpson's 3/8 rule

#include<mpi.h>
#include<iostream>
#include<vector>
#include<cmath>

using namespace std;

double function(double value)
{
 return value*value*exp(-value);
}
  

int main(int argc, char* argv[])
{
 MPI_Init(&argc,&argv);

 int rank,size;
 MPI_Status status;
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 MPI_Comm_size(MPI_COMM_WORLD, &size);

 int n = 105;
 if(rank == 0)
 {
  cout<<"Enter n:";
  cin >> n;
  if(n%3!=0)
  {
   int old_n = n;
   n += (3 - n % 3);
   cout<<" new n = " << old_n << " to " << n ;
   }
 }

 MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

 int part= (n-1) / size;
 int remainder = n % size;

 double xi=0, yi=0;
 double a = 0.0, b= 4.0;
 double ans1 = function(a);
 double ans2 = function(b);
 double h = (b-a) / (double)n;


 if(rank == 0)
 {
  vector<int> A;
  for(int i = 1; i < n; i++){
  A.push_back(i);
  }
 

 for(int p=1; p < size; p++)
 {
  MPI_Send(A.data() + p * part, part, MPI_INT, p, 1, MPI_COMM_WORLD);
 }

 double lsum = 0;
 for(int i = 0; i < part; i++){
  xi = a + (A[i]*h);
  yi = function(xi);
  if(A[i] % 3 == 0)
   lsum += 2.0*yi;
  else 
   lsum += 3.0*yi;
 }

 double finalSum = lsum;
 for(int p =1; p < size; p++) {
  double tempsum;
  MPI_Recv(&tempsum, 1 ,MPI_DOUBLE, p ,2 ,MPI_COMM_WORLD,&status);
  finalSum += tempsum;
 }
 cout<<"\n h = "<<h<<endl;
 double FinalAns = ((3.0*h)/8.0)*(ans1+ans2+finalSum);

 cout << "Final Sum of " << n << " = " << FinalAns << endl;
}

else{
     vector<int> B(part);
     MPI_Recv(B.data(), part, MPI_INT, 0, 1, MPI_COMM_WORLD,&status);
     double lsum = 0;
     for (int i = 0; i < part; i++)
     {
      xi = a + (B[i]*h);
      yi = function(xi);
      if(B[i] % 3 == 0)
       lsum += 2*yi;
      else 
       lsum += 3*yi;
     } 

  MPI_Send(&lsum, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
 }


 MPI_Finalize();

return 0;
}
