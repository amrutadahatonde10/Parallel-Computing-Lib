#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
using namespace std;

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int me,np;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&np);

    int R=0,C=0;
    vector<double>A;
    if(me==0){
        ifstream f("input.txt");
        f>>R>>C; C++;
        A.resize(R*C);
        for(int i=0;i<R*C;i++) f>>A[i];
        f.close();
    }

    MPI_Bcast(&R,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&C,1,MPI_INT,0,MPI_COMM_WORLD);

    if(np>=C){ if(me==0) cout<<"Error: p<n\n"; MPI_Finalize(); return 0;}

    int blk=R/np, ex=R%np;
    int my=(me<ex?blk+1:blk);
    int t=my*C;
    vector<double> L(t);

    vector<int>sc(np),dp(np);
    int s=0;
    for(int i=0;i<np;i++){
        sc[i]=((i<ex?blk+1:blk)*C);
        dp[i]=s; s+=sc[i];
    }

    MPI_Scatterv(A.data(),sc.data(),dp.data(),MPI_DOUBLE,L.data(),t,MPI_DOUBLE,0,MPI_COMM_WORLD);

    int st=(me<ex?me*(blk+1):ex*(blk+1)+(me-ex)*blk);
    vector<double>P(C),T(C);

    for(int k=0;k<R;k++){
        double mx=-1; int rg=-1;
        for(int i=0;i<my;i++){
            int g=st+i;
            if(g>=k && fabs(L[i*C+k])>mx){
                mx=fabs(L[i*C+k]); rg=g;
            }
        }
        struct{double v;int i;}a={mx,rg},b;
        MPI_Allreduce(&a,&b,1,MPI_DOUBLE_INT,MPI_MAXLOC,MPI_COMM_WORLD);

        if(b.v==0){ if(me==0) cout<<"Singular\n"; MPI_Finalize(); return 0;}

        auto own=[&](int r){return (r<ex*(blk+1)?r/(blk+1):ex+(r-ex*(blk+1))/blk);};
        auto lid=[&](int r,int p){return(p<ex?r-p*(blk+1):r-(ex*(blk+1))-(p-ex)*blk);};

        if(b.i!=k){
            int o1=own(k),o2=own(b.i);
            if(o1!=o2){
                if(me==o1){
                    int id=lid(k,me);
                    MPI_Sendrecv(&L[id*C],C,MPI_DOUBLE,o2,2,T.data(),C,MPI_DOUBLE,o2,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    for(int j=0;j<C;j++) L[id*C+j]=T[j];
                }
                if(me==o2){
                    int id=lid(b.i,me);
                    MPI_Sendrecv(&L[id*C],C,MPI_DOUBLE,o1,3,T.data(),C,MPI_DOUBLE,o1,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    for(int j=0;j<C;j++) L[id*C+j]=T[j];
                }
            }
        }

        int rt=own(k);
        if(me==rt){
            int id=lid(k,me);
            double pv=L[id*C+k];
            for(int j=0;j<C;j++) P[j]=L[id*C+j]/pv;
            for(int j=0;j<C;j++) L[id*C+j]=P[j];
        }

        MPI_Bcast(P.data(),C,MPI_DOUBLE,rt,MPI_COMM_WORLD);

        for(int i=0;i<my;i++){
            int g=st+i;
            if(g>k){
                double f=L[i*C+k];
                for(int j=k;j<C;j++) L[i*C+j]-=f*P[j];
            }
        }
    }

    MPI_Gatherv(L.data(),t,MPI_DOUBLE,A.data(),sc.data(),dp.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);

    if(me==0){
        cout<<"Upper:\n";
        for(int i=0;i<R;i++){ for(int j=0;j<C;j++) cout<<A[i*C+j]<<" "; cout<<"\n"; }
        vector<double>X(R);
        for(int i=R-1;i>=0;i--){
            double s=A[i*C+(C-1)];
            for(int j=i+1;j<R;j++) s-=A[i*C+j]*X[j];
            X[i]=s/A[i*C+i];
        }
        cout<<"Solution:\n";
        for(double v:X) cout<<v<<" "; cout<<"\n";
    }

    MPI_Finalize();
}
