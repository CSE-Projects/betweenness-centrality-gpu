#include <bits/stdc++.h>
using namespace std;
long long int a1, a2, start, fin, nodes, edges, e[10000000], v[10000000];
int main(){
    cin>>nodes>>edges;
    for(a1=0;a1<=nodes;++a1) cin>>v[a1];
    for(a1=0;a1<(edges*edges);++a1) cin>>e[a1];
    cout<<nodes<<" "<<edges<<"\n";
    for(a1 = 0; a1 < nodes; ++a1){
        start = v[a1];  
        fin = v[a1+1];
        for(a2=start;a2<fin;++a2){
            cout<<e[a2]<<" ";
        }
        cout<<"\n";
    }
    return 0;
}