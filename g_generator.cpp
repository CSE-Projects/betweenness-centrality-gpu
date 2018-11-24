#include <bits/stdc++.h>

using namespace std;

int main(){

    srand(time(NULL));

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    

    long long int nodes, edges;
    cin>>nodes>>edges;

    vector<vector<long long int> > adjacency(nodes);

    long long int max_limit = (nodes * (nodes-1))/2;
    if(edges > max_limit){cout<<"ERROR: Edges exceed maximum limit, increase nodes/decrease edges"; return 0;}

    set<pair<long long int, long long int> > present_edge;

    for(long long int i = 0; i < edges; ++i){
        long long int node_1 = rand() % nodes;
        long long int node_2 = rand() % nodes;
        if(node_1 == node_2){ --i; continue;}
        if(present_edge.find(make_pair(min(node_1,node_2), max(node_1,node_2))) == present_edge.end()){
            adjacency[node_1].push_back(node_2);
            adjacency[node_2].push_back(node_1);
            present_edge.insert(make_pair(min(node_1,node_2), max(node_1,node_2)));
        }
        else{
            --i;
        }
    }

    cout<<nodes<<" "<<edges<<"\n";
    long long int upto;
    for(long long int i = 0; i < nodes; ++i){
        upto = adjacency[i].size();
        for(long long int j = 0; j <upto; ++j){
            cout<<adjacency[i][j]<<" ";
        }
        cout<<"\n";
    }

    return 0;
}