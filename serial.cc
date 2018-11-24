#include <bits/stdc++.h>

#define n_len 10000000
#define e_len 10000000

using namespace std;

// reset queue
void clear( std::queue<int> &q ) {
   std::queue<int> empty;
   std::swap( q, empty );
}
// reset stack
void clear_stack( std::stack<int> &s ) {
   std::stack<int> empty;
   std::swap( s, empty );
}

// variables
int V[n_len], E[e_len], sigma[n_len], dist[n_len];
float cb[n_len], delta[n_len];
stack<int> s;
queue<int> q;

// MAIN
int main() {

    // ================================ READ INPUT AND MAKE Compressed Adjancency List ====================================
    int nodes, edges;
    cin>>nodes>>edges;
    vector<vector<int> > p(nodes);
        
	// vector<int> V(nodes+1);
	// vector<int> E(2*edges);
    // read graph data in CSR format 
    string line;
    int node = 0;
    int counter = 0;
    getline(cin, line);
    for (int i = 0; i < nodes; ++i) {
        getline(cin, line);
        V[node] = counter;
        istringstream is(line);
        int tmp;
        while (is >> tmp) {
            E[counter] = tmp;
            counter += 1;
        }
        ++node;
    }
    V[node] = counter;
    // Check compressed adj list
    // for (int i = 0; i <= nodes; i++) {
    //     cout<<V[i]<<" ";
    // }
    // cout<<"\n";
    // for (int i = 0; i < 2 * edges ; ++i) {
    //     cout<<E[i]<<" ";
    // }
    // cout<<"\n";
	
	// vector<float> cb(nodes,0.0);
    // BC value stored here
    memset(cb,0,sizeof(nodes));

    // max bc var
    float max_bc = 0.0;
    // ===================================================== MAIN CODE =================================================== 
    clock_t begin = clock();
	for(int i = 0; i < nodes; ++i) {
		// stack<int> s;
		// s.clear();
        // ============== INIT ========================
        clear_stack(s);
        // vector<vector<int> > p(nodes);
        for(int j = 0; j < nodes; ++j){
            if(!p[j].empty()) p[j].clear();
        }
		// vector<int> sigma(nodes,0);
        memset(sigma,0,sizeof(sigma));
		// vector<int> dist(nodes);
        for(int j=0;j<nodes;++j) dist[j]=-1;
		sigma[i]=1;
		dist[i]=0;
		// queue<int> q;
        // q.clear();
		clear(q);
        // ============== distance and sigma calculations ====================
        q.push(i);
		while(!q.empty()){
			int cons=q.front();
			q.pop();
			s.push(cons);
			for(int j = V[cons]; j < V[cons+1]; ++j){
				if(dist[E[j]]<0){
					dist[E[j]]=dist[cons]+1;		
                    q.push(E[j]);
				}	
                if(dist[E[j]]==dist[cons]+1){
                    sigma[E[j]]+=sigma[cons];
                    p[E[j]].push_back(cons);
                }
			}		
		}
        // ============== BC calculation ========================
        // vector<float> delta(nodes,0.0);
        memset(delta,0,sizeof(delta));
        while(!s.empty()){
            int cons=s.top();
            s.pop();
            int upto = p[cons].size();
            for(int j = 0; j < upto;++j){
                delta[p[cons][j]]+=(((float)sigma[p[cons][j]]/sigma[cons])*((float)1+delta[cons]));
            }
            if(cons!=i) {
                    cb[cons]+=delta[cons];
                    max_bc = (max_bc > cb[cons])?max_bc:cb[cons];
            }
        }
	}
    clock_t end = clock();

    // ===================================================== RESULTS ===================================================
    // BC values for all nodes
	for (int i = 0; i < nodes; ++i) {
        cout<<"Node: "<<i<<"  BC: ";
        cout<<fixed<<setprecision(6)<<cb[i]/2.0<<"\n";
    }
    cout<<"\n";

    // Print the time for execution
    double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    cout<<"Execution time: "<<elapsed_time<<endl;
    
    // Max BC value
    cout<<"Max BC value: "<<max_bc/2.0<<endl;

	return 0;
}
