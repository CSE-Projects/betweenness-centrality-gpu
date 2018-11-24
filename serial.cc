#include <bits/stdc++.h>

#define n_len 100005
#define e_len 100005

using namespace std;

// reset queue
void clear( std::queue<long long int> &q ) {
   std::queue<long long int> empty;
   std::swap( q, empty );
}
// reset stack
void clear_stack( std::stack<long long int> &s ) {
   std::stack<long long int> empty;
   std::swap( s, empty );
}

// variables
long long int V[n_len], E[e_len], sigma[n_len], dist[n_len];
float cb[n_len], delta[n_len];
// stack<long long int> s;
// queue<long long int> q;

// MAIN
int main() {

    // ================================ READ INPUT AND MAKE Compressed Adjancency List ====================================
    long long int nodes, edges;
    cin>>nodes>>edges;
    vector<vector<long long int> > p(nodes);
        
	// vector<long long int> V(nodes+1);
	// vector<long long int> E(2*edges);
    // read graph data in CSR format 
    string line;
    long long int node = 0;
    long long int counter = 0;
    getline(cin, line);
    for (long long int i = 0; i < nodes; ++i) {
        getline(cin, line);
        V[node] = counter;
        istringstream is(line);
        long long int tmp;
        while (is >> tmp) {
            E[counter] = tmp;
            counter += 1;
        }
        ++node;
    }
    V[node] = counter;
    // Check compressed adj list
    // for (long long int i = 0; i <= nodes; i++) {
    //     cout<<V[i]<<" ";
    // }
    // cout<<"\n";
    // for (long long int i = 0; i < 2 * edges ; ++i) {
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
	for(long long int i = 0; i < nodes; ++i) {
		// stack<long long int> s;
		// s.clear();
        // ============== INIT ========================
        // clear_stack(s);
        stack<long long int> s;
        // vector<vector<long long int> > p(nodes);
        for(long long int j = 0; j < nodes; ++j){
            p[j].clear();
        }
		// vector<long long int> sigma(nodes,0);
        // memset(sigma,0,sizeof(sigma));
		// vector<long long int> dist(nodes);
        for(long long int j=0;j<nodes;++j) dist[j]=-1,sigma[j]=0,delta[j]=0.0;
		sigma[i]=1;
		dist[i]=0;
		// queue<long long int> q;
        // q.clear();
		// clear(q);
        queue<long long int> q;
        // ============== distance and sigma calculations ====================
        q.push(i);
		while(!q.empty()){
			long long int cons=q.front();
			q.pop();
			s.push(cons);
			for(long long int j = V[cons]; j < V[cons+1]; ++j){
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
        // memset(delta,0,sizeof(delta));
        while(!s.empty()){
            long long int cons=s.top();
            s.pop();
            long long int upto = p[cons].size();
            for(long long int j = 0; j < upto;++j){
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
	// for (long long int i = 0; i < nodes; ++i) {
    //     cout<<"Node: "<<i<<"  BC: ";
    //     cout<<fixed<<setprecision(6)<<cb[i]/2.0<<"\n";
    // }
    cout<<"\n";

    // Prlong long int the time for execution
    double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    cout<<"Execution time: "<<elapsed_time<<endl;
    
    // ================================ Max BC value ====================================
    // Max BC value
    cout<<"Max BC value: "<<max_bc/2.0<<endl;

	return 0;
}
