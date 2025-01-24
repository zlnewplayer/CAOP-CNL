// C++ program to solve the cardinality-constrained assortment optimization under the Cross-Nested Logit model 
// The exact algorithm is based on a tailored branch-and-bound algorithm
// We develop  heurisic algorithms: 1 - sorted-by-revenue, 2 - binary search + greedy algorithm

#include <bits/stdc++.h>
#include <algorithm>
using namespace std;


const int n = 100; // number of items
const int m = 5; // number of nests
const double time_limit = 3600; // computation time limit
int total_capacity; // cardinality constraint
const int beta_c_start = 1, beta_c_end = 3; // different cardinality level, 1 - 0.1n, 2 - 0.2n, 3 - 0.3n
const int overlap_set_start = 0, overlap_set_end = 0; // overlapping rate level, 0 - 1.2, 1 - 2.0, 2 - 3.0, 3 - 1.0, 4 - 1.5, 5 - 2.5
const int instance_num = 20;
const int instance_start_index = 1;
bool brute_force = false;
bool reach_limit = false;
bool overflow = false;
double v0 = 10.0; // preference weight of no-purchase option
double epsilon = 0.00001;
int best_s[n], h3_s[n], zu_s[n], zl_s[n];

int correct_num = 0;

clock_t start_time, end_time;
const int checkInterval = 10000;
ofstream out;

int total_iter = 0;
int branch_num = 0, bound_num = 0, set1_num = 0, alfine_num = 0, all_determined = 0, all_occupied = 0, update_num = 0, last_update_pop = 0;

// Structure for Item which store price and utility value of Item
struct Item
{
	double price;
	double utility;
	double ns_utility[m];
};

// Node structure to store information of decision tree
struct Node
{
	// front_id --> the index of the first item whose decision variable is not fixed
	// back_id --> the index of the last item whose decision variable is not fixed
	// obj --> the value of the objective function of this node only considering the fixed variables
	// bound ---> Upper bound of maximum obj in subtree of this node
	int front_id, back_id;
	int s0[n], s1[n], s[n]; // fixed to 0? fixed to 1? fixed ?
	double obj = 0, bound = 0; // ob_gap represents the gap ratio between the current objective function and the upper bound obj. value of the parameterized problem at this node (gap - obj) / obj
	double a[m], b[m]; // sum of nest-specific utility, sum of revised price * nest-specific utility
	int remain_cap = total_capacity;
     int remain_undeter = n; // the number of undetermined products
};
Node best_node, grandparent_node, parent_node;
double grandparent_z, parent_z;

struct Interval
{
	double start, end;
	int seq[n];
	bool positive[n]; // 1 if value > 0, otherwise 0
};

struct Nest_intervals
{
	vector<Interval> inter;
};

struct cmp2
{
	bool operator() (Node a, Node b)
	{
		return a.obj < b.obj;
	}
};

void copy_node(Node &v, Node &u) { // copy v from u
	v.front_id = u.front_id;
	v.back_id = u.back_id;
	v.obj = u.obj;
	v.bound = u.bound;
	v.remain_cap = u.remain_cap;
     v.remain_undeter = u.remain_undeter;
	for (int i = 0; i < n; i++) {
		v.s0[i] = u.s0[i];
		v.s1[i] = u.s1[i];
		v.s[i] = u.s[i];
	}
	for (int i = 0; i < m; i++) {
		v.a[i] = u.a[i];
		v.b[i] = u.b[i];
	}
}

double update_obj(Item arr[], double g[], Node v) {
	double objective = 0;
	for (int i = 0; i < m; i++)
	{
		if(v.a[i] > 0)
			objective += v.b[i] / pow(v.a[i], 1 - g[i]);
	}
	return objective;
}

// generate the upper bound value of each node in the B&B tree
double generate_upper_bound(Item pro[], double g[], Node v, double z, Nest_intervals nis[]) {
	// decoupling the original problem, each nest is constrained by a cardinality constraint C
	double u_lower = z, u_upper;
	double bd = 0, jac_value = 0;
	double max_ns_obj = 0, ns_obj = 0, a, b;
	for (int i = 0; i < m; i++)
	{
		max_ns_obj = 0;
		a = v.a[i];
		b = v.b[i];
		ns_obj = (a == 0) ? 0 : b / pow(a, 1 - g[i]);
		if (ns_obj > max_ns_obj)
			max_ns_obj = ns_obj;
		u_upper = (v.a[i] > 0) ? max(v.b[i] / v.a[i] + z, pro[v.front_id].price) : pro[v.front_id].price;
		for (int j = 0; j < nis[i].inter.size(); j++)
		{
			if (nis[i].inter[j].end < u_lower) continue;
			else if (nis[i].inter[j].start > u_upper) break;
			else {
				int k = 0, cap = 0, p_id;
				a = v.a[i], b = v.b[i];
				while (cap < v.remain_cap && nis[i].inter[j].positive[k])
				{
					p_id = nis[i].inter[j].seq[k];
					if(v.s[p_id] == 0) {
						a += pro[p_id].ns_utility[i];
						b += pro[p_id].ns_utility[i] * (pro[p_id].price - z);
						cap++;
					}
					k++;
				}
				ns_obj = (a == 0) ? 0 : b / pow(a, 1 - g[i]);
				if (ns_obj > max_ns_obj) {
					max_ns_obj = ns_obj;
				}
			}
		}
		bd += max_ns_obj;
	}
	return bd;
}


double z_upperbound(Item pro[], double g[])
{
	double V[m];
	for (int i = 0; i < m; i++) {
		V[i] = 0;
		vector<double> uti;
		for (int j = 0; j < n; j++) uti.push_back(pro[j].ns_utility[i]);
		std::sort(uti.begin(), uti.end(), greater<double>());
		for (int j = 0; j < total_capacity; j++) V[i] += uti[j];
	}
	double vsum = 0;
	for (int i = 0; i < m; i++) vsum += pow(V[i], g[i]);
	return pro[0].price * vsum / (v0 + vsum);
}


double compute_sbr_rev(Item pro[], double g[], int k, int total_c) // sbr -- sorted-by-revenue
{
	double v[m], r[m], fenzi = 0, fenmu = v0;
	for (int i = 0; i < m; i++) {
		v[i] = 0;
		r[i] = 0;
		for (int j = k; j < k + total_c; j++) {
			v[i] += pro[j].ns_utility[i];
			r[i] += pro[j].ns_utility[i] * pro[j].price;
		}
		fenzi +=  (v[i] == 0) ? 0 : pow(v[i], g[i]) * r[i] / v[i];
		fenmu += pow(v[i], g[i]);
	}
	if (fenmu == 0)
		return 0;
	else
		return fenzi / fenmu;
}

int maxi = 0, maxc = 0;

double z_lowerbound(Item pro[], double g[], int total_cap)
{
	// find the sorted-by-revenue assortment with the largest revenue（N_ik = {i,i+1,...,k}）
	double max_rev = -1, rev;
	for (int c = 1; c <= total_cap; c++)
	{
		for (int i = 0; i < n - c + 1; i++) {
			rev = compute_sbr_rev(pro, g, i, c);
			if (rev > max_rev) {
				max_rev = rev;
				maxi = i;
				maxc = c;
			}
		}
	}
	return max_rev;
}

double evaluate_obj(Item pro[], double g[], double alp[][m], int s[])
{
	double v[m], r[m], fenzi = 0, fenmu = v0;
	for (int i = 0; i < m; i++) {
		v[i] = 0;
		r[i] = 0;
		for (int j = 0; j < n; j++) {
			v[i] += pro[j].ns_utility[i] * s[j];
			r[i] += pro[j].ns_utility[i] * pro[j].price * s[j];
		}
		fenzi += (v[i] == 0) ? 0 : pow(v[i], g[i]) * r[i] / v[i];
		fenmu += pow(v[i], g[i]);
	}
	if (fenmu == 0)
		return 0;
	else
		return fenzi / fenmu;
}


// Brute-Force algorithm for the original problem
double BF(int s[], int start, Item pro[], double g[], double alp[][m], int total_cap)
{
	if(total_cap == 0 || start >= n) {
		return evaluate_obj(pro, g, alp, s);
	}
	double max_val =-1, fval1, fval2;
	s[start] = 1;
	fval1 = BF(s, start + 1, pro, g, alp, total_cap - 1);
	s[start] = 0;
	fval2 = BF(s, start + 1, pro, g, alp, total_cap);
	return max(fval1, fval2);
}

void printAssortment(Node node)
{
	std::cout << "Selected items: ";
	for (int i = 0; i < n; i++)
	{
		if(node.s1[i])
			std::cout << i << " ";
	}
	std::cout << endl;
}

bool judge_set_1(Item arr[], double g[], Node v) 
{
	bool set1 = true;
	int j = v.front_id;
	double f1, f2;
	for (int i = 0; i < m; i++)
	{
		if (arr[j].ns_utility[i] == 0)
			continue;
		f1 = v.a[i] == 0 ? 0 : v.b[i] / pow(v.a[i], 1 - g[i]);
		f2 = (v.b[i] + arr[j].price * arr[j].ns_utility[i]) / pow(v.a[i] + arr[j].ns_utility[i], 1 - g[i]);
		if (f2 < f1)
			return false;
	}
	return true;
}

double ZU(double z, Item pro[], double g[], double alp[][m], Nest_intervals ni[])
{
	Node v;
	for (int i = 0; i < n; i++)
		v.s[i] = v.s1[i] = v.s0[i] = 0;
	for (int i = 0; i < m; i++)
		v.a[i] = v.b[i] = 0;
	v.front_id = 0;
	v.back_id = n-1;
	return generate_upper_bound(pro, g, v, z, ni);
}

double G(double z, Item pro[], double g[], double alp[][m]) 
{
	Item arr[n];
	for (int i = 0; i < n; i++)
	{
		arr[i].price = pro[i].price - z;
		arr[i].utility = pro[i].utility;
		for (int j = 0; j < m; j++)
		{
			arr[i].ns_utility[j] = pow(alp[i][j] * arr[i].utility, 1 / g[j]);
		}
	}
	Node v;
	for (int i = 0; i < n; i++)
		v.s[i] = v.s1[i] = v.s0[i] = 0;
	for (int i = 0; i < m; i++)
		v.a[i] = v.b[i] = 0;
	int occ = 0;
	double cur_obj, old_obj = 0;
	while (occ < total_capacity)
	{
		double max_obj = -1;
		int max_j = -1;
		for (int j = 0; j < n; j++)
		{
			if(v.s[j] == 0) {
				for (int i = 0; i < m; i++)
				{
					v.a[i] += arr[j].ns_utility[i];
					v.b[i] += arr[j].ns_utility[i] * arr[j].price;
				}
				cur_obj = update_obj(arr, g, v);
				if(cur_obj > max_obj) {
					max_obj = cur_obj;
					max_j = j;
				}
				for (int i = 0; i < m; i++)
				{
					v.a[i] -= arr[j].ns_utility[i];
					v.b[i] -= arr[j].ns_utility[i] * arr[j].price;
				}
			}
		}
		if(max_obj > old_obj) {
			v.s1[max_j] = v.s[max_j] = 1;
			for (int i = 0; i < m; i++)
			{
				v.a[i] += arr[max_j].ns_utility[i];
				v.b[i] += arr[max_j].ns_utility[i] * arr[max_j].price;
			}
			occ++;
			old_obj = max_obj;
		}
		else
			break;
	}
	for (int i = 0; i < n; i++)
		h3_s[i] = v.s1[i];
	return update_obj(arr, g, v);
}

int find_branch_id(Item arr[], double g[], Node u)
{
	Node v;
	copy_node(v, u);
	double cur_obj, max_obj = -1, old_obj = u.obj;
	int max_j;
	for (int j = u.front_id; j <= u.back_id; j++)
	{
		if(v.s[j] == 0) { // product j is undetermined 
			for (int i = 0; i < m; i++)
			{
				v.a[i] += arr[j].ns_utility[i];
				v.b[i] += arr[j].ns_utility[i] * arr[j].price;
			}
			cur_obj = update_obj(arr, g, v);
			if(cur_obj > max_obj) {
				max_obj = cur_obj;
				max_j = j;
			}
			for (int i = 0; i < m; i++)
			{
				v.a[i] -= arr[j].ns_utility[i];
				v.b[i] -= arr[j].ns_utility[i] * arr[j].price;
			}
		}
	}
	return max_j;
}

// Returns maximum obj we can get from the optimal assortment
double F(double z, Item pro[], double g[], double alp[][m], Nest_intervals ni[])
{
	//out << "z value: " << z << endl;
	Item arr[n];
	for (int i = 0; i < n; i++)
	{
		arr[i].price = pro[i].price - z;
		arr[i].utility = pro[i].utility;
		for (int j = 0; j < m; j++)
		{
			arr[i].ns_utility[j] = pow(alp[i][j] * arr[i].utility, 1 / g[j]);
		}
	}

	// make a queue for traversing the node
	//queue<Node> Q; // Breadth-First Search
	priority_queue<Node, vector<Node>, cmp2> Q; // Best-First Search, using the priority_queue may be slightly more efficient than using stack
	//stack<Node> Q; // Depth-First Search
	Node u, v1, v2;

	// dummy node at starting
	u.front_id = 0;
	u.back_id = n - 1;
	int i = n - 1;
	while (i >= 0 && arr[i].price <= 0) // important
	{
		u.s0[i] = u.s[i] = 1;
		u.s1[i] = 0;
		u.remain_undeter--;
		u.back_id = --i;
	}
	while (i >= 0)
	{
		u.s0[i] = u.s1[i] = u.s[i] = 0;
		i--;
	}
	for (i = 0; i < m; i++)
		u.a[i] = u.b[i] = 0;
	u.obj = u.bound = 0;
	Q.push(u); // push the root node into the B&B tree

	// One by one extract an item from decision tree
	// compute obj of all children of extracted item and keep saving maxObj
	double maxObj = 0;
	//double maxObj = G(z, pro, g, alp);
	int pop_times = 0, last_update = 0;
     int branch_id = -1;  // denote which product should be branched （added for version 2!）

	while (!Q.empty())
	{
		// Dequeue a node
		//u = Q.front(); // Breadth-First Search
		u = Q.top(); // Best-First Search & Depth-First Search

		Q.pop();
		pop_times++;

		// Variable Fixing Operation
		while(u.remain_cap > 0 && u.remain_undeter <= u.remain_cap && u.front_id <= u.back_id && judge_set_1(arr, g, u)) {
			u.s1[u.front_id] = u.s[u.front_id] = 1;
			for (i = 0; i < m; i++) {
				u.a[i] = u.a[i] + arr[u.front_id].ns_utility[i];

				u.b[i] = u.b[i] + arr[u.front_id].ns_utility[i] * arr[u.front_id].price;
			}
			while (u.s[u.front_id] == 1) u.front_id++;
			u.remain_undeter--;
			u.remain_cap--;
			set1_num++;
		}

		u.obj = update_obj(arr, g, u);
		if (u.obj > maxObj) {
			copy_node(best_node, u);
			maxObj = u.obj;
			std::cout << "Node front: " << u.front_id << " and node back: " << u.back_id << " and remain capacity: " << u.remain_cap << " and Obj: " << u.obj << " and pop times: " << pop_times << endl;
			update_num++;
			last_update = pop_times;
		}
          // If no capacity left or all decision variables are fixed, continues with no branching
		if (u.remain_cap == 0 || u.remain_undeter == 0) {
			alfine_num++;
			if(u.remain_cap == 0) all_occupied++;
			if(u.remain_undeter == 0) all_determined++;
			continue;
		}
          // Check whether to branching
		else {
			u.bound = generate_upper_bound(pro, g, u, z, ni);
			if (u.bound > maxObj) { // it is still necessary to branch
				branch_num++;
				//branch_id = u.front_id; // Branching Rule 1
				branch_id = find_branch_id(arr, g, u); // Branching Rule 2 (IMPORTANT!)
                    // First node: exclude the product
		          copy_node(v1, u);
		          v1.s0[branch_id] = v1.s[branch_id] = 1; // fix current head to 0
				v1.remain_undeter--;
		          while (v1.s[v1.front_id] == 1) v1.front_id++;
				while (v1.s[v1.back_id] == 1) v1.back_id--;
				Q.push(v1);
                    // Second node: include the product
		          copy_node(v2, u);
		          v2.s1[branch_id] = v2.s[branch_id] = 1;
		          v2.remain_cap--;
				v2.remain_undeter--;
		          // update the node's metrics
		          for (i = 0; i < m; i++) {
			          v2.a[i] += arr[branch_id].ns_utility[i];
			          v2.b[i] += arr[branch_id].ns_utility[i] * arr[branch_id].price;
		          }
		          while (v2.s[v2.front_id] == 1) v2.front_id++;
				while (v2.s[v2.back_id] == 1) v2.back_id--;
				v2.obj = update_obj(arr, g, v2);
                    Q.push(v2);
			}
			else {
				bound_num++;
			}
		}
          // Restrict running time within the time limit
		if(pop_times % checkInterval == 0) {
			end_time = clock();
			cout << "queue length: " << Q.size() << ", maxObj: " << maxObj << endl;
			if((double)(end_time - start_time) / CLOCKS_PER_SEC >= time_limit) {
				reach_limit = true;
				break;
			}
			if(Q.size() >= 1000000) {
				reach_limit = true;
                    overflow = true;
				break;
			}
		}
	}
	total_iter++;
	last_update_pop += last_update;
	end_time = clock();
	if((double)(end_time - start_time) / CLOCKS_PER_SEC >= time_limit)
		reach_limit = true;
	return maxObj;
}

void compute_interval_matrix(Item pro[], int nest_id, Nest_intervals &inter_vector) {
	double a[n], b[n], p[n], lend, rend; // left endpoint, right endpoint
	for (int i = 0; i < n; i++)
	{
		a[i] = pro[i].ns_utility[nest_id];
		p[i] = pro[i].price;
		b[i] = a[i] * p[i];
	}
	int bp_num = 0;
	vector<double> break_point;
	break_point.push_back(0);
	bp_num++;
	for (int i = 0; i < n; i++)
	{
		if(a[i] > 0) {
			break_point.push_back(p[i]);
			bp_num++;
			for (int j = i + 1; j < n; j++)
			{
				if(a[j] == 0) continue;
				else {
					if(b[i] < b[j] && p[i] > p[j]) {
						break_point.push_back( (b[j] - b[i]) / (a[j] - a[i]) );
						bp_num++;
					}
					else if(b[i] > b[j] && p[i] < p[j]) {
						break_point.push_back( (b[i] - b[j]) / (a[i] - a[j]) );
						bp_num++;
					}
					else continue;
				}
			}
		}
	} // break points counting completed
	std::sort(break_point.begin(), break_point.end());
	for (int i = 0; i < bp_num - 1; i++)
	{
		lend = break_point[i];
		rend = break_point[i + 1];
		if(lend < rend) {
			Interval new_interval;
			new_interval.start = lend;
			new_interval.end = rend;
			double mid = (lend + rend) / 2;
			double val[n];
			int id[n];
			for (int j = 0; j < n; j++)
			{
				val[j] = b[j] - a[j] * mid;
				id[j] = j;
			}
			// insertion sort by val[j]
			for (int j = 1; j < n; j++)
			{
				double v = val[j];
				int k = j - 1, js = id[j];
				while (k >= 0 && val[k] < v)
				{
					val[k + 1] = val[k];
					id[k + 1] = id[k];
					k--;
				}
				val[k + 1] = v;
				id[k + 1] = js;
			}
			for (int i = 0; i < n; i++) {
				new_interval.seq[i] = id[i];
				new_interval.positive[i] = val[i] > 0 ? 1 : 0;
			}
			inter_vector.inter.push_back(new_interval);
		}
	}
}

bool judge_identical(int zls[], int zus[]) 
{
	for (int i = 0; i < n; i++)
	{
		if(zls[i] != zus[i])
			return false;
	}
	return true;
}


int main()
{
	int beta_c = beta_c_start;
	while (beta_c <= beta_c_end) // traverse three cardinality levels
	{
	int overlap_set = overlap_set_start;
	while (overlap_set <= overlap_set_end)
	{
	string data_name = "pset_m" + to_string(m) +"_n" + to_string(n) + "_o" + to_string(overlap_set) + "_g1_";
	total_capacity = int(ceil(0.1 * n * beta_c));
	string suffix = "c" + to_string(beta_c);
	int m_val, n_val;
	int optimal_num = 0, gap_num = 0; // record exact method related numbers
	int improve_num_h1 = 0, improve_num_h2 = 0; // record number of instances improved over heuristic xxx
	// necessary parameters
	double z, z_upper, z_lower, obf, bf_obj, z_low;
	double z_upper1, z_upper2; // record initial z upper bound values generated by different methods
	double z_lower1, z_lower2; // record heuristic objective value, 1 - sorted-by-revenue, 2 - binary search + greedy
    // input data
	double gamma[m], alpha[n][m];
	Item products[n];
	// timing related 
	double run_time;
	vector<double> heuristic1_t, heuristic2_t, bf_t, t, improve1, improve2, gap_ratio;
	vector<double> heuristic1_obj, heuristic2_obj, our_obj;
	vector<int> iter_num;

	ifstream in;
	// set name
	string s2 = "../Output/Formal_test_PC_v2/" + data_name + "c" + to_string(total_capacity) + "_" + "output_" + suffix + ".txt";
	out.open(s2, ios::out | ios::trunc);
	for (int index = instance_start_index; index < instance_start_index + instance_num; index++)
	{
		reach_limit = false;
          overflow = false;
		total_iter = 0;
		branch_num = 0, bound_num = 0, set1_num = 0, alfine_num = 0, all_determined = 0, all_occupied = 0, update_num = 0, last_update_pop = 0;
		string s1 = "../Data/Formal_test/" + data_name + to_string(index) + ".txt";
		std::cout << "Instance " << index << endl;
		out << "Instance " << index << endl;
		in.open(s1, ios::in);
		in >> m_val;
		in >> n_val;
		for (int i = 0; i < m; i++) in >> gamma[i];
		for (int i = 0; i < n; i++) {
			best_s[i] = 0; // initialize global best assortment
			in >> products[i].price;
			in >> products[i].utility;
		}
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				in >> alpha[i][j];
		in.close();
		// compute nest-specific utility
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				products[i].ns_utility[j] = pow(alpha[i][j] * products[i].utility, 1 / gamma[j]);
			}
		}

		// Compute auxiliary nest-specific interval matrix
		Nest_intervals ninter[m];
		for (int i = 0; i < m; i++)
			compute_interval_matrix(products, i, ninter[i]);
		
		// Brute-Force algorithm (for computing optimal solution)
		if(brute_force) {
			start_time = clock();
			int s[n];
			for (int i = 0; i < n; i++) s[i] = 0;
			bf_obj = BF(s, 0, products, gamma, alpha, total_capacity);
			out << "BF obj: " << bf_obj << endl;
			end_time = clock();
			run_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
			bf_t.push_back(run_time);
			out << "BF run time: " << run_time << endl;
		}

		// initialize three global nodes
		for (int i = 0; i < n; i++)
		{
			grandparent_node.s[i] = grandparent_node.s1[i] = grandparent_node.s0[i] = 0;
			parent_node.s[i] = parent_node.s1[i] = parent_node.s0[i] = 0;
			best_node.s[i] = best_node.s1[i] = best_node.s0[i] = 0;
		}
		
		// main algorithm

		// z_upper generation method 1: set all prices to p1 + decoupling
		z_upper1 = z_upperbound(products, gamma); 
		
		// heuristic 1: sorted-by-revenue assortment
		start_time = clock();
		z_lower1 = z_lowerbound(products, gamma, total_capacity);  
		end_time = clock();
		heuristic1_obj.push_back(z_lower1);
		run_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
		heuristic1_t.push_back(run_time);

		// z_upper generation method 2: decoupling (solving the corresponding CAOP-NL)
		double z_u = z_upper1, z_l = z_lower1, obj_nl;
		while (v0 * z_u > v0 * z_l + epsilon) {
			z = (z_u + z_l) / 2;
			obj_nl = ZU(z, products, gamma, alpha, ninter);
			if(v0 * z < obj_nl)
				z_l = z;
			else
				z_u = z;
		}
		z_upper2 = z_l;

		// heuristic 2: binary search + greedy
		start_time = clock();
		z_lower = 0;
		z_upper = z_upper1;
		double obg;
		while (v0 * z_upper > v0 * z_lower + epsilon)
		{
			z = (z_upper + z_lower) / 2;
			obg = G(z, products, gamma, alpha);
			if(v0 * z < obg)
				z_lower = z;
			else
				z_upper = z;
		}
		z_lower2 = evaluate_obj(products, gamma, alpha, h3_s);
		end_time = clock();
		heuristic2_obj.push_back(z_lower2);
		run_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
		heuristic2_t.push_back(run_time);


		// binary search framework
		z_lower = z_lower2;  // currently decide to use lower bound 2
		z_upper = z_upper2;  // currently decide to use upper bound 2
		
		out << "z_lower: " << z_lower << endl;
		out << "z_upper: " << z_upper << endl;

		start_time = clock();

		obf = F(z_lower, products, gamma, alpha, ninter);
		for (int i = 0; i < n; i++) zl_s[i] = best_node.s1[i];
		if(evaluate_obj(products, gamma, alpha, best_node.s1) > evaluate_obj(products, gamma, alpha, best_s)) {
			out << "Get a better assortment decision!" << endl;
			for (int i = 0; i < n; i++)	best_s[i] = best_node.s1[i];
		}
		if(!reach_limit) {
			obf = F(z_upper, products, gamma, alpha, ninter);
			for (int i = 0; i < n; i++) zu_s[i] = best_node.s1[i];
			if(evaluate_obj(products, gamma, alpha, best_node.s1) > evaluate_obj(products, gamma, alpha, best_s)) {
				out << "Get a better assortment decision!" << endl;
				for (int i = 0; i < n; i++)	best_s[i] = best_node.s1[i];
			}
		}
		while ((v0 * z_upper > v0 * z_lower + epsilon) && !judge_identical(zl_s, zu_s))
		{
			if(reach_limit)
				break;
			z = (z_upper + z_lower) / 2;
			std::cout << "z value: " << z << endl;
			obf = F(z, products, gamma, alpha, ninter);
			if(evaluate_obj(products, gamma, alpha, best_node.s1) > evaluate_obj(products, gamma, alpha, best_s)) {
				//out << "Get a better assortment decision!" << endl;
				for (int i = 0; i < n; i++) best_s[i] = best_node.s1[i];
			}
			if(v0 * z < obf) {
				z_lower = z;
				//out << "z_lower updated to: " << z << endl;
				for (int i = 0; i < n; i++) zl_s[i] = best_node.s1[i];
			}
			else {
				z_upper = z;
				//out << "z_upper updated to: " << z << endl;
				for (int i = 0; i < n; i++) zu_s[i] = best_node.s1[i];
			}
		}
		end_time = clock();
		run_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
          if(overflow)
               run_time = 3600.0; // if there is priority queue overflow, we manually set the time to reach time limit.
		t.push_back(run_time);

		double best_obj = evaluate_obj(products, gamma, alpha, best_s);
		our_obj.push_back(best_obj);
		if(!reach_limit)
			optimal_num++;
		else {
			if((z_upper - best_obj) / z_upper > 0) {
				out << "z_upper: " << fixed << setprecision(3) << z_upper << endl;
				out << "best_obj: " << best_obj << endl;
				out << "gap ratio: " << (z_upper - best_obj) / z_upper << endl;
				gap_ratio.push_back((z_upper - best_obj) / z_upper);
				gap_num++;
			}
		}
		iter_num.push_back(total_iter);

		double imp1 = (best_obj - z_lower1) / z_lower1;
		double imp2 = (best_obj - z_lower2) / z_lower2;
		if(imp1 > 0)
			improve_num_h1++;
		if(imp2 > 0)
			improve_num_h2++;
		improve1.push_back(imp1); 
		improve2.push_back(imp2);
		
		if(best_obj == bf_obj) 
			correct_num++;

		// Output arrangement
		out << "[Heuristics]" << endl;
		out << "Z lower 1: " << fixed << setprecision(3) << z_lower1 << endl;
		out << "Z lower 2: " << z_lower2 << endl;
		out << "Selected items: "; // print the assortment generated by our best heuristic
		for (int i = 0; i < n; i++)
			if(h3_s[i])
				out << i << " ";
		out << endl;
		//out << "maxi: " << maxi << ", maxc: " << maxc << endl; 
		out << "Z lower 1 runtime: " << fixed << setprecision(2) << heuristic1_t.back() << endl;
		out << "Z lower 2 runtime: " << heuristic2_t.back() << endl;
		out << "Improvement over z_lower 1: " << 100 * improve1.back() << "%" << endl;
		out << "Improvement over z_lower 2: " << 100 * improve2.back() << "%" << endl;
		// Results of exact solution method
		out << "[Exact method]" << endl;
		std::cout << "Optimal obj. = " << fixed << setprecision(3) << best_obj << endl;
		out << "Optimal obj. : " << fixed << setprecision(3) << best_obj << endl;
		out << "Run time: " << fixed << setprecision(2) << run_time << "s" << endl;
		out << "Selected items: ";
		for (int i = 0; i < n; i++)
			if(best_s[i])
				out << i << " ";
		out << endl;
		out << "Total iter: " << total_iter << endl;
		out << "Average branch number: " << branch_num / total_iter << ", average bound number: " << bound_num / total_iter << ", average set 1 number: " << set1_num / total_iter << ", average alfine number: " << alfine_num / total_iter << endl;
		out << "(Alfine) average all determined: " << all_determined / total_iter << ", (Alfine) average all occupied: " << all_occupied / total_iter << ", average update: " << update_num / total_iter << ", average find optimal: " << last_update_pop / total_iter << endl;
		out << endl;
	}
	double tmean = std::accumulate(t.begin(), t.end(), 0.0) / t.size();
	double imp1_mean = std::accumulate(improve1.begin(), improve1.end(), 0.0) / improve1.size();
	double imp2_mean = std::accumulate(improve2.begin(), improve2.end(), 0.0) / improve2.size();
	double h1_obj_mean = std::accumulate(heuristic1_obj.begin(), heuristic1_obj.end(), 0.0) / heuristic1_obj.size();
	double h1_time_mean = std::accumulate(heuristic1_t.begin(), heuristic1_t.end(), 0.0) / heuristic1_t.size();
	double h2_obj_mean = std::accumulate(heuristic2_obj.begin(), heuristic2_obj.end(), 0.0) / heuristic2_obj.size();
	double h2_time_mean = std::accumulate(heuristic2_t.begin(), heuristic2_t.end(), 0.0) / heuristic2_t.size();
	double bf_time_mean = std::accumulate(bf_t.begin(), bf_t.end(), 0.0) / bf_t.size();
	double our_obj_mean = std::accumulate(our_obj.begin(), our_obj.end(), 0.0) / our_obj.size();
	double iter_num_mean = std::accumulate(iter_num.begin(), iter_num.end(), 0.0) / iter_num.size();
     vector<double> diff(t.size());
     std::transform(t.begin(), t.end(), diff.begin(),
               std::bind2nd(std::minus<double>(), tmean));
	double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	double stdev = std::sqrt(sq_sum / t.size());
	// Results of heuristics
	out << "[Heuristics Summary]" << endl;
	out << "Mean z_lower1: " << fixed << setprecision(3) << h1_obj_mean << endl;
	out << "Mean z_lower2: " << h2_obj_mean << endl;
	out << "Mean z_lower1 runtime: " << fixed << setprecision(2) << h1_time_mean << endl;
	out << "Mean z_lower2 runtime: " << h2_time_mean << endl;
	out << "Improve num h1: " << improve_num_h1 << endl;
	out << "Improve num h2: " << improve_num_h2 << endl;
	out << "Mean improvement over z_lower 1: " << 100 * imp1_mean << "%" << endl;
	out << "Mean improvement over z_lower 2: " << 100 * imp2_mean << "%" << endl;
	
	// Results of brute force
	if(brute_force) {
		out << "Correct num: " << correct_num << endl;
		out << "Mean brute force time: " << bf_time_mean << endl;
	}
	// Results of exact solution method
	out << "[Exact method Summary]" << endl;
	out << "Mean objective value: " << fixed << setprecision(3) << our_obj_mean << endl;
	out << "Mean run time: " << fixed << setprecision(2) << tmean << endl;
	out << "Standard deviation of run time: " << stdev << endl;
	out << "Convergence num: " << optimal_num << endl;
	out << "Average iteration num: " << fixed << setprecision(1) << iter_num_mean << endl;
	if(gap_num > 0) {
		double gap_ratio_mean = std::accumulate(gap_ratio.begin(), gap_ratio.end(), 0.0) / gap_ratio.size();
		out << "Average Gap ratio: " << fixed << setprecision(2) << 100 * gap_ratio_mean << "%" << endl;
	}
	out.close();
	overlap_set++;
	}
	beta_c++;
	}
	return 0;
}
