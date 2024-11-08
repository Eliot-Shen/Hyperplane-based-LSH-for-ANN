#include <iostream> 
#include <math.h> 
#include <string.h> 
#include <random> 
#include <unordered_set> 
#include <fstream> 
#include <chrono> 
#include "my_priorty_queue.h" 
using namespace std; 
 
#define maxd 130 
#define maxn 25000 
#define maxk 100 
#define NUM_HASH_TABLES 13 //哈希表个数 
#define NUM_BUCKETS 2048   //单表哈希桶个数 = 2 ^ NUM_HyperPlane 
#define BUCKET_SIZE 5000   //哈希桶大小上限 
#define NUM_HyperPlane 11  //超平面个数 
 
int n, d, k; 
int bitset[NUM_HASH_TABLES][NUM_BUCKETS][NUM_HyperPlane]; //embedding的位表示 
int querybitset[NUM_HyperPlane];                          //query的位表示 
bool havebit[NUM_HASH_TABLES][NUM_BUCKETS];      //用于记录该哈希桶是否标好对应位表示 
//embedding 
struct Point 
{ 
    int id; 
    float coordinates[maxd]; 
}; 
Point vec[NUM_HASH_TABLES][NUM_HyperPlane]; //超平面组 
 
//数据集 
struct Dataset{ 
    Point* points; 
    int numPoints; 
 }; 
 
//哈希表 
struct HashTable{ 
    int *bucketlen; //桶内元素计数 
    int **bucket; 
    ~HashTable(); 
}; 
HashTable::~HashTable() { 
    for (int i = 0; i < NUM_BUCKETS; i++) { 
        delete[] bucket[i]; 
    } 
    delete[] bucket; 
    delete[] bucketlen; 
} 
 
//记录最终结果 
struct Result{ 
    int idx; 
    float loss; 
    Result(); 
    Result(int idx, float loss); 
}; 
Result::Result(){ 
} 
Result::Result(int idx, float loss){ 
    idx = idx; 
    loss = loss; 
} 
bool operator < (const Result &x, const Result &y) 
{ 
    return x.loss < y.loss; 
} 
bool operator >= (const Result &x, const Result &y) 
{ 
    return x.loss >= y.loss; 
} 
bool operator > (const Result &x, const Result &y) 
{ 
    return x.loss > y.loss; 
} 
 
//召回率测试 
float cal_recall_k(const unordered_set<int> &gt, const unordered_set<int> &res, 
size_t k) { 
    float recall = 0.f; 
    for (auto x: gt) {
        recall += res.count(x); 
    } 
    return recall / k; 
} 
 
//欧几里得距离计算 
float ltwodistance(Point* point1, Point* point2){ 
    float sum1 = 0.0f; 
    for (register int i = 0; i < d; ++i) { 
        float diff = point1->coordinates[i] - point2->coordinates[i]; 
        sum1 += diff * diff; 
    } 
    return sum1; 
} 
//生成数据集 
void generateDataset(Dataset* dataset) { 
    Point* points = new Point[n+1]; 
    dataset->points = points; 
    dataset->numPoints = n; 
    for(register int i = 0; i < n; ++i) { 
        points[i].id = i; 
        for (register int j = 0; j < d; ++j) { 
            cin >> points[i].coordinates[j]; 
        } 
    } 
 
    mt19937 e; 
    normal_distribution <float> dis(30.0, 2000.0);// 期望为X，标准差为Y的正态分布 
    for(int idx=0;idx<NUM_HASH_TABLES;idx++){ 
        for(register int j=0;j<NUM_HyperPlane;j++){ 
            for(register int i=0;i<d;++i){ 
                vec[idx][j].coordinates[i] = dis(e); 
            } 
        } 
    } 
} 
 
//生成哈希桶的位表示，并将数据集中的embedding归入对应桶中 
float BitGenerate(Point* point1, int tableIndex) { 
    float sum = 0.0f; 
    int bitrecord[NUM_HyperPlane]; 
    for(register int p=0;p<NUM_HyperPlane;p++){ 
        float sumset = 0.0f; 
        for (int i = 0; i < d; ++i) { 
            float diff = point1->coordinates[i] *vec[tableIndex][p].coordinates[i]; 
            sumset += diff; 
        } if(sumset>=0){ 
            bitrecord[p] = 1; 
            sum += pow(2, NUM_HyperPlane-1-p); 
        }else{ 
            bitrecord[p] = 0; 
        } 
    } 
    int bucket = (int)sum % NUM_BUCKETS; 
    if(!havebit[tableIndex][bucket]){ 
        for(register int i=0;i<NUM_HyperPlane;i++){ 
            bitset[tableIndex][bucket][i] = bitrecord[i]; 
        } 
        havebit[tableIndex][bucket] = true; 
    } 
    return sum;//二进制表示对应的十进制即为桶号 
} 
 
unsigned int hashFunction(Point* point, int tableIndex) { 
    unsigned int hash = 0; 
    hash = (unsigned int)BitGenerate(point, tableIndex); 
    return hash; 
} 
//建哈希表 
void buildHashTable(Dataset* dataset, int tableIndex, HashTable 
hashTables[NUM_HASH_TABLES]) { 
    for (int i = 0; i < NUM_HASH_TABLES; i++){ 
        HashTable* hashTable = &hashTables[i]; 
        hashTable->bucketlen = new int[NUM_BUCKETS+10]();    
        hashTable->bucket = new int*[NUM_BUCKETS+10];       //分配行 
        for (int i = 0; i < NUM_BUCKETS+10; i++) { 
            hashTable->bucket[i] = new int[BUCKET_SIZE+10]; //分配列 
        } 
    } 
    //初始化为0 
    for(int i=0;i<NUM_BUCKETS;i++){        
        hashTables[tableIndex].bucketlen[i] = 0; 
        for(int j=0; j < BUCKET_SIZE ; j++){ 
            hashTables[tableIndex].bucket[i][j] = 0; 
        } 
    } 
    //为embedding分配桶，桶内元素计数 
    for (register int i = 0; i < dataset->numPoints; ++i) { 
        unsigned int hash = hashFunction(&dataset->points[i], tableIndex); 
        hashTables[tableIndex].bucket[hash][hashTables[tableIndex].bucketlen[hash]] 
        = dataset->points[i].id;//在桶的末尾插入 
        hashTables[tableIndex].bucketlen[hash]++; 
    } 
} 
 
void searchNearestNeighbors(Dataset* dataset, Point* query, HashTable 
                            hashTables[NUM_HASH_TABLES], unordered_set<int> &finalset) { 
    List<Result> result; //优先级队列 
    result.build_heap();  
 
    int* record = new int[maxn]; //用于去重，记录embedding是否已在候选集中 
    for(int i=0;i<n;i++){ 
        record[i] = 0; 
    } 
    for(int num=0; num < NUM_HASH_TABLES; ++num){ 
        int query_sum = 0; 
        for(register int p = 0; p < NUM_HyperPlane; p++){ 
            float sumset = 0.0f; 
            for (register int i = 0; i < d; ++i) { 
                float diff = query->coordinates[i] * vec[num][p].coordinates[i]; 
                sumset += diff; 
            } 
            if(sumset>=0){ 
                querybitset[p] = 1; 
                query_sum += pow(2, NUM_HyperPlane-1-p); 
            }else{ 
                querybitset[p] = 0; 
            } 
        } 
        // hamming distance == 0 
        int hash = (int)query_sum; 
        for(register int i=0;i<hashTables[num].bucketlen[hash];++i){ 
            int id = hashTables[num].bucket[hash][i]; 
            if(record[id]==0){ 
                Result r1 = Result(); 
                r1.idx = id; 
                float distance = ltwodistance(query, &dataset->points[r1.idx]); 
                r1.loss = distance; 
                result.insert(result.size(), r1); 
                record[id] = 1; 
            } 
        } 
        // hamming distance == 1 
        if(num < NUM_HASH_TABLES/2){  
            int candidates[NUM_HyperPlane]; 
            //计算距离为1时位表示对应的桶 
            for(int i=0;i<NUM_HyperPlane;i++){ 
                if(querybitset[i]==1){ 
                    candidates[i] = query_sum - pow(2, NUM_HyperPlane-1-i); 
                }else{ candidates[i] = query_sum + pow(2, NUM_HyperPlane-1-i); 
                } 
            } 
            //插入候选集 
            for(int i=0;i<NUM_HyperPlane;i++){ 
                int candidate = candidates[i]; 
                for(register int j=0;j < hashTables[num].bucketlen[candidate];++j){ 
                    int id = hashTables[num].bucket[candidate][j]; 
                    if(record[id]==0){ 
                        Result r1 = Result(); 
                        r1.idx = id; 
                        float distance = ltwodistance(query, &dataset->points[r1.idx]); 
                        r1.loss = distance; 
                        result.insert(result.size(), r1); 
                        record[id] = 1; 
                    } 
                }      
            } 
        } 
    } 
    result.heap_sort(); 
    //cout<<"候选集大小："<<result.size()<<endl; 
    int rr = 0; 
    while(finalset.size()<k){  
        finalset.insert(result.entry[rr].idx); 
        rr++; 
    } 
}