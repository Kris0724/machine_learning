#include "stdio.h"
#include "string.h"
#include "memory.h"
#include "math.h"
#include "stdlib.h"
#include "time.h"
#include <getopt.h>

#include <string>
using namespace std;
 

#define GBDT_TERMINAL -1
#define GBDT_TOSPLIT  -2
#define GBDT_INTERIOR -3

//#define DEBUG

#define uint32 unsigned int
#define swap_int(a, b) ((a ^= b), (b ^= a), (a ^= b))

#define SAMPLE_TYPE 1
#define SAMPLE_RATIO 1.0

#define BUFFER_LENGTH 10240

#define NO_VALUE 0x7FFFFFFF

#define LOG_ERROR_(message) fprintf(stderr, "%s:%d:%s(): %s", __FILE__, __LINE__, __FUNCTION__, message); 
#define LOG_WARNING_(message) fprintf(stderr, "%s:%d:%s(): %s", __FILE__, __LINE__, __FUNCTION__, message); 
#define LOG_NOTICE_(message) fprintf(stderr , "%s", message);

typedef struct
{
	int* nodestatus;	//  �����<split, internal, terminal>
 	int* depth;			// ����
 	int* splitid;		// �����split��
 	double* splitvalue; // !<split�����split value
 	int* ndstart;		// !<�����index�����
 	int* ndcount;		// !<���������
 	double* ndavg;		// !<��������
 	//double* vpredict;	// 
 	int* lson;			// ������
 	int* rson;			// ������
 	int nodesize;		//!<������
} gbdt_tree_t; //!< �������



typedef struct
{
	int tree_num;		//!< ������
 	int fea_num;		//!< feature���
 	int data_num;		//!< ��������
 	int sample_num;		//!< ��������
 	int rand_fea_num;	//!< Feature�����

 	double shrink;		//!< ���

 	int gbdt_min_node_size; //!< ������������������
 	int gbdt_max_depth; //!< ������������

 	char train_filename[BUFFER_LENGTH]; //!< ��������
 	char model_filename[BUFFER_LENGTH]; //!< �����
} gbdt_info_t; //!< ����������

typedef struct
{
	gbdt_tree_t** reg_forest; //!< ����
	gbdt_info_t info; //!< GBDT �����

	double* feature_average; //!< Feature ���������
}gbdt_model_t; //!< GBDT ������

typedef struct  
{
	int* fea_pool; //!< ��feature ���
 	double* fvalue_list; //!< �feature i������� x_i
 	double* fv; //!< �������buffer��
 	double* y_list; //!< ���y���
 	int* order_i; //!< �����
} bufset; //!< �����

typedef struct 
{
	int index_b; //!< ���������
 	int index_e; //!< ���������
 	int nodenum; //!< ��������
 	double nodesum; //!< �������y�������
 	double critparent; //!< ������

} nodeinfo; //!< �����

typedef struct  
{
	int bestid; //!< �����feature ID
 	double bestsplit; //!< �����x�
 	int pivot; //!< ���������
 
} splitinfo; //!<�����
 
 /*
 * @brief ��ѵ�������б��������ȡ��FeatureѰ�ҷָ�����λ��
 *
 * @param <IN> gbdt_inf : ģ�͵�������Ϣ�ṹ��
 * @param <IN> data_set : ѵ�����ݳ�
 * @param <IN> x_fea_value : ѵ�������е�featureֵ
 * @param <IN> y_score : ѵ�������е�����Ŀ��ֵ
 * @param <IN> ninf : ���ڵ���Ϣ
 * @param <IN> index : ��������
 * @param <IN> spinf : ���ѵĽڵ���Ϣ
 * @return=-1 : ����ʧ��
 * @return=1 : ���ѵ�����������޷�ѡ�����Էָ��Feature
 * @return=0 : ���ѳɹ�
 * */
 int gbdt_tree_node_split(gbdt_info_t gbdt_inf, bufset* data_set, double *x_fea_value, double *y_score,
     nodeinfo ninf, int* index, splitinfo* spinf);
 
 /*
 * @brief ����һ�ûع���
 *
 * @param <IN> x_fea_value : ѵ�������е�featureֵ
 * @param <IN> y_gradient : ѵ�������е�����Ŀ��ֵ
 * @param <IN> gbdt_inf : ѵ��ģ�͵����ò���
 * @param <IN> data_set : ѵ�����ݳ�
 * @param <IN> index : ��������
 * @param <IN> gbdt_single_tree : �ع�����ָ��
 * @param <IN> nrnodes : ���Ľڵ���Ŀ
 * @return=-1 : ѵ��ʧ��
 * @return=0 : ѵ���ɹ�
 * */
 
 int gbdt_single_tree_estimation(double *x_fea_value, double *y_gradient, 
   gbdt_info_t gbdt_inf, bufset* data_set, 
   int* index, gbdt_tree_t* gbdt_single_tree, int nrnodes );

int gbdt_tree_dfs(gbdt_tree_t *gbdt_single_tree);
 
 /*
 * @brief ʹ��Gradient Boosting Decision Treeģ�ͽ��лع�Ԥ��
 *
 * @param <IN> rf_model : Random Forestģ�ͽṹ��ָ��
 * @param <IN> x_test: �������ݵ�Featureֵ
 * @param <OUT> y_predict : ģ��Ԥ��ֵ
 * @return=-1 : Ԥ��ʧ��
 * @return=1 : Ԥ��ɹ�
 * */
 
 int gbdt_regression_predict(gbdt_model_t* gbdt_model, double *x_test, double& ypredict);
 
 /*
 * @brief ���㵥�ûع�����Ԥ��ֵ
 *
 * @param <IN> x_test : Ԥ�����ݵ�Featureֵ
 * @param <IN> gbdt_single_tree : ���þ�������ָ��
 * @param <IN> ypred : �ع�Ԥ��ֵ
 * @param <IN> shrink : ѧϰ��
 * @return=-1 : Ԥ��ʧ��
 * @return=0 : Ԥ��ɹ�
 * */
 
 int gbdt_tree_predict(double *x_test, gbdt_tree_t *gbdt_single_tree, double& ypred, double shrink);
 
 /*
 * @brief ѵ��Gradient Boosting Decision Treeģ��
 *
 * @param <IN> x_fea_value : ѵ�����ݵ�Featureֵ
 * @param <IN> y_result_score : ѵ�����ݵ�Ŀ��ֵ
 * @param <IN> infbox : ѵ��ģ�͵����ò���
 * @return=-1 : ѵ��ʧ��
 * @return=0 : ѵ���ɹ�
 * */
 
 gbdt_model_t* gbdt_regression_train(double *x_fea_value, double *y_result_score, gbdt_info_t infbox);
 
 /*
 * @brief �������н�������
 *
 * @param <IN> infbox : ģ�����õĽṹ��
 * @param <IN> argc : �����в�������
 * @param <IN> argv : �����в���
 * @return=-1 : ������ȡʧ��
 * @return=0 : ������ȡ�ɹ�
 * */
 
 int read_conf_file(gbdt_info_t& infbox, int argc, char* argv[]);
 /*
 * @brief ��separator��Ϊ�ָ�������һ�н��зָ��洢��items������
 *
 * @param <IN> line : Random Forestģ�ͽṹ��ָ��
 * @param <IN> items[]: �������ݵ�Featureֵ
 * @param <IN> items_num : ģ��Ԥ��ķ������
 * @param <IN> separator : ģ��Ԥ��ķ������
 * @return : �ָ����������ַ���������
 * */
 int splitline(string line, string items[], int items_num, const char separator);
 
 /*
 * @brief ��ѵ���õ�Gradient Boosting Decision Treeģ�ʹ洢���ļ���
 *
 * @param <IN> model_filename : �洢��ģ���ļ���
 * @param <IN> gbdt_model: ���洢��ģ�ͽṹ��ָ��
 * @return=-1 : �洢ʧ��
 * @return=1 : �洢�ɹ�
 * */
 int gbdt_save_model(gbdt_model_t* gbdt_model, char* model_filename);
 /*
 * @brief ���ļ��ж�ȡGradient Boosting Decision Tree��ģ��
 *
 * @param <IN> model_file : ģ���ļ���
 * @return=NULL : ��ȡʧ��
 * @return!=NULL : ��ȡ�ɹ�
 * */
 gbdt_model_t* gbdt_load_model(char* model_file);
 
 /*
 * @brief �ͷ�Gradient Boosting Decision Treeģ��
 *
 * @param <IN> rf_model : gbdt_modelģ�͵�ָ��
 *
 * */
 
 int free_model(gbdt_model_t*& gbdt_model);
 
 void R_qsort_I(double *v, int *I, int i, int j);

