#define LOGURU_IMPLEMENTATION 1
#include "eigen_utils.h"

#include <numeric> //std::iota

using namespace easy_pbr::utils;

void vec2eigen_dynamic(){
    //makes a vector of elements and parses it into a Eigen::VectorXf

    std::vector<float> vec(10); //10 elements
    std::iota (std::begin(vec), std::end(vec), 0); // Fill with 0, 1, ..., 99.

    Eigen::VectorXf eigen_vec=vec2eigen(vec);

    for(int i=0; i<vec.size(); i++){
        if(eigen_vec(i)!=vec[i]){
            LOG(FATAL) << "At position " << i << " the value is not the same. The std::vector has " << vec[i] << " and the eigen vec has " << eigen_vec(i);
        }
    }
}

void vec2eigen_fixed(){
    //makes a vector of elements and parses it into a Eigen::Vector3f

    std::vector<float> vec(3); //3 elements
    std::iota (std::begin(vec), std::end(vec), 0); // Fill with 0, 1, ..., 99.

    Eigen::Vector3f eigen_vec=vec2eigen(vec);

    for(int i=0; i<vec.size(); i++){
        if(eigen_vec(i)!=vec[i]){
            LOG(FATAL) << "At position " << i << " the value is not the same. The std::vector has " << vec[i] << " and the eigen vec has " << eigen_vec(i);
        }
    }
}

void filter(){
    //make a matrix of 5x3 and remove 3 of the rows

    Eigen::MatrixXf mat;
    mat.resize(5,3);
    mat.setRandom();

    //the rows to be kept are specified in a vector of bools of size 5
    std::vector<bool> mask(5, false);
    mask[0]=true;
    mask[2]=true;
    mask[4]=true;


    Eigen::MatrixXf mat_filtered=filter(mat, mask, false); //keep the rows that have a false in the mask


    //we removed rows 0,2,4 check that the rows in filtered correspond to rows 1 and 3 in the original mat
    for(int i=0; i<mat_filtered.cols(); i++){
        if(mat_filtered(0,i)!=mat(1,i)){
            LOG(FATAL) << "At position (0," << i << ") the value is not the same. The original mat has " << mat(1,i) << " and the mat_filtered has " << mat_filtered(0,i);
        }
    }
    for(int i=0; i<mat_filtered.cols(); i++){
        if(mat_filtered(1,i)!=mat(3,i)){
            LOG(FATAL) << "At position (1," << i << ") the value is not the same. The original mat has " << mat(3,i) << " and the mat_filtered has " << mat_filtered(1,i);
        }
    }


}


void filter_and_get_indirection(){
    //make a matrix of 5x3 and remove 3 of the rows

    Eigen::MatrixXf mat;
    mat.resize(5,3);
    mat.setRandom();

    //the rows to be kept are specified in a vector of bools of size 5
    std::vector<bool> mask(5, false);
    mask[0]=true;
    mask[2]=true;
    mask[4]=true;


    std::vector<int> indirection; //will be created with size 5 because we have 5 rows
    Eigen::MatrixXf mat_filtered=filter_return_indirection(indirection, mat, mask, false); //keep the rows that have a false in the mask

    CHECK(indirection.size()==5) << "Indirection should have size the same as the nr of rows in the original eigen mat. So it should be 5 but it is " << indirection.size();

    //we removed rows 0,2,4 and therefore rows 1,3 ended up being rows 0 and 1 in the mat_filtered. The indirection should therefore be [-1,0,-1,1,-1]
    CHECK(indirection[0]==-1) << "Indirection vec is wrong";
    CHECK(indirection[1]==0) << "Indirection vec is wrong";
    CHECK(indirection[2]==-1) << "Indirection vec is wrong";
    CHECK(indirection[3]==1) << "Indirection vec is wrong";
    CHECK(indirection[4]==-1) << "Indirection vec is wrong";

}

void filter_and_get_inverse_indirection(){
    //make a matrix of 5x3 and remove 3 of the rows

    Eigen::MatrixXf mat;
    mat.resize(5,3);
    mat.setRandom();

    //the rows to be kept are specified in a vector of bools of size 5
    std::vector<bool> mask(5, false);
    mask[0]=true;
    mask[2]=true;
    mask[4]=true;


    std::vector<int> inverse_indirection; //will be created with size 2 because only 2 rows are left after filtering
    Eigen::MatrixXf mat_filtered=filter_return_inverse_indirection(inverse_indirection, mat, mask, false); //keep the rows that have a false in the mask

    CHECK(inverse_indirection.size()==2) << "Indirection should have size the same as the nr of rows in the filtered eigen mat. So it should be 2 but it is " << inverse_indirection.size();

    //we removed rows 0,2,4 and therefore rows 1,3 ended up being rows 0 and 1 in the mat_filtered. The inverse_indirection should therefore be [1,3]
    CHECK(inverse_indirection[0]==1) << "Inverse indirection vec is wrong";
    CHECK(inverse_indirection[1]==3) << "Inverse indirection vec is wrong";


}

void filter_and_get_both_indirections(){
    //make a matrix of 5x3 and remove 3 of the rows

    Eigen::MatrixXf mat;
    mat.resize(5,3);
    mat.setRandom();

    //the rows to be kept are specified in a vector of bools of size 5
    std::vector<bool> mask(5, false);
    mask[0]=true;
    mask[2]=true;
    mask[4]=true;


    std::vector<int> indirection; //will be created with size 5 because we have 5 rows
    std::vector<int> inverse_indirection; //will be created with size 2 because only 2 rows are left after filtering
    Eigen::MatrixXf mat_filtered=filter_return_both_indirection(indirection, inverse_indirection, mat, mask, false); //keep the rows that have a false in the mask

    CHECK(indirection.size()==5) << "Indirection should have size the same as the nr of rows in the original eigen mat. So it should be 5 but it is " << indirection.size();

    //we removed rows 0,2,4 and therefore rows 1,3 ended up being rows 0 and 1 in the mat_filtered. The indirection should therefore be [-1,0,-1,1,-1]
    CHECK(indirection[0]==-1) << "Indirection vec is wrong";
    CHECK(indirection[1]==0) << "Indirection vec is wrong";
    CHECK(indirection[2]==-1) << "Indirection vec is wrong";
    CHECK(indirection[3]==1) << "Indirection vec is wrong";
    CHECK(indirection[4]==-1) << "Indirection vec is wrong";

    CHECK(inverse_indirection.size()==2) << "Indirection should have size the same as the nr of rows in the filtered eigen mat. So it should be 2 but it is " << inverse_indirection.size();

    //we removed rows 0,2,4 and therefore rows 1,3 ended up being rows 0 and 1 in the mat_filtered. The inverse_indirection should therefore be [1,3]
    CHECK(inverse_indirection[0]==1) << "Inverse indirection vec is wrong";
    CHECK(inverse_indirection[1]==3) << "Inverse indirection vec is wrong";


}

void filter_and_apply_indirection(){
    //make a matrix of 5x3 and remove 3 of the rows

    Eigen::MatrixXf mat;
    mat.resize(5,3);
    mat.setRandom();

    //the rows to be kept are specified in a vector of bools of size 5
    std::vector<bool> mask(5, false);
    mask[0]=true;
    mask[2]=true;
    mask[4]=true;


    std::vector<int> indirection; //will be created with size 5 because we have 5 rows
    Eigen::MatrixXf mat_filtered=filter_return_indirection(indirection, mat, mask, false); //keep the rows that have a false in the mask

    //if there is another matrix that has indices pointing into mat, the indices are no longer valid when they point into mat_filtered. Changing the indices is done with apply_indirection()
    Eigen::MatrixXi indices_mat;
    indices_mat.resize(5,3);
    for(int i=0; i<5; i++){
        indices_mat(i,0)=i;
        indices_mat(i,1)=i; //do not index above 5 because we have only 5 rows
        indices_mat(i,2)=i; //do not index above 5 because we have only 5 rows
    }

    //if any of the indexes point into a row that was removed, the row of indexes is removed. This is useful for reindexing faces of a triangular mesh after removing vertices with filter()
    Eigen::MatrixXi indices_mat_filtered;
    indices_mat_filtered=filter_apply_indirection(indirection, indices_mat);

    CHECK(indices_mat_filtered.rows()==2) << "Indices mat filtered should have 2 rows. However it has " << indices_mat_filtered.rows();

    //they should index only up until 2 now because we only have 2 rows left in the matrix
    int max=indices_mat_filtered.maxCoeff();
    CHECK(max==1) << "Max index should be 1 because we can only index the row 0 and 1. However, the max is " << max;


}


int main(int argc, char *argv[]) {

    vec2eigen_dynamic();
    vec2eigen_fixed();
    filter();
    filter_and_get_indirection();
    filter_and_get_inverse_indirection();
    filter_and_get_both_indirections();
    filter_and_apply_indirection();



    return 0;

}
