#pragma once

#include <vector>
#include <fstream>
//https://stackoverflow.com/a/217605
#include <algorithm>
#include <functional>

#include <Eigen/Geometry>
#include <Eigen/StdVector>


// //loguru
#include <loguru.hpp>


typedef std::vector<float> row_type_f;
typedef std::vector<row_type_f> matrix_type_f;

typedef std::vector<double> row_type_d;
typedef std::vector<row_type_f> matrix_type_d;

typedef std::vector<int> row_type_i;
typedef std::vector<row_type_i> matrix_type_i;

typedef std::vector<bool> row_type_b;
typedef std::vector<row_type_b> matrix_type_b;

namespace radu{
namespace utils{

// //append new_mat into a preallocated mat and check if we have enough size to append there
// template<class T>
// inline void eigen_append_in_preallocated(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& new_mat){

// }


inline void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

inline void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

inline void eigen2file(Eigen::MatrixXd& src, std::string pathAndName)
{
      std::ofstream fichier(pathAndName);
      if(fichier.is_open())  // si l'ouverture a réussi
      {
        // instructions
        fichier << src << "\n";
        fichier.close();  // on referme le fichier
      }
      else  // sinon
      {
        LOG(FATAL) << "Error opening " << pathAndName;
      }
 }

//when using dyanmic vector we don't need an eigen alocator
template<class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> vec2eigen( const std::vector<  Eigen::Matrix<T, Eigen::Dynamic, 1>>& std_vec )
{
    if(std_vec.size()==0){
        //return empty eigen mat
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec;
        return eigen_vec;
    }

    const int dim=std_vec[0].size();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec(std_vec.size(),dim);
    for (size_t i = 0; i < std_vec.size(); ++i) {
        eigen_vec.row(i)=std_vec[i];
    }
    return eigen_vec;

}


//for the case of using fixed sized vector like Vector3f, it also works with dynamic types if you write the template explicitly
template<class T, int rows >
// Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> vec2eigen( const std::vector<  Eigen::Matrix<T, rows, 1>, Eigen::aligned_allocator< Eigen::Matrix<T, rows, 1>>   >& std_vec )
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> vec2eigen( const std::vector<  Eigen::Matrix<T, rows, 1>>   & std_vec )
{
    if(std_vec.size()==0){
        //return empty eigen mat
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec;
        return eigen_vec;
    }

    const int dim=std_vec[0].size();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec(std_vec.size(),dim);
    for (size_t i = 0; i < std_vec.size(); ++i) {
        eigen_vec.row(i)=std_vec[i];
    }
    return eigen_vec;

}


//for the case of using a vector of double of floats or whatever numeric type. It will only take numeric types and not anything else https://stackoverflow.com/a/14294277
template<
    class T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> vec2eigen( const std::vector< T >& std_vec )
{
    if(std_vec.size()==0){
        //return empty eigen mat
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec;
        return eigen_vec;
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec(std_vec.size(),1);
    for (size_t i = 0; i < std_vec.size(); ++i) {
        eigen_vec(i)=std_vec[i];
    }
    return eigen_vec;

}


//when using dyanmic vector we don't need an eigen alocator
template<class T>
std::vector< T > eigen2vec( const Eigen::Matrix<T, Eigen::Dynamic, 1>& eigen )
{
    std::vector< T > std_vec(eigen.rows());

    for (int i = 0; i < eigen.rows(); ++i) {
        std_vec[i] =  eigen(i);
    }

    return std_vec;

}


//filters the rows of an eigen matrix and returns only those for which the mask is equal to the keep
template <class T>
inline T filter_impl(std::vector<int>&indirection, std::vector<int>&inverse_indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks ){
    if(eigen_mat.rows()!=(int)mask.size() || eigen_mat.size()==0 ){
        LOG_IF_S(WARNING, do_checks) << "filter: Eigen matrix and mask don't have the same size: " << eigen_mat.rows() << " and " << mask.size() ;
        return eigen_mat;
    }

    int nr_elem_to_keep=std::count(mask.begin(), mask.end(), keep);
    T new_eigen_mat( nr_elem_to_keep, eigen_mat.cols() );

    indirection.resize(eigen_mat.rows(),-1);
    inverse_indirection.resize(nr_elem_to_keep, -1);


    int insert_idx=0;
    for (int i = 0; i < eigen_mat.rows(); ++i) {
        if(mask[i]==keep){
            new_eigen_mat.row(insert_idx)=eigen_mat.row(i);
            indirection[i]=insert_idx;
            inverse_indirection[insert_idx]=i;
            insert_idx++;
        }
    }

    return new_eigen_mat;

}

template <class T>
inline T filter( const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks=true){

    std::vector<int> indirection;
    std::vector<int> inverse_indirection;
    return filter_impl(indirection, inverse_indirection, eigen_mat, mask, keep, do_checks);

}

template <class T>
inline T filter_return_indirection(std::vector<int>&indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks=true){

    std::vector<int> inverse_indirection;
    return filter_impl(indirection, inverse_indirection, eigen_mat, mask, keep, do_checks);

}

template <class T>
inline T filter_return_inverse_indirection(std::vector<int>&inverse_indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks=true ){

    std::vector<int> indirection;
    return filter_impl(indirection, inverse_indirection, eigen_mat, mask, keep, do_checks);

}

template <class T>
inline T filter_return_both_indirection(std::vector<int>&indirection, std::vector<int>&inverse_indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks=true ){

    return filter_impl(indirection, inverse_indirection, eigen_mat, mask, keep, do_checks);

}

//gets rid of the faces that are redirected to a -1 or edges that are also indirected into a -1
inline Eigen::MatrixXi filter_apply_indirection(const std::vector<int>&indirection, const Eigen::MatrixXi& eigen_mat ){

    if(!eigen_mat.size()){  //it's empty
        return eigen_mat;
    }

    if(eigen_mat.maxCoeff() > (int)indirection.size()){
        LOG(FATAL) << "filter apply_indirection: eigen_mat is indexing indirection at a higher position than allowed" << eigen_mat.maxCoeff() << " " << indirection.size();
    }

    std::vector<Eigen::VectorXi> new_eigen_mat_vec;

    for (int i = 0; i < eigen_mat.rows(); ++i) {

        Eigen::VectorXi row= eigen_mat.row(i);
        bool should_keep=true;
        for (int j = 0; j < row.size(); ++j) {
            if (indirection[row(j)]==-1){
                //it points to a an already removed point so we will not keep it
                should_keep=false;
            }else{
                //it point to a valid vertex so we change the idx so that it point to that one
                row(j) = indirection[row(j)];
            }
        }

        if(should_keep){
            new_eigen_mat_vec.push_back(row);
        }

    }

    return vec2eigen(new_eigen_mat_vec);

}

//gets rid of the faces that are redirected to a -1 or edges that are also indirected into a -1 AND also returns a mask (size eigen_mat x 1) with value of TRUE for those which were keps
inline Eigen::MatrixXi filter_apply_indirection_return_mask(std::vector<bool>& mask_kept, const std::vector<int>&indirection, const Eigen::MatrixXi& eigen_mat ){

    if(!eigen_mat.size()){  //it's empty
        return eigen_mat;
    }

    if(eigen_mat.maxCoeff() > (int)indirection.size()){
        LOG(FATAL) << "filter apply_indirection: eigen_mat is indexing indirection at a higher position than allowed" << eigen_mat.maxCoeff() << " " << indirection.size();
    }

    std::vector<Eigen::VectorXi> new_eigen_mat_vec;
    mask_kept.resize(eigen_mat.rows(),false);

    for (int i = 0; i < eigen_mat.rows(); ++i) {
        // LOG_IF_S(INFO,debug) << "getting row " << i;
        Eigen::VectorXi row= eigen_mat.row(i);
        // LOG_IF_S(INFO,debug) << "row is" << row;
        bool should_keep=true;
        for (int j = 0; j < row.size(); ++j) {
            // LOG_IF_S(INFO,debug) << "value at column " << j << " is " << row(j);
            // LOG_IF_S(INFO,debug) << "indirection has size " << indirection.size();
            // LOG_IF_S(INFO,debug) << "value of indirection at is  " <<  indirection[row(j)];
            if (indirection[row(j)]==-1){
                //it points to a an already removed point so we will not keep it
                should_keep=false;
            }else{
                //it point to a valid vertex so we change the idx so that it point to that one
                row(j) = indirection[row(j)];
            }
        }

        if(should_keep){
            // LOG_IF_S(INFO,debug) << "pushing new row " <<  row;
            new_eigen_mat_vec.push_back(row);
            // LOG_IF_S(INFO,debug) << "setting face " << i << " to kept";
            mask_kept[i]=true;
        }

    }

    // LOG_IF_S(INFO,debug) << "finished, doing a vec2eigen ";
    return vec2eigen(new_eigen_mat_vec);

}



//filters that does not actually remove the points, but just sets them to zero
template <class T>
inline T filter_set_zero_impl(std::vector<int>&indirection, std::vector<int>&inverse_indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks ){
    if(eigen_mat.rows()!=(int)mask.size() || eigen_mat.size()==0 ){
        LOG_IF_S(WARNING, do_checks) << "filter: Eigen matrix and mask don't have the same size: " << eigen_mat.rows() << " and " << mask.size() ;
        return eigen_mat;
    }

    T new_eigen_mat( eigen_mat.rows(), eigen_mat.cols() );
    new_eigen_mat.setZero();

    indirection.resize(eigen_mat.rows(),-1);
    inverse_indirection.resize(eigen_mat.rows(), -1);


    for (int i = 0; i < eigen_mat.rows(); ++i) {
        if(mask[i]==keep){
            new_eigen_mat.row(i)=eigen_mat.row(i);
            indirection[i]=i;
            inverse_indirection[i]=i;
        }
    }

    return new_eigen_mat;

}

//sets the corresponding rows to zero instead of removing them
template <class T>
inline T filter_set_zero( const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks=true){

    std::vector<int> indirection;
    std::vector<int> inverse_indirection;
    return filter_set_zero_impl(indirection, inverse_indirection, eigen_mat, mask, keep, do_checks);

}

template <class T>
inline T filter_set_zero_return_indirection(std::vector<int>&indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks=true){

    std::vector<int> inverse_indirection;
    return filter_set_zero_impl(indirection, inverse_indirection, eigen_mat, mask, keep, do_checks);

}



template <class T>
inline T concat(const T& mat_1, const T& mat_2){

    if(mat_1.cols()!=mat_2.cols() && mat_1.cols()!=0 && mat_2.cols()!=0){
        LOG(FATAL) << "concat: Eigen matrices don't have the same nr of columns: " << mat_1.cols() << " and " << mat_2.cols() ;
    }


    T mat_new(mat_1.rows() + mat_2.rows(), mat_1.cols());
    mat_new << mat_1, mat_2;
    return mat_new;
}


//transform from a vector of xyz and qx,qy,qz,qw into a full Affine3d
template <class T>
inline Eigen::Transform<T,3,Eigen::Affine>  tf_vec2matrix(const Eigen::Matrix< T, Eigen::Dynamic, 1 >& tf_vec){

    CHECK(tf_vec.rows()==7) << "Expecting 7 elements corresponding to x,y,z, qx,qy,qz,qw";

    Eigen::Transform<T,3,Eigen::Affine> tf;
    tf.setIdentity();
    tf.translation().x() = tf_vec[0];
    tf.translation().y() = tf_vec[1];
    tf.translation().z() = tf_vec[2];
    Eigen::Quaternion<T> q;
    q.x() = tf_vec[3];
    q.y() = tf_vec[4];
    q.z() = tf_vec[5];
    q.w() = tf_vec[6];
    q.normalize();
    tf.linear()=q.toRotationMatrix();

    return tf;
}
//transform from a vector of strings xyz and qx,qy,qz,qw into a full Affine3d
template <class T>
inline Eigen::Transform<T,3,Eigen::Affine>  tf_vecstring2matrix(const std::vector<std::string>& tokens ){

    CHECK(tokens.size()==7) << "Expecting 7 elements corresponding to x,y,z, qx,qy,qz,qw";

    Eigen::Transform<T,3,Eigen::Affine> tf;
    tf.setIdentity();
    tf.translation().x() = std::stod(tokens[0]);
    tf.translation().y() = std::stod(tokens[1]);
    tf.translation().z() = std::stod(tokens[2]);
    Eigen::Quaternion<T> q;
    q.x() = std::stod(tokens[3]);
    q.y() = std::stod(tokens[4]);
    q.z() = std::stod(tokens[5]);
    q.w() = std::stod(tokens[6]);
    q.normalize();
    tf.linear()=q.toRotationMatrix();

    return tf;
}
//transform from a Affine3d into a vec containing x,y,z, qx,qy,qz,qw
template <class T>
inline Eigen::Matrix< T, Eigen::Dynamic, 1 >  tf_matrix2vec(const Eigen::Transform<T,3,Eigen::Affine> & tf){

    Eigen::Quaternion<T> q (tf.linear());
    auto t = tf.translation();

    Eigen::Matrix< T, Eigen::Dynamic, 1 > vec;
    vec.resize(7);
    vec[0]=t.x();
    vec[1]=t.y();
    vec[2]=t.z();
    vec[3]=q.x();
    vec[4]=q.y();
    vec[5]=q.z();
    vec[6]=q.w();

    return vec;
}
//transform from a Affine3d into a vector of strings containing x,y,z, qx,qy,qz,qw
template <class T>
inline std::vector<std::string>  tf_matrix2vecstring(const Eigen::Transform<T,3,Eigen::Affine> & tf){

    Eigen::Matrix< T, Eigen::Dynamic, 1 > vec;
    vec=tf_matrix2vec(tf);

    std::vector<std::string> tokens;
    for (int i=0; i<vec.rows(); i++){
        std::string num_string=std::to_string(vec(i));
        tokens.push_back(num_string);
    }


    return tokens;
}

//transform from a vector of fx,fy,cx,cy into a full Matrix3d
template <class T>
inline Eigen::Matrix<T,3,3>  K_vec2matrix(const Eigen::Matrix< T, Eigen::Dynamic, 1 >& K_vec){

    CHECK(K_vec.rows()==4) << "Expecting 4 elements corresponding to fx,fy,cx,cy";

    Eigen::Matrix<T,3,3> K;
    K.setIdentity();
    K(0,0)= K_vec[0];
    K(1,1)= K_vec[1];
    K(0,2)= K_vec[2];
    K(1,2)= K_vec[3];

    return K;
}
//transform from a  Matrix3d
template <class T>
inline Eigen::Matrix< T, Eigen::Dynamic, 1 >  K_matrix2vec(const Eigen::Matrix<T,3,3>& K){

    Eigen::Matrix< T, Eigen::Dynamic, 1 > K_vec;
    K_vec.resize(4);
    K_vec(0)=K(0,0);
    K_vec(1)=K(1,1);
    K_vec(2)=K(0,2);
    K_vec(3)=K(1,2);

    return K_vec;
}

//from a series of tokens, fills a martrix, assuming row major;
template <typename T, int R, int C>
inline void  tokens2matrix(const std::vector<std::string> tokens, Eigen::Matrix<T,R,C>& matrix, bool rowmajor=true){

    size_t matrix_size=matrix.rows()*matrix.cols();
    CHECK(matrix_size==tokens.size()) << "The nr of tokens does not correspond to the matrix size. Tokens has size " << tokens.size() << " matrix has rows and cols " << matrix.rows() <<" x " << matrix.cols();


    //get the tokens into an vector;
    std::vector<T> array;
    for(size_t i=0;i<tokens.size();i++){
        array.push_back( std::stod(tokens[i]) );
    }

    if(rowmajor){
        matrix = Eigen::Map<const Eigen::Matrix<T,R,C,Eigen::RowMajor> >(array.data() );
    }else{
        matrix = Eigen::Map<const Eigen::Matrix<T,R,C> >(array.data() );
    }


}
// //in the case of a dynamic matrix we need to provide a size estimate to the MAP https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
//https://eigen.tuxfamily.org/dox/classEigen_1_1Map.html
template <typename T>
inline void  tokens2matrix(const std::vector<std::string> tokens, Eigen::Matrix<T,Eigen::Dynamic,1>& matrix ){

    size_t matrix_size=matrix.rows()*matrix.cols();
    CHECK(matrix_size==tokens.size()) << "The nr of tokens does not correspond to the matrix size. Tokens has size " << tokens.size() << " matrix has rows and cols " << matrix.rows() <<" x " << matrix.cols();

    //get the tokens into an vector;
    std::vector<T> array;
    for(size_t i=0;i<tokens.size();i++){
        array.push_back( std::stod(tokens[i]) );
    }

    matrix = Eigen::Map<const Eigen::Matrix<T,Eigen::Dynamic,1> >(array.data(), tokens.size());

}



//has function for eigen matrices and vectors  https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
template<typename T>
struct MatrixHash : std::unary_function<T, size_t> {
    std::size_t operator()(T const& matrix) const {
        // Note that it is oblivious to the storage order of Eigen matrix (column- or
        // row-major). It will give you the same hash value for two different matrices if they
        // are the transpose of each other in different storage order.
        size_t seed = 0;
        for (int i = 0; i < matrix.size(); ++i) {
        auto elem = *(matrix.data() + i);
        seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};


} //namespace utils
} //namespace radu
