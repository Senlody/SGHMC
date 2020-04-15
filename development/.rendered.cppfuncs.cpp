
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <functional>

namespace py = pybind11;
using std::default_random_engine;
using std::normal_distribution;
using std::bind;
using namespace Eigen;
    
//#@jit('double[:](double[:],double[:],double,double[:,:],double[:,:],int32)')
//#def update_r(r,gU,eps,fric,sqrt_noise,p):
//#    return r - eps*gU - fric@r + sqrt_noise@np.random.normal(size=(p))

default_random_engine re{};
normal_distribution<double> norm(0,1);
auto rnorm = bind(norm, re);

VectorXd update_r(VectorXd r, VectorXd gU,double eps,MatrixXd fric,MatrixXd sqrt_noise){
    int p = r.rows();
    VectorXd noise = VectorXd::Zero(p).unaryExpr([](double x){ return rnorm();});
    return r.array() - eps*gU.array() - (fric*r).array() + (sqrt_noise*noise).array();
}

PYBIND11_MODULE(cppfuncs, m) {
    m.def("update_r", &update_r);
}