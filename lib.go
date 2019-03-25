package caffe2

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -g -Wno-sign-compare -Wno-unused-function  -I/Users/abhiutd/workspace/scratch/caffe2/pytorch -I/Users/abhiutd/workspace/scratch/caffe2/pytorch/aten/src -I/Users/abhiutd/workspace/scratch/caffe2/pytorch/build_android -I/Users/abhiutd/workspace/scratch/caffe2/pytorch/build_host_protoc/include
// #cgo LDFLAGS: -lstdc++ -L/Users/abhiutd/workspace/scratch/android-ndk -llog -L/Users/abhiutd/workspace/scratch/caffe2/pytorch/build_android/lib -lcaffe2_detectron_ops -lonnxifi_dummy -lcaffe2 -lcaffe2_protos -lcpuinfo -lnnpack -lnnpack_reference_layers -lqnnpack -lpthreadpool -lprotobuf
import "C"
