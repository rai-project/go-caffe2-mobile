#define _GLIBCXX_USE_CXX11_ABI 0

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include <caffe2/core/common.h>
#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/core/observer.h>
#include <caffe2/core/operator.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/core/types.h>
//#include <caffe2/onnx/backend.h>
//#include <caffe2/onnx/backend_rep.h>

#include <caffe2/proto/caffe2.pb.h>

#include <caffe2/core/tensor.h>

#include "predictor.hpp"

#if 0
#define DEBUG_STMT std ::cout << __func__ << "  " << __LINE__ << "\n";
#else
#define DEBUG_STMT
#endif

using namespace caffe2;
using std::string;

/* Pair (label, confidence) representing a prediction. */
using Prediction = std::pair<int, float>;

/*
  Predictor class takes in model files (init_net.pb and pred_net.pb) batch size and device mode for inference
*/
class Predictor {
	public:
		Predictor(NetDef *init_net, NetDef *net_def, int batch, int mode);
		void Predict(float* inputData, std::string input_type, const int batch, const int channels, const int width, const int height);

		Workspace *ws_{nullptr};
		NetBase *net_;
		std::vector<string> input_names_;
		std::vector<string> output_names_;
		int width_, height_, channels_;
		int batch_ = 1;
		int pred_len_;
		int mode_ = 0;
		void *result_{nullptr};
};

Predictor::Predictor(NetDef *init_net, NetDef *net_def, int batch, int mode) {
	/* Load the network. */
	ws_ = new Workspace();
	mode_ = mode;
	ws_->RunNetOnce(*init_net);
	for(auto in : net_def->external_input()) {

		auto* blob = ws_->GetBlob(in);
		if(!blob) {
			ws_->CreateBlob(in);	
		}
		input_names_.emplace_back(in);
	}
	for(auto out : net_def->external_output()) {

		auto* blob = ws_->GetBlob(out);
		if(!blob) {
			ws_->CreateBlob(out);
		}
		output_names_.emplace_back(out);
	}
	if(!net_def->has_name()) {
		net_def->set_name("go-caffe2");	
	}
	net_ = ws_->CreateNet(*net_def);

	assert(net_ != nullptr);
	mode_ = mode;
	batch_ = batch;
}

void Predictor::Predict(float* inputData, std::string input_type, const int batch, const int channels, const int width, const int height) {

	if(result_ != nullptr) {
		free(result_);
		result_ = nullptr;
	}

	batch_ = batch;
	const auto data_size = batch_ * channels * width * height;

	std::vector<float> data(data_size);
	std::copy(inputData, inputData + data_size, data.begin());

	std::vector<int64_t> dims({batch_, channels, width, height});

	auto input_name = input_names_[0];
	auto *blob = ws_->GetBlob(input_name);
	if(blob == nullptr) {
		blob = ws_->CreateBlob(input_name);
	}

	// using CPU by default
	auto tensor = BlobGetMutableTensor(blob, caffe2::CPU);
	tensor->Resize(dims);
	tensor->ShareExternalPointer(data.data());

	if(!net_->Run()) {
		throw std::runtime_error("invalid run");
	}

	auto output_name = output_names_[0];
	auto *output_blob = ws_->GetBlob(output_name);
	if(output_blob == nullptr) {
		throw std::runtime_error("output blob does not exist");	
	}
	auto output_tensor = output_blob->Get<TensorCPU>().Clone();
	pred_len_ = output_tensor.size() / batch_;
	result_ = (void *)malloc(output_tensor.nbytes());
	memcpy(result_, output_tensor.raw_data(), output_tensor.nbytes());

}

static void set_operator_engine(NetDef *net, int mode) {
	// using CPU as of now
	// need to enable usage of different accelerators underneath
	//net->mutable_device_option()->set_device_type(TypeToProto(mode));
	net->mutable_device_option()->set_device_type(TypeToProto(DeviceType::CPU));
  
	for (int i = 0; i < net->op_size(); i++) {
		caffe2::OperatorDef *op_def = net->mutable_op(i);
		//op_def->mutable_device_option()->set_device_type(TypeToProto(mode));
		op_def->mutable_device_option()->set_device_type(TypeToProto(DeviceType::CPU));
  }
}

PredictorContext NewCaffe2(char *init_net_file, char *pred_net_file, int batch,
                          int mode) {
	try {
		int mode_temp = 0;
		if (mode == 1) mode_temp = 1;
		NetDef init_net, pred_net;
		if(!ReadProtoFromFile(init_net_file, &init_net)) {
			throw std::runtime_error("cannot read init net file");	
		}
		set_operator_engine(&init_net, mode);
		if(!ReadProtoFromFile(pred_net_file, &pred_net)) {
			throw std::runtime_error("cannot read pred net file");	
		}
		set_operator_engine(&pred_net, mode);
		const auto ctx = new Predictor(&init_net, &pred_net, batch,
                                   mode_temp);
		return (void *)ctx;
  } catch (const std::invalid_argument &ex) {
		errno = EINVAL;
		return nullptr;
  }

}

void SetModeCaffe2(int mode) {
	if(mode == 1) {
		// Do nothing as of now
	}
}

void InitCaffe2() {}

void PredictCaffe2(PredictorContext pred, float* inputData, const char* input_type, const int batch, const int channels, const int width, const int height) {
	auto predictor = (Predictor *)pred;
	if (predictor == nullptr) {
		return;
	}
	
	predictor->Predict(inputData, input_type, batch, channels, width, height);
	return;
}

const float*GetPredictionsCaffe2(PredictorContext pred) {
	auto predictor = (Predictor *)pred;
	if (predictor == nullptr) {
		return nullptr;
	}
	if(predictor->result_ == nullptr) {
		throw std::runtime_error("expected a non-nil result");	
	}
	return (float*)predictor->result_;
}

void DeleteCaffe2(PredictorContext pred) {
	auto predictor = (Predictor *)pred;
	if (predictor == nullptr) {
		return;
	}
	if(predictor->ws_ != nullptr) {
		delete predictor->ws_;	
	}
	if(predictor->result_) {
		free(predictor->result_);	
	}
	delete predictor;
}

int GetWidthCaffe2(PredictorContext pred) {
	auto predictor = (Predictor *)pred;
	if (predictor == nullptr) {
		return 0;
	}
	return predictor->width_;
}

int GetHeightCaffe2(PredictorContext pred) {
	auto predictor = (Predictor *)pred;
	if (predictor == nullptr) {
		return 0;
	}
	return predictor->height_;
}

int GetChannelsCaffe2(PredictorContext pred) {
	auto predictor = (Predictor *)pred;
	if (predictor == nullptr) {
		return 0;
	}
	return predictor->channels_;
}

int GetPredLenCaffe2(PredictorContext pred) {
	auto predictor = (Predictor *)pred;
	if (predictor == nullptr) {
		return 0;
	}
	return predictor->pred_len_;
}

