# Go Bindings for Caffe2 ( deployed on mobile devices using gomobile )

## Description

The purpose of this repository is to provide a common predictor codebase to enable deployment of inference models on both Android and IOS applications. We achieve this through [gomobile](https://github.com/golang/mobile) - a command line tool which generates Java and Objective-C bindings of a golang codebase. The generated binding is then deployed as part of the Android/IOS applications as an external library.

## API Functions

New()
```
 func New(model string, mode, batch int) (*PredictorData, error) {}
```

SetUseX() (X=CPU/GPU/DSP/VisualCore)
```
CPUMode = 0
GPUMode = 1
DSPMode = 2
VisualCore = 3

func SetUseCPU() {}
```

Predict()
```
func Predict(p *PredictorData, data []byte) error {}
```

ReadPredictionOutput()
```
func ReadPredictionOutput(p *PredictorData, labelFile string) (string, error) {}
```

Close()
```
func Close(p *PredictorData) {}
```

## Worklow

1. Add caffe2 as a library dependency to your mobile application
2. Add go-caffe2 binding as a library dependency to your mobile application
3. Add model and labels file to the application ensuring that the filepaths are accessible expilcitly
4. Create a new predictor using New()
5. Perform inference using Predict()
6. Read Top-1 predicted label using ReadPredictionOutput()
7. Delete predictor using Close()

## Caffe2 Installation

## Gomobile

## Use Other Library Paths

## Demo applications


