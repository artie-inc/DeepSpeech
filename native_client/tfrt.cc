#include "tfrt.h"
#include <chrono>
#include <tensorrt>


using namespace tensorflow;
using std::vector;
using namespace std::chrono;
using namespace std;


TFModelState::TFModelState()
  : ModelState()
  , mmap_env_(nullptr)
  , session_(nullptr)
{
}



int
TFModelState::init(const char* model_path,
                   unsigned int n_features,
                   unsigned int n_context,
                   const char* alphabet_path,
                   unsigned int beam_width)
{
  int err = ModelState::init(model_path, n_features, n_context, alphabet_path, beam_width);
  if (err != DS_ERR_OK) {
    return err;
  }

  IBuilder* builder = createInferBuilder(gLogger);
  INetworkDefinition* network = builder->createNetwork();
  IUFFParser* parser = createUffParser();
  parser->registerInput("Input_0", DimsCHW(1, 28, 28), UffInputOrder::kNCHW);
  parser->registerOutput("Binary_3");
  parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT);
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(1 << 20);
  ICudaEngine* engine = builder->buildCudaEngine(*network);

  IHostMemory *serializedModel = engine->serialize();
  // store model to disk
  // <…>
  serializedModel->destroy();


  parser->destroy();
  network->destroy();
  builder->destroy();
}


int createEngine()
{
  IRuntime* runtime = createInferRuntime(gLogger);
  ICudaEngine* engine = runtime->deserializeCudaEngine(modelData, modelSize, nullptr);

}

int infer() {
  IExecutionContext *context = engine->createExecutionContext();
  int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
  int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

  void* buffers[2];
  buffers[inputIndex] = inputbuffer;
  buffers[outputIndex] = outputBuffer;

  // The final argument to enqueue() is an optional CUDA event which will be signaled 
  // when the input buffers have been consumed and their memory may be safely reused.
  context->enqueue(batchSize, buffers, stream, nullptr);
}