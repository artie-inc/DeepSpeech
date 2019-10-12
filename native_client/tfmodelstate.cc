#include "tfmodelstate.h"

#include "workspace_status.h"

#include "tensorflow/core/kernels/batching_util/basic_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/shared_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include  "tensorflow/core/kernels/batching_util/periodic_function.h"

#include "tensorflow_serving/batching/batching_session.h"

using namespace tensorflow;
// using namespace tensorflow::serving;

using std::vector;

TFModelState::TFModelState()
  : ModelState()
  , mmap_env_(nullptr)
  , session_(nullptr)
{
}

TFModelState::~TFModelState()
{
  if (session_) {
    Status status = session_->Close();
    if (!status.ok()) {
      std::cerr << "Error closing TensorFlow session: " << status << std::endl;
    }
  }
  delete mmap_env_;
}

class FakeTask : public tensorflow::serving::BatchTask {
 public:
  explicit FakeTask(size_t size) : size_(size) {}

  ~FakeTask() override = default;

  size_t size() const override { return size_; }

  private:
  const size_t size_;

  // TF_DISALLOW_COPY_AND_ASSIGN(FakeTask);
};



int
TFModelState::init(const char* model_path,
                   unsigned int beam_width)
{
  int err = ModelState::init(model_path, beam_width);
  if (err != DS_ERR_OK) {
    return err;
  }

  Status status;
  SessionOptions options;

  mmap_env_ = new MemmappedEnv(Env::Default());

  bool is_mmap = std::string(model_path).find(".pbmm") != std::string::npos;
  if (!is_mmap) {
    std::cerr << "Warning: reading entire model file into memory. Transform model file into an mmapped graph to reduce heap usage." << std::endl;
  } else {
    status = mmap_env_->InitializeFromFile(model_path);
    if (!status.ok()) {
      std::cerr << status << std::endl;
      return DS_ERR_FAIL_INIT_MMAP;
    }

    options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(::OptimizerOptions::L0);
    options.env = mmap_env_;
  }

  // session_ = tfSession_.get();


  status = NewSession(options, &session_);

  // session_ = tfSession_.get();
  tfSession_ = std::unique_ptr<tensorflow::Session>(session_);


  if (!status.ok()) {
    std::cerr << status << std::endl;
    return DS_ERR_FAIL_INIT_SESS;
  }
  std::cout << "TFModelState::init() created NewSession" << " typeid=" << typeid(session_).name() << std::endl;

  tensorflow::serving::BasicBatchScheduler<tensorflow::serving::BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;  // fits two 2-unit tasks
  // schedule_options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
  schedule_options.batch_timeout_micros = 100;  // won't trigger
  // schedule_options.num_batch_threads = 1;
  schedule_options.num_batch_threads = 8;
  
  // std::unique_ptr<Session> tfSession(session_);
  
  // auto tfSession = std::make_unique<Session>(&session_);

  tensorflow::serving::TensorSignature signature = {
      {"input_node", "input_lengths", "previous_state_c", "previous_state_h"},
      {"logits", "new_state_c", "new_state_h"} 
  };

  // std::unique_ptr<Session> batching_session;
  tensorflow::serving::BatchingSessionOptions batching_session_options;
  std::cout << "TFModelState::init() pushing1" << std::endl;
  //batching_session_options.allowed_batch_sizes.push_back(1);
  std::cout << "TFModelState::init() pushing2" << std::endl;
  batching_session_options.allowed_batch_sizes.push_back(2);
  std::cout << "TFModelState::init() pushing3" << std::endl;
  batching_session_options.allowed_batch_sizes.push_back(3);
  std::cout << "TFModelState::init() pushing4" << std::endl;  
  batching_session_options.allowed_batch_sizes.push_back(4);
  std::cout << "TFModelState::init() pushing DONE" << std::endl;
  tensorflow::serving::CreateBasicBatchingSession(schedule_options, 
      batching_session_options, signature, std::move(tfSession_), &batching_session);

  std::cout << "TFModelState::init() created BatchingSession\n";
// {{"x"}, {"y"}}



  if (is_mmap) {
    status = ReadBinaryProto(mmap_env_,
                             MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
                             &graph_def_);
  } else {
    status = ReadBinaryProto(Env::Default(), model_path, &graph_def_);
  }
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return DS_ERR_FAIL_READ_PROTOBUF;
  }

  status = session_->Create(graph_def_);
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return DS_ERR_FAIL_CREATE_SESS;
  }

  std::vector<tensorflow::Tensor> version_output;
  status = session_->Run({}, {
    "metadata_version"
  }, {}, &version_output);
  if (!status.ok()) {
    std::cerr << "Unable to fetch graph version: " << status << std::endl;
    return DS_ERR_MODEL_INCOMPATIBLE;
  }

  int graph_version = version_output[0].scalar<int>()();
  if (graph_version < ds_graph_version()) {
    std::cerr << "Specified model file version (" << graph_version << ") is "
              << "incompatible with minimum version supported by this client ("
              << ds_graph_version() << "). See "
              << "https://github.com/mozilla/DeepSpeech/blob/"
              << ds_git_version() << "/doc/USING.rst#model-compatibility "
              << "for more information" << std::endl;
    return DS_ERR_MODEL_INCOMPATIBLE;
  }

  std::vector<tensorflow::Tensor> metadata_outputs;
  status = session_->Run({}, {
    "metadata_sample_rate",
    "metadata_feature_win_len",
    "metadata_feature_win_step",
    "metadata_alphabet",
  }, {}, &metadata_outputs);
  if (!status.ok()) {
    std::cout << "Unable to fetch metadata: " << status << std::endl;
    return DS_ERR_MODEL_INCOMPATIBLE;
  }

  sample_rate_ = metadata_outputs[0].scalar<int>()();
  int win_len_ms = metadata_outputs[1].scalar<int>()();
  int win_step_ms = metadata_outputs[2].scalar<int>()();
  audio_win_len_ = sample_rate_ * (win_len_ms / 1000.0);
  audio_win_step_ = sample_rate_ * (win_step_ms / 1000.0);

  string serialized_alphabet = metadata_outputs[3].scalar<string>()();
  err = alphabet_.deserialize(serialized_alphabet.data(), serialized_alphabet.size());
  if (err != 0) {
    return DS_ERR_INVALID_ALPHABET;
  }

  assert(sample_rate_ > 0);
  assert(audio_win_len_ > 0);
  assert(audio_win_step_ > 0);

  for (int i = 0; i < graph_def_.node_size(); ++i) {
    NodeDef node = graph_def_.node(i);
    if (node.name() == "input_node") {
      const auto& shape = node.attr().at("shape").shape();
      n_steps_ = shape.dim(1).size();
      n_context_ = (shape.dim(2).size()-1)/2;
      n_features_ = shape.dim(3).size();
      mfcc_feats_per_timestep_ = shape.dim(2).size() * shape.dim(3).size();
    } else if (node.name() == "previous_state_c") {
      const auto& shape = node.attr().at("shape").shape();
      state_size_ = shape.dim(1).size();
    } else if (node.name() == "logits_shape") {
      Tensor logits_shape = Tensor(DT_INT32, TensorShape({3}));
      if (!logits_shape.FromProto(node.attr().at("value").tensor())) {
        continue;
      }

      int final_dim_size = logits_shape.vec<int>()(2) - 1;
      if (final_dim_size != alphabet_.GetSize()) {
        std::cerr << "Error: Alphabet size does not match loaded model: alphabet "
                  << "has size " << alphabet_.GetSize()
                  << ", but model has " << final_dim_size
                  << " classes in its output. Make sure you're passing an alphabet "
                  << "file with the same size as the one used for training."
                  << std::endl;
        return DS_ERR_INVALID_ALPHABET;
      }
    }
  }

  if (n_context_ == -1 || n_features_ == -1) {
    std::cerr << "Error: Could not infer input shape from model file. "
              << "Make sure input_node is a 4D tensor with shape "
              << "[batch_size=1, time, window_size, n_features]."
              << std::endl;
    return DS_ERR_INVALID_SHAPE;
  }

  return DS_ERR_OK;
}

Tensor
tensor_from_vector(const std::vector<float>& vec, const TensorShape& shape)
{
  Tensor ret(DT_FLOAT, shape);
  auto ret_mapped = ret.flat<float>();
  int i;
  for (i = 0; i < vec.size(); ++i) {
    ret_mapped(i) = vec[i];
  }
  for (; i < shape.num_elements(); ++i) {
    ret_mapped(i) = 0.f;
  }
  return ret;
}

void
copy_tensor_to_vector(const Tensor& tensor, vector<float>& vec, int num_elements = -1)
{
  auto tensor_mapped = tensor.flat<float>();
  if (num_elements == -1) {
    num_elements = tensor.shape().num_elements();
  }
  for (int i = 0; i < num_elements; ++i) {
    vec.push_back(tensor_mapped(i));
  }
}

void
TFModelState::infer(const std::vector<float>& mfcc,
                    unsigned int n_frames,
                    const std::vector<float>& previous_state_c,
                    const std::vector<float>& previous_state_h,
                    vector<float>& logits_output,
                    vector<float>& state_c_output,
                    vector<float>& state_h_output)
{
  std::cout << "TFModelState::infer() start\n";
  const size_t num_classes = alphabet_.GetSize() + 1; // +1 for blank

  Tensor input = tensor_from_vector(mfcc, TensorShape({BATCH_SIZE, n_steps_, 2*n_context_+1, n_features_}));
  Tensor previous_state_c_t = tensor_from_vector(previous_state_c, TensorShape({BATCH_SIZE, (long long)state_size_}));
  Tensor previous_state_h_t = tensor_from_vector(previous_state_h, TensorShape({BATCH_SIZE, (long long)state_size_}));

  Tensor input_lengths(DT_INT32, TensorShape({1}));
  input_lengths.scalar<int>()() = n_frames;

  vector<Tensor> outputs;
  // Status status = session_->Run(
  std::cout << "TFModelState::infer() calling run()\n";
  // Status status = tfSession_->Run(    
  Status status = batching_session.get()->Run(        
    {
     {"input_node", input},
     {"input_lengths", input_lengths},
     {"previous_state_c", previous_state_c_t},
     {"previous_state_h", previous_state_h_t}
    },
    {"logits", "new_state_c", "new_state_h"},
    {},
    &outputs);
  std::cout << "TFModelState::infer() run() complete\n";
  if (!status.ok()) {
    std::cerr << "Error running session: " << status << "\n";
    return;
  }

  copy_tensor_to_vector(outputs[0], logits_output, n_frames * BATCH_SIZE * num_classes);

  state_c_output.clear();
  state_c_output.reserve(state_size_);
  copy_tensor_to_vector(outputs[1], state_c_output);

  state_h_output.clear();
  state_h_output.reserve(state_size_);
  copy_tensor_to_vector(outputs[2], state_h_output);
}

void
TFModelState::compute_mfcc(const vector<float>& samples, vector<float>& mfcc_output)
{
  Tensor input = tensor_from_vector(samples, TensorShape({audio_win_len_}));

  vector<Tensor> outputs;
  Status status = session_->Run({{"input_samples", input}}, {"mfccs"}, {}, &outputs);

  if (!status.ok()) {
    std::cerr << "Error running session: " << status << "\n";
    return;
  }

  // The feature computation graph is hardcoded to one audio length for now
  const int n_windows = 1;
  assert(outputs[0].shape().num_elements() / n_features_ == n_windows);
  copy_tensor_to_vector(outputs[0], mfcc_output);
}
