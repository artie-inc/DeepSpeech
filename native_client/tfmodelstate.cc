#include <thread>
#include <chrono>
#include <ctime>   

#include "tfmodelstate.h"

#include "workspace_status.h"

#include "tensorflow_serving/batching/batching_session.h"

using namespace tensorflow;
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

uint64_t timeSinceEpochMillisec3() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}


int
TFModelState::init(const char* model_path,
                   unsigned int beam_width,
                   int max_batch_size,
                   int batch_timeout_micros,
                   int num_batch_threads)
{
  int err = ModelState::init(model_path, beam_width, max_batch_size, batch_timeout_micros, num_batch_threads);
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

  options.config.mutable_graph_options()
    ->mutable_optimizer_options()
    ->set_opt_level(::OptimizerOptions::L0);

  options.config.set_allow_soft_placement(true);
  options.config.set_log_device_placement(true);


  status = NewSession(options, &session_);
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return DS_ERR_FAIL_INIT_SESS;
  }
  std::cout << "TFModelState::init() created NewSession" << " typeid=" << typeid(session_).name() << std::endl;

  if(max_batch_size > 1) {
    tensorflow::serving::BasicBatchScheduler<tensorflow::serving::BatchingSessionTask>::Options schedule_options;
    schedule_options.max_batch_size = max_batch_size;  // fits two 2-unit tasks
    schedule_options.batch_timeout_micros = batch_timeout_micros;  
    schedule_options.num_batch_threads = num_batch_threads;
    
    tensorflow::serving::TensorSignature signature = {
        {"input_node", "input_lengths", "previous_state_c", "previous_state_h"},
        {"logits", "new_state_c", "new_state_h"} 
    };

    tensorflow::serving::BatchingSessionOptions batching_session_options;
    batching_session_options.allowed_batch_sizes.push_back(max_batch_size);

    tf_session_ = std::unique_ptr<tensorflow::Session>(session_);
    tensorflow::serving::CreateBasicBatchingSession(schedule_options, 
        batching_session_options, signature, std::move(tf_session_), &batching_session_);

    std::cout << "TFModelState::init() created BatchingSession" <<  std::endl;
  }

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
  return infer(mfcc, n_frames, previous_state_c, previous_state_h, logits_output, state_c_output, state_h_output, false);
}

void
TFModelState::infer(const std::vector<float>& mfcc,
                    unsigned int n_frames,
                    const std::vector<float>& previous_state_c,
                    const std::vector<float>& previous_state_h,
                    vector<float>& logits_output,
                    vector<float>& state_c_output,
                    vector<float>& state_h_output, bool doProfile)
{
  // std::cout << "TFModelState::infer() start\n";

  auto t_start = std::chrono::high_resolution_clock::now();

  const size_t num_classes = alphabet_.GetSize() + 1; // +1 for blank

  Tensor input = tensor_from_vector(mfcc, TensorShape({BATCH_SIZE, n_steps_, 2*n_context_+1, n_features_}));
  Tensor previous_state_c_t = tensor_from_vector(previous_state_c, TensorShape({BATCH_SIZE, (long long)state_size_}));
  Tensor previous_state_h_t = tensor_from_vector(previous_state_h, TensorShape({BATCH_SIZE, (long long)state_size_}));

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_time_ms_tensors = std::chrono::duration<double,  std::milli>(t_end-t_start).count();
  t_start = std::chrono::high_resolution_clock::now();

  Tensor input_lengths(DT_INT32, TensorShape({1}));
  input_lengths.scalar<int>()() = n_frames;

  vector<Tensor> outputs;

  tensorflow::Session* session_to_run;
  if(batching_session_) {
    session_to_run = batching_session_.get();
  }
  else {
    session_to_run = session_;
  }

  t_end = std::chrono::high_resolution_clock::now();
  double elapsed_time_ms_sess = std::chrono::duration<double,  std::milli>(t_end-t_start).count();
  t_start = std::chrono::high_resolution_clock::now();

  // std::cout << "TFModelState::infer() calling run()\n";
  Status status = session_to_run->Run(        
    {
     {"input_node", input},
     {"input_lengths", input_lengths},
     {"previous_state_c", previous_state_c_t},
     {"previous_state_h", previous_state_h_t}
    },
    {"logits", "new_state_c", "new_state_h"},
    {},
    &outputs);

  t_end = std::chrono::high_resolution_clock::now();
  double elapsed_time_ms_run = std::chrono::duration<double,  std::milli>(t_end-t_start).count();
  t_start = std::chrono::high_resolution_clock::now();


  // std::cout << "TFModelState::infer() run() complete\n";
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

  t_end = std::chrono::high_resolution_clock::now();
  double elapsed_time_ms_copy = std::chrono::duration<double,  std::milli>(t_end-t_start).count();
  t_start = std::chrono::high_resolution_clock::now();

  if(doProfile) {
    std::cout << timeSinceEpochMillisec3() << " " << std::this_thread::get_id()
    << " TFModelState::infer() "
    << " tensors=" << elapsed_time_ms_tensors 
    << " sess=" << elapsed_time_ms_sess 
    << " run=" << elapsed_time_ms_run
    << " copy=" << elapsed_time_ms_copy 
    << std::endl;
  }
}

void
TFModelState::compute_mfcc(const vector<float>& samples, vector<float>& mfcc_output)
{
  return compute_mfcc(samples, mfcc_output, false);
}

void
TFModelState::compute_mfcc(const vector<float>& samples, vector<float>& mfcc_output, bool doProfile)
{

    // options.config.mutable_graph_options()
    //   ->mutable_optimizer_options()
    //   ->set_opt_level(::OptimizerOptions::L0);

  auto t_start = std::chrono::high_resolution_clock::now();
  Tensor input = tensor_from_vector(samples, TensorShape({audio_win_len_}));
  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_time_ms_1 = std::chrono::duration<double,  std::milli>(t_end-t_start).count();
  t_start = std::chrono::high_resolution_clock::now();
  
  vector<Tensor> outputs;
  Status status = session_->Run({{"input_samples", input}}, {"mfccs"}, {}, &outputs, doProfile);
  t_end = std::chrono::high_resolution_clock::now();
  double elapsed_time_ms_run = std::chrono::duration<double,  std::milli>(t_end-t_start).count();
  t_start = std::chrono::high_resolution_clock::now();

  if (!status.ok()) {
    std::cerr << "Error running session: " << status << "\n";
    return;
  }

  // The feature computation graph is hardcoded to one audio length for now
  const int n_windows = 1;
  assert(outputs[0].shape().num_elements() / n_features_ == n_windows);
  copy_tensor_to_vector(outputs[0], mfcc_output);
  t_end = std::chrono::high_resolution_clock::now();
  double elapsed_time_ms_copy = std::chrono::duration<double,  std::milli>(t_end-t_start).count();

  if(doProfile && (elapsed_time_ms_1 + elapsed_time_ms_run + elapsed_time_ms_copy > 1000)) {
    std::cout << "TFModelState::compute_mfcc() ms1=" << elapsed_time_ms_1 << " run=" << elapsed_time_ms_run << " copy=" << elapsed_time_ms_copy << std::endl;
  }
}



