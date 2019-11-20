#ifndef TFMODELSTATE_H
#define TFMODELSTATE_H

#include <vector>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/memmapped_file_system.h"

#include "modelstate.h"

struct TFModelState : public ModelState
{
  tensorflow::MemmappedEnv* mmap_env_;
  tensorflow::Session* session_;
  std::unique_ptr<tensorflow::Session> tf_session_;
  std::unique_ptr<tensorflow::Session> batching_session_;
  tensorflow::GraphDef graph_def_;

  TFModelState();
  virtual ~TFModelState();

  virtual int init(const char* model_path,
                   unsigned int beam_width,
                   int max_batch_size,
                   int batch_timeout_micros,
                   int num_batch_threads
                   ) override;

  virtual void infer(const std::vector<float>& mfcc,
                     unsigned int n_frames,
                     const std::vector<float>& previous_state_c,
                     const std::vector<float>& previous_state_h,
                     std::vector<float>& logits_output,
                     std::vector<float>& state_c_output,
                     std::vector<float>& state_h_output) override;

  virtual void infer(const std::vector<float>& mfcc,
                     unsigned int n_frames,
                     const std::vector<float>& previous_state_c,
                     const std::vector<float>& previous_state_h,
                     std::vector<float>& logits_output,
                     std::vector<float>& state_c_output,
                     std::vector<float>& state_h_output, bool doProfile) override;

  virtual void compute_mfcc(const std::vector<float>& audio_buffer,
                            std::vector<float>& mfcc_output) override;

  virtual void compute_mfcc(const std::vector<float>& audio_buffer,
                            std::vector<float>& mfcc_output, bool doProfile) override;
};

#endif // TFMODELSTATE_H
