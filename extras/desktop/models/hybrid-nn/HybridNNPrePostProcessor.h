#ifndef ANIRA_HYBRIDNNPREPOSTPROCESSOR_H
#define ANIRA_HYBRIDNNPREPOSTPROCESSOR_H

#include "HybridNNConfig.h"
#include <anira/anira.h>

class HybridNNPrePostProcessor : public anira::PrePostProcessor
{
public:
    virtual void pre_process(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
        int64_t num_batches;
        int64_t num_input_samples;
        int64_t num_output_samples;
#ifdef USE_LIBTORCH
        if (current_inference_backend == anira::LIBTORCH) {
            num_batches = config.m_model_input_shape_torch[0];
            num_input_samples = config.m_model_input_shape_torch[2];
            num_output_samples = config.m_model_output_shape_torch[1];
        }
#endif
#ifdef USE_ONNXRUNTIME
        if (current_inference_backend == anira::ONNX) {
            num_batches = config.m_model_input_shape_onnx[0];
            num_input_samples = config.m_model_input_shape_onnx[2];
            num_output_samples = config.m_model_output_shape_onnx[1];
        }
#endif
#ifdef USE_TFLITE
        if (current_inference_backend == anira::TFLITE) {
            num_batches = config.m_model_input_shape_tflite[0];
            num_input_samples = config.m_model_input_shape_tflite[1];
            num_output_samples = config.m_model_output_shape_tflite[1];
        }
#endif 
        else if (current_inference_backend == anira::NONE) {
#if USE_LIBTORCH
            num_batches = config.m_model_input_shape_torch[0];
            num_input_samples = config.m_model_input_shape_torch[2];
            num_output_samples = config.m_model_output_shape_torch[1];
#elif USE_ONNXRUNTIME
            num_batches = config.m_model_input_shape_onnx[0];
            num_input_samples = config.m_model_input_shape_onnx[2];
            num_output_samples = config.m_model_output_shape_onnx[1];
#elif USE_TFLITE
            num_batches = config.m_model_input_shape_tflite[0];
            num_input_samples = config.m_model_input_shape_tflite[1];
            num_output_samples = config.m_model_output_shape_tflite[1];
#endif
        }
        
        if (current_inference_backend != anira::LIBTORCH && current_inference_backend != anira::ONNX && current_inference_backend != anira::TFLITE && current_inference_backend != anira::NONE) {
            throw std::runtime_error("Invalid inference backend");
        }
            
        for (size_t batch = 0; batch < num_batches; batch++) {
            size_t base_index = batch * num_input_samples;
            pop_samples_from_buffer(input, output, num_output_samples, num_input_samples-num_output_samples, base_index);
        }
    };
    
    anira::InferenceConfig config = hybridnn_config;
};

#endif //ANIRA_HYBRIDNNPREPOSTPROCESSOR_H
