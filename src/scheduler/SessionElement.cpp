#include <anira/scheduler/SessionElement.h>

namespace anira {

SessionElement::SessionElement(int newSessionID, PrePostProcessor& pp_processor, InferenceConfig& inference_config) :
    m_session_id(newSessionID),
    m_pp_processor(pp_processor),
    m_inference_config(inference_config),
    m_default_processor(m_inference_config),
    m_custom_processor(&m_default_processor)
{
}

SessionElement::ThreadSafeStruct::ThreadSafeStruct(size_t num_input_samples, size_t num_output_samples, size_t num_input_channels, size_t num_output_channels) {
    m_processed_model_input.resize(num_input_channels, num_input_samples);
    m_raw_model_output.resize(num_output_channels, num_output_samples);
}

void SessionElement::clear() {
    m_send_buffer.clear_with_positions();
    m_receive_buffer.clear_with_positions();
    m_time_stamps.clear();
    m_inference_queue.clear();
}

void SessionElement::prepare(HostAudioConfig new_config) {
    m_host_config = new_config;

    // TODO: forcing init until we find a better way
    // @ANIRA: should allow for overwriting the maxSecs and maxStructs ?
    
    int maxSecs = 20;
    int maxStructs = 20;
    
    m_send_buffer.initialize_with_positions(m_inference_config.m_num_audio_channels[Input], (size_t) m_host_config.m_host_sample_rate * maxSecs); // 20 secs
    m_receive_buffer.initialize_with_positions(m_inference_config.m_num_audio_channels[Output], (size_t) m_host_config.m_host_sample_rate * maxSecs); // 20 secs

    // Compute the queue size: ensure enough structures to handle hits considering thread count and processing time?
    size_t n_structs = maxStructs;
    
    // How big are the input and output buffers
    size_t num_input_samples = m_inference_config.m_input_sizes[m_inference_config.m_index_audio_data[Input]] / m_inference_config.m_num_audio_channels[Input];
    size_t num_output_samples = m_inference_config.m_output_sizes[m_inference_config.m_index_audio_data[Output]] / m_inference_config.m_num_audio_channels[Output];
    size_t num_input_channels = m_inference_config.m_num_audio_channels[Input];
    size_t num_output_channels = m_inference_config.m_num_audio_channels[Output];

    for (int i = 0; i < n_structs; ++i) {
        m_inference_queue.emplace_back(std::make_unique<ThreadSafeStruct>(num_input_samples, num_output_samples, num_input_channels, num_output_channels));
    }

    m_time_stamps.reserve(n_structs);
}

template <typename T> void SessionElement::set_processor(std::shared_ptr<T>& processor) {
#ifdef USE_LIBTORCH
    if (std::is_same<T, LibtorchProcessor>::value) {
        m_libtorch_processor = std::dynamic_pointer_cast<LibtorchProcessor>(processor);
    }
#endif
#ifdef USE_ONNXRUNTIME
    if (std::is_same<T, OnnxRuntimeProcessor>::value) {
        m_onnx_processor = std::dynamic_pointer_cast<OnnxRuntimeProcessor>(processor);
    }
#endif
#ifdef USE_TFLITE
    if (std::is_same<T, TFLiteProcessor>::value) {
        m_tflite_processor = std::dynamic_pointer_cast<TFLiteProcessor>(processor);
    }
#endif
}

#ifdef USE_LIBTORCH
template void SessionElement::set_processor<LibtorchProcessor>(std::shared_ptr<LibtorchProcessor>& processor);
#endif
#ifdef USE_ONNXRUNTIME
template void SessionElement::set_processor<OnnxRuntimeProcessor>(std::shared_ptr<OnnxRuntimeProcessor>& processor);
#endif
#ifdef USE_TFLITE
template void SessionElement::set_processor<TFLiteProcessor>(std::shared_ptr<TFLiteProcessor>& processor);
#endif

} // namespace anira