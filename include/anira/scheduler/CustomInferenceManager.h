#ifndef CUSTOM_INFERENCEMANAGER_H
#define CUSTOM_INFERENCEMANAGER_H

class InferenceManager
{
public:
    InferenceManager();
    ~InferenceManager();

    anira::InferenceBackend get_backend() const;
    int getLatency() const;
    double getSampleRate() const;

    void prepareToPlay (double sampleRate, int samplesPerBlock);
    void process_submit(juce::AudioBuffer<float> input_data);
    void process_request(float* const* output_data);
    void setNonRealtime (bool IsNonRealtime);

private:
    void processesNonRealtimeSubmit(const juce::AudioBuffer<float>& input_data);
    void processesNonRealtimeRequest(float* const* output_data);
    void prepare(anira::HostAudioConfig config);
    void process_input(juce::AudioBuffer<float> &buffer);
    void process_output(float* const* output_data, size_t num_samples);
    void set_backend(anira::InferenceBackend new_inference_backend);
    void clear_data(float* const* data, size_t input_samples, size_t num_channels);
    int max_num_inferences(int m_host_buffer_size, int model_output_size);

private:
    int m_latency { 0 };
    double m_sampleRate { 0.0 };
    std::atomic<bool> nonRealtimeMode { false };
    CustomPrePostProcessor pp_processor;
    std::shared_ptr<anira::Context> m_context;
    anira::InferenceConfig& m_inference_config;
    std::shared_ptr<anira::SessionElement> m_session;
    anira::HostAudioConfig m_spec;
    std::atomic<int> m_inference_counter {0};    
};

#endif //CUSTOM_INFERENCEMANAGER_H
