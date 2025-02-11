

InferenceManager::InferenceManager():
m_context(anira::Context::get_instance(context_config)),
m_inference_config(inference_config),
m_session(m_context->create_session(pp_processor, inference_config, nullptr))
{
    
}

InferenceManager::~InferenceManager() {
    m_context->release_session(m_session);
}

void InferenceManager::prepareToPlay (double sampleRate, int samplesPerBlock) {
    constexpr auto inferenceMaxTime { 0.1 };
    m_latency = static_cast<int> (std::round (inferenceMaxTime * sampleRate));
    m_sampleRate = sampleRate;

    anira::HostAudioConfig host_config {
            (size_t) samplesPerBlock,
            sampleRate
        };

    prepare(host_config);
    set_backend(anira::ONNX);
}

int InferenceManager::getLatency() const {
    return m_latency;
}

double InferenceManager::getSampleRate() const {
    return m_sampleRate;
}

void InferenceManager::processesNonRealtimeSubmit(const juce::AudioBuffer<float>& buffer) {
    anira::AudioBufferF offline_model_input = anira::AudioBufferF(1, modelInputFullSize);
    anira::AudioBufferF offline_raw_model_output = anira::AudioBufferF(1, modelOutputFullSize);
    
    auto write_ptr = offline_model_input.get_write_pointer(0);
    int numChannels = buffer.getNumChannels();
    int numSamples = buffer.getNumSamples();
    
    for (int sample = 0; sample < numSamples; ++sample) {
        for (int channel = 0; channel < numChannels; ++channel) {
            int index = sample * numChannels + channel;
            write_ptr[index] = buffer.getSample(channel, sample);
        }
    }
    
    m_session->m_onnx_processor->process(offline_model_input, offline_raw_model_output, m_session);
    m_session->m_pp_processor.push_samples_to_buffer(offline_raw_model_output, m_session->m_receive_buffer);
}

void InferenceManager::processesNonRealtimeRequest(float* const* output) {
    for (size_t channel = 0; channel < m_inference_config.m_num_audio_channels[anira::Output]; ++channel) {
        for (size_t sample = 0; sample < modelOutputFullSize; ++sample) {
            output[channel][sample] = m_session->m_receive_buffer.pop_sample(channel);
        }
    }
}

void InferenceManager::setNonRealtime (bool IsNonRealtime) {
    nonRealtimeMode = IsNonRealtime;
}

void InferenceManager::set_backend(anira::InferenceBackend new_inference_backend) {
    m_session->m_currentBackend.store(new_inference_backend, std::memory_order_relaxed);
}

anira::InferenceBackend InferenceManager::get_backend() const {
    return m_session->m_currentBackend.load(std::memory_order_relaxed);
}

void InferenceManager::prepare(anira::HostAudioConfig new_config) {
    m_spec = new_config;

    m_context->prepare(m_session, m_spec);
    m_inference_counter.store(0);
}

void InferenceManager::process_submit(juce::AudioBuffer<float> input_data) {
    if (nonRealtimeMode) {
        processesNonRealtimeSubmit(input_data);
    } else {
        process_input(input_data); // put samples into ring send_buffer
        m_context->new_data_submitted(m_session);
    }
}

void InferenceManager::process_request(float* const* output) {
    if (nonRealtimeMode) {
        processesNonRealtimeRequest(output);
    } else {
        m_context->new_data_request(m_session, 0); // second argument 0 unused
        process_output(output, modelOutputFullSize);
    }
}

void InferenceManager::process_input(juce::AudioBuffer<float> &buffer) {
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
            for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
                m_session->m_send_buffer.push_sample(0, buffer.getSample(channel, sample));
        }
    }
}

void InferenceManager::process_output(float* const* output_data, size_t num_samples) {
    while (m_inference_counter.load() > 0) {
        if (m_session->m_receive_buffer.get_available_samples(0) >= 2 * (size_t) num_samples) {
            for (size_t channel = 0; channel < m_inference_config.m_num_audio_channels[anira::Output]; ++channel) {
                for (size_t sample = 0; sample < num_samples; ++sample) {
                    m_session->m_receive_buffer.pop_sample(channel);
                }
            }
            m_inference_counter.fetch_sub(1);
            std::cout << "[WARNING] Catch up samples in session: " << m_session->m_session_id << "!" << std::endl;
        }
        else {
            break;
        }
    }
    if (m_session->m_receive_buffer.get_available_samples(0) >= (size_t) num_samples) {
        for (size_t channel = 0; channel < m_inference_config.m_num_audio_channels[anira::Output]; ++channel) {
            for (size_t sample = 0; sample < num_samples; ++sample) {
                output_data[channel][sample] = m_session->m_receive_buffer.pop_sample(channel);
//                std::cout << "output_data[" << channel << "][" << sample << "]: " << output_data[channel][sample] << std::endl;
            }
        }
    } else {
        clear_data(output_data, num_samples, m_inference_config.m_num_audio_channels[anira::Output]);
        m_inference_counter.fetch_add(1);
        std::cout << "[WARNING] Missing samples in session: " << m_session->m_session_id << "!" << std::endl;
    }
}

void InferenceManager::clear_data(float* const* data, size_t num_samples, size_t num_channels) {
    for (size_t channel = 0; channel < num_channels; ++channel) {
        for (size_t sample = 0; sample < num_samples; ++sample) {
            data[channel][sample] = 0.f;
        }
    }
}
