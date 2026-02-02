#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <fstream>


namespace NN {




    enum class ActivationType { Sigmoid = 's', Tanh = 't', ReLU = 'r', LeakyReLU = 'l' };

    class Layer {
    public:
        int numNodesIn;
        int numNodesOut;
        ActivationType actType;

        std::vector<float> weights;
        std::vector<float> biases;

        // MEMORY: We must store inputs and outputs to calculate gradients later
        std::vector<float> lastInputs;
        std::vector<float> lastOutputs; // Outputs BEFORE activation (z) or AFTER (a)?
                                        // Usually easier to store 'After' for Sigmoid/Tanh derivatives.
        Layer(int nIn, int nOut, ActivationType act)
        : numNodesIn(nIn), numNodesOut(nOut), actType(act) {
            weights.resize(numNodesIn * numNodesOut);
            biases.resize(numNodesOut);

            // 1. Setup Random Device (Hardware entropy)
            std::random_device rd;
            // 2. Setup the Generator (Mersenne Twister - high quality)
            std::mt19937 gen(rd());
            // 3. Define the range (-1.0 to 1.0)
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            // Fill weights
            for(auto& w : weights) {
                w = dis(gen);
            }
            // Fill biases (optional, usually 0 is fine, but random is okay too)
            for(auto& b : biases) {
                b = dis(gen);
            }
        }

        // 1. FORWARD PASS
        std::vector<float> calculateOutput(const std::vector<float>& inputs) {
            this->lastInputs = inputs; // SAVE INPUTS for backprop
            std::vector<float> outputs(numNodesOut, 0.0f);

            for (int out = 0; out < numNodesOut; out++) {
                float sum = biases[out];
                for (int in = 0; in < numNodesIn; in++) {
                    sum += inputs[in] * weights[in * numNodesOut + out];
                }
                outputs[out] = activation(sum);
            }
            this->lastOutputs = outputs; // SAVE OUTPUTS
            return outputs;
        }

        // 2. BACKWARD PASS (Gradient Descent)
        // Returns: Gradients for the PREVIOUS layer
        std::vector<float> backPropagate(const std::vector<float>& outputGradients, float learningRate) {
            std::vector<float> inputGradients(numNodesIn, 0.0f);

            for (int out = 0; out < numNodesOut; out++) {
                // Calculate 'delta' = error_term * derivative_of_activation
                // We use lastOutputs[out] because Sigmoid derivative depends on the output value
                float derivative = activationDerivative(lastOutputs[out]);
                float delta = outputGradients[out] * derivative;

                // Update Biases
                biases[out] -= learningRate * delta;

                // Update Weights AND Calculate Input Gradients
                for (int in = 0; in < numNodesIn; in++) {
                    int weightIndex = in * numNodesOut + out;

                    // Accumulate gradient to pass back to previous layer
                    // (Chain Rule: dC/dInput = dC/dOutput * dOutput/dInput)
                    inputGradients[in] += delta * weights[weightIndex];

                    // Update Weight
                    // W_new = W_old - (LearningRate * Delta * Input)
                    weights[weightIndex] -= learningRate * delta * lastInputs[in];
                }
            }
            return inputGradients;
        }

    private:
        float activation(float x) {
            switch (actType) {
                case ActivationType::Tanh: return std::tanh(x);
                case ActivationType::Sigmoid: return 1.0f / (1.0f + std::exp(-x));
                case ActivationType::ReLU: return std::max(0.0f, x);
                case ActivationType::LeakyReLU: return (x > 0) ? x : 0.01f * x;
                default: return x;
            }
        }

        // Calculates f'(x). Note: We pass the ACTIVATED value (y), not x, for efficiency
        float activationDerivative(float y) {
            switch (actType) {
                case ActivationType::Tanh:
                    return 1.0f - (y * y); // d/dx tanh(x) = 1 - tanh^2(x)
                case ActivationType::Sigmoid:
                    return y * (1.0f - y); // d/dx sig(x) = sig(x)(1 - sig(x))
                case ActivationType::ReLU:
                    return (y > 0.0f) ? 1.0f : 0.0f;
                case ActivationType::LeakyReLU:
                    return (y > 0.0f) ? 1.0f : 0.01f;
                default: return 1.0f;
            }
        }
    };

    class NeuralNetwork {
    public:

        std::vector<Layer> layers;

        NeuralNetwork(const std::vector<int>& topology) {
            for (size_t i = 0; i < topology.size() - 1; i++) {
                // Last layer usually Sigmoid/Linear, Hidden usually ReLU/Tanh
                ActivationType act = (i == topology.size() - 2) ? ActivationType::Sigmoid : ActivationType::Tanh;
                layers.emplace_back(topology[i], topology[i + 1], act);
            }
        }

        std::vector<float> feedForward(std::vector<float> inputs) {
            for (auto& layer : layers) {
                inputs = layer.calculateOutput(inputs);
            }
            return inputs;
        }
        void save(const std::string& filename) {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Error saving model!" << std::endl;
                return;
            }

            // 1. Save Number of Layers
            int numLayers = layers.size();
            file.write((char*)&numLayers, sizeof(int));

            // 2. Save Each Layer
            for (auto& layer : layers) {
                // Save Architecture
                file.write((char*)&layer.numNodesIn, sizeof(int));
                file.write((char*)&layer.numNodesOut, sizeof(int));
                file.write((char*)&layer.actType, sizeof(ActivationType));

                // Save Data
                file.write((char*)layer.weights.data(), layer.weights.size() * sizeof(float));
                file.write((char*)layer.biases.data(), layer.biases.size() * sizeof(float));
            }
            file.close();
            std::cout << "Model saved to " << filename << std::endl;
        }

        void load(const std::string& filename) {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Error loading model!" << std::endl;
                return;
            }

            layers.clear();
            int numLayers;
            file.read((char*)&numLayers, sizeof(int));

            for (int i = 0; i < numLayers; i++) {
                int nIn, nOut;
                ActivationType act;

                // Read Architecture
                file.read((char*)&nIn, sizeof(int));
                file.read((char*)&nOut, sizeof(int));
                file.read((char*)&act, sizeof(ActivationType));

                // Create Layer
                layers.emplace_back(nIn, nOut, act);

                // Read Data
                auto& l = layers.back();
                file.read((char*)l.weights.data(), l.weights.size() * sizeof(float));
                file.read((char*)l.biases.data(), l.biases.size() * sizeof(float));
            }
            file.close();
            std::cout << "Model loaded from " << filename << std::endl;
        }


        // THE TRAINING FUNCTION
        void train(const std::vector<float>& inputs, const std::vector<float>& targets, float learningRate) {
            // 1. Forward Pass (Fill the "memory" of the layers)
            std::vector<float> results = feedForward(inputs);

            // 2. Calculate Initial Gradients (Derivative of Cost Function MSE)
            // Gradient = (Predicted - Target)
            std::vector<float> gradients;
            for(size_t i=0; i<results.size(); i++) {
                gradients.push_back(results[i] - targets[i]);
            }

            // 3. Backward Pass (Loop reversed)
            for (int i = layers.size() - 1; i >= 0; i--) {
                gradients = layers[i].backPropagate(gradients, learningRate);
            }
        }
    };
}
