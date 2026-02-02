#include "../lib/network.h"
#include "../lib/image_processing.h"
#include <iostream>
#include <filesystem>

int main() {
    // 1. Define the Base Path exactly as you found it
    // Note: I preserved the spelling "NeuralNetwok" from your path
    std::string basePath = "/home/manuel/Projects/NeuralNetwok/dataset/MNIST_CSV/";

    std::cout << "Loading data from: " << basePath << std::endl;

    // 2. Load Data using the base path
    auto trainingData = ImgProc::MnistLoader::load(
        basePath + "train-images.idx3-ubyte",
        basePath + "train-labels.idx1-ubyte"
    );

    auto testData = ImgProc::MnistLoader::load(
        basePath + "t10k-images.idx3-ubyte",
        basePath + "t10k-labels.idx1-ubyte"
    );

    if (trainingData.empty() || testData.empty()) {
        std::cerr << "CRITICAL ERROR: Still cannot open files." << std::endl;
        return 1;
    }

    // 3. Setup Network
    NN::NeuralNetwork net({784, 128, 10});

    // 4. Training Loop
    std::cout << "Training on " << trainingData.size() << " images..." << std::endl;

    // We run for 1 epoch first to test speed
    for (const auto& img : trainingData) {
        net.train(img.pixels, img.target, 0.05f); // 0.05 is a safer learning rate for MNIST
    }

    std::cout << "Training Complete. Testing first image..." << std::endl;

    // 5. Verify it actually learned something
    // We pass the first TEST image (not training image) to see if it generalizes
    auto prediction = net.feedForward(testData[0].pixels);
    std::cout << "Target: " << testData[0].label << std::endl;

    // Find the digit with the highest probability
    int bestGuess = 0;
    float maxProb = -1.0f;
    for(int i=0; i<10; i++) {
        std::cout << i << ": " << prediction[i] << "\n";
        if (prediction[i] > maxProb) {
            maxProb = prediction[i];
            bestGuess = i;
        }
    }
    std::cout << "\nNetwork Guessed: " << bestGuess << std::endl;

    return 0;
}
