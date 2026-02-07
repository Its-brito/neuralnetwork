#include "../lib/network.h"
#include "../lib/image_processing.h"
#include <iostream>
#include <filesystem>
#include <SFML/Graphics.hpp>
#include <iomanip> // For std::setprecision
#include <random>

// Helper to find the index of the max value
int getPrediction(const std::vector<float>& output) {
    int maxIndex = 0;
    float maxVal = output[0];
    for(size_t i = 1; i < output.size(); i++) {
        if(output[i] > maxVal) {
            maxVal = output[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

int main() {
    // --- PART 1: SETUP & TRAINING ---
    // Update path !!!!!!!!!!!!!!!!!        
    std::string basePath = "/home/manuel/Projects/NeuralNetwok/dataset/MNIST_CSV/";

    // Check if fonts exist (Common Linux path)
    std::string fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
    if (!std::filesystem::exists(fontPath)) {
        // Fallback for some other Linux distros
        fontPath = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf";
    }

    std::cout << "Loading Data..." << std::endl;
    auto trainingData = ImgProc::MnistLoader::load(basePath + "train-images.idx3-ubyte", basePath + "train-labels.idx1-ubyte");
    auto testData = ImgProc::MnistLoader::load(basePath + "t10k-images.idx3-ubyte", basePath + "t10k-labels.idx1-ubyte");

    if (trainingData.empty() || testData.empty()) return 1;

    NN::NeuralNetwork net({784, 64, 10}); // Neural Network Layers - - -

    // Setup RNG for on-the-fly augmentation (scaling + translation)
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> shiftDist(-2, 2); // shift by -2..2 pixels
    std::bernoulli_distribution translateProb(0.5); // 50% chance to translate
    std::bernoulli_distribution scaleProb(0.5); // 50% chance to scale
    std::uniform_real_distribution<float> scaleDist(0.85f, 1.15f); // scale range

    int epochs = 5; // number of epochs to train
    // Augmentation schedule: only apply augmentation for epochs in [augment_start, augment_end]
    int augment_start = 2; // 1-based epoch index when augmentation begins
    int augment_end = 5;   // 1-based epoch index when augmentation ends
    std::cout << "Training (" << epochs << " epochs, with augmentation)..." << std::endl;

    for (int e = 0; e < epochs; ++e) {
        std::shuffle(trainingData.begin(), trainingData.end(), rng);
        std::cout << "Epoch " << (e + 1) << "/" << epochs << "\n";

        bool augmentEnabledThisEpoch = ( (e+1) >= augment_start && (e+1) <= augment_end );

        for (const auto& img : trainingData) {
            // 1) Train on the original image
            net.train(img.pixels, img.target, 0.05f);

            // 2) Optionally create an augmented copy and train on it as well
            bool didAug = false;
            std::vector<float> aug = img.pixels;

            if (augmentEnabledThisEpoch && scaleProb(rng)) {
                float s = scaleDist(rng);
                aug = ImgProc::MnistLoader::scaleImage(aug, s);
                didAug = true;
            }
            if (augmentEnabledThisEpoch && translateProb(rng)) {
                int dx = shiftDist(rng);
                int dy = shiftDist(rng);
                aug = ImgProc::MnistLoader::translateImage(aug, dx, dy);
                didAug = true;
            }

            if (didAug) {
                net.train(aug, img.target, 0.05f);
            }
        }
    }
    std::cout << "Training Complete." << std::endl;

    // EXPORT THE MODEL
    net.save("mnist_model.bin");




    // --- PART 2: GRAPHICAL INTERFACE ---

    // Window (800x600) to fit text on the right
    sf::RenderWindow window(sf::VideoMode(800,600), "MNIST Neural Net Viewer");
    window.setFramerateLimit(60);

    // 1. Load Font
    sf::Font font;
    if (!font.loadFromFile(fontPath)) {
        std::cerr << "ERROR: Could not find font at " << fontPath << std::endl;
        std::cerr << "Please copy a .ttf file to your project folder and update the path." << std::endl;
        return 1;
    }

    // 2. Setup Text Objects
    sf::Text lblPrediction;
    lblPrediction.setFont(font);
    lblPrediction.setCharacterSize(24);
    lblPrediction.setFillColor(sf::Color::White);
    lblPrediction.setPosition(600, 50);

    sf::Text lblTarget;
    lblTarget.setFont(font);
    lblTarget.setCharacterSize(20);
    lblTarget.setFillColor(sf::Color::Cyan); // Cyan for truth
    lblTarget.setPosition(600, 100);

    sf::Text lblConfidence;
    lblConfidence.setFont(font);
    lblConfidence.setCharacterSize(16);
    lblConfidence.setFillColor(sf::Color::Green);
    lblConfidence.setPosition(600, 150);

    // 3. Setup Button
    sf::RectangleShape btnNext(sf::Vector2f(160, 50));
    btnNext.setFillColor(sf::Color(50, 50, 50)); // Dark Grey
    btnNext.setOutlineThickness(2);
    btnNext.setOutlineColor(sf::Color::White);
    btnNext.setPosition(600, 500);

    sf::Text btnText;
    btnText.setFont(font);
    btnText.setString("Next Image");
    btnText.setCharacterSize(20);
    btnText.setFillColor(sf::Color::White);
    // Center text in button approximately
    btnText.setPosition(615, 510);

    // Grid Settings
    float scale = 20.0f;
    int currentImageIdx = 0;
    bool needsUpdate = true;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();

            // CLICK HANDLING
            if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    // Get mouse position relative to window
                    sf::Vector2i mousePos = sf::Mouse::getPosition(window);

                    // Check if mouse is inside the button rectangle
                    if (btnNext.getGlobalBounds().contains(static_cast<float>(mousePos.x), static_cast<float>(mousePos.y))) {
                        currentImageIdx = (currentImageIdx + 1) % testData.size();
                        needsUpdate = true;

                        // Visual feedback: click color
                        btnNext.setFillColor(sf::Color(100, 100, 100));
                    }
                }
            }

            // Restore button color on release
            if (event.type == sf::Event::MouseButtonReleased) {
                btnNext.setFillColor(sf::Color(50, 50, 50));
            }
        }

        // --- UPDATE LOGIC ---
        if (needsUpdate) {
            auto& img = testData[currentImageIdx];
            auto output = net.feedForward(img.pixels);
            int guess = getPrediction(output);
            float confidence = output[guess] * 100.0f; // To percentage

            // Update Text
            lblPrediction.setString("Prediction: " + std::to_string(guess));
            lblTarget.setString("Actual Label: " + std::to_string(img.label));

            // Formatting confidence string
            std::stringstream ss;
            ss << "Confidence: " << std::fixed << std::setprecision(1) << confidence << "%";
            lblConfidence.setString(ss.str());

            // Color code the prediction
            if (guess == img.label) lblPrediction.setFillColor(sf::Color::Green);
            else lblPrediction.setFillColor(sf::Color::Red);

            needsUpdate = false;
        }

        // --- DRAWING ---
        window.clear(sf::Color::Black);

        // 1. Draw Grid (Left side)
        const auto& pixels = testData[currentImageIdx].pixels;
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int i = y * 28 + x;
                float brightness = pixels[i];

                sf::RectangleShape rect(sf::Vector2f(scale, scale));
                rect.setPosition(x * scale + 20, y * scale + 20);

                sf::Uint8 val = static_cast<sf::Uint8>(brightness * 255.0f);
                rect.setFillColor(sf::Color(val, val, val));
                window.draw(rect);
            }
        }

        // 2. Draw Interface (Right side)
        window.draw(lblPrediction);
        window.draw(lblTarget);
        window.draw(lblConfidence);

        // Draw Button
        window.draw(btnNext);
        window.draw(btnText);

        window.display();
    }

    return 0;
}
