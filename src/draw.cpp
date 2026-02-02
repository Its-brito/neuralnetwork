#include "../lib/network.h"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// Helper to find best guess
int getPrediction(const std::vector<float>& values) {
    return std::distance(values.begin(), std::max_element(values.begin(), values.end()));
}

int main() {
    // 1. Load the Trained Model
    // Safe dummy values. load() will overwrite them anyway.
    NN::NeuralNetwork net({1, 1, 1});
    net.load("mnist_model.bin"); // Load structure and weights from file

    if (net.layers.empty()) {
        std::cerr << "Could not load model. Run the trainer first!" << std::endl;
        return 1;
    }

    // 2. Setup Drawing Window
    const int CELL_SIZE = 20;
    const int GRID_SIZE = 28;
    sf::RenderWindow window(sf::VideoMode(GRID_SIZE * CELL_SIZE + 200, GRID_SIZE * CELL_SIZE), "Draw a Digit");
    window.setFramerateLimit(60);

    // This vector represents the 28x28 grid (0.0 = black, 1.0 = white)
    std::vector<float> canvas(GRID_SIZE * GRID_SIZE, 0.0f);



    sf::Font font;
    // Use a robust check. If load fails, we MUST exit, otherwise it segfaults later.
    if (!font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) {
        std::cerr << "ERROR: Font not found!" << std::endl;
        std::cerr << "Please copy a .ttf file to the execution folder." << std::endl;
        return 1;
    }

    sf::Text text;
    text.setFont(font);
    text.setCharacterSize(20);
    text.setPosition(GRID_SIZE * CELL_SIZE + 20, 50);

    bool drawing = false;
    bool erasing = false;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();

            // Clear Screen with 'C' or Space
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::C || event.key.code == sf::Keyboard::Space) {
                    std::fill(canvas.begin(), canvas.end(), 0.0f);
                }
            }

            // Mouse Input
            if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) drawing = true;
                if (event.mouseButton.button == sf::Mouse::Right) erasing = true;
            }
            if (event.type == sf::Event::MouseButtonReleased) {
                drawing = false;
                erasing = false;
            }
        }

        // Handle Drawing Logic
        if (drawing || erasing) {
            sf::Vector2i pos = sf::Mouse::getPosition(window);
            int x = pos.x / CELL_SIZE;
            int y = pos.y / CELL_SIZE;

            if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
                int index = y * GRID_SIZE + x;
                float val = drawing ? 1.0f : 0.0f;

                // Set the pixel
                canvas[index] = val;

                // Simple "Brush" effect (color neighbors slightly)
                if(drawing) {
                   if (x+1 < 28) canvas[y*28 + (x+1)] = std::max(canvas[y*28 + (x+1)], 0.5f);
                   if (x-1 >= 0) canvas[y*28 + (x-1)] = std::max(canvas[y*28 + (x-1)], 0.5f);
                   if (y+1 < 28) canvas[(y+1)*28 + x] = std::max(canvas[(y+1)*28 + x], 0.5f);
                   if (y-1 >= 0) canvas[(y-1)*28 + x] = std::max(canvas[(y-1)*28 + x], 0.5f);
                }
            }
        }

        // Real-time Prediction
        auto output = net.feedForward(canvas);
        int guess = getPrediction(output);
        float conf = output[guess];

        // Update Text
        std::string info = "Prediction: " + std::to_string(guess) + "\n\n";
        info += "Confidence: \n" + std::to_string((int)(conf * 100)) + "%\n\n";
        info += "[Left Click] Draw\n[Right Click] Erase\n[Space] Clear";
        text.setString(info);

        // Render
        window.clear(sf::Color::Black);

        // Draw Grid
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                int i = y * GRID_SIZE + x;
                sf::RectangleShape pixel(sf::Vector2f(CELL_SIZE - 1, CELL_SIZE - 1));
                pixel.setPosition(x * CELL_SIZE, y * CELL_SIZE);

                int colorVal = (int)(canvas[i] * 255);
                pixel.setFillColor(sf::Color(colorVal, colorVal, colorVal));
                window.draw(pixel);
            }
        }

        window.draw(text);
        window.display();
    }
    return 0;
}
