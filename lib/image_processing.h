#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint> // for uint32_t

namespace ImgProc {

    // Same structure as before, compatible with your Network
    struct Image {
        std::vector<float> pixels;  // 784 values (0.0 - 1.0)
        std::vector<float> target;  // One-hot encoded label
        int label;                  // Raw integer (0-9)
    };

    class MnistLoader {
    private:
        // MNIST files use Big-Endian integers. Most PCs are Little-Endian.
        // We must swap the bytes to read the integers correctly.
        static uint32_t swapEndian(uint32_t val) {
            return ((val << 24) & 0xFF000000) |
                   ((val << 8)  & 0x00FF0000) |
                   ((val >> 8)  & 0x0000FF00) |
                   ((val >> 24) & 0x000000FF);
        }

    public:
        static std::vector<Image> load(const std::string& imagePath, const std::string& labelPath) {
            std::vector<Image> dataset;

            std::ifstream imgFile(imagePath, std::ios::binary);
            std::ifstream lblFile(labelPath, std::ios::binary);

            if (!imgFile.is_open() || !lblFile.is_open()) {
                std::cerr << "Error: Could not open MNIST files." << std::endl;
                std::cerr << "Checked paths: " << imagePath << " & " << labelPath << std::endl;
                return dataset;
            }

            // 1. READ HEADERS
            uint32_t magicImg, numImg, rows, cols;
            uint32_t magicLbl, numLbl;

            imgFile.read(reinterpret_cast<char*>(&magicImg), 4);
            imgFile.read(reinterpret_cast<char*>(&numImg), 4);
            imgFile.read(reinterpret_cast<char*>(&rows), 4);
            imgFile.read(reinterpret_cast<char*>(&cols), 4);

            lblFile.read(reinterpret_cast<char*>(&magicLbl), 4);
            lblFile.read(reinterpret_cast<char*>(&numLbl), 4);

            // Convert Endianness
            magicImg = swapEndian(magicImg);
            numImg   = swapEndian(numImg);
            rows     = swapEndian(rows);
            cols     = swapEndian(cols);
            magicLbl = swapEndian(magicLbl);
            numLbl   = swapEndian(numLbl);

            // 2. SANITY CHECKS
            // Magic numbers: 2051 for images, 2049 for labels
            if (magicImg != 2051 || magicLbl != 2049) {
                std::cerr << "Error: Invalid MNIST magic numbers!" << std::endl;
                return dataset;
            }
            if (numImg != numLbl) {
                std::cerr << "Error: Image count doesn't match label count!" << std::endl;
                return dataset;
            }

            std::cout << "Loading " << numImg << " images (" << rows << "x" << cols << ")..." << std::endl;

            // 3. READ DATA
            int imageSize = rows * cols; // 28 * 28 = 784
            dataset.resize(numImg);

            // Buffers to hold raw bytes
            // Note: Reading byte-by-byte is slow, better to read chunks or map memory,
            // but this is simple and sufficient for 60k images.
            for (uint32_t i = 0; i < numImg; i++) {
                Image& img = dataset[i];
                img.pixels.reserve(imageSize);

                // Read Label (1 byte)
                unsigned char labelByte;
                lblFile.read(reinterpret_cast<char*>(&labelByte), 1);
                img.label = static_cast<int>(labelByte);

                // Create Target Vector (One-Hot)
                img.target.assign(10, 0.0f);
                if (img.label >= 0 && img.label < 10) {
                    img.target[img.label] = 1.0f;
                }

                // Read Pixels (784 bytes)
                // We read into a temporary buffer to minimize file I/O calls
                std::vector<unsigned char> pixelBuffer(imageSize);
                imgFile.read(reinterpret_cast<char*>(pixelBuffer.data()), imageSize);

                for (int j = 0; j < imageSize; j++) {
                    // Normalize 0-255 -> 0.0-1.0
                    img.pixels.push_back(static_cast<float>(pixelBuffer[j]) / 255.0f);
                }
            }

            std::cout << "Done. Loaded " << dataset.size() << " samples." << std::endl;
            return dataset;
        }
    };
}
