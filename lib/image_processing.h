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

        // Translate a single 28x28 image by (dx,dy). Positive dx moves image right,
        // positive dy moves image down. Empty areas are filled with 0.0f.
        static std::vector<float> translateImage(const std::vector<float>& pixels, int dx, int dy, int width = 28, int height = 28) {
            std::vector<float> out(width * height, 0.0f);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int sx = x - dx; // source x
                    int sy = y - dy; // source y
                    if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                        out[y * width + x] = pixels[sy * width + sx];
                    } else {
                        out[y * width + x] = 0.0f;
                    }
                }
            }
            return out;
        }

        // Scale an image about its center using bilinear interpolation.
        // scale > 1.0 zooms in, scale < 1.0 zooms out. Result is always dstW x dstH.
        static std::vector<float> scaleImage(const std::vector<float>& src, float scale, int srcW = 28, int srcH = 28, int dstW = 28, int dstH = 28) {
            std::vector<float> out(dstW * dstH, 0.0f);

            const float cx_src = (srcW - 1) / 2.0f;
            const float cy_src = (srcH - 1) / 2.0f;
            const float cx_dst = (dstW - 1) / 2.0f;
            const float cy_dst = (dstH - 1) / 2.0f;

            for (int y = 0; y < dstH; ++y) {
                for (int x = 0; x < dstW; ++x) {
                    // Map dst pixel to source coordinate (inverse transform)
                    float sx = (x - cx_dst) / scale + cx_src;
                    float sy = (y - cy_dst) / scale + cy_src;

                    if (sx < 0 || sx >= srcW - 1 || sy < 0 || sy >= srcH - 1) {
                        out[y * dstW + x] = 0.0f;
                        continue;
                    }

                    int x0 = static_cast<int>(std::floor(sx));
                    int y0 = static_cast<int>(std::floor(sy));
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    float wx = sx - x0;
                    float wy = sy - y0;

                    float v00 = src[y0 * srcW + x0];
                    float v10 = src[y0 * srcW + x1];
                    float v01 = src[y1 * srcW + x0];
                    float v11 = src[y1 * srcW + x1];

                    float v0 = v00 * (1 - wx) + v10 * wx;
                    float v1 = v01 * (1 - wx) + v11 * wx;
                    float v = v0 * (1 - wy) + v1 * wy;

                    out[y * dstW + x] = v;
                }
            }

            return out;
        }
    };
}
