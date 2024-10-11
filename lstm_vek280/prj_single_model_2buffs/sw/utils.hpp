/* Original License Notice:
 * MIT License
 *
 * Copyright (C) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of Advanced Micro Devices, Inc. shall not be used in advertising or
 * otherwise to promote the sale, use or other dealings in this Software without prior written authorization from
 * Advanced Micro Devices, Inc.
 */

/* Additional Contributions:
 *
 * Added by: Adewale Adetomi
 * Date: 2024
 * 
 * Contributions include all the functions after read_data_from_file.
 * 
 * These modifications are also provided under the MIT License.
 */

#pragma once

#include <vector>
#include <complex>

static const float kFloatTolerance = 0.009;

// Compare 2 floats
inline bool isSame(float a, float b) {
    //    return a==b;

    if (a != 0.0) {
        return (fabs((a - b) / a) <= kFloatTolerance);
    } else {
        return a == b;
    }
}

template <typename T>
size_t num_bytes(const std::vector<T>& vec) {
    return sizeof(T) * vec.size();
}

template <typename T>
void check_size(std::string name, const std::vector<T>& vec) {
    if (num_bytes(vec) / 1024 > 128) {
        printf("Error: size of %s (%dKB) exceeds the capacity of test harness buffers (128KB).\n", name.c_str(),
               num_bytes(vec) / 1024);
        exit(-1);
    }
}

template <typename T>
void read_data_from_file(const std::string file_name, std::vector<T>& vec) {
    std::ifstream input_file(file_name, std::ifstream::in);
    if (!input_file.is_open()) {
        printf("Failed to open data file %s for reading\n", file_name.c_str());
        throw std::runtime_error("Failed to open data file.\n");
    }

    T val;
    while (input_file >> val) {
        vec.push_back(val);
    }

    input_file.close();
}

template <typename T>
void read_data_from_file(const std::string file_name, std::vector<std::complex<T> >& vec) {
    std::ifstream input_file(file_name, std::ifstream::in);
    if (!input_file.is_open()) {
        printf("Failed to open data file %s for reading\n", file_name.c_str());
        throw std::runtime_error("Failed to open data file.\n");
    }

    T val;
    std::complex<T> cpx;
    while (input_file >> val) {
        cpx.real(val);
        if (input_file >> val) {
            cpx.imag(val);
        } else {
            cpx.imag(0);
        }
        vec.push_back(cpx);
    }

    input_file.close();
}

// Function to create directories recursively
void create_directories(const std::string& path) {
    size_t pos = 0;
    std::string current_path;

    while ((pos = path.find('/', pos)) != std::string::npos) {
        current_path = path.substr(0, pos++);
        if (!current_path.empty() && mkdir(current_path.c_str(), 0755) && errno != EEXIST) {
            throw std::runtime_error("Failed to create directory: " + current_path + " - " + strerror(errno));
        }
    }

    if (mkdir(path.c_str(), 0755) && errno != EEXIST) {
        throw std::runtime_error("Failed to create directory: " + path + " - " + strerror(errno));
    }
}

template <typename T>
void write_data_to_file(const std::string file_name, const std::vector<T>& vec) {
    std::ofstream output_file(file_name, std::ofstream::out);
    if (!output_file.is_open()) {
        printf("Failed to open data file %s for writing\n", file_name.c_str());
        throw std::runtime_error("Failed to open data file.\n");
    }

    for (const auto& val : vec) {
        output_file << val << "\n";
    }

    output_file.close();
}

void convert_float_vector_to_bfloat16_array(const std::vector<float>& input_vector, uint16_t* output_array) {
    size_t length = input_vector.size();
    for (size_t i = 0; i < length; ++i) {
        uint32_t float_bits;
        std::memcpy(&float_bits, &input_vector[i], sizeof(float_bits));

        uint16_t sign = (float_bits >> 16) & 0x8000;     // Top bit of 32-bit value
        uint16_t exponent = (float_bits >> 23) & 0xFF;   // Exponent
        uint16_t mantissa = (float_bits >> 16) & 0x007F; // Top 7 bits of mantissa

        output_array[i] = sign | (exponent << 7) | mantissa;
    }
}