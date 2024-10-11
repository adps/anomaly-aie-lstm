#include <fstream>
#include <cstring>
#include <sys/stat.h>
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_graph.h"
#include "xrt/xrt_aie.h"
#include "../aie/config.h"
#include "models.h"
#include "utils.hpp"



int main(int argc, char* argv[])
{
	//////////////////////////////////////////
	// Open xclbin
	//////////////////////////////////////////
    char* xclbinFile = argv[1];
	auto device = xrt::device(0);
	if(device == nullptr) {
		throw std::runtime_error("No valid device handle found. Ensure the correct xclOpen index is being used.");
	}
	auto xclbin_uuid = device.load_xclbin(xclbinFile);
	auto ghdl = xrt::graph(device, xclbin_uuid, "g");

	printf("xclbinFile loaded!\n");

	//////////////////////////////////////////
	// Prepare data
	//////////////////////////////////////////
    
    size_t num_inputs[NPAR];
	size_t num_outputs[NPAR];

    size_t LSTM_1A_BLOCK_SIZE_in_Bytes[NPAR];
    size_t LSTM_1B_BLOCK_SIZE_in_Bytes[NPAR];
    size_t LSTM_2A_BLOCK_SIZE_in_Bytes[NPAR];
    size_t LSTM_2B_BLOCK_SIZE_in_Bytes[NPAR];
    size_t DENSE_BLOCK_SIZE_in_Bytes[NPAR];
    size_t XIN_NUM_OF_SAMPLES[NPAR];
    size_t HOUT_NUM_OF_SAMPLES[NPAR];
    size_t ITERATION[NPAR];
    // size_t HOUT_NUM_OF_SAMPLES_TO_SAVE[NPAR];

    /**
     * Note that an array of array [NPAR][NTDM] is not needed here as
     * all the TDM models share the same graph and all their model 
     * data are loaded together. The read_data_from_file() function
     * is used to append all the model data for each graph
     */
    std::array<std::vector<float>, NPAR> lstm1a;
    std::array<std::vector<float>, NPAR> lstm1b;
    std::array<std::vector<float>, NPAR> inputs;
    std::array<std::vector<float>, NPAR> lstm2a;
    std::array<std::vector<float>, NPAR> lstm2b;
    std::array<std::vector<float>, NPAR> dense;
    std::array<std::vector<float>, NPAR> outputs;

    // Declare variables for model buffer mapping
    std::array<xrt::aie::bo, NPAR> lstm1a_buffer;
    std::array<xrt::aie::bo, NPAR> lstm1b_buffer;
    std::array<xrt::aie::bo, NPAR> lstm2a_buffer;
    std::array<xrt::aie::bo, NPAR> lstm2b_buffer;
    std::array<xrt::aie::bo, NPAR> dense_buffer;
    std::array<uint16_t*, NPAR> lstm1aArray;
    std::array<uint16_t*, NPAR> lstm1bArray;
    std::array<uint16_t*, NPAR> lstm2aArray;
    std::array<uint16_t*, NPAR> lstm2bArray;
    std::array<uint16_t*, NPAR> denseArray;

    // Declare variables for input memory mapping
    std::array<xrt::bo, NPAR> in_bohdl;
	std::array<uint32_t*, NPAR> in_bomapped;

    // Declare variables for output memory mapping
    std::array<xrt::bo, NPAR> out_bohdl;
    std::array<float*, NPAR> out_bomapped;

    // Declare data mover kernel control variables
    std::array<xrt::kernel, NPAR> mm2s_khdl;
    std::array<xrt::kernel, NPAR> s2mm_khdl;

    // Declare data mover kernel run control variables
    std::array<xrt::run, NPAR> s2mm_rhdl;
    std::array<xrt::run, NPAR> mm2s_rhdl;

    char label[64];

	for (size_t i = 0; i < NPAR; i++) {

		for (size_t j = 0; j < NTDM; j++) {
            std::string fname = "./data/" + std::string(PROJECT_MODELS[i][j]);
            read_data_from_file(fname + "/lstm1_x2fp_part1.txt", lstm1a[i]); // TDM models are appended
            read_data_from_file(fname + "/lstm1_x2fp_part2.txt", lstm1b[i]);
            read_data_from_file(fname + "/lstm2_x2fp_part1.txt", lstm2a[i]);
            read_data_from_file(fname + "/lstm2_x2fp_part2.txt", lstm2b[i]);
            read_data_from_file(fname + "/dense_x2fp.txt", dense[i]);
        }

        if (1 == NTDM) {
            std::string fname = "./data/" + std::string(PROJECT_MODELS[i][0]);
            read_data_from_file(fname + "/xin.txt", inputs[i]);
        } else {
            // Input data has to be interleaved
            for (size_t k = 0; k < NITERATIONS; k++) {
                for (size_t j = 0; j < NTDM; j++) {
                    std::string fname = "./data/" + std::string(PROJECT_MODELS[i][j]);
                    read_data_from_file(fname + "/xin.txt", inputs[i], true, k*NSENSORS, NSENSORS);
                }
            }
        }

        LSTM_1A_BLOCK_SIZE_in_Bytes[i]  = lstm1a[i].size() * sizeof(float);
        LSTM_1B_BLOCK_SIZE_in_Bytes[i]  = lstm1b[i].size() * sizeof(float);
        LSTM_2A_BLOCK_SIZE_in_Bytes[i]  = lstm2a[i].size() * sizeof(float);
        LSTM_2B_BLOCK_SIZE_in_Bytes[i]  = lstm2b[i].size() * sizeof(float);
        DENSE_BLOCK_SIZE_in_Bytes[i]    = dense[i].size()  * sizeof(float);
        XIN_NUM_OF_SAMPLES[i]           = inputs[i].size();

        HOUT_NUM_OF_SAMPLES[i]          = (NDENSE * XIN_NUM_OF_SAMPLES[i]) / NSENSORS;
        // ITERATION[i]				     = HOUT_NUM_OF_SAMPLES[i] / NDENSE;
        // HOUT_NUM_OF_SAMPLES_TO_SAVE[i]  = ((XIN_NUM_OF_SAMPLES[i] / NSENSORS) - SAMPLE_SIZE - NPREDICTIONS) * NDENSE;

        outputs[i].resize(HOUT_NUM_OF_SAMPLES[i]);

        num_inputs[i] = inputs[i].size();
        num_outputs[i] = outputs[i].size();

        /** 
         * Make sure the number of samples agrees with what the graph expects.
         * One graph iteration requires NSENSORS float inputs and produces NDENSE float outputs.
         * These values are defined in aie/config.h
         */
        if (num_inputs[i] % (NSENSORS*NTDM) != 0) {
			printf("Error: number of inputs (%ld) must be a multiple of NSENSORS*NTDM (%d)\n", num_inputs[i], (NSENSORS*NTDM));
			return -1;
		}
		if (num_outputs[i] % (NDENSE*NTDM) != 0) {
			printf("Error: number of ouputs (%ld) must be a multiple of NDENSE*NTDM (%d)\n", num_outputs[i], (NDENSE*NTDM));
			return -1;
		}
		
        lstm1a_buffer[i] = xrt::aie::bo (device, LSTM_1A_BLOCK_SIZE_in_Bytes[i], xrt::bo::flags::normal, /*memory group*/0); //Only non-cacheable buffer is supported
        lstm1aArray[i] = lstm1a_buffer[i].map<uint16_t *>();
        
        lstm1b_buffer[i] = xrt::aie::bo (device, LSTM_1B_BLOCK_SIZE_in_Bytes[i], xrt::bo::flags::normal, 0);
        lstm1bArray[i] = lstm1b_buffer[i].map<uint16_t *>();

        lstm2a_buffer[i] = xrt::aie::bo (device, LSTM_2A_BLOCK_SIZE_in_Bytes[i], xrt::bo::flags::normal, 0);
        lstm2aArray[i] = lstm2a_buffer[i].map<uint16_t *>();

        lstm2b_buffer[i] = xrt::aie::bo (device, LSTM_2B_BLOCK_SIZE_in_Bytes[i], xrt::bo::flags::normal, 0);
        lstm2bArray[i] = lstm2b_buffer[i].map<uint16_t *>();

        dense_buffer[i] = xrt::aie::bo (device, DENSE_BLOCK_SIZE_in_Bytes[i], xrt::bo::flags::normal, 0);
        denseArray[i] = dense_buffer[i].map<uint16_t *>();
    

        //////////////////////////////////////////
        // Initialization
        // Cast the model to bfloat16
        // and initialize the input array
        //////////////////////////////////////////
        convert_float_vector_to_bfloat16_array(lstm1a[i], lstm1aArray[i]);
        convert_float_vector_to_bfloat16_array(lstm1b[i], lstm1bArray[i]);
        convert_float_vector_to_bfloat16_array(lstm2a[i], lstm2aArray[i]);
        convert_float_vector_to_bfloat16_array(lstm2b[i], lstm2bArray[i]);
        convert_float_vector_to_bfloat16_array(dense[i], denseArray[i]);

        sprintf(label,"g.lstm_in1a[%zu]", i);
        lstm1a_buffer[i].async(label, XCL_BO_SYNC_BO_GMIO_TO_AIE, LSTM_1A_BLOCK_SIZE_in_Bytes[i], /*offset*/0);
        sprintf(label,"g.lstm_in1b[%zu]", i);
        lstm1b_buffer[i].async(label, XCL_BO_SYNC_BO_GMIO_TO_AIE, LSTM_1B_BLOCK_SIZE_in_Bytes[i], 0);
        sprintf(label,"g.lstm_in2a[%zu]", i);
        lstm2a_buffer[i].async(label, XCL_BO_SYNC_BO_GMIO_TO_AIE, LSTM_2A_BLOCK_SIZE_in_Bytes[i], 0);
        sprintf(label,"g.lstm_in2b[%zu]", i);
        lstm2b_buffer[i].async(label, XCL_BO_SYNC_BO_GMIO_TO_AIE, LSTM_2B_BLOCK_SIZE_in_Bytes[i], 0);
        sprintf(label,"g.dense_in[%zu]", i);
        dense_buffer[i].async(label, XCL_BO_SYNC_BO_GMIO_TO_AIE, DENSE_BLOCK_SIZE_in_Bytes[i], 0);


        //////////////////////////////////////////
        // Input Memory
        // Allocating the input size of XIN_NUM_OF_SAMPLES to MM2S
        // MM2S module transfers input data from PL to the AI Engine
        //////////////////////////////////////////
		in_bohdl[i] = xrt::bo(device, XIN_NUM_OF_SAMPLES[i] * sizeof(float), 0, 0);
		in_bomapped[i] = in_bohdl[i].map<uint32_t *>();
		memcpy(in_bomapped[i], inputs[i].data(), XIN_NUM_OF_SAMPLES[i] * sizeof(float));
    	printf("Input memory virtual addr %p\n", (void*)in_bomapped[i]);

		std::cout << "in_bohdl[" << i << "] sync started" << std::endl;
		in_bohdl[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
		std::cout << "in_bohdl[" << i << "] sync done" << std::endl;


        //////////////////////////////////////////
        // output memory
        // Allocating the output size of HOUT_NUM_OF_SAMPLES to S2MM
        // S2MM module receives the output data from AI Engine
        //////////////////////////////////////////
        out_bohdl[i] = xrt::bo(device, HOUT_NUM_OF_SAMPLES[i] * sizeof(float), 0, 0);
        out_bomapped[i] = out_bohdl[i].map<float*>();
        for (size_t j = 0; j < HOUT_NUM_OF_SAMPLES[i]; ++j) {
            out_bomapped[i][j] = 0.;
        }
        printf("Output memory virtual addr %p\n", (void*)out_bomapped[i]);


        //////////////////////////////////////////
        // mm2s ip
        // Using the xrtPLKernelOpen function to manually control the PL Kernel
        // that is outside of the AI Engine graph
        //////////////////////////////////////////
        sprintf(label,"mm2s:{mm2s_%zu}", i);
        mm2s_khdl[i] = xrt::kernel(device, xclbin_uuid, label);
    

        //////////////////////////////////////////
        // s2mm ip
        // Using the xrtPLKernelOpen function to manually control the PL Kernel
        // that is outside of the AI Engine graph
        //////////////////////////////////////////
        sprintf(label,"s2mm:{s2mm_%zu}", i);
        s2mm_khdl[i] = xrt::kernel(device, xclbin_uuid, label);


        //////////////////////////////////////////
        // Start data mover kernels
        //////////////////////////////////////////
        std::cout << "Run s2mm and mm2s" << std::endl;
        s2mm_rhdl[i] = s2mm_khdl[i](out_bohdl[i], nullptr, HOUT_NUM_OF_SAMPLES[i]); // start s2mm
        mm2s_rhdl[i] = mm2s_khdl[i](in_bohdl[i], nullptr, XIN_NUM_OF_SAMPLES[i]); // start mm2s

	}


	//////////////////////////////////////////
	// Graph execution for AIE
	//////////////////////////////////////////
	ghdl.run(NITERATIONS);
	std::cout << "Graph running..." << std::endl;
	// Wait for graph for some cycles
	ghdl.end();
	std::cout << "Graph run completed!" << std::endl;


    //////////////////////////////////////////
    // Wait for mm2s and s2mm runs
    //////////////////////////////////////////
	for (size_t i = 0; i < NPAR; i++) {
		mm2s_rhdl[i].wait();
		s2mm_rhdl[i].wait();
		out_bohdl[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
	}
	std::cout << "mm2s and s2mm runs completed!" << std::endl;
	std::cout << "GMIO transactions finished" << std::endl;


    //////////////////////////////////////////
    // Write output to file
    //////////////////////////////////////////  
	for (size_t i = 0; i < NPAR; i++) {
		for(size_t j = 0; j < HOUT_NUM_OF_SAMPLES[i]; j++) {
			outputs[i][j] = out_bomapped[i][j];
		}
	}

	std::string outputPath = "output";
    std::array<std::array<std::string, NTDM>, NPAR> outFilename;

    for (size_t i = 0; i < NPAR; ++i) {
        for (size_t j = 0; j < NTDM; ++j) {
            std::string fname = outputPath + "/hout_" + PROJECT_MODELS[i][j] + ".txt";
            outFilename[i][j] = fname;
        }
    }

	try {
        create_directories(outputPath);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

	for (size_t i = 0; i < NPAR; i++) {
		size_t portion_size = outputs[i].size() / NTDM;
		for (size_t j = 0; j < NTDM; ++j) {
			try {
				auto start_iter = outputs[i].begin() + j * portion_size;
				auto end_iter = (j == NTDM - 1) ? outputs[i].end() : start_iter + portion_size;
				// write_data_to_file(outFilename[i][j], std::vector<float>(start_iter, end_iter));
                std::vector<float> vec(start_iter + ((SAMPLE_SIZE + NPREDICTIONS) * NDENSE), end_iter);
                write_data_to_file(outFilename[i][j], vec);
				std::cout << "Output for model [" << i << "][" << j << "] successfully written to " << outFilename[i][j] << std::endl;
			} catch (const std::runtime_error& e) {
				std::cerr << e.what() << std::endl;
			}
		}
	}


	return 0;
}
