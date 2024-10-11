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
    std::vector<float> lstm1a;
    std::vector<float> lstm1b;
    std::vector<float> inputs;
    std::vector<float> lstm2a;
    std::vector<float> lstm2b;
    std::vector<float> dense;
    
    std::string channelId = PROJECT_MODELS[0][0];

    read_data_from_file("./data/" + channelId + "/lstm1_x2fp_part1.txt", lstm1a);
    read_data_from_file("./data/" + channelId + "/lstm1_x2fp_part2.txt", lstm1b);
	read_data_from_file("./data/" + channelId + "/xin.txt", inputs);
    read_data_from_file("./data/" + channelId + "/lstm2_x2fp_part1.txt", lstm2a);
    read_data_from_file("./data/" + channelId + "/lstm2_x2fp_part2.txt", lstm2b);
    read_data_from_file("./data/" + channelId + "/dense_x2fp.txt", dense);

    const size_t LSTM_1A_BLOCK_SIZE_in_Bytes  = lstm1a.size() * sizeof(float);
    const size_t LSTM_1B_BLOCK_SIZE_in_Bytes  = lstm1b.size() * sizeof(float);
    const size_t LSTM_2A_BLOCK_SIZE_in_Bytes  = lstm2a.size() * sizeof(float);
    const size_t LSTM_2B_BLOCK_SIZE_in_Bytes  = lstm2b.size() * sizeof(float);
    const size_t DENSE_BLOCK_SIZE_in_Bytes    = dense.size()  * sizeof(float);
    const size_t XIN_NUM_OF_SAMPLES           = inputs.size();

    const size_t HOUT_NUM_OF_SAMPLES          = (NDENSE * XIN_NUM_OF_SAMPLES) / NSENSORS;
    const size_t ITERATION 				      = HOUT_NUM_OF_SAMPLES / NDENSE;
    const size_t HOUT_NUM_OF_SAMPLES_TO_SAVE  = ((XIN_NUM_OF_SAMPLES / NSENSORS) - SAMPLE_SIZE - NPREDICTIONS) * NDENSE;

    auto num_inputs = inputs.size();

    /** 
	 * Make sure the number of samples agrees with what the graph expects.
     * One graph iteration requires NSENSORS float inputs and produces NDENSE float outputs.
     * These values are defined in aie/config.h
	 */
    if (num_inputs % NSENSORS != 0) {
        printf("Error: number of inputs (%ld) must be a multiple of NSENSORS (%d)\n", num_inputs, NSENSORS);
        return -1;
    }

	auto lstm1a_buffer = xrt::aie::bo (device, LSTM_1A_BLOCK_SIZE_in_Bytes, xrt::bo::flags::normal, /*memory group*/0); //Only non-cacheable buffer is supported
	uint16_t * lstm1aArray = lstm1a_buffer.map<uint16_t *>();
	auto lstm1b_buffer = xrt::aie::bo (device, LSTM_1B_BLOCK_SIZE_in_Bytes, xrt::bo::flags::normal, 0);
	uint16_t * lstm1bArray = lstm1b_buffer.map<uint16_t *>();
	auto lstm2a_buffer = xrt::aie::bo (device, LSTM_2A_BLOCK_SIZE_in_Bytes, xrt::bo::flags::normal, 0);
	uint16_t * lstm2aArray = lstm2a_buffer.map<uint16_t *>();
	auto lstm2b_buffer = xrt::aie::bo (device, LSTM_2B_BLOCK_SIZE_in_Bytes, xrt::bo::flags::normal, 0);
	uint16_t * lstm2bArray = lstm2b_buffer.map<uint16_t *>();
	auto dense_buffer = xrt::aie::bo (device, DENSE_BLOCK_SIZE_in_Bytes, xrt::bo::flags::normal, 0);
	uint16_t * denseArray = dense_buffer.map<uint16_t *>();

	//////////////////////////////////////////
	// Initialization
	// Cast the model to bfloat16
	// and initialize the input array
	//////////////////////////////////////////
    convert_float_vector_to_bfloat16_array(lstm1a, lstm1aArray);
	convert_float_vector_to_bfloat16_array(lstm1b, lstm1bArray);
	convert_float_vector_to_bfloat16_array(lstm2a, lstm2aArray);
	convert_float_vector_to_bfloat16_array(lstm2b, lstm2bArray);
	convert_float_vector_to_bfloat16_array(dense, denseArray);

	lstm1a_buffer.async("g.lstm_in1a", XCL_BO_SYNC_BO_GMIO_TO_AIE, LSTM_1A_BLOCK_SIZE_in_Bytes, /*offset*/0);
	lstm1b_buffer.async("g.lstm_in1b", XCL_BO_SYNC_BO_GMIO_TO_AIE, LSTM_1B_BLOCK_SIZE_in_Bytes, 0);
	lstm2a_buffer.async("g.lstm_in2a", XCL_BO_SYNC_BO_GMIO_TO_AIE, LSTM_2A_BLOCK_SIZE_in_Bytes, 0);
	lstm2b_buffer.async("g.lstm_in2b", XCL_BO_SYNC_BO_GMIO_TO_AIE, LSTM_2B_BLOCK_SIZE_in_Bytes, 0);
	dense_buffer.async("g.dense_in", XCL_BO_SYNC_BO_GMIO_TO_AIE, DENSE_BLOCK_SIZE_in_Bytes, 0);

	//////////////////////////////////////////
    // Input Memory
    // Allocating the input size of XIN_NUM_OF_SAMPLES to MM2S
    // MM2S module transfers input data from PL to the AI Engine
    //////////////////////////////////////////
    auto in_bohdl = xrt::bo(device, XIN_NUM_OF_SAMPLES * sizeof(float), 0, 0);
    auto in_bomapped = in_bohdl.map<uint32_t*>();
    if (in_bomapped == nullptr) {
        std::cerr << "Failed to map input buffer!" << std::endl;
        return -1;  // or handle the error as appropriate
    }
	memcpy(in_bomapped, inputs.data(), XIN_NUM_OF_SAMPLES * sizeof(float));
    printf("Input memory virtual addr: %p\n", (void*)in_bomapped);
 
    std::cout << "in_bohdl sync started\n";
    in_bohdl.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::cout << "in_bohdl sync done\n";
 
    //////////////////////////////////////////
    // Output Memory
    // Allocating the output size of HOUT_NUM_OF_SAMPLES to S2MM
    // S2MM module receives the output data from AI Engine
    //////////////////////////////////////////
    auto out_bohdl = xrt::bo(device, HOUT_NUM_OF_SAMPLES * sizeof(float), 0, 0);
    auto out_bomapped = out_bohdl.map<float*>();
    if (out_bomapped == nullptr) {
        std::cerr << "Failed to map output buffer!" << std::endl;
        return -1;  // or handle the error as appropriate
    }
    for (size_t i = 0; i < HOUT_NUM_OF_SAMPLES; ++i) {
     out_bomapped[i] = 0.;
    }
    // memset(out_bomapped, 0xABCDEF00, sizeOut * sizeof(int));
    printf("Output memory virtual addr %p\n", (void*)out_bomapped);
 
    //////////////////////////////////////////
    // mm2s IP
    // Using the xrtPLKernelOpen function to manually control the PL Kernel
    // that is outside of the AI Engine graph
    //////////////////////////////////////////
    auto mm2s_khdl = xrt::kernel(device, xclbin_uuid, "mm2s");
 
    //////////////////////////////////////////
    // s2mm IP
    // Using the xrtPLKernelOpen function to manually control the PL Kernel
    // that is outside of the AI Engine graph
    //////////////////////////////////////////
    auto s2mm_khdl = xrt::kernel(device, xclbin_uuid, "s2mm");

	//////////////////////////////////////////
	// Graph execution for AIE
	//////////////////////////////////////////
	// Input data mover kernel run
    auto s2mm_rhdl = s2mm_khdl(out_bohdl, nullptr, HOUT_NUM_OF_SAMPLES); // start s2mm
    printf("s2mm run started\n");
    // Output data mover kernel run
    auto mm2s_rhdl = mm2s_khdl(in_bohdl, nullptr, XIN_NUM_OF_SAMPLES); // start mm2s
    printf("mm2s run started\n");
	// Grpah run
	ghdl.run(ITERATION);
	std::cout << "Graph running..." << std::endl;
	// Wait for graph for some cycles
	ghdl.end();
	
	std::cout << "Graph run completed!" << std::endl;

	//////////////////////////////////////////
    // Wait for mm2s done
    //////////////////////////////////////////  
    mm2s_rhdl.wait();
    std::cout << "mm2s completed!\n";
 
    //////////////////////////////////////////
    // Wait for s2mm done
    //////////////////////////////////////////  
    auto state = s2mm_rhdl.wait();
    std::cout << "s2mm completed with status(" << state << ")\n";
    out_bohdl.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

	//////////////////////////////////////////
    // Write output to file
    //////////////////////////////////////////   
	std::cout << "GMIO transactions finished" << std::endl;
	std::string outputPath = "output";
    std::string outFilename = outputPath + "/hout_" + channelId + ".txt";
    
	try {
        create_directories(outputPath);
        // Create a vector from out_bomapped
        std::vector<float> vec(out_bomapped + (HOUT_NUM_OF_SAMPLES - HOUT_NUM_OF_SAMPLES_TO_SAVE), out_bomapped + HOUT_NUM_OF_SAMPLES);
        write_data_to_file(outFilename, vec);
        std::cout << "Output successfully written to " << outFilename << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

	return 0;
}
