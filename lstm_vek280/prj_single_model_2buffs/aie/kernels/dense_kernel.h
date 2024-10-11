#pragma once

#include "adf.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
using namespace adf;


template<size_t NUM_NEURONS, size_t NUM_FEATURES, typename ELEMENT_TYPE, typename ELEMENT_ACC_TYPE, size_t VSIZE, size_t TDMUX> class DENSE_KERNEL
{
private:

    alignas(64) ELEMENT_TYPE tmp_acc[NUM_NEURONS];
    alignas(64) ELEMENT_TYPE y_buf[NUM_NEURONS];

    uint32 step_counter=0;
	uint32 first_data=1;
	uint32 counter=0;
    uint32 tdmux_pos=0;

public:
    DENSE_KERNEL();

    void dense_func(
    		input_async_buffer<ELEMENT_TYPE,extents<NUM_NEURONS*(NUM_FEATURES+1)*TDMUX>> &model_in,
			input_stream<float> *x_in,
			output_stream<float> *y_out);

    //user needs to write this function to register necessary info
    static void registerKernelClass()
    {
        REGISTER_FUNCTION(DENSE_KERNEL::dense_func);
    }
};
