#pragma once

#include "adf.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>

using namespace adf;

template<size_t NUM_NEURONS, size_t NUM_FEATURES, typename ELEMENT_TYPE, typename ELEMENT_ACC_TYPE, size_t VSIZE, size_t VSIZEH, size_t TDMUX, size_t PADDING> class LSTM_KERNEL
{
private:
    alignas(128) ELEMENT_TYPE h[TDMUX][NUM_NEURONS];
    alignas(128) ELEMENT_TYPE c[TDMUX][NUM_NEURONS];
    alignas(128) ELEMENT_TYPE tmp_acc[NUM_NEURONS*4];

    uint32 step_counter=0;
    uint32 tdmux_pos=0;
    uint32 first_data=1;

    aie::vector<ELEMENT_TYPE,VSIZEH> logistic_fn(
       		aie::vector<ELEMENT_TYPE,VSIZEH> x);
    aie::vector<ELEMENT_TYPE,VSIZEH> htan_fn(
            aie::vector<ELEMENT_TYPE,VSIZEH> x);

public:
    LSTM_KERNEL();

    void lstm_func(
    		input_async_buffer<ELEMENT_TYPE,extents<NUM_NEURONS*4*(NUM_FEATURES+NUM_NEURONS+1+PADDING)/2*TDMUX>> &model1_in,
            input_async_buffer<ELEMENT_TYPE,extents<NUM_NEURONS*4*(NUM_FEATURES+NUM_NEURONS+1-PADDING)/2*TDMUX>> &model2_in,
			input_stream<float> *x_in,	
			output_stream<float> *h_out);

    //user needs to write this function to register necessary info
    static void registerKernelClass()
    {
        REGISTER_FUNCTION(LSTM_KERNEL::lstm_func);
    }
};
