#pragma once

#include <lstm_kernel.h>
#include <dense_kernel.h>
#include <config.h>

#include "adf.h"

using namespace adf;

class mygraph : public graph
{
public:
    input_gmio lstm_in1a;
    input_gmio lstm_in1b;
    input_plio xin;
    input_gmio lstm_in2a;
    input_gmio lstm_in2b;
    input_gmio dense_in;
    output_plio hout;

    kernel k1;
    kernel k2;
    kernel k3;

    mygraph()
    {
        lstm_in1a   = input_gmio::create("lstm_in1a", 64/*burst_lenght*/, 1000/*bandwidth*/);
        lstm_in1b   = input_gmio::create("lstm_in1b", 64, 1000);
        xin         = input_plio::create("DataIn_0", plio_32_bits, "data/A2/xin_A-2.txt");
        lstm_in2a   = input_gmio::create("lstm_in2a", 64, 1000);
        lstm_in2b   = input_gmio::create("lstm_in2b", 64, 1000);
        dense_in    = input_gmio::create("dense_in", 64, 1000);
        hout        = output_plio::create("DataOut_0", plio_32_bits, "output/hout_A-2.txt");

        k1 = kernel::create_object<LSTM_KERNEL<NNEURONS1,NSENSORS,bfloat16,accfloat,64,16,NTDM,0>>();
        runtime<ratio>(k1) = 0.3;
        source(k1) = "aie/kernels/lstm_kernel.cpp";

        k2 = kernel::create_object<LSTM_KERNEL<NNEURONS2,NNEURONS1,bfloat16,accfloat,64,16,NTDM,1>>();
        runtime<ratio>(k2) = 0.5;
        source(k2) = "aie/kernels/lstm_kernel.cpp";

        k3 = kernel::create_object<DENSE_KERNEL<NDENSE,NNEURONS2,bfloat16,accfloat,16,NTDM>>();
        runtime<ratio>(k3) = 0.1;
        source(k3) = "aie/kernels/dense_kernel.cpp";

        connect net1 (lstm_in1a.out[0],k1.in[0]);
        single_buffer(k1.in[0]);
        connect net1a (lstm_in1b.out[0],k1.in[1]);
        single_buffer(k1.in[1]);
       
        connect<stream> net2 (xin.out[0], k1.in[2]);
        connect<stream> net3 (k1.out[0], k2.in[2]);

        connect net4 (lstm_in2a.out[0], k2.in[0]);
        single_buffer(k2.in[0]);
        connect net4a (lstm_in2b.out[0], k2.in[1]);
        single_buffer(k2.in[1]);
        connect<stream> net5 (k2.out[0],k3.in[1]);

        connect net6 (dense_in.out[0], k3.in[0]);
        single_buffer(k3.in[0]);
        connect<stream> net7(k3.out[0],hout.in[0]);
    }
};