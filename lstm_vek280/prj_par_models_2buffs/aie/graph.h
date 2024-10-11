#pragma once

#include <lstm_kernel.h>
#include <dense_kernel.h>
#include <config.h>

#include "adf.h"

using namespace adf;

class mygraph : public graph
{
public:
    input_gmio  lstm_in1a[NPAR];
    input_gmio  lstm_in1b[NPAR];
    input_plio  xin[NPAR];
    input_gmio  lstm_in2a[NPAR];
    input_gmio  lstm_in2b[NPAR];
    input_gmio  dense_in[NPAR];
    output_plio hout[NPAR];

    kernel k1[NPAR];
    kernel k2[NPAR];
    kernel k3[NPAR];

    mygraph()
    {
        char label[64];
        char fname[64];

        for (size_t i=0; i<NPAR; i++) {
            sprintf(label, "lstm_in1a_%d", i);
            lstm_in1a[i] = input_gmio::create(label, 64/*burst_lenght*/, 1000/*bandwidth*/); //256/*burst_lenght*/,10000/*bandwidth*/
            sprintf(label, "lstm_in1b_%d", i);
            lstm_in1b[i] = input_gmio::create(label, 64, 1000);
            sprintf(label, "DataIn_%d", i);
            sprintf(fname, "data/xin_A-2_%d.txt", i);
            xin[i]  = input_plio::create(label, plio_32_bits, fname);
            sprintf(label, "lstm_in2a_%d", i);
            lstm_in2a[i] = input_gmio::create(label, 64, 1000);
            sprintf(label, "lstm_in2b_%d", i);
            lstm_in2b[i] = input_gmio::create(label, 64, 1000);
            sprintf(label, "dense_in_%d", i);
            dense_in[i]  = input_gmio::create(label, 64, 1000);
            sprintf(label, "DataOut_%d", i);
            sprintf(fname, "data/hout_A-2_%d.txt", i);
            hout[i] = output_plio::create(label, plio_32_bits, fname);

            k1[i] = kernel::create_object<LSTM_KERNEL<NNEURONS1,NSENSORS,bfloat16,accfloat,64,16,NTDM,0>>();
            runtime<ratio>(k1[i]) = 0.3;
            source(k1[i]) = "aie/kernels/lstm_kernel.cpp";

            k2[i] = kernel::create_object<LSTM_KERNEL<NNEURONS2,NNEURONS1,bfloat16,accfloat,64,16,NTDM,1>>();
            runtime<ratio>(k2[i]) = 0.5;
            source(k2[i]) = "aie/kernels/lstm_kernel.cpp";

            k3[i] = kernel::create_object<DENSE_KERNEL<NDENSE,NNEURONS2,bfloat16,accfloat,16,NTDM>>();
            runtime<ratio>(k3[i]) = 0.1;
            source(k3[i]) = "aie/kernels/dense_kernel.cpp";


            connect net1 (lstm_in1a[i].out[0],k1[i].in[0]);
            single_buffer(k1[i].in[0]);
            connect net1a (lstm_in1b[i].out[0],k1[i].in[1]);
            single_buffer(k1[i].in[1]);
        
            connect<stream> net2 (xin[i].out[0], k1[i].in[2]);

            connect<stream> net3 (k1[i].out[0], k2[i].in[2]);

            connect net4 (lstm_in2a[i].out[0], k2[i].in[0]);
            single_buffer(k2[i].in[0]);
            connect net4a (lstm_in2b[i].out[0], k2[i].in[1]);
            single_buffer(k2[i].in[1]);
            connect<stream> net5 (k2[i].out[0],k3[i].in[1]);

            connect net6 (dense_in[i].out[0], k3[i].in[0]);
            single_buffer(k3[i].in[0]);
            connect<stream> net7(k3[i].out[0],hout[i].in[0]);
        }
    }
};