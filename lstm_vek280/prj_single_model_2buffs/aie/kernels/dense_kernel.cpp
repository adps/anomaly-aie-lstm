#include <dense_kernel.h>
//#include <stdio.h>
using namespace adf;

template<size_t NUM_NEURONS, size_t NUM_FEATURES, typename ELEMENT_TYPE,typename ELEMENT_ACC_TYPE, size_t VSIZE, size_t TDMUX> DENSE_KERNEL<NUM_NEURONS, NUM_FEATURES, ELEMENT_TYPE, ELEMENT_ACC_TYPE,VSIZE,TDMUX> :: DENSE_KERNEL()
{
	for (int i = 0; i < NUM_NEURONS; i++) {
	        y_buf[i] = 0;
	}
	step_counter = 0;
	first_data =1;
	counter = 0;
    tdmux_pos =0;

}

template<size_t NUM_NEURONS, size_t NUM_FEATURES, typename ELEMENT_TYPE, typename ELEMENT_ACC_TYPE, size_t VSIZE, size_t TDMUX> void DENSE_KERNEL<NUM_NEURONS, NUM_FEATURES, ELEMENT_TYPE, ELEMENT_ACC_TYPE,VSIZE,TDMUX>::dense_func(
		input_async_buffer<ELEMENT_TYPE,extents<NUM_NEURONS*(NUM_FEATURES+1)*TDMUX>> &model_in,
		input_stream<float> *x_in,
		output_stream<float> *y_out)
{
	ELEMENT_TYPE x;
	{
		if (first_data != 0)
			{
				model_in.acquire();
				first_data=0;
			}
		do {// use do..until to repeat loop across all steps
		if (step_counter==0)
			{
			printf("*************\nDense Iteration %d\n",counter);
			counter++;
			}

		if (step_counter<NUM_FEATURES)
		{
			float x_tmp;
			x_tmp=readincr(x_in);
			x= (bfloat16) x_tmp;
		}
	
		aie::accum<ELEMENT_ACC_TYPE,VSIZE> acc;
		aie::vector<ELEMENT_TYPE,VSIZE> vec;
		aie::vector<ELEMENT_TYPE,VSIZE> vec3;

		auto pmodel = aie::begin_vector<VSIZE>(model_in);
		pmodel+=NUM_NEURONS*step_counter/VSIZE+(tdmux_pos*(NUM_NEURONS*(NUM_FEATURES+1)/VSIZE));

		for (int k=0; k<NUM_NEURONS; k+= VSIZE)
		{
			//readincr(model_in, vec);
			vec = *pmodel++;
			//{ELEMENT_TYPE tmp_vec[VSIZE];aie::store_v(tmp_vec,vec);for(int i=0;i<VSIZE;i++) printf("%2.3f ",(float) tmp_vec[i]);printf("\n");}

			if (step_counter==0) {
				// First Feature Input  - use Mul to "zero" accumulator
				acc = aie::mul(vec,x);
				aie::store_v(tmp_acc+k,acc.template to_vector<ELEMENT_TYPE>());
			} else if (step_counter< NUM_FEATURES) {
				// Remaining Feature Inputs
				vec3= aie::load_v<VSIZE>(tmp_acc+k);
				acc.from_vector(vec3);
				acc = aie::mac(acc,vec,x);
				aie::store_v(tmp_acc+k,acc.template to_vector<ELEMENT_TYPE>());
				//{ELEMENT_TYPE tmp_vec[VSIZE];aie::store_v(tmp_vec,acc.template to_vector<ELEMENT_TYPE>());for(int i=0;i<VSIZE;i++) printf("%2.3f ",(float) tmp_vec[i]);printf("\n");}
			}
			else {
				// Bias
				vec3= aie::load_v<VSIZE>(tmp_acc+k);
				acc.from_vector(vec3);
				acc = aie::add(acc,vec);
				aie::store_v(y_buf+k,acc.template to_vector<ELEMENT_TYPE>());
			}
		}

		if ((step_counter<NUM_NEURONS)&&(first_data==0))
		{
			float y_data;
			y_data = (float) y_buf[step_counter];
			writeincr(y_out,y_data);
			
		}
		if (step_counter < NUM_FEATURES)
			step_counter++;
		else
		{
			if (tdmux_pos != TDMUX-1)
				tdmux_pos++;
			else
			{
				tdmux_pos = 0;		
			}
			step_counter = 0;
		}

		} while (step_counter != 0);
	}
}

