#include <lstm_kernel.h>
#include <stdio.h>
using namespace adf;

template<size_t NUM_NEURONS, size_t NUM_FEATURES, typename ELEMENT_TYPE,typename ELEMENT_ACC_TYPE, size_t VSIZE, size_t VSIZEH, size_t TDMUX,size_t PADDING> LSTM_KERNEL<NUM_NEURONS, NUM_FEATURES, ELEMENT_TYPE, ELEMENT_ACC_TYPE,VSIZE,VSIZEH,TDMUX,PADDING> :: LSTM_KERNEL()
{
    for (int k=0;k<TDMUX;k++)
    	for (int i = 0; i < NUM_NEURONS; i++) {
    		h[k][i] = 0;
    		c[k][i] = 0;
    	}
    step_counter = 0;
    tdmux_pos =0;
    first_data = 1;
}

// Logistic Function Approximation
template<size_t NUM_NEURONS, size_t NUM_FEATURES, typename ELEMENT_TYPE, typename ELEMENT_ACC_TYPE, size_t VSIZE, size_t VSIZEH, size_t TDMUX,size_t PADDING> aie::vector<ELEMENT_TYPE,VSIZEH> LSTM_KERNEL<NUM_NEURONS, NUM_FEATURES, ELEMENT_TYPE, ELEMENT_ACC_TYPE,VSIZE,VSIZEH,TDMUX,PADDING>::logistic_fn(aie::vector<ELEMENT_TYPE,VSIZEH> x)
{
	aie::vector<ELEMENT_TYPE,VSIZEH> vec;
	aie::accum<ELEMENT_ACC_TYPE,VSIZEH> acc;

	if (1) {
		acc = aie::mul(x, (ELEMENT_TYPE) 0.2);// Apply approx sigmoid [min(max(x/5+0.5,0),1)] to f
		vec = acc.template to_vector<ELEMENT_TYPE>();
		vec = aie::add(vec, (ELEMENT_TYPE) 0.5);// Apply approx sigmoid to f
		vec = aie::max(vec, (ELEMENT_TYPE) 0); // Apply approx sigmoid to f
		vec = aie::min(vec, (ELEMENT_TYPE) 1); // Apply approx sigmoid to f (TO DO support other ELEMENT_TYPE)
	}

	return vec;
}

// Hyperbolic Tangent Approximation
template<size_t NUM_NEURONS, size_t NUM_FEATURES, typename ELEMENT_TYPE, typename ELEMENT_ACC_TYPE, size_t VSIZE, size_t VSIZEH, size_t TDMUX,size_t PADDING> aie::vector<ELEMENT_TYPE,VSIZEH> LSTM_KERNEL<NUM_NEURONS, NUM_FEATURES, ELEMENT_TYPE, ELEMENT_ACC_TYPE,VSIZE, VSIZEH,TDMUX, PADDING>::htan_fn(aie::vector<ELEMENT_TYPE,VSIZEH> x)
{
	aie::vector<ELEMENT_TYPE,VSIZEH> vec;
	aie::accum<ELEMENT_ACC_TYPE,VSIZEH> acc;

	if (0) {
		//[min(max(x,-1),1)] very simple approx
		vec = aie::min(x, (ELEMENT_TYPE) 1.0);
		vec = aie::max(vec, (ELEMENT_TYPE) -1.0);
	}

	if (1)
		{ // 5 Stage linear approximation
			aie::mask<VSIZEH> msk_lt;
			aie::mask<VSIZEH> msk_ge;
			aie::vector<ELEMENT_TYPE,VSIZEH> vtemp;

			vec = aie::zeros<ELEMENT_TYPE,VSIZEH>();

			// x < -2 , y = -1
			vtemp=aie::broadcast<ELEMENT_TYPE,VSIZEH>((ELEMENT_TYPE) -1.0);
			msk_lt = aie::lt(x,(ELEMENT_TYPE) -2.0);
			vtemp=aie::select((ELEMENT_TYPE) 0.0, vtemp, msk_lt);
			vec = aie::add(vec,vtemp);


			// If -2 < x < -0.7, y = 0.2x-0.6
			acc = aie::mul(x, (ELEMENT_TYPE) 0.2);
			acc = aie::add(acc, (ELEMENT_TYPE) -0.6);
			vtemp = acc.template to_vector<ELEMENT_TYPE>();
			msk_lt = aie::lt(x,(ELEMENT_TYPE) -0.7);
			msk_ge = aie::ge(x,(ELEMENT_TYPE) -2.0);
			vtemp=aie::select((ELEMENT_TYPE) 0.0, vtemp, msk_lt);
			vtemp=aie::select((ELEMENT_TYPE) 0.0, vtemp, msk_ge);
			vec = aie::add(vec,vtemp);


			// If -0.7 < x < 0.7, y = x
			msk_lt = aie::lt(x,(ELEMENT_TYPE) 0.7);
			msk_ge = aie::ge(x,(ELEMENT_TYPE) -0.7);
			vtemp=aie::select((ELEMENT_TYPE) 0.0, x, msk_lt);
			vtemp=aie::select((ELEMENT_TYPE) 0.0, vtemp, msk_ge);
			vec = aie::add(vec,vtemp);


			// If 0.7 < x < 2.0, y = 0.2x+0.6
			acc = aie::mul(x, (ELEMENT_TYPE) 0.2);
			acc = aie::add(acc, (ELEMENT_TYPE) 0.6);
			vtemp = acc.template to_vector<ELEMENT_TYPE>();
			msk_lt = aie::lt(x,(ELEMENT_TYPE) 2.0);
			msk_ge = aie::ge(x,(ELEMENT_TYPE) 0.7);
			vtemp=aie::select((ELEMENT_TYPE) 0.0, vtemp, msk_lt);
			vtemp=aie::select((ELEMENT_TYPE) 0.0, vtemp, msk_ge);
			vec = aie::add(vec,vtemp);

			// If x>2 , y = 1.0
			vtemp=aie::broadcast<ELEMENT_TYPE,VSIZEH>((ELEMENT_TYPE) 1.0);
			msk_ge = aie::ge(x,(ELEMENT_TYPE) 2.0);
			vtemp=aie::select((ELEMENT_TYPE) 0.0, vtemp, msk_ge);
			vec = aie::add(vec,vtemp);


		}


	return vec;
}


template<size_t NUM_NEURONS, size_t NUM_FEATURES, typename ELEMENT_TYPE, typename ELEMENT_ACC_TYPE, size_t VSIZE, size_t VSIZEH, size_t TDMUX,size_t PADDING> void LSTM_KERNEL<NUM_NEURONS, NUM_FEATURES, ELEMENT_TYPE, ELEMENT_ACC_TYPE,VSIZE, VSIZEH, TDMUX, PADDING>::lstm_func(
		input_async_buffer<ELEMENT_TYPE,extents<NUM_NEURONS*4*(NUM_FEATURES+NUM_NEURONS+1+PADDING)/2*TDMUX>> &model1_in,
        input_async_buffer<ELEMENT_TYPE,extents<NUM_NEURONS*4*(NUM_FEATURES+NUM_NEURONS+1-PADDING)/2*TDMUX>> &model2_in,
		input_stream<float> *x_in,
		output_stream<float> *h_out)
{
	ELEMENT_TYPE x;



	
	{
		if (first_data != 0)
        {
			model1_in.acquire();
            model2_in.acquire();
            first_data=0;
        }
        
        auto pmodel1 = aie::begin_vector<VSIZE>(model1_in);
        auto pmodel2 = aie::begin_vector<VSIZE>(model2_in);
        int use_pmodel = 0;
        if (step_counter<(NUM_FEATURES+NUM_NEURONS+1+PADDING)/2)
        {
	        use_pmodel = 1;
	        pmodel1+=(NUM_NEURONS*4*step_counter/VSIZE)+tdmux_pos*(NUM_NEURONS*4*(NUM_FEATURES + NUM_NEURONS+1+PADDING)/2/VSIZE);
        }
        else {
            use_pmodel = 2;
            int shift = (NUM_FEATURES+NUM_NEURONS+1+PADDING)/2;
	        pmodel2+=(NUM_NEURONS*4*(step_counter-shift)/VSIZE)+tdmux_pos*(NUM_NEURONS*4*(NUM_FEATURES + NUM_NEURONS+1-PADDING)/2/VSIZE);
        }
	//printf("*************\nLSTM<%d,%d> Iteration %d\n",NUM_FEATURES,NUM_NEURONS,step_counter);


	do { // use do..until to repeat loop across all steps
		if (step_counter<(NUM_FEATURES+NUM_NEURONS+1+PADDING)/2)
		{
			use_pmodel = 1;
		}
		else {
			use_pmodel = 2;
		}
	
		if (step_counter<NUM_FEATURES) {
			float tmp_x;
			tmp_x=readincr(x_in);
			x= (bfloat16) tmp_x;
			//printf("LSTM: %d : %d %3.6f\n",NUM_FEATURES,step_counter,(float) x);
		}
		
		
		aie::accum<ELEMENT_ACC_TYPE,VSIZE> acc;
		aie::vector<ELEMENT_TYPE,VSIZE> vec;
		aie::vector<ELEMENT_TYPE,VSIZE> vec2;
		aie::vector<ELEMENT_TYPE,VSIZE> vec3;

		aie::accum<ELEMENT_ACC_TYPE,VSIZEH> acch;
		aie::vector<ELEMENT_TYPE,VSIZEH> vech;
		aie::vector<ELEMENT_TYPE,VSIZEH> vec2h;
		aie::vector<ELEMENT_TYPE,VSIZEH> vec3h;



		for (int k=0; k<NUM_NEURONS*4; k+= VSIZE)
		{
			if (use_pmodel==1)
				vec = *pmodel1++;
			else
				vec = *pmodel2++;;

			//if ((k==0) & (step_counter==0))
			//	{alignas(64) ELEMENT_TYPE tmp_vec[VSIZE];aie::store_v(tmp_vec,vec);for(int i=0;i<VSIZE;i++) printf("%2.3f ",(float) tmp_vec[i]);printf("\n");}
			if (step_counter==0) {
				// First Feature Input  - use Mul to "zero" accumulator 
				acc = aie::mul(vec,x);
				aie::store_v(tmp_acc+k,acc.template to_vector<ELEMENT_TYPE>());  
			}
			else if (step_counter< NUM_FEATURES) {
				// Remaining Feature Inputs
				vec3= aie::load_v<VSIZE>(tmp_acc+k);
				acc.from_vector(vec3);
				acc = aie::mac(acc,vec,x);
				aie::store_v(tmp_acc+k,acc.template to_vector<ELEMENT_TYPE>());
			}
			else if (step_counter < NUM_FEATURES + NUM_NEURONS){
				// Internal State updates from hidden feedback
				vec3= aie::load_v<VSIZE>(tmp_acc+k);
				acc.from_vector(vec3);
				acc = aie::mac(acc,vec,h[tdmux_pos][step_counter-NUM_FEATURES]);
				aie::store_v(tmp_acc+k,acc.template to_vector<ELEMENT_TYPE>());
			}
			else {
				// Bias
				vec3= aie::load_v<VSIZE>(tmp_acc+k);
				acc.from_vector(vec3);
				acc = aie::add(acc,vec);
				aie::store_v(tmp_acc+k,acc.template to_vector<ELEMENT_TYPE>());
			}
			

		}
	//printf("LSTM: %d : %d : ",NUM_FEATURES,step_counter);
	//        for(int i=0;i<10;i++) printf("%2.6f ",(float) tmp_acc[i]);
	//        printf("\n");

		if (step_counter == NUM_FEATURES + NUM_NEURONS){

			// Hadamard products
			// Ref Model Weights in Order f i o c
			// Keras Model Weights (test model) in Order  i f c o  (or i c f o ?)
			const int FPOS = 1;
			const int IPOS = 0;
			const int CPOS = 2;
			const int OPOS = 3;

			// Hadamard products
			for (int k=0; k<NUM_NEURONS; k+= VSIZEH) {
				// f
				vech = aie::load_unaligned_v<VSIZEH>(tmp_acc+NUM_NEURONS*FPOS+k);
				vech = logistic_fn(vech);
				vec2h = aie::load_v<VSIZEH>(c[tdmux_pos]+k);
				acch = aie::mul(vech,vec2h);  // ft * c(t-1)
				// i
				vec3h = aie::load_unaligned_v<VSIZEH>(tmp_acc+NUM_NEURONS*IPOS+k);
				vech=acch.template to_vector<ELEMENT_TYPE>(); // Convert back to vector type (TO DO support other ELEMENT_TYPE)
				vec3h = logistic_fn(vec3h);
				// c
				vec2h = aie::load_unaligned_v<VSIZEH>(tmp_acc+NUM_NEURONS*CPOS+k);
				// Hyperbolic Tangent Approximation [min(max(x,-1),1)]
				vec2h = htan_fn(vec2h);
				acch = aie::mul(vec3h,vec2h); // it * c_bar_t
				acch = aie::add(acch,vech); // ft * c(t-1) + it * c_bar_t
				vech=acch.template to_vector<ELEMENT_TYPE>(); // Convert back to vector type (TO DO support other ELEMENT_TYPE)
				aie::store_v(c[tdmux_pos]+k,vech);
				// Hyperbolic Tangent Approximation	[min(max(x,-1),1)]
				vech = htan_fn(vech);
				// o
				vec2h = aie::load_unaligned_v<VSIZEH>(tmp_acc+NUM_NEURONS*OPOS+k);
				vec2h = logistic_fn(vec2h);
				acch = aie::mul(vech,vec2h);  // ot * o_h(ct)
				vech=acch.template to_vector<ELEMENT_TYPE>(); // Convert back to vector type (TO DO support other ELEMENT_TYPE)
				aie::store_v(h[tdmux_pos]+k,vech);


				}
		}
		if ((step_counter<NUM_NEURONS)&& (first_data==0))
		{
			float h_data = h[tdmux_pos][step_counter];
		// printf("LSTM %d : %d : %2.5f\n",NUM_FEATURES,step_counter, (float)  h[tdmux_pos][step_counter]);
			writeincr(h_out,h_data);  
			
		}
		if (step_counter < NUM_FEATURES + NUM_NEURONS )
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
		} while (step_counter !=0);
	}
}

