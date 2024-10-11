/**
 * Important Note
 * 
 * This file is programmatically modified at compile time and should not be manually changed
 * If you need to change the commented values below, use the config.yaml file in the telemanom folder
 * after it is cloned from GitHub
 */
#pragma once

#define NSENSORS 25 // number of sensors
#define NNEURONS1 80
#define NNEURONS2 80
#define NDENSE 16
#define NPAR 1 // number of graphs in parallel
#define NTDM 1 // number of models time-multiplexed in each graph
#define SAMPLE_SIZE 250 // number of previous timesteps provided to model to predict future values
#define NPREDICTIONS 10 // number of steps ahead to predict
