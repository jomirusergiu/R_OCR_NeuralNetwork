//
//  Main.cpp
//  R_BP_NeuralNetwork
//
//  Created by RocKK on 04/01/14.
//  Copyright (c) 2014 RocKK.
//  All rights reserved.
//
//  Redistribution and use in source and binary forms are permitted
//  provided that the above copyright notice and this paragraph are
//  duplicated in all such forms and that any documentation,
//  advertising materials, and other materials related to such
//  distribution and use acknowledge that the software was developed
//  by the RocKK.  The name of the
//  RocKK may not be used to endorse or promote products derived
//  from this software without specific prior written permission.
//  THIS SOFTWARE IS PROVIDED ''AS IS'' AND WITHOUT ANY EXPRESS OR
//  IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Special thanks to Daniel Rios. Written based on his work.

#include <iostream>
#include <stdio.h>
#include "R_BP_Net.h"
#include "R_BP_Net.cpp"
using namespace std;
#define PATTERN_COUNT 64
#define PATTERN_SIZE 64
#define NETWORK_INPUTNEURONS 3
#define NETWORK_OUTPUT 64
#define HIDDEN_LAYERS 0
#define EPOCHS 1000

int main()
{

    //Create some patterns
    cout << "Creating Patterns" << endl;

    //first input values
    float firstPattern[PATTERN_COUNT][PATTERN_SIZE]=
    {
        	 {0},{0},{0},{1},{1},{0},{0},{0},
             {0},{0},{1},{1},{1},{0},{0},{0},
             {0},{0},{0},{1},{1},{0},{0},{0},
             {0},{0},{0},{1},{1},{0},{0},{0},
             {0},{0},{0},{1},{1},{0},{0},{0},
             {0},{0},{0},{1},{1},{0},{0},{0},
             {1},{1},{1},{1},{1},{1},{1},{1},
             {1},{1},{1},{1},{1},{1},{1},{1}
    };

    //second input values
    float secondPattern[PATTERN_COUNT][PATTERN_SIZE]=
    {
        	 {0},{0},{1},{1},{1},{1},{0},{0},
             {0},{1},{1},{0},{0},{1},{1},{0},
             {0},{0},{0},{0},{0},{1},{1},{0},
             {0},{0},{0},{0},{1},{1},{0},{0},
             {0},{0},{0},{1},{1},{0},{0},{0},
             {0},{0},{1},{1},{0},{0},{0},{0},
             {1},{1},{1},{0},{0},{0},{1},{1},
             {1},{1},{1},{1},{1},{1},{1},{1} 
    };

    //first desired output values
    float firstDesiredOut[PATTERN_COUNT][NETWORK_OUTPUT]=
    {
        	 {0},{0},{0},{1},{1},{0},{0},{0},
             {0},{0},{1},{1},{1},{0},{0},{0},
             {0},{0},{0},{1},{1},{0},{0},{0},
             {0},{0},{0},{1},{1},{0},{0},{0},
             {0},{0},{0},{1},{1},{0},{0},{0},
             {0},{0},{0},{1},{1},{0},{0},{0},
             {1},{1},{1},{1},{1},{1},{1},{1},
             {1},{1},{1},{1},{1},{1},{1},{1} 
    };

    //second desired output values
    float secondDesiredOut[PATTERN_COUNT][NETWORK_OUTPUT]=
    {
        	 {0},{0},{1},{1},{1},{1},{0},{0},
             {0},{1},{1},{0},{0},{1},{1},{0},
             {0},{0},{0},{0},{0},{1},{1},{0},
             {0},{0},{0},{0},{1},{1},{0},{0},
             {0},{0},{0},{1},{1},{0},{0},{0},
             {0},{0},{1},{1},{0},{0},{0},{0},
             {1},{1},{1},{0},{0},{0},{1},{1},
             {1},{1},{1},{1},{1},{1},{1},{1}  
    };


    //We create the networks
    cout << "Creating Neural networks" << endl;

    R_BP_Net net;//Our neural network object
    net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,HIDDEN_LAYERS,HIDDEN_LAYERS);
    int i,j;
    float error;

    //Display input for all patterns
    cout << "Displaying Input for all patterns" << endl << endl;
    for(i=0;i<PATTERN_COUNT;i++)
    {
        if(i % 8 == 0 && i!= 0)
    		cout << "" << endl;
        printf("%.0f ",firstDesiredOut[i][0]);

    }
    cout << endl << endl;
    for(i=0;i<PATTERN_COUNT;i++)
    {
        if(i % 8 == 0 && i!= 0)
    		cout << "" << endl;
        printf("%.0f ",secondDesiredOut[i][0]);

    }
    cout << endl << endl;

    //Start the neural network training, 1.000 times
    cout << "Starting Neural Network training [1.000 epochs]" << endl;
    for(i=0;i<EPOCHS;i++)
    {
        error=0;
        for(j=0;j<PATTERN_COUNT;j++)
        {
            error+=net.train(firstDesiredOut[j],firstPattern[j],0.2f,0.2f);
        }
        error/=PATTERN_COUNT;
        //display error
        cout << "ERROR:" << error << "\r";
    }

    cout << "Neural network finshed training, testing all patterns" << endl << endl;
    //once trained test all patterns
    for(i=0;i<PATTERN_COUNT;i++)
    {
        net.propagate(firstPattern[i]);
    //display result
        if(i % 8 == 0 && i!= 0)
    		cout << "" << endl;
        printf("%.2f ",net.getOutput().neurons[0]->output);

    }
    cout << endl << endl;
    for(i=0;i<PATTERN_COUNT;i++)
    {
        net.propagate(secondPattern[i]);
    //display result
        if(i % 8 == 0 && i!= 0)
    		cout << "" << endl;
        printf("%.2f ",net.getOutput().neurons[0]->output);

    }
    cout << endl << endl;

    cout << "Aproximating, and normalising values" << endl << endl;
    //once trained test all patterns
    for(i=0;i<PATTERN_COUNT;i++)
    {
        net.propagate(firstPattern[i]);
    //display result
        if(i % 8 == 0 && i!= 0)
    		cout << "" << endl;
        printf("%.f ",net.getOutput().neurons[0]->output);

    }
    cout << endl << endl;
    for(i=0;i<PATTERN_COUNT;i++)
    {
        net.propagate(secondPattern[i]);
    //display result
        if(i % 8 == 0 && i!= 0)
    		cout << "" << endl;
        printf("%.f ",net.getOutput().neurons[0]->output);

    }
    cout << endl << endl;

    return 0;
}