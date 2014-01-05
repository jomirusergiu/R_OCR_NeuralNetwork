R_OCR_NeuralNetwork
=================

R_OCR_NeuralNetwork is a small C++ project showing how to implement Backpropagation Neural Network for creating a basic OCR (Optical Character Recognition).

Optical Character Recognition, usually abbreviated to OCR, is the mechanical or electronic conversion of scanned or photoed images of typewritten or printed text into machine-encoded/computer-readable text. It is widely used as a form of data entry from some sort of original paper data source, whether passport documents, invoices, bank statement, receipts, business card, mail, or any number of printed records. It is a common method of digitizing printed texts so that they can be electronically edited, searched, stored more compactly, displayed on-line, and used in machine processes such as machine translation, text-to-speech, key data extraction and text mining. OCR is a field of research in pattern recognition, artificial intelligence and computer vision.

Early versions needed to be programmed with images of each character, and worked on one font at a time. "Intelligent" systems with a high degree of recognition accuracy for most fonts are now common. Some commercial systems are capable of reproducing formatted output that closely approximates the original scanned page including images, columns and other non-textual components.

Backpropagation, an abbreviation for "backward propagation of errors", is a common method of training artificial neural networks. From a desired output, the network learns from many 
inputs, similar to the way a child learns to identify a dog from examples of dogs.

It is a supervised learning method, and is a generalization of the delta rule. It requires a dataset of the desired output for many inputs, making up the training set. It is most useful 
for feed-forward networks (networks that have no feedback, or simply, that have no connections that loop). Backpropagation requires that the activation function used by the artificial 
neurons (or "nodes") be differentiable.

You can change the precision level of final outputs, and may not have to aproximate/normalize them, by changing the number of epochs.

Usage
-------------

```C++
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
```
Output
--------

```BASH
[rockk@arch Demo]$ g++ main.cpp 
[rockk@arch Demo]$ ./a.out
Creating Patterns
Creating Neural networks
Displaying Input for all patterns

0 0 0 1 1 0 0 0 
0 0 1 1 1 0 0 0 
0 0 0 1 1 0 0 0 
0 0 0 1 1 0 0 0 
0 0 0 1 1 0 0 0 
0 0 0 1 1 0 0 0 
1 1 1 1 1 1 1 1 
1 1 1 1 1 1 1 1 

0 0 1 1 1 1 0 0 
0 1 1 0 0 1 1 0 
0 0 0 0 0 1 1 0 
0 0 0 0 1 1 0 0 
0 0 0 1 1 0 0 0 
0 0 1 1 0 0 0 0 
1 1 1 0 0 0 1 1 
1 1 1 1 1 1 1 1 

Starting Neural Network training [1.000 epochs]
Neural network finshed training, testing all patterns

0.04 0.04 0.04 0.97 0.97 0.04 0.04 0.04 
0.04 0.04 0.97 0.97 0.97 0.04 0.04 0.04 
0.04 0.04 0.04 0.97 0.97 0.04 0.04 0.04 
0.04 0.04 0.04 0.97 0.97 0.04 0.04 0.04 
0.04 0.04 0.04 0.97 0.97 0.04 0.04 0.04 
0.04 0.04 0.04 0.97 0.97 0.04 0.04 0.04 
0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 
0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 

0.04 0.04 0.97 0.97 0.97 0.97 0.04 0.04 
0.04 0.97 0.97 0.04 0.04 0.97 0.97 0.04 
0.04 0.04 0.04 0.04 0.04 0.97 0.97 0.04 
0.04 0.04 0.04 0.04 0.97 0.97 0.04 0.04 
0.04 0.04 0.04 0.97 0.97 0.04 0.04 0.04 
0.04 0.04 0.97 0.97 0.04 0.04 0.04 0.04 
0.97 0.97 0.97 0.04 0.04 0.04 0.97 0.97 
0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 

Aproximating, and normalising values

0 0 0 1 1 0 0 0 
0 0 1 1 1 0 0 0 
0 0 0 1 1 0 0 0 
0 0 0 1 1 0 0 0 
0 0 0 1 1 0 0 0 
0 0 0 1 1 0 0 0 
1 1 1 1 1 1 1 1 
1 1 1 1 1 1 1 1 

0 0 1 1 1 1 0 0 
0 1 1 0 0 1 1 0 
0 0 0 0 0 1 1 0 
0 0 0 0 1 1 0 0 
0 0 0 1 1 0 0 0 
0 0 1 1 0 0 0 0 
1 1 1 0 0 0 1 1 
1 1 1 1 1 1 1 1 
```

License
--------

This code is under the BSD license.
