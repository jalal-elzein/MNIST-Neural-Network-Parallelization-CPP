
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

using namespace std;

// Testing image file name
const string testing_image_fn = "mnist/t10k-images.idx3-ubyte";

// Testing label file name
const string testing_label_fn = "mnist/t10k-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network.dat";

// Report file name
const string report_fn = "testing-report.dat";

// Number of testing samples
const int nTesting = 10000;

// Image size in MNIST database
const int width = 28;
const int height = 28;

// n1 = Number of input neurons
// n2 = Number of hidden neurons
// n3 = Number of output neurons

const int n1 = width * height; // = 784, without bias neuron 
const int n2 = 128; 
const int n3 = 10; // Ten classes: 0 - 9

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double *w1[n1 + 1], *out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *w2[n2 + 1], *in2, *out2;

// Layer 3 - Output layer
double *in3, *out3;
double expected[n3 + 1];

// Image. In MNIST: 28x28 gray scale images.
int d[width + 1][height + 1];

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

// +--------------------+
// | About the software |
// +--------------------+

void about() {
	// Details
	cout << "*************************************************" << endl;
	cout << "*** Testing Neural Network for MNIST database ***" << endl;
	cout << "*************************************************" << endl;
	cout << endl;
	cout << "No. input neurons: " << n1 << endl;
	cout << "No. hidden neurons: " << n2 << endl;
	cout << "No. output neurons: " << n3 << endl;
	cout << endl;
	cout << "Testing image data: " << testing_image_fn << endl;
	cout << "Testing label data: " << testing_label_fn << endl;
	cout << "No. testing sample: " << nTesting << endl << endl;
}

// +-----------------------------------+
// | Memory allocation for the network |
// +-----------------------------------+

void init_array() {
	// Layer 1 - Layer 2 = Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        w1[i] = new double [n2 + 1];
    }
    
    out1 = new double [n1 + 1];

	// Layer 2 - Layer 3 = Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        w2[i] = new double [n3 + 1];
    }
    
    in2 = new double [n2 + 1];
    out2 = new double [n2 + 1];

	// Layer 3 - Output layer
    in3 = new double [n3 + 1];
    out3 = new double [n3 + 1];
}

// +----------------------------------------+
// | Load model of a trained Neural Network |
// +----------------------------------------+

void load_model(string file_name) {
	ifstream file(file_name.c_str(), ios::in);
	
	// Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
			file >> w1[i][j];
		}
    }
	
	// Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
			file >> w2[i][j];
		}
    }
	
	file.close();
}

// +------------------+
// | Sigmoid function |
// +------------------+

// double sigmoid(double x) {
//     return 1.0 / (1.0 + exp(-x));
// }

// +------------------------------+
// | Forward process - Perceptron |
// +------------------------------+
__global__ void compute_in2(float* out1, float* w1, float* in2, int n1, int n2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        in2[idx] = 0.0;
        for (int i = 1; i <= n1; ++i) {
            in2[idx] += out1[i] * w1[i * n2 + idx];
        }
    }
}
__global__ void compute_in3(float* out2, float* w2, float* in3, int n2, int n3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n3) {
        in3[idx] = 0.0;
        for (int i = 1; i <= n2; ++i) {
            in2[idx] += out2[i] * w2[i * n3 + idx];
        }
    }
}
// __global__ void compute_in2(float* out1, float* w1, float* in2, int n1, int n2) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n2) {
//         float sum = 0.0;
//         for (int i = 1; i <= n1; ++i) {
//             sum += out1[i] * w1[i * n2 + idx];
//         }
//         in2[idx] = sum;
//     }
// }

// __global__ void compute_in3(float* out2, float* w2, float* in3, int n2, int n3) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n3) {
//         float sum = 0.0;
//         for (int i = 1; i <= n2; ++i) {
//             sum += out2[i] * w2[i * n3 + idx];
//         }
//         in3[idx] = sum;
//     }
// }

__global__ void sigmoid_kernel(float* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = 1.0 / (1.0 + expf(-arr[idx]));
    }
}

void perceptron() {
    int block_size = 256;
    int num_blocks = (n2 + block_size - 1) / block_size;
    compute_in2<<<num_blocks, block_size>>>(out1, w1, in2, n1, n2);

    block_size = 256;
    num_blocks = (n3 + block_size - 1) / block_size;
    compute_in3<<<num_blocks, block_size>>>(out2, w2, in3, n2, n3);

    block_size = 256;
    num_blocks = (n2 + block_size - 1) / block_size;
    sigmoid_kernel<<<num_blocks, block_size>>>(in2, n2);

    block_size = 256;
    num_blocks = (n3 + block_size - 1) / block_size;
    sigmoid_kernel<<<num_blocks, block_size>>>(in3, n3);
}


// +---------------+
// | Norm L2 error |
// +---------------+

double square_error(){
    double res = 0.0;
    for (int i = 1; i <= n3; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
	}
    res *= 0.5;
    return res;
}

// +--------------------------------------------------------------+
// | Reading input - gray scale image and the corresponding label |
// +--------------------------------------------------------------+

int input() {
	// Reading image
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
				d[i][j] = 0; 
			} else {
				d[i][j] = 1;
			}
        }
	}

    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1[pos] = d[i][j];
        }
	}

	// Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= n3; ++i) {
		expected[i] = 0.0;
	}
    expected[number + 1] = 1.0;
        
    return (int)(number);
}

// +--------------+
// | Main Program |
// +--------------+

int main(int argc, char *argv[]) {
	about();
	
    report.open(report_fn.c_str(), ios::out);
    image.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(testing_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

	// Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
		
	// Neural Network Initialization
    init_array(); // Memory allocation
    load_model(model_fn); // Load model (weight matrices) of a trained Neural Network
    
    int nCorrect = 0;
    for (int sample = 1; sample <= nTesting; ++sample) {
        cout << "Sample " << sample << endl;
        
        // Getting (image, label)
        int label = input();
		
		// Classification - Perceptron procedure
        perceptron();
        
        // Prediction
        int predict = 1;
        for (int i = 2; i <= n3; ++i) {
			if (out3[i] > out3[predict]) {
				predict = i;
			}
		}
		--predict;

		// Write down the classification result and the squared error
		double error = square_error();
		printf("Error: %0.6lf\n", error);
		
		if (label == predict) {
			++nCorrect;
			cout << "Classification: YES. Label = " << label << ". Predict = " << predict << endl << endl;
			report << "Sample " << sample << ": YES. Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
		} else {
			cout << "Classification: NO.  Label = " << label << ". Predict = " << predict << endl;
			cout << "Image:" << endl;
			for (int j = 1; j <= height; ++j) {
				for (int i = 1; i <= width; ++i) {
					cout << d[i][j];
				}
				cout << endl;
			}
			cout << endl;
			report << "Sample " << sample << ": NO.  Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
		}
    }

	// Summary
    double accuracy = (double)(nCorrect) / nTesting * 100.0;
    cout << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
    printf("Accuracy: %0.2lf\n", accuracy);
    
    report << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
    report << "Accuracy: " << accuracy << endl;

    report.close();
    image.close();
    label.close();
    
    return 0;
}