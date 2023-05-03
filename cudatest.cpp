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
#include <curand_kernel.h>

using namespace std;

// Training image file name
const string training_image_fn = "mnist/train-images.idx3-ubyte";

// Training label file name
const string training_label_fn = "mnist/train-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network.dat";

// Report file name
const string report_fn = "training-report.dat";

// Number of training samples
const int nTraining = 60000;

// Image size in MNIST database
const int width = 28;
const int height = 28;

const int n1 = width * height; // = 784, without bias neuron 
const int n2 = 128; 
const int n3 = 10; // Ten classes: 0 - 9
const int epochs = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

double *w1[n1 + 1], *delta1[n1 + 1], *out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *w2[n2 + 1], *delta2[n2 + 1], *in2, *out2, *theta2;

// Layer 3 - Output layer
double *in3, *out3, *theta3;
double expected[n3 + 1];

// Image. In MNIST: 28x28 gray scale images.
int d[width + 1][height + 1];

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

void about() {
	// Details
	cout << "**************************************************" << endl;
	cout << "*** Training Neural Network for MNIST database ***" << endl;
	cout << "**************************************************" << endl;
	cout << endl;
	cout << "No. input neurons: " << n1 << endl;
	cout << "No. hidden neurons: " << n2 << endl;
	cout << "No. output neurons: " << n3 << endl;
	cout << endl;
	cout << "No. iterations: " << epochs << endl;
	cout << "Learning rate: " << learning_rate << endl;
	cout << "Momentum: " << momentum << endl;
	cout << "Epsilon: " << epsilon << endl;
	cout << endl;
	cout << "Training image data: " << training_image_fn << endl;
	cout << "Training label data: " << training_label_fn << endl;
	cout << "No. training sample: " << nTraining << endl << endl;
}

__global__ void init_weights(double *w, int n, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < m) {
        curandState_t state;
        curand_init((unsigned long long) clock() + row + col, 0, 0, &state);
        int sign = curand(&state) % 2;
        float rand_num = curand_uniform(&state);
        w[row * m + col] = (double)(rand_num * 0.6);
        if (sign == 1) {
            w[row * m + col] = - w[row * m + col];
        }
    }
}

void init_array() {
    // Layer 1 - Layer 2 = Input layer - Hidden layer
    cudaMalloc((void**)&w1, n1 * (n2 + 1) * sizeof(double));
    cudaMalloc((void**)&delta1, n1 * (n2 + 1) * sizeof(double));
    cudaMalloc((void**)&out1, (n1 + 1) * sizeof(double));
    
    // Layer 2 - Layer 3 = Hidden layer - Output layer
    cudaMalloc((void**)&w2, n2 * (n3 + 1) * sizeof(double));
    cudaMalloc((void**)&delta2, n2 * (n3 + 1) * sizeof(double));
    cudaMalloc((void**)&in2, (n2 + 1) * sizeof(double));
    cudaMalloc((void**)&out2, (n2 + 1) * sizeof(double));
    cudaMalloc((void**)&theta2, (n2 + 1) * sizeof(double));
    
    // Layer 3 - Output layer
    cudaMalloc((void**)&in3, (n3 + 1) * sizeof(double));
    cudaMalloc((void**)&out3, (n3 + 1) * sizeof(double));
    cudaMalloc((void**)&theta3, (n3 + 1) * sizeof(double));

    dim3 blockDim(16, 16);
    dim3 gridDim((n2 + blockDim.x - 1) / blockDim.x, (n1 + blockDim.y - 1) / blockDim.y);

    init_weights<<<gridDim, blockDim>>>(w1, n1, n2 + 1);
    init_weights<<<gridDim, blockDim>>>(delta1, n1, n2 + 1);
    init_weights<<<1, n1>>>(out1, n1 + 1, 1);

    gridDim = dim3((n3 + blockDim.x - 1) / blockDim.x, (n2 + blockDim.y - 1) / blockDim.y);

    init_weights<<<gridDim, blockDim>>>
}
__global__ void init_weights_kernel(double *w1, double *w2, int n1, int n2, int n3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n1 && j < n2) {
        int sign = rand() % 2;
        w1[i * (n2+1) + j] = (double)(rand() % 6) / 10.0;
        if (sign == 1) {
            w1[i * (n2+1) + j] = - w1[i * (n2+1) + j];
        }
    }

    if (i < n2 && j < n3) {
        int sign = rand() % 2;
        w2[i * (n3+1) + j] = (double)(rand() % 10 + 1) / (10.0 * n3);
        if (sign == 1) {
            w2[i * (n3+1) + j] = - w2[i * (n3+1) + j];
        }
    }
}

void init_array()
{
    // Allocate memory for weights and deltas
    double *d_w1, *d_w2;
    cudaMalloc(&d_w1, sizeof(double) * (n1+1) * (n2+1));
    cudaMalloc(&d_w2, sizeof(double) * (n2+1) * (n3+1));

    // Initialize weights in parallel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(ceil(n1 / 16.0), ceil(n2 / 16.0));
    init_weights_kernel<<<dimGrid, dimBlock>>>(d_w1, d_w2, n1, n2, n3);

    // Copy weights to host memory
    cudaMemcpy(w1, d_w1, sizeof(double) * (n1+1) * (n2+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(w2, d_w2, sizeof(double) * (n2+1) * (n3+1), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_w1);
    cudaFree(d_w2);
}
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
__global__ void perceptron_kernel(double *out1, double *w1, double *in2, double *out2, double *w2, double *in3, double *out3, int n1, int n2, int n3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n2) {
        in2[i+1] = 0.0;
        for (int j = 1; j <= n1; ++j) {
            in2[i+1] += out1[j] * w1[j*n2 + i];
        }
        out2[i+1] = sigmoid(in2[i+1]);
    }
    
    if (i < n3) {
        in3[i+1] = 0.0;
        for (int j = 1; j <= n2; ++j) {
            in3[i+1] += out2[j] * w2[j*n3 + i];
        }
        out3[i+1] = sigmoid(in3[i+1]);
    }
}

void perceptron() {
    double *d_out1, *d_w1, *d_in2, *d_out2, *d_w2, *d_in3, *d_out3;
    
    // Allocate device memory
    cudaMalloc(&d_out1, (n1+1)*sizeof(double));
    cudaMalloc(&d_w1, (n1*n2+1)*sizeof(double));
    cudaMalloc(&d_in2, (n2+1)*sizeof(double));
    cudaMalloc(&d_out2, (n2+1)*sizeof(double));
    cudaMalloc(&d_w2, (n2*n3+1)*sizeof(double));
    cudaMalloc(&d_in3, (n3+1)*sizeof(double));
    cudaMalloc(&d_out3, (n3+1)*sizeof(double));
    
    // Copy data to device memory
    cudaMemcpy(d_out1, out1, (n1+1)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1, w1[0], (n1*n2+1)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, w2[0], (n2*n3+1)*sizeof(double), cudaMemcpyHostToDevice);
    
    // Define block and grid sizes
    int block_size = 256;
    int num_blocks = (n2 + block_size - 1) / block_size;
    
    // Call kernel to compute Layer 2
    perceptron_kernel<<<num_blocks, block_size>>>(d_out1, d_w1, d_in2, d_out2, d_w2, d_in3, d_out3, n1, n2, n3);
    
    // Copy data back to host memory
    cudaMemcpy(out2, d_out2, (n2+1)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(out3, d_out3, (n3+1)*sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_out1);
    cudaFree(d_w1);
    cudaFree(d_in2);
    cudaFree(d_out2);
    cudaFree(d_w2);
    cudaFree(d_in3);
    cudaFree(d_out3);
}
double square_error(){
    double res = 0.0;
    for (int i = 1; i <= n3; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
	}
    res *= 0.5;
    return res;
}
void back_propagation() {
    double sum;

    for (int i = 1; i <= n3; ++i) {
        theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
	}

    for (int i = 1; i <= n2; ++i) {
        sum = 0.0;
        for (int j = 1; j <= n3; ++j) {
            sum += w2[i][j] * theta3[j];
		}
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
            w2[i][j] += delta2[i][j];
        }
	}

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1 ; j <= n2 ; j++ ) {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
            w1[i][j] += delta1[i][j];
        }
	}
}

int learning_process() {
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
			delta1[i][j] = 0.0;
		}
	}

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
			delta2[i][j] = 0.0;
		}
	}

    for (int i = 1; i <= epochs; ++i) {
        perceptron();
        back_propagation();
        if (square_error() < epsilon) {
			return i;
		}
    }
    return epochs;
}

void input() {
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
	
	cout << "Image:" << endl;
	for (int j = 1; j <= height; ++j) {
		for (int i = 1; i <= width; ++i) {
			cout << d[i][j];
		}
		cout << endl;
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
    
    cout << "Label: " << (int)(number) << endl;
}

// +------------------------+
// | Saving weights to file |
// +------------------------+

void write_matrix(string file_name) {
    ofstream file(file_name.c_str(), ios::out);
	
	// Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
			file << w1[i][j] << " ";
		}
		file << endl;
    }
	
	// Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
			file << w2[i][j] << " ";
		}
        file << endl;
    }
	
	file.close();
}

// +--------------+
// | Main Program |
// +--------------+

int main(int argc, char *argv[]) {
	about();
	
    report.open(report_fn.c_str(), ios::out);
    image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

	// Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
		
	// Neural Network Initialization
    init_array();
    
    for (int sample = 1; sample <= nTraining; ++sample) {
        cout << "Sample " << sample << endl;
        
        // Getting (image, label)
        input();
		
		// Learning process: Perceptron (Forward procedure) - Back propagation
        int nIterations = learning_process();

		// Write down the squared error
		cout << "No. iterations: " << nIterations << endl;
        printf("Error: %0.6lf\n\n", square_error());
        report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << square_error() << endl;
		
		// Save the current network (weights)
		if (sample % 100 == 0) {
			cout << "Saving the network to " << model_fn << " file." << endl;
			write_matrix(model_fn);
		}
    }
	
	// Save the final network
    write_matrix(model_fn);

    report.close();
    image.close();
    label.close();
    
    return 0;
}
