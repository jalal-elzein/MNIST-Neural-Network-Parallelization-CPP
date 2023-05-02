#include <iostream> 

// INCLUDE THIS LIBRARY TO ACCESS THE FUNCTIONS 
#include <chrono>

int main () {
    // START THE TIMER 
    auto start = std::chrono::high_resolution_clock::now();

    // DO SOME COMPUTATION HERE 
    int x = 0;
    for (int i = 0; i < 1000; i++) {
        x += i;
    }

    // END TIMER 
    auto end = std::chrono::high_resolution_clock::now();

    // CALCULATE THE TIME INTERVAL 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // DISPLAY THE DURATION
    std::cout << "Duration: " << duration.count() << "\n";
}