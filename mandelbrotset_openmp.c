#include <stdio.h>
#include <time.h>
#include <omp.h> // Include OpenMP header
#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));
    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg;
    int temp;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n"); // PGM file format header
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");
    
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            temp = image[i][j];
            fprintf(pgmimg, "%d ", temp);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main() {
    int image[HEIGHT][WIDTH];
    double AVG = 0;
    int N = 10; // Number of trials
    double total_time[N];
    struct complex c;
    int i, j;  // Declare loop variables before OpenMP directive

    for (int k = 0; k < N; k++) {
        double start_time = omp_get_wtime(); // Start measuring time

        /* OpenMP Parallelization:
           - The outer loop over rows (i) is parallelized.
           - Row Wise Chunking: Each thread processes one full row of pixels at a time.
           - Dynamic Scheduling with chunk size of 1 (schedule(dynamic, 1)):
             - Threads dynamically pick up one row at a time.
             - Helps with load balancing since computation time per row varies.
           - Variables:
             - private(i, j, c): Each thread has its own copy, preventing race conditions.
             - shared(image): All threads can write to the image array safely because each thread works on a different row.
        */
        #pragma omp parallel for schedule(dynamic, 1) private(i, j, c) shared(image)
        for (i = 0; i < HEIGHT; i++) {
            for (j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image[i][j] = cal_pixel(c);
            }
        }

        double end_time = omp_get_wtime(); // End measuring time
        total_time[k] = end_time - start_time;
        printf("Execution time of trial [%d]: %f seconds\n", k, total_time[k]);
        AVG += total_time[k];
    }

    save_pgm("mandelbrot.pgm", image);
    printf("The average execution time of 10 trials is: %f ms\n", (AVG / N) * 1000);
    
    return 0;
}
