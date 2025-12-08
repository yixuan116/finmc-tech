/*
 * OpenMP Monte Carlo Benchmark Demo (HPC, Step 8).
 *
 * This is an optional C + OpenMP demo illustrating the same Monte Carlo pattern
 * used in the main Python pipeline (scenario_mc.py).
 *
 * Conceptually:
 * - The single-threaded loop here mirrors the baseline NumPy vectorization.
 * - The OpenMP parallel loop mirrors the Numba `@njit(parallel=True)` + `prange` kernel.
 *
 * Compile example (Linux/macOS with OpenMP):
 *   gcc -O3 -fopenmp finmc_tech/hpc_demos/openmp_mc_demo.c -o openmp_mc_demo
 *   # Note: On macOS with clang, you may need: -Xpreprocessor -fopenmp -lomp
 *
 * Run:
 *   mkdir -p results/step8
 *   ./openmp_mc_demo                    # Default: 12 months (1Y)
 *   ./openmp_mc_demo --steps 36         # 3Y
 *   ./openmp_mc_demo --steps 60         # 5Y
 *   ./openmp_mc_demo --steps 120        # 10Y
 *
 * Output:
 *   Appends runtime results into results/step8/hpc_benchmark_openmp.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

// Default parameters matching project defaults
#define DEFAULT_N_SIMS 100000
#define DEFAULT_N_STEPS 12
#define MU 0.01
#define SIGMA 0.40
#define S0 100.0
#define STEPS_PER_YEAR 12  // Monthly steps

// Multi-horizon configuration
#define NUM_HORIZONS 4
static const int HORIZON_STEPS[] = {12, 36, 60, 120};
static const char* HORIZON_LABELS[] = {"1Y", "3Y", "5Y", "10Y"};

// Simple Box-Muller transform for normal distribution
double rand_normal() {
    double u1 = ((double) rand() / RAND_MAX);
    double u2 = ((double) rand() / RAND_MAX);
    // Avoid log(0)
    if (u1 < 1e-9) u1 = 1e-9;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Helper to write CSV row
void append_csv_row(const char *filename, const char *backend, const char *mode, 
                   int n_sims, int n_steps, int n_ranks, double time_sec) {
    FILE *fp = fopen(filename, "a");
    if (!fp) {
        // Try to create if append fails (though "a" usually creates)
        fp = fopen(filename, "w");
        if (!fp) {
            fprintf(stderr, "Error opening file %s\n", filename);
            return;
        }
        // Write header if new file
        fprintf(fp, "backend,mode,n_sims,n_steps,n_ranks,time_sec\n");
    } else {
        // Check if empty to write header
        fseek(fp, 0, SEEK_END);
        long size = ftell(fp);
        if (size == 0) {
            fprintf(fp, "backend,mode,n_sims,n_steps,n_ranks,time_sec\n");
        }
    }
    
    fprintf(fp, "%s,%s,%d,%d,%d,%.6f\n", backend, mode, n_sims, n_steps, n_ranks, time_sec);
    fclose(fp);
}

// Helper function to parse command line arguments
void parse_args(int argc, char *argv[], int *n_sims, int *n_steps) {
    *n_sims = DEFAULT_N_SIMS;
    *n_steps = DEFAULT_N_STEPS;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            *n_steps = atoi(argv[i + 1]);
            i++; // Skip next argument
        } else if (strcmp(argv[i], "--sims") == 0 && i + 1 < argc) {
            *n_sims = atoi(argv[i + 1]);
            i++; // Skip next argument
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--steps N] [--sims N]\n", argv[0]);
            printf("  --steps N    Number of time steps (months). Default: 12 (1Y)\n");
            printf("               Common values: 12 (1Y), 36 (3Y), 60 (5Y), 120 (10Y)\n");
            printf("  --sims N     Total number of Monte Carlo simulations. Default: 100000\n");
            exit(0);
        }
    }
}

int main(int argc, char *argv[]) {
    int n_sims, n_steps_arg;
    parse_args(argc, argv, &n_sims, &n_steps_arg);
    
    // Check if --steps was explicitly provided
    int steps_provided = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--steps") == 0) {
            steps_provided = 1;
            break;
        }
    }
    
    const char *csv_path = "results/step8/hpc_benchmark_openmp.csv";
    
    // Check if directory exists (simple check)
    struct stat st = {0};
    if (stat("results/step8", &st) == -1) {
        printf("Warning: results/step8 directory may not exist.\n");
    }
    
    // Determine which horizons to run
    int horizons_to_run[NUM_HORIZONS];
    int num_to_run = 0;
    
    if (steps_provided) {
        // Run only the specified horizon
        horizons_to_run[0] = n_steps_arg;
        num_to_run = 1;
        printf("Starting OpenMP Monte Carlo Demo...\n");
        printf("Sims: %d, Steps: %d\n", n_sims, n_steps_arg);
    } else {
        // Run all horizons (1Y, 3Y, 5Y)
        for (int i = 0; i < NUM_HORIZONS; i++) {
            horizons_to_run[i] = HORIZON_STEPS[i];
        }
        num_to_run = NUM_HORIZONS;
        printf("Starting Multi-Horizon OpenMP Monte Carlo Demo...\n");
        printf("Sims: %d\n", n_sims);
    }
    
    // Allocate memory once (reused for all horizons)
    double *terminals_seq = malloc(n_sims * sizeof(double));
    double *terminals_par = malloc(n_sims * sizeof(double));
    
    // Loop over horizons
    for (int h = 0; h < num_to_run; ++h) {
        int n_steps = horizons_to_run[h];
        const char *hz_label = NULL;
        
        // Find corresponding label
        for (int i = 0; i < NUM_HORIZONS; i++) {
            if (HORIZON_STEPS[i] == n_steps) {
                hz_label = HORIZON_LABELS[i];
                break;
            }
        }
        
        if (hz_label == NULL) {
            // Fallback for custom steps
            char label_buf[32];
            snprintf(label_buf, sizeof(label_buf), "%dM", n_steps);
            hz_label = label_buf;
        }
        
        printf("\n=== Horizon %s (%d steps) ===\n", hz_label, n_steps);
        
        // --- Sequential Run ---
        printf("\n1. Running Sequential Baseline...\n");
        srand(42); // Seed for reproducibility
        
        double start_seq = omp_get_wtime();
        
        // [HPC-OpenMP] Baseline: single-threaded loop over Monte Carlo paths.
        for (int i = 0; i < n_sims; ++i) {
            double s_t = S0;
            double sigma_step = SIGMA / sqrt((double)STEPS_PER_YEAR); // Monthly steps
            
            for (int t = 0; t < n_steps; ++t) {
                double eps = rand_normal();
                double r_t = MU + sigma_step * eps;
                if (r_t < -0.99) r_t = -0.99;
                s_t = s_t * (1.0 + r_t);
            }
            terminals_seq[i] = s_t;
        }
        
        double end_seq = omp_get_wtime();
        double time_seq = end_seq - start_seq;
        printf("✓ Sequential finished in %.4f seconds\n", time_seq);

        // --- OpenMP Parallel Run ---
        printf("\n2. Running OpenMP Parallel...\n");
        srand(42); // Reset seed (approximate, as threads will diverge in rand usage)
        
        double start_par = omp_get_wtime();
        
        // [HPC-OpenMP] Enable path-level parallelism over Monte Carlo simulations.
        // This mirrors the Numba `@njit(parallel=True)` + `prange` pattern in Python.
        #pragma omp parallel for
        for (int i = 0; i < n_sims; ++i) {
            // Each thread needs its own random state ideally, each i = 1 path
            // but for this simple demo we use thread-unsafe rand() with caution 
            // or assume thread-safe implementation (e.g. glibc with locks).
            // For high-performance, use reentrant rand_r or C++ <random>.
            // Here we focus on the loop structure.
            
            unsigned int seed = i; // Per-path seed for rand_r
            double s_t = S0;
            double sigma_step = SIGMA / sqrt((double)STEPS_PER_YEAR);
            
            for (int t = 0; t < n_steps; ++t) {
                // Simple thread-safe random generation
                double u1 = ((double) rand_r(&seed) / RAND_MAX);
                double u2 = ((double) rand_r(&seed) / RAND_MAX);
                if (u1 < 1e-9) u1 = 1e-9;
                double eps = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                
                double r_t = MU + sigma_step * eps;
                if (r_t < -0.99) r_t = -0.99;
                s_t = s_t * (1.0 + r_t);
            }
            terminals_par[i] = s_t;
        }
        
        double end_par = omp_get_wtime();
        double time_par = end_par - start_par;
        printf("✓ OpenMP Parallel finished in %.4f seconds\n", time_par);
        printf("  Speedup: %.2fx\n", time_seq / time_par);

        // --- Save Results ---
        append_csv_row(csv_path, "openmp_c", "sequential", n_sims, n_steps, 1, time_seq);
        append_csv_row(csv_path, "openmp_c", "openmp_parallel", n_sims, n_steps, 1, time_par);
    }
    
    printf("\nAll results appended to: %s\n", csv_path);

    free(terminals_seq);
    free(terminals_par);
    return 0;
}

