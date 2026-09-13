#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

/* ============================================================
 * 2D Ising Model — OpenMP Metropolis Implementation
 *
 * Parallel implementation using a red-black checkerboard
 * decomposition of a square lattice with periodic boundaries.
 *
 * Natural units are used: k_B = 1.
 * ============================================================ */


/* ---------------- Model parameters ---------------- */

#define L 16
#define N (L * L)

#define J 1.0
#define B 0.0


/* ---------------- Simulation parameters ---------------- */

#ifdef QUICK_TEST

#define EQUILIBRATION_T 2.0
#define EQUILIBRATION_SWEEPS 50

#define T_MIN 1.5
#define T_MAX 3.5
#define T_STEP 0.5

#define THERMALIZATION_SWEEPS 100
#define MEASUREMENT_SWEEPS 200
#define SAMPLE_INTERVAL 10

#else

#define EQUILIBRATION_T 2.0
#define EQUILIBRATION_SWEEPS 1000

#define T_MIN 1.5
#define T_MAX 3.5
#define T_STEP 0.05

#define THERMALIZATION_SWEEPS 5000
#define MEASUREMENT_SWEEPS 10000
#define SAMPLE_INTERVAL 10

#endif


/* ---------------- Output files ---------------- */

#ifdef QUICK_TEST

#define EQUILIBRATION_FILE \
    "results/data/test/openmp/equilibration.csv"

#define THERMODYNAMICS_FILE \
    "results/data/test/openmp/thermodynamics.csv"

#define CORRELATION_FILE \
    "results/data/test/openmp/correlation.csv"

#else

#define EQUILIBRATION_FILE \
    "results/data/openmp/equilibration.csv"

#define THERMODYNAMICS_FILE \
    "results/data/openmp/thermodynamics.csv"

#define CORRELATION_FILE \
    "results/data/openmp/correlation.csv"

#endif

/* ---------------- Lattice ---------------- */

static int grid[L][L];


/* ============================================================
 * Random number generator
 * ============================================================ */

/*
 * Each OpenMP thread has its own pseudo-random-number-generator
 * state. This avoids race conditions associated with rand().
 *
 * xorshift32 is used because it is simple, fast and sufficient
 * for this educational Monte Carlo implementation.
 */

static inline uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    *state = x;

    return x;
}


/*
 * Uniform pseudo-random number in [0,1).
 */
static inline double uniform_random(uint32_t *state)
{
    return (double)xorshift32(state)
           / ((double)UINT32_MAX + 1.0);
}


/* ============================================================
 * Utility functions
 * ============================================================ */

static inline int pbc(int i)
{
    return (i + L) % L;
}


/*
 * Initialize lattice spins randomly.
 *
 * Initialization is kept serial because its computational cost
 * is negligible compared with the Monte Carlo evolution.
 */
void initialize_lattice(uint32_t *seed)
{
    for (int i = 0; i < L; i++) {

        for (int j = 0; j < L; j++) {

            grid[i][j] =
                (xorshift32(seed) & 1U)
                ? 1
                : -1;
        }
    }
}


/*
 * Energy change associated with flipping one spin.
 */
static inline double delta_energy(
    int row,
    int col
)
{
    const int neighbor_sum =
        grid[pbc(row - 1)][col]
        + grid[pbc(row + 1)][col]
        + grid[row][pbc(col - 1)]
        + grid[row][pbc(col + 1)];

    return
        2.0 * J
        * grid[row][col]
        * neighbor_sum
        +
        2.0 * B
        * grid[row][col];
}


/* ============================================================
 * Thermodynamic observables
 * ============================================================ */

double total_energy(void)
{
    double energy = 0.0;

    #pragma omp parallel for collapse(2) reduction(+:energy)
    for (int i = 0; i < L; i++) {

        for (int j = 0; j < L; j++) {

            energy +=
                -J
                * grid[i][j]
                * (
                    grid[pbc(i + 1)][j]
                    + grid[i][pbc(j + 1)]
                  )
                -
                B * grid[i][j];
        }
    }

    return energy;
}


double total_magnetization(void)
{
    double magnetization = 0.0;

    #pragma omp parallel for collapse(2) reduction(+:magnetization)
    for (int i = 0; i < L; i++) {

        for (int j = 0; j < L; j++) {

            magnetization +=
                grid[i][j];
        }
    }

    return magnetization;
}


/* ============================================================
 * Parallel Metropolis sweep
 * ============================================================ */

/*
 * A square lattice is bipartite.
 *
 * We divide it into two checkerboard sublattices:
 *
 *      A B A B ...
 *      B A B A ...
 *      A B A B ...
 *
 * Spins belonging to the same color are not nearest neighbors.
 * Therefore all spins of one color can be updated simultaneously
 * without data races.
 *
 * One Monte Carlo sweep consists of updating both colors.
 */
void metropolis_sweep_openmp(
    double temperature,
    uint32_t *thread_states
)
{
    for (int color = 0; color < 2; color++) {

        #pragma omp parallel
        {
            const int tid =
                omp_get_thread_num();

            uint32_t local_state =
                thread_states[tid];


            #pragma omp for schedule(static)
            for (int row = 0; row < L; row++) {

                /*
                 * Choose columns belonging to the current
                 * checkerboard color.
                 */
                const int first_col =
                    (row + color) % 2;

                for (
                    int col = first_col;
                    col < L;
                    col += 2
                ) {

                    const double dE =
                        delta_energy(
                            row,
                            col
                        );


                    if (
                        dE <= 0.0
                        ||
                        uniform_random(
                            &local_state
                        )
                        <
                        exp(
                            -dE
                            / temperature
                        )
                    ) {

                        grid[row][col] *= -1;
                    }
                }
            }


            /*
             * Store the updated RNG state for this thread.
             */
            thread_states[tid] =
                local_state;
        }
    }
}


/* ============================================================
 * Equilibration analysis
 * ============================================================ */

void run_equilibration(
    uint32_t *thread_states,
    uint32_t *initialization_seed
)
{
    FILE *file =
        fopen(
            EQUILIBRATION_FILE,
            "w"
        );


    if (file == NULL) {

        perror(
            "Could not open equilibration output file"
        );

        exit(EXIT_FAILURE);
    }


    fprintf(
        file,
        "sweep,"
        "energy_per_spin,"
        "abs_magnetization_per_spin\n"
    );


    initialize_lattice(
        initialization_seed
    );


    for (
        int sweep = 0;
        sweep <= EQUILIBRATION_SWEEPS;
        sweep++
    ) {

        metropolis_sweep_openmp(
            EQUILIBRATION_T,
            thread_states
        );


        const double energy =
            total_energy()
            / N;


        const double abs_magnetization =
            fabs(
                total_magnetization()
            )
            / N;


        fprintf(
            file,
            "%d,%.10f,%.10f\n",
            sweep,
            energy,
            abs_magnetization
        );
    }


    fclose(file);


    printf(
        "OpenMP equilibration analysis completed.\n"
    );
}


/* ============================================================
 * Thermodynamic and correlation analysis
 * ============================================================ */

void run_thermodynamic_analysis(
    uint32_t *thread_states,
    uint32_t *initialization_seed
)
{
    FILE *thermo_file =
        fopen(
            THERMODYNAMICS_FILE,
            "w"
        );


    FILE *corr_file =
        fopen(
            CORRELATION_FILE,
            "w"
        );


    if (
        thermo_file == NULL
        ||
        corr_file == NULL
    ) {

        perror(
            "Could not open analysis output files"
        );


        if (thermo_file != NULL) {
            fclose(thermo_file);
        }


        if (corr_file != NULL) {
            fclose(corr_file);
        }


        exit(EXIT_FAILURE);
    }


    fprintf(
        thermo_file,
        "temperature,"
        "energy_per_spin,"
        "abs_magnetization_per_spin,"
        "heat_capacity,"
        "susceptibility\n"
    );


    fprintf(
        corr_file,
        "temperature"
    );


    for (
        int r = 1;
        r <= L / 2;
        r++
    ) {

        fprintf(
            corr_file,
            ",C_%d",
            r
        );
    }


    fprintf(
        corr_file,
        "\n"
    );


    for (
        double temperature = T_MIN;
        temperature <= T_MAX + 1e-12;
        temperature += T_STEP
    ) {

        initialize_lattice(
            initialization_seed
        );


        /* ---------------- Thermalization ---------------- */

        for (
            int sweep = 0;
            sweep < THERMALIZATION_SWEEPS;
            sweep++
        ) {

            metropolis_sweep_openmp(
                temperature,
                thread_states
            );
        }


        /* ---------------- Measurements ---------------- */

        double E_sum = 0.0;
        double E2_sum = 0.0;

        double M_sum = 0.0;
        double M2_sum = 0.0;
        double abs_M_sum = 0.0;

        double correlation[
            L / 2 + 1
        ] = {0.0};

        int samples = 0;


        for (
            int sweep = 0;
            sweep < MEASUREMENT_SWEEPS;
            sweep++
        ) {

            metropolis_sweep_openmp(
                temperature,
                thread_states
            );


            if (
                sweep
                % SAMPLE_INTERVAL
                != 0
            ) {
                continue;
            }


            const double E =
                total_energy();


            const double M =
                total_magnetization();


            E_sum += E;
            E2_sum += E * E;

            M_sum += M;
            M2_sum += M * M;

            abs_M_sum +=
                fabs(M);


            /*
             * Spatial spin-spin correlation.
             *
             * The reduction is implemented manually because
             * correlation[] is an array.
             */
            #pragma omp parallel
            {
                double local_correlation[
                    L / 2 + 1
                ] = {0.0};


                #pragma omp for collapse(2) schedule(static)
                for (
                    int row = 0;
                    row < L;
                    row++
                ) {

                    for (
                        int col = 0;
                        col < L;
                        col++
                    ) {

                        for (
                            int r = 1;
                            r <= L / 2;
                            r++
                        ) {

                            local_correlation[r] +=
                                grid[row][col]
                                *
                                grid[
                                    row
                                ][
                                    pbc(
                                        col + r
                                    )
                                ];
                        }
                    }
                }


                #pragma omp critical
                {
                    for (
                        int r = 1;
                        r <= L / 2;
                        r++
                    ) {

                        correlation[r] +=
                            local_correlation[r];
                    }
                }
            }


            samples++;
        }


        /* ---------------- Ensemble averages ---------------- */

        const double E_mean =
            E_sum
            / samples;


        const double E2_mean =
            E2_sum
            / samples;


        const double M_mean =
            M_sum
            / samples;


        const double M2_mean =
            M2_sum
            / samples;


        const double abs_M_mean =
            abs_M_sum
            / samples;


        const double heat_capacity =
            (
                E2_mean
                -
                E_mean * E_mean
            )
            /
            (
                N
                *
                temperature
                *
                temperature
            );


        const double susceptibility =
            (
                M2_mean
                -
                M_mean * M_mean
            )
            /
            (
                N
                *
                temperature
            );


        fprintf(
            thermo_file,
            "%.6f,"
            "%.10f,"
            "%.10f,"
            "%.10f,"
            "%.10f\n",
            temperature,
            E_mean / N,
            abs_M_mean / N,
            heat_capacity,
            susceptibility
        );


        /*
         * Connected correlation:
         *
         * C(r) =
         * <s_i s_{i+r}> - <s>²
         */
        const double mean_spin =
            M_mean
            / N;


        fprintf(
            corr_file,
            "%.6f",
            temperature
        );


        for (
            int r = 1;
            r <= L / 2;
            r++
        ) {

            const double correlation_mean =
                correlation[r]
                /
                (
                    (double)samples
                    *
                    N
                );


            const double connected_correlation =
                correlation_mean
                -
                mean_spin
                *
                mean_spin;


            fprintf(
                corr_file,
                ",%.10f",
                connected_correlation
            );
        }


        fprintf(
            corr_file,
            "\n"
        );


        printf(
            "T = %.2f completed.\n",
            temperature
        );
    }


    fclose(
        thermo_file
    );


    fclose(
        corr_file
    );


    printf(
        "OpenMP thermodynamic analysis completed.\n"
    );
}


/* ============================================================
 * Main
 * ============================================================ */

int main(
    int argc,
    char *argv[]
)
{
    /*
     * Fixed default seed for reproducibility.
     */
    uint32_t seed =
        12345U;


    if (argc > 1) {

        seed =
            (uint32_t)
            strtoul(
                argv[1],
                NULL,
                10
            );
    }


    const int max_threads =
        omp_get_max_threads();


    /*
     * One random-number state per OpenMP thread.
     */
    uint32_t *thread_states =
        malloc(
            (size_t)max_threads
            *
            sizeof(uint32_t)
        );


    if (
        thread_states == NULL
    ) {

        fprintf(
            stderr,
            "Could not allocate RNG states.\n"
        );

        return EXIT_FAILURE;
    }


    /*
     * Different deterministic seed for each thread.
     */
    for (
        int tid = 0;
        tid < max_threads;
        tid++
    ) {

        thread_states[tid] =
            seed
            +
            0x9E3779B9U
            *
            (uint32_t)(
                tid + 1
            );


        /*
         * xorshift32 must never receive state = 0.
         */
        if (
            thread_states[tid]
            == 0U
        ) {

            thread_states[tid] =
                (uint32_t)(
                    tid + 1
                );
        }
    }


    uint32_t initialization_seed =
        seed ^ 0xA5A5A5A5U;


    if (
        initialization_seed
        == 0U
    ) {

        initialization_seed =
            1U;
    }


    printf(
        "2D Ising model — OpenMP implementation\n"
    );


    printf(
        "L = %d, N = %d, seed = %u\n",
        L,
        N,
        seed
    );


    printf(
        "OpenMP threads = %d\n",
        max_threads
    );


    run_equilibration(
        thread_states,
        &initialization_seed
    );


    run_thermodynamic_analysis(
        thread_states,
        &initialization_seed
    );


    free(
        thread_states
    );


    return EXIT_SUCCESS;
}
