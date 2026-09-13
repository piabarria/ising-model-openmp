#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/* ============================================================
 * 2D Ising Model — Serial Metropolis Implementation
 *
 * Square lattice with periodic boundary conditions.
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
    "results/data/test/serial/equilibration.csv"

#define THERMODYNAMICS_FILE \
    "results/data/test/serial/thermodynamics.csv"

#define CORRELATION_FILE \
    "results/data/test/serial/correlation.csv"

#else

#define EQUILIBRATION_FILE \
    "results/data/serial/equilibration.csv"

#define THERMODYNAMICS_FILE \
    "results/data/serial/thermodynamics.csv"

#define CORRELATION_FILE \
    "results/data/serial/correlation.csv"

#endif

/* ---------------- Lattice ---------------- */

static int grid[L][L];


/* ============================================================
 * Utility functions
 * ============================================================ */

/*
 * Periodic boundary conditions.
 */
static inline int pbc(int i)
{
    return (i + L) % L;
}


/*
 * Initialize the lattice with random spins:
 *
 *      s_i = +1 or -1
 *
 * with equal probability.
 */
void initialize_lattice(void)
{
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            grid[i][j] = (rand() % 2 == 0) ? 1 : -1;
        }
    }
}


/*
 * Energy change associated with flipping the spin at (row, col).
 *
 * For the Ising Hamiltonian
 *
 *      H = -J Σ_<ij> s_i s_j - B Σ_i s_i
 *
 * the energy difference is
 *
 *      ΔE = 2 J s_i Σ_nn s_j + 2 B s_i .
 */
double delta_energy(int row, int col)
{
    const int neighbor_sum =
        grid[pbc(row - 1)][col]
        + grid[pbc(row + 1)][col]
        + grid[row][pbc(col - 1)]
        + grid[row][pbc(col + 1)];

    return 2.0 * J * grid[row][col] * neighbor_sum
           + 2.0 * B * grid[row][col];
}


/*
 * Total lattice energy.
 *
 * Only the right and lower neighbors are counted to avoid
 * double-counting pair interactions.
 */
double total_energy(void)
{
    double energy = 0.0;

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {

            energy +=
                -J * grid[i][j]
                * (
                    grid[pbc(i + 1)][j]
                    + grid[i][pbc(j + 1)]
                  )
                - B * grid[i][j];
        }
    }

    return energy;
}


/*
 * Total magnetization
 *
 *      M = Σ_i s_i .
 */
double total_magnetization(void)
{
    double magnetization = 0.0;

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            magnetization += grid[i][j];
        }
    }

    return magnetization;
}


/*
 * Perform one Metropolis spin-flip attempt.
 */
void metropolis_step(double temperature)
{
    const int row = rand() % L;
    const int col = rand() % L;

    const double dE = delta_energy(row, col);

    if (
        dE <= 0.0
        || ((double) rand() / RAND_MAX) < exp(-dE / temperature)
    ) {
        grid[row][col] *= -1;
    }
}


/*
 * One Monte Carlo sweep corresponds to N attempted spin flips.
 */
void metropolis_sweep(double temperature)
{
    for (int i = 0; i < N; i++) {
        metropolis_step(temperature);
    }
}


/* ============================================================
 * Equilibration analysis
 * ============================================================ */

void run_equilibration(void)
{
    FILE *file = fopen(EQUILIBRATION_FILE, "w");

    if (file == NULL) {
        perror("Could not open equilibration output file");
        exit(EXIT_FAILURE);
    }

    fprintf(
        file,
        "sweep,energy_per_spin,abs_magnetization_per_spin\n"
    );

    initialize_lattice();

    for (
        int sweep = 0;
        sweep <= EQUILIBRATION_SWEEPS;
        sweep++
    ) {

        metropolis_sweep(EQUILIBRATION_T);

        const double energy =
            total_energy() / N;

        const double abs_magnetization =
            fabs(total_magnetization()) / N;

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
        "Serial equilibration analysis completed.\n"
    );
}


/* ============================================================
 * Thermodynamic and correlation analysis
 * ============================================================ */

void run_thermodynamic_analysis(void)
{
    FILE *thermo_file =
        fopen(THERMODYNAMICS_FILE, "w");

    FILE *corr_file =
        fopen(CORRELATION_FILE, "w");

    if (thermo_file == NULL || corr_file == NULL) {

        perror("Could not open analysis output files");

        if (thermo_file != NULL) {
            fclose(thermo_file);
        }

        if (corr_file != NULL) {
            fclose(corr_file);
        }

        exit(EXIT_FAILURE);
    }


    /*
     * Thermodynamic output:
     *
     * T
     * <E>/N
     * <|M|>/N
     * heat capacity per spin
     * susceptibility per spin
     */
    fprintf(
        thermo_file,
        "temperature,"
        "energy_per_spin,"
        "abs_magnetization_per_spin,"
        "heat_capacity,"
        "susceptibility\n"
    );


    /*
     * Correlation output:
     *
     * C(r) for r = 1,...,L/2.
     */
    fprintf(corr_file, "temperature");

    for (int r = 1; r <= L / 2; r++) {
        fprintf(corr_file, ",C_%d", r);
    }

    fprintf(corr_file, "\n");


    for (
        double temperature = T_MIN;
        temperature <= T_MAX + 1e-12;
        temperature += T_STEP
    ) {

        initialize_lattice();


        /* ---------------- Thermalization ---------------- */

        for (
            int sweep = 0;
            sweep < THERMALIZATION_SWEEPS;
            sweep++
        ) {
            metropolis_sweep(temperature);
        }


        /* ---------------- Measurements ---------------- */

        double E_sum = 0.0;
        double E2_sum = 0.0;

        double M_sum = 0.0;
        double M2_sum = 0.0;
        double abs_M_sum = 0.0;

        double correlation[L / 2 + 1] = {0.0};

        int samples = 0;


        for (
            int sweep = 0;
            sweep < MEASUREMENT_SWEEPS;
            sweep++
        ) {

            metropolis_sweep(temperature);


            /*
             * Measurements are separated by several sweeps
             * to reduce correlations between consecutive samples.
             */
            if (sweep % SAMPLE_INTERVAL != 0) {
                continue;
            }


            const double E =
                total_energy();

            const double M =
                total_magnetization();


            E_sum += E;
            E2_sum += E * E;

            /*
             * Keep signed and absolute magnetization separately.
             *
             * Signed M is required for the standard susceptibility,
             * while |M| is useful as the finite-lattice order
             * parameter.
             */
            M_sum += M;
            M2_sum += M * M;
            abs_M_sum += fabs(M);


            /*
             * Spatial spin-spin correlation along lattice rows.
             */
            for (int row = 0; row < L; row++) {

                for (int col = 0; col < L; col++) {

                    for (int r = 1; r <= L / 2; r++) {

                        correlation[r] +=
                            grid[row][col]
                            * grid[row][pbc(col + r)];
                    }
                }
            }

            samples++;
        }


        /* ---------------- Ensemble averages ---------------- */

        const double E_mean =
            E_sum / samples;

        const double E2_mean =
            E2_sum / samples;

        const double M_mean =
            M_sum / samples;

        const double M2_mean =
            M2_sum / samples;

        const double abs_M_mean =
            abs_M_sum / samples;


        /*
         * Heat capacity per spin:
         *
         * C_V / N = ( <E²> - <E>² ) / (N T²)
         */
        const double heat_capacity =
            (E2_mean - E_mean * E_mean)
            / (N * temperature * temperature);


        /*
         * Magnetic susceptibility per spin:
         *
         * χ / N = ( <M²> - <M>² ) / (N T)
         */
        const double susceptibility =
            (M2_mean - M_mean * M_mean)
            / (N * temperature);


        fprintf(
            thermo_file,
            "%.6f,%.10f,%.10f,%.10f,%.10f\n",
            temperature,
            E_mean / N,
            abs_M_mean / N,
            heat_capacity,
            susceptibility
        );


        /*
         * Connected spatial correlation:
         *
         * C(r) = <s_i s_{i+r}> - <s>²
         */
        const double mean_spin =
            M_mean / N;


        fprintf(
            corr_file,
            "%.6f",
            temperature
        );


        for (int r = 1; r <= L / 2; r++) {

            const double correlation_mean =
                correlation[r]
                / ((double) samples * N);

            const double connected_correlation =
                correlation_mean
                - mean_spin * mean_spin;

            fprintf(
                corr_file,
                ",%.10f",
                connected_correlation
            );
        }

        fprintf(corr_file, "\n");


        printf(
            "T = %.2f completed.\n",
            temperature
        );
    }


    fclose(thermo_file);
    fclose(corr_file);

    printf(
        "Serial thermodynamic analysis completed.\n"
    );
}


/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char *argv[])
{
    /*
     * A fixed default seed makes the simulation reproducible.
     * A different seed can be supplied from the command line:
     *
     *      ./ising_serial 42
     */
    unsigned int seed = 12345;

    if (argc > 1) {
        seed = (unsigned int) strtoul(
            argv[1],
            NULL,
            10
        );
    }

    srand(seed);

    printf(
        "2D Ising model — serial implementation\n"
    );

    printf(
        "L = %d, N = %d, seed = %u\n",
        L,
        N,
        seed
    );


    run_equilibration();

    run_thermodynamic_analysis();

    return EXIT_SUCCESS;
}
