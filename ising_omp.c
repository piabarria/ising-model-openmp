#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define L 16          // Tamaño de la red (L x L)
#define N (L * L)
#define J 1.0         // Constante de interacción
#define B 0.0         // Campo magnético

int grid[L][L];
unsigned int thread_seeds[64]; // Soporte para múltiples hilos (hasta 64 en este ejemplo)

// Condiciones de contorno periódicas
int pbc(int i) {
    return (i + L) % L;
}

// Inicializar la red aleatoriamente
void init_grid() {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            grid[i][j] = (rand() % 2 == 0) ? 1 : -1;
        }
    }
}

// Calcular la diferencia de energía si se invierte el spin en (r, c)
// Es thread-safe si se usa el esquema Red-Black
double calc_dE(int r, int c) {
    int sum_neighbors = grid[pbc(r-1)][c] + grid[pbc(r+1)][c] +
                        grid[r][pbc(c-1)] + grid[r][pbc(c+1)];
    return 2.0 * J * grid[r][c] * sum_neighbors + 2.0 * B * grid[r][c];
}

// Energía total del sistema (con reducción OpenMP)
double total_energy() {
    double E = 0;
    #pragma omp parallel for reduction(+:E) collapse(2)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            E += -J * grid[i][j] * (grid[pbc(i+1)][j] + grid[i][pbc(j+1)]) - B * grid[i][j];
        }
    }
    return E;
}

// Magnetización total (con reducción OpenMP)
double total_magnetization() {
    double M = 0;
    #pragma omp parallel for reduction(+:M) collapse(2)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            M += grid[i][j];
        }
    }
    return M;
}

// Un sweep COMPLETO (N intentos) de Metrópolis usando Red-Black
void metropolis_sweep_omp(double T) {
    // color 0 = "casillas rojas", color 1 = "casillas negras"
    for (int color = 0; color < 2; color++) {
        #pragma omp parallel for
        for (int i = 0; i < L; i++) {
            int tid = omp_get_thread_num();
            // El inicio de la columna j depende de la paridad de la fila i y del color
            for (int j = (i + color) % 2; j < L; j += 2) {
                double dE = calc_dE(i, j);

                // Usamos rand_r() para garantizar thread-safety en la generación aleatoria
                if (dE < 0 || ((double)rand_r(&thread_seeds[tid]) / RAND_MAX) < exp(-dE / T)) {
                    grid[i][j] *= -1; // Se acepta la inversión
                }
            }
        }
    }
}

// --- EJERCICIO 1: Tiempo de equilibración ---
void ejercicio1() {
    FILE *f = fopen("ej1_equilibrio.csv", "w");
    fprintf(f, "iteracion,energia,magnetizacion\n");

    init_grid();
    double T = 2.0;
    int max_sweeps = 1000; // 1000 sweeps equivalen a ~256,000 iteraciones en L=16

    for (int sweep = 0; sweep <= max_sweeps; sweep++) {
        metropolis_sweep_omp(T);

        // Guardamos algunas muestras
        if (sweep % 5 == 0) {
            double E = total_energy() / N;
            double M = fabs(total_magnetization()) / N;
            fprintf(f, "%d,%f,%f\n", sweep * N, E, M); // Multiplicamos por N para mantener la escala del doc
        }
    }
    fclose(f);
    printf("Ejercicio 1 completado en paralelo.\n");
}

// --- EJERCICIOS 2 y 3: Transición de fase y Correlación ---
void ejercicios2_y_3() {
    FILE *f2 = fopen("ej2_termodinamica.csv", "w");
    FILE *f3 = fopen("ej3_correlacion.csv", "w");

    fprintf(f2, "T,E_mean,M_mean,Cv,Chi\n");
    fprintf(f3, "T");
    for(int r = 1; r <= L/2; r++) fprintf(f3, ",C_%d", r);
    fprintf(f3, "\n");

    int therm_sweeps = 5000;
    int meas_sweeps = 10000;

    for (double T = 1.5; T <= 3.5; T += 0.05) {
        init_grid();

        // Termalización
        for (int i = 0; i < therm_sweeps; i++) {
            metropolis_sweep_omp(T);
        }

        double E_sum = 0, E2_sum = 0;
        double M_sum = 0, M2_sum = 0;
        double C_r[L/2 + 1];
        for(int r=0; r<=L/2; r++) C_r[r] = 0;

        int samples = 0;

        // Medición
        for (int i = 0; i < meas_sweeps; i++) {
            metropolis_sweep_omp(T);

            // Tomar muestras cada 10 sweeps para evitar autocorrelación
            if (i % 10 == 0) {
                double E = total_energy();
                double M = fabs(total_magnetization());

                E_sum += E; E2_sum += E*E;
                M_sum += M; M2_sum += M*M;

                // Paralelizamos el cálculo de la correlación de forma segura
                #pragma omp parallel
                {
                    double thread_C_r[L/2 + 1] = {0};

                    #pragma omp for collapse(2)
                    for(int row=0; row<L; row++) {
                        for(int col=0; col<L; col++) {
                            for(int r=1; r<=L/2; r++) {
                                thread_C_r[r] += grid[row][col] * grid[row][pbc(col+r)];
                            }
                        }
                    }

                    // Combinamos los resultados de cada hilo críticamente
                    #pragma omp critical
                    {
                        for(int r=1; r<=L/2; r++) C_r[r] += thread_C_r[r];
                    }
                }
                samples++;
            }
        }

        double E_mean = E_sum / samples;
        double E2_mean = E2_sum / samples;
        double M_mean = M_sum / samples;
        double M2_mean = M2_sum / samples;

        double Cv = (E2_mean - E_mean*E_mean) / (T * T * N);
        double Chi = (M2_mean - M_mean*M_mean) / (T * N);

        fprintf(f2, "%f,%f,%f,%f,%f\n", T, E_mean/N, M_mean/N, Cv, Chi);

        fprintf(f3, "%f", T);
        for(int r = 1; r <= L/2; r++) {
            double c_val = (C_r[r] / (samples * N)) - ((M_mean/N)*(M_mean/N));
            fprintf(f3, ",%f", c_val);
        }
        fprintf(f3, "\n");
    }

    fclose(f2);
    fclose(f3);
    printf("Ejercicios 2 y 3 completados en paralelo.\n");
}

int main() {
    srand(time(NULL));

    // Inicializar semillas independientes para cada hilo
    int max_threads = omp_get_max_threads();
    for(int i = 0; i < max_threads; i++) {
        thread_seeds[i] = rand();
    }

    printf("Ejecutando con %d hilos...\n", max_threads);

    ejercicio1();
    ejercicios2_y_3();

    return 0;
}
