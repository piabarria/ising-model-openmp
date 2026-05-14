#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define L 16          // Tamaño de la red (L x L)
#define N (L * L)
#define J 1.0         // Constante de interacción
#define B 0.0         // Campo magnético

int grid[L][L];

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
double calc_dE(int r, int c) {
    int sum_neighbors = grid[pbc(r-1)][c] + grid[pbc(r+1)][c] +
                        grid[r][pbc(c-1)] + grid[r][pbc(c+1)];
    // H = -J * S_i * S_j - B * S_i
    return 2.0 * J * grid[r][c] * sum_neighbors + 2.0 * B * grid[r][c];
}

// Energía total del sistema
double total_energy() {
    double E = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            E += -J * grid[i][j] * (grid[pbc(i+1)][j] + grid[i][pbc(j+1)]) - B * grid[i][j];
        }
    }
    return E;
}

// Magnetización total
double total_magnetization() {
    double M = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            M += grid[i][j];
        }
    }
    return M;
}

// Un paso de Metrópolis (intenta invertir 1 spin aleatorio)
void metropolis_step(double T) {
    int r = rand() % L;
    int c = rand() % L;

    double dE = calc_dE(r, c);

    if (dE < 0 || ((double)rand() / RAND_MAX) < exp(-dE / T)) {
        grid[r][c] *= -1; // Se acepta la inversión
    }
}

// --- EJERCICIO 1: Tiempo de equilibración ---
void ejercicio1() {
    FILE *f = fopen("ej1_equilibrio.csv", "w");
    fprintf(f, "iteracion,energia,magnetizacion\n");

    init_grid();
    double T = 2.0; // T por debajo de Tc para ver ordenamiento
    int max_iter = 200000;

    for (int iter = 0; iter <= max_iter; iter++) {
        metropolis_step(T);
        if (iter % 100 == 0) { // Guardar cada 100 pasos para no saturar
            double E = total_energy() / N;
            double M = fabs(total_magnetization()) / N;
            fprintf(f, "%d,%f,%f\n", iter, E, M);
        }
    }
    fclose(f);
    printf("Ejercicio 1 completado.\n");
}

// --- EJERCICIOS 2 y 3: Transición de fase y Correlación ---
void ejercicios2_y_3() {
    FILE *f2 = fopen("ej2_termodinamica.csv", "w");
    FILE *f3 = fopen("ej3_correlacion.csv", "w");

    fprintf(f2, "T,E_mean,M_mean,Cv,Chi\n");

    // Encabezado para correlación: T, C(1), C(2), ..., C(L/2)
    fprintf(f3, "T");
    for(int r = 1; r <= L/2; r++) fprintf(f3, ",C_%d", r);
    fprintf(f3, "\n");

    int therm_sweeps = 5000;  // Sweeps para termalizar
    int meas_sweeps = 10000;  // Sweeps para medir

    // Barrido de temperaturas
    for (double T = 1.5; T <= 3.5; T += 0.05) {
        init_grid();

        // Termalización
        for (int i = 0; i < therm_sweeps * N; i++) {
            metropolis_step(T);
        }

        double E_sum = 0, E2_sum = 0;
        double M_sum = 0, M2_sum = 0;
        double C_r[L/2 + 1];
        for(int r=0; r<=L/2; r++) C_r[r] = 0;

        int samples = 0;

        // Medición
        for (int i = 0; i < meas_sweeps; i++) {
            for(int j=0; j<N; j++) metropolis_step(T); // 1 sweep

            // Tomar muestras cada 10 sweeps para reducir autocorrelación
            if (i % 10 == 0) {
                double E = total_energy();
                double M = fabs(total_magnetization());

                E_sum += E; E2_sum += E*E;
                M_sum += M; M2_sum += M*M;

                // Función de correlación a lo largo de las filas
                for(int row=0; row<L; row++) {
                    for(int col=0; col<L; col++) {
                        for(int r=1; r<=L/2; r++) {
                            C_r[r] += grid[row][col] * grid[row][pbc(col+r)];
                        }
                    }
                }
                samples++;
            }
        }

        // Promedios termodinámicos
        double E_mean = E_sum / samples;
        double E2_mean = E2_sum / samples;
        double M_mean = M_sum / samples;
        double M2_mean = M2_sum / samples;

        double Cv = (E2_mean - E_mean*E_mean) / (T * T * N);
        double Chi = (M2_mean - M_mean*M_mean) / (T * N);

        fprintf(f2, "%f,%f,%f,%f,%f\n", T, E_mean/N, M_mean/N, Cv, Chi);

        // Promedios de correlación
        fprintf(f3, "%f", T);
        for(int r = 1; r <= L/2; r++) {
            double c_val = (C_r[r] / (samples * N)) - ((M_mean/N)*(M_mean/N));
            fprintf(f3, ",%f", c_val);
        }
        fprintf(f3, "\n");
    }

    fclose(f2);
    fclose(f3);
    printf("Ejercicios 2 y 3 completados.\n");
}

int main() {
    srand(time(NULL));
    ejercicio1();
    ejercicios2_y_3();
    return 0;
}