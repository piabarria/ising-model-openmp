CC = gcc

CFLAGS = -O3 -std=c11 -Wall -Wextra -pedantic
OMPFLAGS = -fopenmp
LDLIBS = -lm

BIN_DIR = bin
SRC_DIR = src

SERIAL_SRC = $(SRC_DIR)/ising_serial.c
OPENMP_SRC = $(SRC_DIR)/ising_openmp.c

SERIAL_BIN = $(BIN_DIR)/ising_serial
OPENMP_BIN = $(BIN_DIR)/ising_openmp

SERIAL_QUICK_BIN = $(BIN_DIR)/ising_serial_quick
OPENMP_QUICK_BIN = $(BIN_DIR)/ising_openmp_quick


.PHONY: all serial openmp quick quick-serial quick-openmp \
        run-serial run-openmp run-quick plots clean directories


all: serial openmp


directories:
	mkdir -p $(BIN_DIR)
	mkdir -p results/data/serial
	mkdir -p results/data/openmp
	mkdir -p results/data/test/serial
	mkdir -p results/data/test/openmp
	mkdir -p results/figures


serial: directories
	$(CC) $(CFLAGS) $(SERIAL_SRC) $(LDLIBS) -o $(SERIAL_BIN)


openmp: directories
	$(CC) $(CFLAGS) $(OMPFLAGS) $(OPENMP_SRC) $(LDLIBS) -o $(OPENMP_BIN)


quick: quick-serial quick-openmp


quick-serial: directories
	$(CC) $(CFLAGS) -DQUICK_TEST $(SERIAL_SRC) $(LDLIBS) -o $(SERIAL_QUICK_BIN)


quick-openmp: directories
	$(CC) $(CFLAGS) $(OMPFLAGS) -DQUICK_TEST $(OPENMP_SRC) $(LDLIBS) -o $(OPENMP_QUICK_BIN)


run-serial: serial
	./$(SERIAL_BIN)


run-openmp: openmp
	./$(OPENMP_BIN)


run-quick: quick
	./$(SERIAL_QUICK_BIN)
	./$(OPENMP_QUICK_BIN)


plots:
	python3 scripts/plot_results.py


clean:
	rm -rf $(BIN_DIR)
