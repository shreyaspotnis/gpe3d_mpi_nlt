CC=mpicc
CFLAGS=-O2
LIBS=-lfftw3 -lfftw3_mpi

all: bin/gpe3d_mpi_nlt

bin/gpe3d_mpi_nlt: src/gpe3d_mpi_nlt.c src/ini.c src/configuration.c
	$(CC) $(CFLAGS) $(LIBS) src/ini.c src/gpe3d_mpi_nlt.c src/configuration.c -o bin/gpe3d_mpi_nlt

clean:
	rm bin/gpe3d_mpi_nlt
