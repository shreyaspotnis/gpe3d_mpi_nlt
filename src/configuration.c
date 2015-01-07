#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>

#include "common.h"
#include "configuration.h"

int read_config(configuration *cfg, int argc, char **argv) {
    char *input_filename = "parms.ini";
    // read inputs from file
    if(rank == MASTER_RANK) {
        int success = FALSE;
        int c;
        FILE *fp;
        success = TRUE;
        while ((c = getopt(argc, argv, "f:")) != EOF) {
            switch(c) {
            case 'f':
                input_filename = optarg;
                success = TRUE;
                break;
            } // end of switch(c)
        } // end of while

        if (ini_parse(input_filename, handler, cfg) < 0) {
                fprintf(stderr, "Cannot load parms file: %s\n", input_filename);
                success = FALSE;
        }
        if(check_config(cfg) != 0)
            success = FALSE;
        process_config(cfg);
        if(!success)
            MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // send configuration to all processors
    MPI_Bcast(cfg, sizeof(configuration), MPI_CHAR, MASTER_RANK,
              MPI_COMM_WORLD);
    return 0;
}


// Returns 1 if something wrong with config file, prints error to stderr
int check_config(configuration *cfg) {
    // find if Nx, Ny, Nz are powers of 2
    if(!POW2(cfg->Nx)) {
        fprintf(stderr, "Nx:%d is not a power of 2\n", cfg->Nx);
        return 1;
    }
    if(!POW2(cfg->Ny)) {
        fprintf(stderr, "Ny:%d is not a power of 2\n", cfg->Ny);
        return 1;
    }
    if(!POW2(cfg->Nz)) {
        fprintf(stderr, "Nz:%d is not a power of 2\n", cfg->Nz);
        return 1;
    }
    return 0;

}

// Calculate quantities required for the simulation based on inputs in
// configuration
int process_config(configuration *cfg) {

    cfg->dkx = 2.0*PI/(cfg->dx * cfg->Nx);
    cfg->dky = 2.0*PI/(cfg->dy * cfg->Ny);
    cfg->dkz = 2.0*PI/(cfg->dz * cfg->Nz);

    cfg->mu_theory = pow(3.0 * cfg->kappa * cfg->alpha * cfg->gamma_z /
                         PI, 1.0/3.0);

    cfg->lx_theory = cfg->mu_theory / cfg->alpha;
    cfg->ry_theory = sqrt(2.0 * cfg->mu_theory);
    cfg->rz_theory = cfg->ry_theory / cfg->gamma_z;

    return 0;
}

static int handler(void* user, const char* section, const char* name,
                   const char* value) {
    configuration* pconfig = (configuration*)user;

    #define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0
    if (MATCH("sim", "nx"))
        pconfig->Nx = atoi(value);
    else if (MATCH("sim", "ny"))
        pconfig->Ny = atoi(value);
    else if (MATCH("sim", "nz"))
        pconfig->Nz = atoi(value);
    else if (MATCH("sim", "imag_time"))
        pconfig->imag_time = atoi(value);
    else if (MATCH("sim", "nt"))
        pconfig->Nt = atoi(value);
    else if (MATCH("sim", "nt_imag"))
        pconfig->Nt_imag = atoi(value);
    else if (MATCH("sim", "nt_store"))
        pconfig->Nt_store = atoi(value);
    else if (MATCH("sim", "dt"))
        pconfig->dt = atof(value);
    else if (MATCH("sim", "dx"))
        pconfig->dx = atof(value);
    else if (MATCH("sim", "dy"))
        pconfig->dy = atof(value);
    else if (MATCH("sim", "dz"))
        pconfig->dz = atof(value);
    else if (MATCH("sim", "gamma_z"))
        pconfig->gamma_z = atof(value);
    else if (MATCH("sim", "kappa"))
        pconfig->kappa = atof(value);
    else if (MATCH("sim", "alpha"))
        pconfig->alpha = atof(value);
    else if (MATCH("sim", "barrier_height"))
        pconfig->barrier_height = atof(value);
    else if (MATCH("sim", "barrier_height_imag"))
        pconfig->barrier_height_imag = atof(value);
    else if (MATCH("sim", "barrier_waist"))
        pconfig->barrier_waist = atof(value);
    else if (MATCH("sim", "barrier_pos"))
        pconfig->barrier_pos = atof(value);
    else if (MATCH("sim", "barrier_rayleigh_range"))
        pconfig->barrier_rayleigh_range = atof(value);
    else if (MATCH("sim", "store_dir"))
        strcpy(pconfig->store_dir, value);
    else if (MATCH("sim", "padding_start_index"))
        pconfig->padding_start_index = atoi(value);
    else if (MATCH("sim", "padding_slope"))
        pconfig->padding_slope = atof(value);
    else if (MATCH("sim", "loss_factor"))
        pconfig->loss_factor = atof(value);
    else if (MATCH("sim", "nt_ramp"))
        pconfig->nt_ramp = atoi(value);
    else
        return 0;  /* unknown section/name, error */
    return 1;
}

int print_int(FILE *fp, const char *variable_name, int variable) {
    fprintf(fp, "%s:\t%d\n", variable_name, variable);
    return 0;
}

int print_double(FILE * fp, const char *variable_name, double variable) {
    fprintf(fp, "%s:\t%f\n", variable_name, variable);
    return 0;
}

int print_configuration(FILE *fp, configuration *cfg) {
    fprintf(fp, "Input parameters:\n");
    print_int(fp, "Nx", cfg->Nx);
    print_int(fp, "Ny", cfg->Ny);
    print_int(fp, "Nz", cfg->Nz);
    fprintf(fp, "\n");

    print_int(fp, "Nt", cfg->Nt);
    print_int(fp, "Nt_store", cfg->Nt_store);
    print_int(fp, "imag_time", cfg->imag_time);
    print_int(fp, "nt_ramp", cfg->nt_ramp);
    print_double(fp, "dt", cfg->dt);
    fprintf(fp, "\n");

    print_double(fp, "dx", cfg->dx);
    print_double(fp, "dy", cfg->dy);
    print_double(fp, "dz", cfg->dz);
    fprintf(fp, "\n");

    print_double(fp, "gamma_z", cfg->gamma_z);
    print_double(fp, "kappa", cfg->kappa);
    print_double(fp, "alpha", cfg->alpha);
    print_double(fp, "barrier_height", cfg->barrier_height);
    print_double(fp, "barrier_height_imag", cfg->barrier_height_imag);
    print_double(fp, "barrier_waist", cfg->barrier_waist);
    print_double(fp, "barrier_rayleigh_range", cfg->barrier_rayleigh_range);
    print_int(fp, "padding_start_index", cfg->padding_start_index);
    print_double(fp, "padding_slope", cfg->padding_slope);
    print_double(fp, "loss_factor", cfg->loss_factor);

    fprintf(fp, "Calculated parameters:\n");
    print_double(fp, "dkx", cfg->dkx);
    print_double(fp, "dky", cfg->dky);
    print_double(fp, "dkz", cfg->dkz);
    print_double(fp, "mu_theory", cfg->mu_theory);
    print_double(fp, "lx_theory", cfg->lx_theory);
    print_double(fp, "ry_theory", cfg->ry_theory);
    print_double(fp, "rz_theory", cfg->rz_theory);
    fprintf(fp, "\n");
    return 0;

}
