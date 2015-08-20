#ifndef CONFIGURATION_H
#define CONFIGURATION_H

// Loads paramaters from parms.ini and stores it in this structure
typedef struct configuration {
    int Nx;
    int Nx_local;
    int x_start_local;
    int Ny;
    int Nz;
    int Nt;
    int Nt_imag;
    int Nt_store;
    int imag_time;
    int padding_start_index;
    int nt_ramp;

    double dt;
    double dx, dy, dz;

    // simulation specific stuff
    double gamma_z;
    double kappa;
    double alpha;
    double barrier_height;
    double barrier_height_imag;
    double barrier_waist;
    double barrier_rayleigh_range;
    double barrier_pos;
    double padding_slope;
    double loss_factor;

    // calculated stuff
    double mu_theory;
    double lx_theory, ry_theory, rz_theory;
    double dkx, dky, dkz;

    // log and save file names
    char store_dir[256];

} configuration;

int read_config(configuration *cfg, int argc, char **argv);
int check_config(configuration *cfg);
int process_config(configuration *cfg);
int print_config(FILE *fp, configuration *cfg);

int print_int(FILE *fp, const char *variable_name, int variable);
int print_double(FILE *fp, const char *variable_name, double variable);

// for reading INI files
static int handler(void* user, const char* section, const char* name,
                   const char* value);

#endif
