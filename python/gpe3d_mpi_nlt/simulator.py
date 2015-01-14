"""Calculates non-dimensional constants and saves them in parms.ini."""

from rbconstants import *
from common import to_nK
import ConfigParser
import numpy as np
from sciscript import prependscratch
import shutil
import os
import pickle


class Simulator(object):
    # Inputs start here
    Nx = 256
    Ny = 128
    Nz = 128

    dx = 1./16.
    dy = 1./4.
    dz = 1./8.

    Nt = 320000
    Nt_imag = 2000
    dt = 0.001
    Nt_store = 801

    log_n0 = 9.0

    fz = 86.6
    a = 8.4  # m/s^2
    barrier_height_imag_t = 550e-9  # nK
    barrier_height_t = 340e-9  # nK
    barrier_waist = 1.3e-6  # um
    ramp_time = 5e-3  # ms

    barrier_rayleigh_range = 8e-6  # um
    padding_height_t = 500e-9  # nK
    bg_decay_time = 3.3  # s

    def calculate_derived_quantities(self):
        """Calculate quantities required for the simulation, but dependent on
        the input parameters."""
        self.n_atoms = np.exp(self.log_n0)
        self.fy = self.fz / 2.0
        self.barrier_pos_index = self.Nx / 4
        self.barrier_pos = (self.barrier_pos_index -
                            (self.Nx - 1) / 2.0) * self.dx
        self.padding_start_index = self.Nx / 2

        self.omega_y = 2.0 * pi * self.fy
        self.omega_z = 2.0 * pi * self.fz

        self.gamma_z = self.fz / self.fy

        self.length_scale = (hbar / m_rb / self.omega_y) ** 0.5
        self.time_scale = 1.0 / self.omega_y
        self.energy_scale = hbar * self.omega_y

        self.barrier_height_imag_nd = (self.barrier_height_imag_t * kb /
                                       self.energy_scale)
        self.barrier_height_nd = self.barrier_height_t * kb / self.energy_scale
        self.barrier_waist_nd = self.barrier_waist / self.length_scale
        self.barrier_rayleigh_range_nd = (self.barrier_rayleigh_range /
                                          self.length_scale)

        self.kappa = (4.0 * pi * a_bg * self.n_atoms / self.length_scale)
        self.alpha = self.a / self.length_scale / self.omega_y ** 2

        self.mu_nd = (3.0 * self.kappa * self.alpha
                      * self.gamma_z / pi) ** 0.3333
        self.ry_nd = (2.0 * self.mu_nd) ** 0.5
        self.rz_nd = self.ry_nd / self.gamma_z
        self.lx_nd = self.mu_nd / self.alpha

        self.padding_slope_nd = (self.padding_height_t * kb / self.energy_scale
                                 / (self.Nx - self.padding_start_index))
        self.bg_decay_time_nd = self.bg_decay_time / self.time_scale
        self.loss_factor = np.exp(-self.dt/self.bg_decay_time_nd/2.0/2.0)
        self.nt_ramp = int(self.ramp_time / self.time_scale / self.dt)

    def print_stuff(self):
        print('dx_real(um)', self.dx*self.length_scale*1e6)
        print('dy_real(um)', self.dy*self.length_scale*1e6)
        print('dz_real(um)', self.dz*self.length_scale*1e6)
        print('dt(ms)', self.dt*self.time_scale*1e3)
        print('kappa:', self.kappa)
        print('alpha', self.alpha)
        print('mu_nd', self.mu_nd)
        print('ry_nd', self.ry_nd)
        print('rz_nd', self.rz_nd)
        print('lx_nd', self.lx_nd)
        print('mu(nK)', to_nK(self.mu_nd * self.energy_scale))
        print('ry(um)', self.ry_nd * self.length_scale * 1e6)
        print('rz(um)', self.rz_nd * self.length_scale * 1e6)
        print('lx(um)', self.lx_nd * self.length_scale * 1e6)
        print('length x', self.dx * self.Nx)
        print('length y', self.dy * self.Ny)
        print('length z', self.dz * self.Nz)
        print('barrier_height_imag_nd', self.barrier_height_imag_nd)
        print('barrier_height_nd', self.barrier_height_nd)
        print('barrier_waist(um)', self.barrier_waist * 1e6)
        print('barrier_waist_nd', self.barrier_waist_nd)
        print('barrier_rayleigh_range(um)', self.barrier_rayleigh_range * 1e6)
        print('barrier_rayleigh_range_nd', self.barrier_rayleigh_range_nd)
        print('dt(us)', self.dt * self.time_scale * 1e6)
        print('total time (ms)', self.dt * self.Nt * self.time_scale * 1e3)
        print('total trap cycles', self.dt*self.Nt*self.time_scale*self.fy)
        print('padding_slope_nd', self.padding_slope_nd)
        print('loss_factor', self.loss_factor)
        print('nt_ramp', self.nt_ramp)

    def get_save_folder_name(self, base_save_folder):
        """Returns the path in which to write contents, based on barrier height
        and atom number. Appends this folder to base_save_folder."""
        folder_name_fmt = 'bh0_{0:.0f}_bh1_{1:.0f}_logn0_{2:.1f}'

        folder_name = folder_name_fmt.format(self.barrier_height_imag_t*1e9,
                                             self.barrier_height_t*1e9,
                                             self.log_n0)

        store_dir = os.path.join(base_save_folder, folder_name)
        return store_dir

    def create_config_files(self, store_dir):
        """Creates the following files in store_dir:

        parms.ini - All the simulation parameters needed for the 3D GPE
                    simulation.

        jobscript - A bash file which describes the MPI job to run. This file
                    is copied from the scripts/ folder

        sim.pkl - A pickle of itself, used to later load the sim parameters.
        """
        if not os.path.exists(store_dir):

            os.makedirs(store_dir)
        parms_file_name = os.path.join(store_dir, 'parms.ini')
        with open(parms_file_name, 'w') as f:
            f.write('[sim]')

        parms = ConfigParser.SafeConfigParser(allow_no_value=True)
        parms.read(parms_file_name)

        parms.set('sim', 'Nx', str(self.Nx))
        parms.set('sim', 'Ny', str(self.Ny))
        parms.set('sim', 'Nz', str(self.Nz))
        parms.set('sim', 'dx', str(self.dx))
        parms.set('sim', 'dy', str(self.dy))
        parms.set('sim', 'dz', str(self.dz))
        parms.set('sim', 'gamma_z', str(self.gamma_z))
        parms.set('sim', 'kappa', str(self.kappa))
        parms.set('sim', 'alpha', str(self.alpha))
        parms.set('sim', 'imag_time', str(1))
        parms.set('sim', 'nt', str(self.Nt))
        parms.set('sim', 'nt_store', str(self.Nt_store))
        parms.set('sim', 'dt', str(self.dt))
        parms.set('sim', 'barrier_height', str(self.barrier_height_nd))
        parms.set('sim', 'barrier_height_imag',
                  str(self.barrier_height_imag_nd))
        parms.set('sim', 'barrier_waist',
                  str(self.barrier_waist_nd))
        parms.set('sim', 'barrier_rayleigh_range',
                  str(self.barrier_rayleigh_range_nd))
        parms.set('sim', 'store_dir', store_dir)
        parms.set('sim', 'padding_start_index', str(self.padding_start_index))
        parms.set('sim', 'padding_slope', str(self.padding_slope_nd))
        parms.set('sim', 'nt_imag', str(self.Nt_imag))
        parms.set('sim', 'barrier_pos', str(self.barrier_pos))
        parms.set('sim', 'loss_factor', str(self.loss_factor))
        parms.set('sim', 'nt_ramp', str(self.nt_ramp))

        curr_file_dir = os.path.dirname(os.path.realpath(__file__))
        jobscript_path = os.path.join(curr_file_dir, '../../scripts/jobscript')

        with open(parms_file_name, 'w') as f:
            parms.write(f)
        shutil.copyfile(jobscript_path, os.path.join(store_dir, 'jobscript'))

        pickle_file_name = os.path.join(store_dir, 'sim.pkl')
        self.pickle(pickle_file_name)

    def pickle(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)


def load_sim_from_file(fname):
    with open(fname, 'rb') as f:
        sim = pickle.load(f)
    return sim


def load_sim_from_folder(folder):
    with open(os.path.join(folder, 'sim.pkl'), 'rb') as f:
        sim = pickle.load(f)

    # check if the simulation has already run
    ex_vals_path = os.path.join(folder, 'ex_vals')
    if os.path.exists(ex_vals_path):
        ex_vals = dict_from_data_file(ex_vals_path)
        sim.ex_vals = ex_vals
    return sim


def write_submission_script(base_save_folder):
    """Writes a script to base_save_folder. When this script is run, it
    recursively goes to all sub-directories in base_save_folder and runs
    "qsub jobscript"
    """
    string = 'find . -type d -exec sh -c \'(cd {} && qsub jobscript)\' \';\''
    with open(os.path.join(base_save_folder, 'job_submit.sh'), 'w') as f:
        f.write(string)


def dict_from_data_file(full_path):
    """Reads tabulated data and returns a dict with each col being an entry
       in the dict."""
    with open(full_path) as fp:
        first_line = fp.readline()
    ex_val = np.loadtxt(full_path)
    var_names = first_line.split('\t')
    var_names[0] = var_names[0][1:]  # remove # from the first index
    var_names[-1] = var_names[-1][:-1]  # remove new line from the last index

    data_dict = {}
    for col, vn in zip(ex_val.T, var_names):
        data_dict[vn] = col
    return data_dict


def get_ex_vals_dict(folder_name):
    return dict_from_data_file(os.path.join(folder_name, 'ex_vals'))


def create_batch_jobs(base_save_folder):
    logn_atoms_array = np.array([10., 10.5, 11., 11.5, 12., 12.5])
    bht_array = np.array([280, 300, 320, 350])*1e-9
    sims = []
    for logn in logn_atoms_array:
        for bht in bht_array:
            sim = Simulator()
            sim.barrier_height_t = bht
            sim.log_n0 = logn
            sim.calculate_derived_quantities()
            sim.print_stuff()
            save_folder = self.get_save_folder_name(base_save_folder)
            sim.create_config_files(save_folder)
            sims.append(sim)

    write_submission_script(base_save_folder)


def create_single_job(base_save_folder):
    sim = Simulator()
    sim.barrier_height_t = 330e-9
    sim.log_n0 = 10.0

    sim.calculate_derived_quantities()
    sim.print_stuff()
    save_folder = sim.get_save_folder_name(base_save_folder)
    sim.create_config_files(save_folder)

    sim1 = load_sim_from_folder(save_folder)
    sim1.print_stuff()


if __name__ == '__main__':
    base_save_folder = prependscratch('nl6_debug/')
    create_single_job(base_save_folder)
