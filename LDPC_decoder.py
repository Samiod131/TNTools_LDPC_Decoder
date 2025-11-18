import time
import json
import warnings
import sys

import numpy as np
import pickle

import TNTools as tnt
import code_gen as cg

'''
Classical LDPC decoder using TNTools MPS-MPO formalism. This files reads a list
of parameters from a json file dictionnary. 

To run this program run it into the terminal such as:

    ^^^ python LDPC_decoder_pyqec.py PATH

Where PATH is the path leading to where the parameter lists are. This is also
where results will be stored in a txt file.

'''


'''
Theses are functions for creating mpo xor checks following TNTools format
'''


def bitflip_array(p, n):
    '''
    Return 0-array of n bits with fixed bit flip probability p
    '''
    arr = np.random.choice([0, 1], n, p=[1-p, p])
    return arr


def xor_4t2():
    '''
    This function returns the four legged tensor fusion of a 3 variables xor
    tensor and a 3 legged copy tensor. Basic tensor of the decomposition of an
    xor gate into MPO format.
    '''
    # creating initial 3 legs xor tensor
    reverse_id = np.array([[0., 1.], [1., 0.]])
    xor_3t2 = np.zeros((2, 2, 2))
    xor_3t2[0, :, :] = reverse_id
    xor_3t2[1, :, :] = np.identity(2)

    # Creating initial delta tensor
    delta = np.zeros((2, 2, 2))
    delta[0, :, :] = np.array([[1., 0.], [0., 0.]])
    delta[1, :, :] = np.array([[0., 0.], [0., 1.]])

    # contracting the two to generate generic xor MPO tensor in correct format
    tens = np.tensordot(delta, xor_3t2, axes=([2], [0]))

    return tens


def parity_TN(check, par_tens=xor_4t2(), adjacency=True):
    '''
    Returns the MPO decomposition of a parity check gates plus the position of
    the first and last tensors in the chain. 
    '''
    # Creating simple cross tensor
    cross = np.tensordot(np.identity(2), np.identity(2), axes=0)
    # adapting receiving of input, adjacency matrix rows of check positions
    if adjacency == True:
        nonzeros = np.core.fromnumeric.nonzero(check)[0]
    else:
        nonzeros = np.array([c for c in check])

    # init list and tens dimensions
    mpo = []
    tens_shape = par_tens.shape

    # We save the first element of the MPO, index fixed linked to #tens. parity
    if nonzeros.size % 2 == 0:
        mpo.append(par_tens[:, :, 0, :].reshape(
            (tens_shape[0], tens_shape[1], 1, tens_shape[3])))
    else:
        mpo.append(par_tens[:, :, 1, :].reshape(
            (tens_shape[0], tens_shape[1], 1, tens_shape[3])))

    # we assign a tensor at each position of MPO with check or not
    for i in range(int(nonzeros[0]+1), int(nonzeros[-1])):
        if i in nonzeros:
            mpo.append(par_tens)
        else:
            mpo.append(cross)
    # We save the last tensor
    mpo.append(par_tens[:, :, :, 0].reshape(
        (tens_shape[0], tens_shape[1], tens_shape[2], 1)))

    # We return the list of tensors, the positions of the mps that are assigned
    # to the first and last mpo tensors.

    return mpo, nonzeros[0], nonzeros[-1]


def classical_ldpc_decoding(entry, checks, decoder_par, svd_function_par, main_comp_finder_par, sparse_entry=False):
    '''
    Full classical ldpc schedule:
        -entry: received message in np array;
        -checks: list of parity checks positions;
        -decoder_par: dictionnary of the decoder params;
        -svd_function_par: dict. of the svd function params used in the decoder; 
        -main_comp_finder_par: dict of main component finder params.

    '''
    # Initiating svd function
    def svd_func(_m):
        return tnt.reduced_svd(_m, **svd_function_par)

    # Creating the initial mps state class
    if sparse_entry:
        entry_mps = tnt.binary_mps_from_sparse(entry)
    else:
        entry_mps = tnt.binary_mps(entry)
    state_mps = tnt.MpsStateCanon(entry_mps, orth_pos=None, svd_func=svd_func)
    state_mps.create_orth()  # an orth center is created at the last site by default

    # Inputing boltzmann weight
    boltz_mpo = tnt.boltz_mpo(state_mps.length, **decoder_par)
    state_mps.mpo_contract(boltz_mpo)

    # Procedure to protect from complete state kill
    try:
        # Filtering loop
        for check in checks:
            c_mpo, begin, _ = parity_TN(check)
            state_mps.mpo_contract(c_mpo, begin)

        # check for nans and infs in array
        NanInf, _ = state_mps.nans_infs_find()
        if NanInf is True:
            warnings.warn(
                'Final mps is invalid (NANs or infs). Try again with less approximative svd function.')
            main_comp = None
        else:
            # Main component extraction
            main_comp = state_mps.main_component(**main_comp_finder_par)
    except ValueError:
        warnings.warn(
            'Total norm elimination of the state. Try again with less approximative svd function.')
        main_comp = None

    return main_comp


'''
Function for data saving
'''


def results_saving(results, filename='results.txt'):
    '''
    Saves all params and results to a txt file. 
    '''
    # Unpackig each sub_dictionnary
    labels = []
    values = []
    for sub_dict in results.values():
        labels = labels+list(sub_dict.keys())
        values = values+list(sub_dict.values())

    save_to_file(parameters=values, filename=filename, header=labels)

    return labels, values


def save_to_file(parameters, filename, header=False, buffer=30):
    '''
    Saving a list of elements into a text, csv or dat file. Allows for header
    selection when no precedent file already existing.
    '''
    try:
        with open(filename) as _:
            pass

    except FileNotFoundError:
        if header is False:
            print('The file doesn\'t exist. Creating one for data.')

        else:
            print('File doesn\'t exist, Creating one with given header.')
            header_line = ''
            for i, param in enumerate(header):
                if i == len(parameters)-1:
                    param_add = str(param)
                else:
                    param_add = str(param)+','
                skip = buffer-len(param_add)
                if skip <= 0:
                    param_add += ' '*buffer
                else:
                    param_add += ' '*skip
                header_line += param_add
            save = open(filename, 'a')
            print(header_line, file=save)
            save.close()

    step_line = ''
    for i, param in enumerate(parameters):
        if i == len(parameters)-1:
            param_add = str(param)
        else:
            param_add = str(param)+','
        skip = buffer-len(param_add)
        if skip <= 0:
            param_add += ' '*buffer
        else:
            param_add += ' '*skip
        step_line += param_add

    save = open(filename, 'a')
    print(step_line, file=save)
    save.close()


'''
Decoder class + schedule
'''


class TN_LDPC_Decoder:
    '''
    Decoder class fitting for qeclab/qecstruct format. 
    '''

    def __init__(self, parity_mat, decoder_par, svd_function_par, main_comp_finder_par):
        self.checks = parity_mat
        self.decoder_par = decoder_par
        self.svd_function_par = svd_function_par
        self.main_comp_finder_par = main_comp_finder_par

    def decode(self, entry):
        try:
            output = classical_ldpc_decoding(
                entry, self.checks, self.decoder_par, self.svd_function_par, self.main_comp_finder_par)
        except:
            output = None
        return output


class TimedDecoder:
    '''
    To keep time of each decoding procedure. 
    '''

    def __init__(self, decoder):
        self.decoder = decoder
        self.times = []

    def decode(self, entry):
        before = time.time()
        output = self.decoder.decode(entry)
        after = time.time()

        self.times.append(after-before)

        return output


def code_select(bit_degree, check_degree, code_size_mult, phys_err_rt, num_times, codes_path):
    '''
    Selects the right code, and the number of successive experiences from
    given parameters.
    '''
    try:
        with open(codes_path+'/'+str(bit_degree)+'_' + str(check_degree)+'_'+str(code_size_mult)+'.pkl', "rb") as file:
            ldpc_code = pickle.load(file)[2]
    except:
        warnings.warn(
            'No detected code for selected class. Generating a random one.')
        ldpc_code = cg.safe_code_gen(
            bit_deg=bit_degree, check_deg=check_degree, nmult=code_size_mult, rcmk=True)

    return ldpc_code, phys_err_rt, num_times, check_degree*code_size_mult


def new_run_func(code_selector, decoder_par, svd_function_par, main_comp_finder_par, codes_path):
    '''
    Run one serie of experience for a set of parameters and returns the required
    parameters results in a dictionnary format.
    '''

    # Gets the right LDPC code or the testing
    ldpc_code, phys_err_rt, num_times, entry_size = code_select(
        **code_selector, codes_path=codes_path)

    decoder = TimedDecoder(TN_LDPC_Decoder(
        parity_mat=ldpc_code, decoder_par=decoder_par, svd_function_par=svd_function_par, main_comp_finder_par=main_comp_finder_par))

    # Default case for minimal noise cutting in svd function
    if svd_function_par['err_th'] == 'default':
        # minimum possible probability value
        minimal_val = decoder_par['b_prob']**(
            code_selector['check_degree']*code_selector['code_size_mult'])
        # Going one order below for safety
        svd_function_par['err_th'] = minimal_val/10

    # Default max bond for dephased dmrg is the same as for the whole schedule
    if main_comp_finder_par['chi_max'] == 'default':
        main_comp_finder_par['chi_max'] = svd_function_par['max_len']

    # Create list for storing all failures / successes
    failures = []


    for _ in range(num_times):
        # Generate Random entry with given error rate
        entry = bitflip_array(p=phys_err_rt, n=entry_size)

        # Run decoding procedure on entry
        output = decoder.decode(entry)

        # Check if output is the all 0 codeword
        if output is None:
            failures.append(1)
        elif np.allclose(output, np.zeros(len(output))):
            failures.append(0)
        else:
            failures.append(1)

    results = {"results": {
        "failure_rt": np.mean(failures),
        "fail_std": np.std(failures),
        "avg_time": np.mean(decoder.times),
        "time_std": np.std(decoder.times)
    }}

    return results


def run_decode_frm_file(start_from='param_dict_list.json', send_to='results_list.txt', codes_path='selected_codes'):
    '''
    Runs a batch of decoding procedures studies for a set of parameters from a
    list of dictionnaries in a file.
    '''
    # read file
    with open(start_from) as myfile:
        params_list = json.load(myfile)

    # list of parameters studies
    iter = 1

    # progress bar
    for param_set in params_list:
        print('[Plotting point '+str(iter)+' of '+str(len(params_list))+']')
        # run decoding procedure
        results = new_run_func(codes_path=codes_path, **param_set)
        # Put params+ results into one dictionary
        data_point = {**param_set, **results}

        results_saving(data_point, filename=send_to)
        iter += 1


if __name__ == "__main__":
    # Calling this file with a repo address with contained parameters sets.
    if len(sys.argv) > 1:
        path = str(sys.argv[1])  # retrieving path given
    else:
        path = 'LDPC_params_results'

    run_decode_frm_file(start_from=path+'/param_dict_list.json',
                        send_to=path+'/results_list.txt', codes_path=path+'/selected_codes/')
