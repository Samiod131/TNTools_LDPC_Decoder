import time
import json
import warnings
import sys

import numpy as np

import tntools_ldpc_decoder as tnt_ldpc
import code_gen

"""
Evaluation of LDPC decoder code example.
"""


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


def bitflip_array(p, n):
    '''
    Return 0-array of n bits with fixed bit flip probability p
    '''
    arr = np.random.choice([0, 1], n, p=[1-p, p])
    return arr


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
    ldpc_code = code_gen.safe_code_gen(
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

    decoder = TimedDecoder(tnt_ldpc.TN_LDPC_Decoder(
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
