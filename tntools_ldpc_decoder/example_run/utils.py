import json

import numpy as np

import utils

"""
File edit and params utils.
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


def save_to_file(parameters, filename, header=False, path='./', buffer=20):

    # Check if result file already exist, and initiate it if not
    try:
        with open(path+filename) as _:
            pass

    except FileNotFoundError:
        if header is False:
            print('The file doesn\'t exist. Creating one for data.')

        else:
            print('File doesn\'t exist, Creating one with given header.')
            header_line = ''
            # enumerate all header elements and convert to a atring line
            for i, param in enumerate(header):
                # If parameter is last of list, do not separate w. comma & buffer
                if i == len(parameters)-1:
                    param_add = str(param)
                else:
                    param_add = str(param)+','

                # Put space for regular column look
                skip = buffer-len(param_add)

                # If value string larger than buffer. Just add full buffer value.
                if skip <= 0:
                    param_add += ' '*buffer
                else:
                    param_add += ' '*skip

                # Add parameter string to the whole lign
                header_line += param_add

            # Print string line on file and close it
            save = open(path+filename, 'a')
            print(header_line, file=save)
            save.close()

    step_line = ''
    # see header printing fo context on top
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

    save = open(path+filename, 'a')
    print(step_line, file=save)
    save.close()


def import_data_file(path_to_file, header=True):
    '''
    Reads the data saved by the save_to_file function. Returns a numpy array
    without fixed variable types.
    '''
    names = True
    if header is False:
        names = None
    data = np.genfromtxt(path_to_file,
                         skip_header=0,
                         skip_footer=0,
                         names=names,
                         dtype=None,
                         delimiter=',',
                         deletechars=" !#$%&'()*+, -./:;<=>?@[\\]^{|}~",
                         autostrip=True,
                         replace_space='_')
    return data


def generate_all_batches_lists(large_file):
    '''
    Returns a list of lists of all the parameters values, one for each intended
    experiment.
    '''
    gigalist = []
    loop = 0
    for a_list in large_file:
        for parameter in large_file[a_list]:
            dump = []
            if loop == 0:
                gigalist = large_file[a_list][parameter]
            elif loop == 1:
                for x in gigalist:
                    for y in large_file[a_list][parameter]:
                        dump.append([x, y])
                gigalist = dump
            else:
                for x in gigalist:
                    for y in large_file[a_list][parameter]:
                        dump.append([x+[y]][0])
                gigalist = dump
            loop += 1

    return gigalist


def gen_dict_from_lists(large_file, params_lists):
    '''
    Generates a dictionnary for each sub list of parameters using the original
    ''large file''. Returns a list of all these dictionnaries. 
    '''
    full_dicts = []
    for par_list in params_lists:
        # Creates a copy of the original largefile.
        dump = copy.deepcopy(large_file)
        iter = 0
        # Replaces each param value list by a single value specific to one
        # experiment. Then appends it to the list of dictionnaries.
        for sub_dic in dump:
            for dict_param in dump[sub_dic]:
                dump[sub_dic][dict_param] = par_list[iter]
                iter += 1
        full_dicts.append(dump)

    return full_dicts


def create_params_dict_list(file_from='batch_params.json', file_to='param_dict_list.json'):
    '''
    Takes a dictionnary of parameter value lists and return all possible
    combinations set of parameters values in a list of dictionnaries. 
    '''
    with open(file_from) as myfile:
        params = json.load(myfile)

    # Generate the list of dictionnary
    params_lists = generate_all_batches_lists(params)
    full_dict = gen_dict_from_lists(params, params_lists)

    # Overwrite previous json file with same name and closes it
    out_file = open(file_to, "w")
    json.dump(full_dict, out_file, indent=4)
    out_file.close()