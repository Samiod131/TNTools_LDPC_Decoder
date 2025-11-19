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