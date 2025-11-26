import warnings

import numpy as np

import tntools as tnt


"""
Classical LDPC decoder using TNTools MPS-MPO formalism. 
"""

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
        output = classical_ldpc_decoding(
            entry, self.checks, self.decoder_par, self.svd_function_par, self.main_comp_finder_par)
