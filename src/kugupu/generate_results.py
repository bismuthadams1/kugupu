#    kugupu - molecular networks for change transport
#    Copyright (C) 2019  Micaela Matta and Richard J Gowers
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""runs the whole thing from head to toe"""

import numpy as np
from tqdm import tqdm
import MDAnalysis as mda
from typing import Literal
from ocelotml import load_models, predict_from_molecule, predict_from_list
from pymatgen.core.structure import Molecule
from typing import List, Dict, Any, Optional, Tuple

from .models_abc import MODELS_AVAILABLE, CouplingModel

from . import logger
from . import KugupuResults
from .dimers import find_dimers
from ._yaehmop import run_dimer, run_fragment
from ._hamiltonian_reduce import find_psi

# ocelotml_model = load_models('hh')
print("models available")
print(MODELS_AVAILABLE)

# Elements known to yaehmop (default eht_parms at least...)
REF_ELEMS = set('AC AG AL AM AR AS AT AU B BA BE BI BK BR C CA CD CE CF CL CM '
                'CO CR CS CU DY ER ES EU F FE FM FR GA GD GE H HE HF HG HO I '
                'IN IR K KR LA LI LR LU MD MG MN MO N NA NB ND NE NI NO NP O '
                'OS P PA PB PD PM PO PR PT PU RA RB RE RH RN RU S SB SC SE SI '
                'SM SN SR TA TB TC TE TH TI TL TM U UNQ V W XE Y YB ZN ZR'
                ''.split())

def _check_universe(universe):
    """Check universe is suitable for calculations

    Checks
    - existence of bonds
    - existence of names OR types
    - that all names are an element recognised by yaehmop

    Raises
    ------
    ValueError
      if anything wrong
    """
    if not hasattr(universe.atoms, 'bonds'):
        raise ValueError("Universe has no bond information. "
                         "Maybe try guess_bonds?")
    if not (hasattr(universe.atoms, 'names')):
        raise ValueError("Universe has no names, "
                         "these need to be set to element names")
    elems = set(universe.atoms.names)
    elems = {v.upper() for v in elems}
    new = elems - REF_ELEMS
    if new:  # if any user supplied elements not in yaehmop elements
        raise ValueError("Unknown elements found: {} "
                         "yaehmop knows of: {}".format(new, REF_ELEMS))


def _single_frame(fragments, nn_cutoff, degeneracy, state):
    """Results for a single frame

    Parameters
    ----------
    fragments : list of AtomGroup
      all fragments in system
    nn_cutoff : float
      distance for dimer pairs
    degeneracy : numpy array
      degenerate states per fragment
    state : str
      'homo' or 'lumo'

    Returns
    -------
    H_frag : numpy array
      coupling matrix
    """
    # make sure that all fragments are whole
    # ie a fragment isn't split between periodic images
    for frag in fragments:
        mda.lib.mdamath.make_whole(frag)
    dimers = find_dimers(fragments, nn_cutoff)

    size = degeneracy.sum()
    H_frag = np.zeros((size, size))
    # start and stop indices for each fragment
    stops = np.cumsum(degeneracy)
    starts = np.r_[0, stops[:-1]]
    diag = np.arange(size)  # diagonal indices
    wave = dict()  # wavefunctions for each fragment

    for (i, j), ags in tqdm(sorted(dimers.items())):
        # indices for indexing H_frag for each fragment
        ix, iy = starts[i], stops[i]
        jx, jy = starts[j], stops[j]

        logger.debug('Calculating dimer {}-{}'.format(i, j))
        # call Yaehmop
        Hij, frag_i, frag_j = run_dimer(ags)

        # lazily calculate the wave function for i and j
        try:
            # If we already did fragment i, just retrieve psi
            psi_i = wave[i]
        except KeyError:
            # If we didn't have fragment i already done,
            # calculate the state energy and wavefunction
            e_i, psi_i = find_psi(frag_i[0], frag_i[1], frag_i[2],
                                  state, degeneracy[i])
            # fill diagonal with energy of states
            # this only has to be one as self contribution does not change
            H_frag[diag[ix:iy], diag[ix:iy]] = e_i
            # store the wavefunction for future use
            wave[i] = psi_i

        try:
            psi_j = wave[j]
        except KeyError:
            e_j, psi_j = find_psi(frag_j[0], frag_j[1], frag_j[2],
                                  state, degeneracy[j])
            H_frag[diag[jx:jy], diag[jx:jy]] = e_j
            wave[j] = psi_j


        # H = <psi_i|Hij|psi_j>
        H_frag[ix:iy, jx:jy] = abs(psi_i.T.dot(Hij).dot(psi_j))
        H_frag[jx:jy, ix:iy] = H_frag[ix:iy, jx:jy]

    # do single fragment calculations for all missing
    for i in (set(range(len(degeneracy))) - set(wave.keys())):
        ix, iy = starts[i], stops[i]
        logger.debug('Calculating lone fragment {}'.format(i))
        H, S, ele = run_fragment(fragments[i])

        e_i, psi_i = find_psi(H, S, ele, state, degeneracy[i])

        H_frag[diag[ix:iy], diag[ix:iy]] = e_i
        # don't need to save the psi for this fragment
        #wave[i] = psi_i

    return H_frag


def _dask_single(top, trj, frame, nn_cutoff, degeneracy, state):
    """Dask helper function for calculating a single frame

    Parameters
    ----------
    top : pickle
      pickled MDAnalysis Topology, usually broadcasted to workers
    trj : str
      filename to the trajectory file
    frame : int
      index of the frame to analyse
    nn_cutoff, degeneracy, state
      same as for _single_frame

    Reheats the MDAnalysis Universe, loads correct frame then calls _single_frame
    """
    # load the Universe
    u = mda.Universe(top)
    u.load_new(trj)
    # select correct frame
    u.trajectory[frame]

    res = _single_frame(u.atoms.fragments, nn_cutoff, degeneracy, state)

    return res


def _dask_coupling(client,
                   u, nn_cutoff, degeneracy, state,
                   start=None, stop=None, step=None):
    import dask

    frames = np.arange(len(u.trajectory))
    # distribute this to all workers at start
    future_top = client.scatter(u._topology, broadcast=True)

    futures = []
    for i in frames[start:stop:step]:
        futures.append(dask.delayed(_dask_single)(future_top, u.trajectory.filename,
                                                  i, nn_cutoff, degeneracy, state))

    return client.compute(dask.delayed(np.stack)(futures)).result()


def coupling_matrix(u,
                    nn_cutoff, state, degeneracy=None,
                    start=None, stop=None, step=None, client: Optional[bool] = None, 
                    model: Literal['yaehmop', 'chadML'] = 'yaehmop' ):
    """Generate Hamiltonian matrix H_frag for each frame in trajectory

    Parameters
    ----------
    u : mda.Universe
      Universe to analyse.  All fragments from this Universe will be used
    nn_cutoff : float
      maximum distance (in A) of closest approach between dimers to
      consider neighbours (leading to a tight binding calculation).  A
      value of 5.0 A is usually adequate.
    degeneracy : int or array or dict or None
      number of degenerate states on each fragment to consider.
      int value - single value given to all fragments
      array - value for each fragment
      dict - mapping of resname to degeneracy of fragment
      None - automatically determine degeneracy
    state : str
      'HOMO' or 'LUMO'
    start, stop, step : int, optional
      slice through Universe trajectory
    client : dask distributed Client, optional
      if given, coupling matrix will be calculated in parallel as a
      dask distributed job using this client

    Returns
    -------
    results : KugupuResults namedtuple
      namedtuple of results, with attributes:
      - frames: index of each frame analysed
      - degeneracy: degeneracy for each fragment in Universe
      - H_frag: coupling between different fragments
    """
    _check_universe(u)

    if not model == 'yaehmop':
        if client:
          from dask.distributed import Client
          model_instance = MODELS_AVAILABLE["ocelotml"](local=False, server_id=Client())
        else:
          model_instance = MODELS_AVAILABLE["ocelotml"](local=True)

    Hs, frames = [], []

    nframes = len(u.trajectory[start:stop:step])
    # top_pickle = u._topology
    # traj_filename = u.trajectory.filename

    logger.info("Processing {} frames".format(nframes))

    if degeneracy is not None:
        # we need to pass a vector n_frags long
        if isinstance(degeneracy, int):
            # if only one value is given the elements are all the same
            degeneracy = np.full(len(u.atoms.fragments), degeneracy)
        elif isinstance(degeneracy, dict):
            # if our system is multi-component,
            # different residues have different degeneracy
            deg_arr = np.zeros(len(u.atoms.fragments))
            for i, frag in enumerate(u.atoms.fragments):
                # for a molecule with more than 1 residue,
                #  only the 1st one is checked
                deg_arr[i] = degeneracy[frag.residues[0].resname]
            degeneracy = deg_arr

    if client is None:
      for i, ts in enumerate(u.trajectory[start:stop:step]):
          logger.info("Processing frame {} of {}"
                      "".format(i + 1, nframes))
          if model == 'yaehmop':
            H_frag = _single_frame(u.atoms.fragments, nn_cutoff, degeneracy, state)
          elif model == 'ocelotml':
            if model_instance.local:
              fragments = u.atoms.fragments
              H_frag = model_instance(
                  fragments,
                  nn_cutoff=nn_cutoff,
                  degeneracy=degeneracy,
                  state=state,
              )
          frames.append(ts.frame)
          Hs.append(H_frag)
    else:
          H_frag = model_instance(
              u,
              nn_cutoff=nn_cutoff,
              degeneracy=degeneracy,
              state = "HOMO",
              start = start,
              stop = stop,
              step = step,
          )
          # H_frag = _ocelotl_frame(u.atoms.fragments, nn_cutoff, degeneracy)

    # H_frag = np.stack(Hs)
    # frames = np.array(frames)
    H_all = np.stack(Hs)
    frames_arr = np.array(frames)

    return KugupuResults(frames=frames_arr, H_frag=H_all, degeneracy=degeneracy)


    # else:
    #     H_frag = _dask_coupling(client, u,
    #                             nn_cutoff, degeneracy, state,
    #                             start, stop, step)
    #     frames = np.arange(len(u.trajectory))[start:stop:step]

    # logger.info('Done!')
    # return KugupuResults(
    #     frames=frames,
    #     H_frag=H_frag,
    #     degeneracy=degeneracy,
    # )

def  _ocelotl_frame(fragments, nn_cutoff, degeneracy):
    """Calculate the coupling from the fragments

    Parameters
    ----------
    fragments : list of AtomGroup
          all fragments in system    
    """
    for frag in fragments:
        mda.lib.mdamath.make_whole(frag)
    dimers = find_dimers(fragments, nn_cutoff)
    dimers_pymat = [_atomgroup_to_pymatgen_molecule(dimer) for dimer in dimers.items()]
    dimers_dict = dict(zip(dimers.keys(),dimers_pymat))

    size = degeneracy.sum()
    H_frag = np.zeros((size, size))
    # start and stop indices for each fragment
    stops = np.cumsum(degeneracy)
    starts = np.r_[0, stops[:-1]]
    diag = np.arange(size)  # diagonal indices
    wave = dict()  # wavefunctions for each fragment
    print(dimers_pymat[0].to_file('dimer.xyz'))
    predictions = predict_from_list(dimers_pymat, ocelotml_model)
    print(predictions)

    for (i, j), ags in tqdm(sorted(dimers_dict.items())):
        # indices for indexing H_frag for each fragment
        ix, iy = starts[i], stops[i]
        jx, jy = starts[j], stops[j]

        H_frag[diag[ix:iy], diag[ix:iy]] = predict_from_molecule(
            molecule = ags,
            model = ocelotml_model
        )
        wave[i] = 0
        wave[j] = 0

    
        # do single fragment calculations for all missing
    for i in (set(range(len(degeneracy))) - set(wave.keys())):
        ix, iy = starts[i], stops[i]
        e_i = predict_from_molecule(
            molecule  = _atomgroup_to_pymatgen_molecule(fragments[i]),
            model=ocelotml_model
        )
        H_frag[diag[ix:iy], diag[ix:iy]] = e_i
        # don't need to save the psi for this fragment
        # wave[i] = psi_i

    return H_frag

def _atomgroup_to_pymatgen_molecule(atomgroup) -> Molecule:
    """Takes a tuple of AtomGroup objects and turns them into a single pymatgen object

    Parameters
    ----------
    atomgroup: AtomGroup

    Returns
    -------
    mol: Molecule

    
    """
    if isinstance(atomgroup, tuple): 
      # print(atomgroup)
      atomgroup =  sum(atomgroup[-1])

    elements = atomgroup.names
    coords = atomgroup.positions  # NumPy array of shape (n_atoms, 3)
    
    mol = Molecule(elements, coords)
    return mol