
import MDAnalysis as mda
import kugupu as kgp

def main():

    u = mda.Universe('./datafiles/C6.data', './datafiles/C6.dcd')
    def add_names(u):
        # Guesses atom names based upon masses
        def approx_equal(x, y):
            return abs(x - y) < 0.1
        
        # mapping of atom mass to element
        massdict = {}
        for m in set(u.atoms.masses):
            for elem, elem_mass in mda.guesser.tables.masses.items():
                if approx_equal(m, elem_mass):
                    massdict[m] = elem
                    break
            else:
                raise ValueError
                
        u.add_TopologyAttr('names')
        for m, e in massdict.items():
            u.atoms[u.atoms.masses == m].names = e

    add_names(u)
    res = kgp.coupling_matrix(u, nn_cutoff=5.0, state='lumo', degeneracy=1, stop=3, model='xtb')
    kgp.save_results('xtb_res.hdf5', res)
if __name__ == "__main__":
    main()