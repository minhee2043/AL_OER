"""
Feature extraction from surface alloy structures.

This is a placeholder - copy your motif_to_feature.py content here.
Then update the imports to:
    from active_learning.utils.helpers import count_atoms
"""

from ase import Atoms
import numpy as np
from active_learning.utils.helpers import count_atoms


class Slab(object):
    '''Slab class for extracting structural fingerprints from surface alloy structures.
    
    The slab structure uses tags to identify different layers:
    - tag 0: Adsorbate atoms (O, OH, etc.)
    - tag 1: Surface layer (1st metal layer)
    - tag 2: Subsurface layer (2nd metal layer)
    '''

    def __init__(self, atoms=None):
        self.atoms = atoms      # ASE Atoms object of the slab structure
        self.slab3x3 = None     # Slab repeated 3x3x1 for periodic boundary handling
        self.site = None        # Adsorption site type: 'hcp' or 'fcc'

    def get_site(self):
        '''Classify the hollow adsorption site as either 'hcp' or 'fcc'.
        
        Returns:
        - For hcp site: (metal_symbol, atom_id) of the sublayer atom directly below
        - For fcc site: (None, None) as there's no atom directly below
        
        Updates self.site to 'hcp' or 'fcc'
        '''
        # Get the 3 closest surface atoms forming the adsorption triangle
        ensSym, ensIds = self.closest(layer=1, start=1, stop=3)
        slab3x3 = self.slab3x3

        # Get x,y positions of the 3 ensemble atoms
        ensPos = slab3x3.get_positions()[ensIds][:, 0:2]
        triangle = Triangle(ensPos[0], ensPos[1], ensPos[2])

        # Get all subsurface (tag=2) atom IDs and their x,y positions
        subIds = [atom.index for atom in slab3x3 if atom.tag == 2]
        subPos = slab3x3.get_positions()[subIds][:, 0:2]

        # Check if any subsurface atom lies within the adsorption triangle
        count, hcp = 0, False
        for i, pos in enumerate(subPos):
            if triangle.within(pos):
                count += 1
                hcp = True
                subId = subIds[i]
                symbols = slab3x3.get_chemical_symbols()
                subMetal = symbols[subId]
        
        if count > 1:
            print('Warning: %i metals detected in hcp hole' % count)
        
        if hcp:
            self.site = 'hcp'
            return subMetal, subId
        else:
            self.site = 'fcc'
            return None, None

    def adsorbate_id(self):
        '''Return the atom ID of the central adsorbate in the 3x3 extended slab.
        
        The slab is extended 3x3 to handle periodic boundaries properly.
        Returns the ID of the adsorbate at the center cell.
        '''
        # Extend slab 3x3 to handle periodic boundaries
        n = 3
        slab3x3 = self.atoms.repeat((n, n, 1))
        self.slab3x3 = slab3x3

        # Get IDs of all adsorbate atoms (tag=0) in the 3x3 slab
        adsIds = np.array([atom.index for atom in slab3x3 if atom.tag == 0])

        # Number of atoms per adsorbate (e.g., 1 for O, 2 for OH)
        nAdsAtoms = int(len(adsIds) / (n * n))

        # Extract IDs of the adsorbing atoms (first atom of each adsorbate)
        keepIds = range(0, len(adsIds), nAdsAtoms)
        adsIds = adsIds[keepIds]

        # Reshape to 3x3 grid and return central adsorbate ID
        adsIds = adsIds.reshape((n, n))
        nHalf = int(n / 2)
        return adsIds[nHalf, nHalf]

    def closest(self, layer, start, stop):
        '''Return symbols and IDs of atoms closest to the adsorbate in the specified layer.
        
        Parameters:
        layer : int     Layer tag (1=surface, 2=subsurface, etc.)
        start : int     Start index (1-indexed, inclusive)
        stop  : int     Stop index (1-indexed, inclusive)
        
        Returns:
        zoneSym : list  Chemical symbols of atoms in distance range [start, stop]
        ids     : list  Atom IDs sorted by their atom index
        
        '''
        adsId = self.adsorbate_id()
        slab3x3 = self.slab3x3
        
        # Get all atoms in the specified layer
        layerIds = [atom.index for atom in slab3x3 if atom.tag == layer]
        
        # Calculate distances from adsorbate to all layer atoms
        layerDist = slab3x3.get_distances(adsId, layerIds)
        
        # Sort atoms by distance and select range [start, stop]
        indexedDist = [[layerIds[i], layerDist[i]] for i in range(len(layerDist))]
        sortedDist = sorted(indexedDist, key=lambda x: x[1])[start-1:stop]
        
        # Extract IDs from sorted distance list
        ids = [i for i, dist in sortedDist]

        # Get chemical symbols and sort by atom ID 
        symbols = np.array(slab3x3.get_chemical_symbols())
        zoneSym = list(symbols[ids])
        indexedSym = [[zoneSym[i], ids[i]] for i in range(len(ids))]
        sortedSym = sorted(indexedSym, key=lambda x: x[1])
        
        zoneSym = [metal for metal, atomId in sortedSym]
        ids = [atomId for metal, atomId in sortedSym]
        
        return zoneSym, ids

    def ensemble(self, onTop=False):
        '''Return ensemble atom symbols (atoms forming the adsorption site).
        
        onTop : bool    If True, return 1 atom (on-top site)
                        If False, return 3 atoms (hollow site)
        '''
        if onTop:
            return self.closest(layer=1, start=1, stop=1)[0]
        else:
            return self.closest(layer=1, start=1, stop=3)[0]

    def surface(self, onTop=False):
        '''Return surface neighbor symbols around the ensemble (with double counting).
        
        onTop : bool    If True, return 6 atoms
                        If False, return 12 atoms (3 atoms counted twice + 6 atoms once)
        
        Note: This includes weighted counting for statistical averaging.
        '''
        if onTop:
            return self.closest(layer=1, start=2, stop=7)[0]
        else:
            return (self.closest(layer=1, start=4, stop=6)[0] * 2 +
                    self.closest(layer=1, start=7, stop=12)[0])

    def subsurface(self, onTop=False):
        '''Return subsurface neighbor symbols (with double counting for some atoms).
        
        onTop : bool    If True, return 3 atoms
                        If False, return 6-7 atoms depending on site type
                        - fcc site: 6 atoms (3 atoms counted twice)
                        - hcp site: 7 atoms (1 atom counted three times + 6 atoms once)
        
        Note: This includes weighted counting for statistical averaging.
        '''
        if onTop:
            return self.closest(layer=2, start=1, stop=3)[0]

        if self.site is None:
            self.get_site()

        if self.site == 'fcc':
            return (self.closest(layer=2, start=1, stop=3)[0] * 2 +
                    self.closest(layer=2, start=4, stop=6)[0])
        elif self.site == 'hcp':
            return (self.closest(layer=2, start=1, stop=1)[0] * 3 +
                    self.closest(layer=2, start=2, stop=7)[0])
        else:
            raise ValueError('The site was not classified')

    def surface_near(self, onTop=False):
        '''Return nearest surface neighbor symbols around the ensemble.
        
        onTop : bool    If True, return 6 nearest neighbors (on-top)
                        If False, return 3 nearest neighbors (hollow)
        '''
        if onTop:
            return self.closest(layer=1, start=2, stop=7)[0]
        else:
            return self.closest(layer=1, start=4, stop=6)[0]

    def surface_far(self):
        '''Return the 6 furthest surface neighbor symbols around the ensemble.
        
        Note: Only valid for hollow sites, not on-top.
        '''
        return self.closest(layer=1, start=7, stop=12)[0]

    def subsurface_near(self, onTop=False):
        '''Return nearest subsurface neighbor symbols.
        
        onTop : bool    If True, return 3 nearest subsurface atoms
                        If False, return 1 atom (hcp) or 3 atoms (fcc)
        '''
        if onTop:
            return self.closest(layer=2, start=1, stop=3)[0]

        # Determine site type if not already done
        if self.site is None:
            self.get_site()

        if self.site == 'fcc':
            return self.closest(layer=2, start=1, stop=3)[0]
        elif self.site == 'hcp':
            return self.closest(layer=2, start=1, stop=1)[0]
        else:
            raise ValueError('The site was not classified as hcp or fcc')

    def subsurface_far(self):
        '''Return furthest subsurface neighbor symbols.
        
        Returns 6 atoms (hcp) or 3 atoms (fcc).
        Note: Only valid for hollow sites, not on-top.
        '''
        if self.site is None:
            self.get_site()

        if self.site == 'hcp':
            return self.closest(layer=2, start=2, stop=7)[0]
        elif self.site == 'fcc':
            return self.closest(layer=2, start=4, stop=6)[0]
        else:
            raise ValueError('The hollow site is neither fcc nor hcp')

    def onTop(self, onTopDist=0.70):
        '''Detect if the adsorbate is on an on-top site or hollow site.
        
        Returns True if the x,y distance between adsorbate and closest surface 
        metal is below the threshold (indicating on-top adsorption).
        
        Parameters:
        onTopDist : float   Maximum x,y distance (in Angstroms) to classify as on-top
                            Default: 0.70 Å
        
        Returns:
        bool    True if on-top site, False if hollow site
        
        Usage:
            slab = Slab(atoms)
            is_ontop = slab.onTop()
            feature = slab.features(['Ni','Fe','Co'], onTop=is_ontop, zones=[...])
        '''
        adsId = self.adsorbate_id()
        metalId = self.closest(1, 1, 1)[1]  # Closest surface metal id
        [dist] = self.slab3x3.get_distance(adsId, metalId, vector=True)
        xyDist = np.sqrt(dist[0]**2 + dist[1]**2)

        if xyDist < onTopDist:
            return True
        else:
            return False

    def features(self, metals, onTop=False, zones=['ens', 's', 'ss']):
        '''Extract fingerprint features by counting metals in each specified zone.
        
        Parameters:
        metals : list   Reference metals for counting, e.g., ['Ni', 'Fe', 'Co']
        onTop  : bool   True for on-top adsorption, False for hollow site
        zones  : list   Zone identifiers to include in fingerprint:
                        
                        BASIC ZONES (with weighted counting):
                        'ens' : Ensemble - atoms forming adsorption site
                                Hollow: 3 atoms  |  On-top: 1 atom
                        's'   : Surface neighbors (full, with double counting)
                                Hollow: 12 atoms |  On-top: 6 atoms
                        'ss'  : Subsurface neighbors (full, with double counting)
                                Hollow: 6-7 atoms (fcc/hcp) | On-top: 3 atoms
                        
                        SPLIT ZONES (without weighted counting):
                        'sn'  : Surface near - nearest surface neighbors
                                Hollow: 3 atoms  |  On-top: 6 atoms
                        'sf'  : Surface far - furthest surface neighbors
                                Hollow: 6 atoms  |  On-top: N/A
                        'ssn' : Subsurface near - nearest subsurface neighbors
                                Hollow: 1-3 atoms (hcp/fcc) | On-top: 3 atoms
                        'ssf' : Subsurface far - furthest subsurface neighbors
                                Hollow: 3-6 atoms (fcc/hcp) | On-top: N/A
        
        Returns:
        feature : list  Counts of each metal in each zone, concatenated
                        For N metals and M zones: N×M elements total
        
        USAGE EXAMPLES:
        
        1. Hollow site with combined zones (your case):
           zones=['ens', 'sf', 'ssf', 'sn', 'ssn']
           Result for ['Ni','Fe','Co']: 15 elements
           [Ni1,Fe1,Co1, Ni2,Fe2,Co2, Ni3,Fe3,Co3, Ni4,Fe4,Co4, Ni5,Fe5,Co5]
           - Zone 1 (ens): 3 atoms
           - Zone 2 (sf): 6 atoms  
           - Zone 3 (ssf): 3-6 atoms
           - Zone 4 (sn): 3 atoms
           - Zone 5 (ssn): 1-3 atoms
        
        2. On-top site with basic zones:
           onTop=True, zones=['ens', 's', 'ss']
           Result for ['Ni','Fe','Co']: 9 elements
           [Ni_ens,Fe_ens,Co_ens, Ni_s,Fe_s,Co_s, Ni_ss,Fe_ss,Co_ss]
           - Zone 1 (ens): 1 atom
           - Zone 2 (s): 6 atoms
           - Zone 3 (ss): 3 atoms

        '''
        feature = []

        for zone in zones:
            if zone == 'ens':
                # Ensemble zone: 3 atoms (hollow) or 1 atom (on-top)
                siteSym = self.ensemble(onTop)
                siteCount = count_atoms(siteSym, metals)
                feature += siteCount
                
            elif zone == 's':
                # Full surface zone with weighted counting
                # Hollow: 12 atoms (3 counted twice + 6 once)
                # On-top: 6 atoms
                sSym = self.surface(onTop)
                sCount = count_atoms(sSym, metals)
                feature += sCount
                
            elif zone == 'ss':
                # Full subsurface zone with weighted counting
                # Hollow hcp: 7 atoms (1 counted three times + 6 once)
                # Hollow fcc: 6 atoms (3 counted twice)
                # On-top: 3 atoms
                ssSym = self.subsurface(onTop)
                ssCount = count_atoms(ssSym, metals)
                feature += ssCount
                
            elif zone == 'sn':
                # Surface near: 3 atoms (hollow) or 6 atoms (on-top)
                snSym = self.surface_near(onTop)
                snCount = count_atoms(snSym, metals)
                feature += snCount
                
            elif zone == 'ssn':
                # Subsurface near: 1 atom (hcp) or 3 atoms (fcc/on-top)
                ssnSym = self.subsurface_near(onTop)
                ssnCount = count_atoms(ssnSym, metals)
                feature += ssnCount
                
            elif zone == 'sf':
                # Surface far: 6 atoms (hollow only)
                sfSym = self.surface_far()
                sfCount = count_atoms(sfSym, metals)
                feature += sfCount
                
            elif zone == 'ssf':
                # Subsurface far: 6 atoms (hcp) or 3 atoms (fcc), hollow only
                ssfSym = self.subsurface_far()
                ssfCount = count_atoms(ssfSym, metals)
                feature += ssfCount
                
        return feature


# --- Helper Triangle class for classifying hollow sites --- #

class Triangle(object):
    '''Triangle class for determining if a point lies within a triangle.
    
    Used to classify hollow sites as hcp (atom underneath) or fcc (no atom underneath).
    '''

    def __init__(self, p1, p2, p3):
        '''Initialize triangle with three 2D vertices.'''
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)

    def within(self, p):
        '''Check if point p is within the triangle using barycentric coordinates.
        
        Parameters:
        p : array-like  2D point [x, y]
        
        Returns:
        bool    True if point is inside triangle, False otherwise
        
        Implementation based on: http://mathworld.wolfram.com/TriangleInterior.html
        '''
        v = np.array(p)
        v0 = self.p1
        v1 = self.p2 - v0
        v2 = self.p3 - v0
        
        # Calculate barycentric coordinates using cross products
        detvv2 = np.cross(v, v2)
        detv0v2 = np.cross(v0, v2)
        detv1v2 = np.cross(v1, v2)
        detvv1 = np.cross(v, v1)
        detv0v1 = np.cross(v0, v1)
        
        a = (detvv2 - detv0v2) / detv1v2
        b = -(detvv1 - detv0v1) / detv1v2
        
        # Point is inside if both barycentric coordinates are positive and sum < 1
        if a > 0 and b > 0 and a + b < 1:
            return True
        else:
            return False

