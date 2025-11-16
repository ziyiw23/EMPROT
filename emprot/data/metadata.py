import pandas as pd
import os
from typing import Dict, List, Tuple


class MetadataManager:
    """
    Manages protein metadata from CSV files and provides utilities for protein filtering.
    """
    
    def __init__(self, metadata_path: str):
        """
        Initialize metadata manager.
        
        Args:
            metadata_path: Path to the metadata CSV file
        """
        self.metadata_path = metadata_path
        self.metadata_df = self._load_metadata()
    
    def _load_metadata(self) -> pd.DataFrame:
        """
        Load and clean metadata from CSV file.
        
        Returns:
            Cleaned metadata DataFrame
        """
        metadata_df = pd.read_csv(self.metadata_path, header=0)
        metadata_df.columns = metadata_df.columns.str.strip()
        if 'Unnamed: 0' in metadata_df.columns:
            metadata_df = metadata_df.drop(columns=['Unnamed: 0'])
            
        metadata_df['Dynamic id'] = metadata_df['Dynamic id'].astype(str).str.strip()
        metadata_df.set_index('Dynamic id', inplace=True)
        
        return metadata_df
    
    def get_protein_info(self, dynamic_id: str) -> pd.Series:
        """
        Get protein information by dynamic ID.
        
        Args:
            dynamic_id: Dynamic ID of the protein
            
        Returns:
            Protein information as pandas Series
        """
        return self.metadata_df.loc[dynamic_id]
    
    def get_all_dynamic_ids(self) -> List[str]:
        """
        Get all dynamic IDs from metadata.
        
        Returns:
            List of all dynamic IDs
        """
        return list(self.metadata_df.index)
    
    def filter_proteins_by_size(self, protein_sizes: Dict[str, int], 
                               min_residues: int = 50, 
                               max_residues: int = 1000) -> List[str]:
        """
        Filter proteins by size criteria.
        
        Args:
            protein_sizes: Dictionary mapping dynamic_id to number of residues
            min_residues: Minimum number of residues
            max_residues: Maximum number of residues
            
        Returns:
            List of dynamic IDs that pass the filter
        """
        valid_proteins = []
        
        for dynamic_id, num_residues in protein_sizes.items():
            if min_residues <= num_residues <= max_residues:
                valid_proteins.append(dynamic_id)
            else:
                print(f"Filtered protein {dynamic_id}: {num_residues} residues")
        
        return valid_proteins


class TrajectoryCatalog:
    """
    Manages mapping between PDB files and trajectory files.
    """
    
    def __init__(self):
        self.catalog = {}
    
    def build_catalog(self, traj_names: List[str]) -> Dict[str, str]:
        """
        Build catalog mapping PDB names to XTC trajectory names.
        
        Args:
            traj_names: List of trajectory directory names
            
        Returns:
            Dictionary mapping PDB names to XTC names
        """
        catalog = {}
        for traj_name in traj_names:
            parts = traj_name.split('_')
            pdb_name = "_".join(parts[:3]) + ".pdb"
            xtc_name = "_".join([parts[4], "trj", parts[2]]) + ".xtc"
            catalog[pdb_name] = xtc_name
        
        self.catalog = catalog
        return catalog
    
    def get_xtc_name(self, pdb_name: str) -> str:
        """
        Get XTC trajectory name for a given PDB name.
        
        Args:
            pdb_name: PDB file name
            
        Returns:
            Corresponding XTC file name
        """
        return self.catalog.get(pdb_name)
    
    def get_pdb_name(self, xtc_name: str) -> str:
        """
        Get PDB file name for a given XTC name.
        
        Args:
            xtc_name: XTC file name
            
        Returns:
            Corresponding PDB file name
        """
        for pdb_name, xtc in self.catalog.items():
            if xtc == xtc_name:
                return pdb_name
        return None 