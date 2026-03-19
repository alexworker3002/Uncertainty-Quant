# Phase 1: The Diagnostic - Core THE computation
from .dmt_accelerator import prepare_filtration
from .persistence_homology import PersistenceDiagram, extract_persistence
from .hallucination_energy import compute_the, the_from_probability
