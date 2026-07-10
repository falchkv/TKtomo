"""TKtomo: pre-processing and inspection toolkit for ptycho-tomography.

The library layer (``io``, ``align``, ``recon``, ``messaging``) imports without
the GUI or heavy reconstruction stacks; those dependencies are imported lazily
inside the functions that need them.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
