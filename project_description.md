TKtomo description:

A Python software for assisting with ptycho tomo reconstruction. We will leverage existing libraries for both ptycho and tomographic reconstruction, but want to do pre-processing tasks like image alignment etc. within this software. 

priorities:
  - Good tools for color maps and colorscales is important.
  - Responsiveness of the UI is important.
  - Modularity so that different libraries and methods can be interchanged.
  - The UIs must be runnable as python programs without modifications.
  - It is not necessary that all UIs are visible at once. They can be separate programs, but transferring parameters and images between them will be necessary for some of them.

Assumptions:
- Assume ptycho projections already exist.

Implementation tasks:
- Create a UI to visually inspect sinograms. Scrolling between slices must be possible.
- Create a UI to visually inspect tomograms. Slicing and scrolling between slices must be possible. It must be possible to reproject the tomogram and show the resulting projection.
- Create a UI to visually inspect projections. It must be possible to scroll between projections.
- Create a UI to tweak image alignment between two images. It must be possible to enter manual values, tweak values with a slider. There should also be a button to apply automatic alignment methods, using current state as initial condition if applicable. Alignment methods used should be selectable from a dropdown menu. There should be a button and a hotkey to undo the automatic alignment. This UI must be able to transfer its results to the sinogram UI to update the sinogram.