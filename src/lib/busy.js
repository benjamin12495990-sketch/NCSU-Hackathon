// Shared flag: the 3D view skips its redraws while the neural network trains,
// so training gets the processor even on a machine with slow graphics.
export const busy = { training: false }
