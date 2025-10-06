# Image-Gs_Implementation
This is a custom implementation of https://arxiv.org/pdf/2407.01866.
Not complete.
This isn't a full implementation.
It's not optimized.
This is the minimal amount of math behind it, a minimal working example of the concepts of the paper.
Analogy with the FDF project of 42 and other heightmap programs:
On normal heightmap programs we pass the points into a function, isometric projection, and then we print those points to the screen.
Here I initialize randomly the 2D Gaussians and, with PyTorch, find the right value to represent the image as 2D Gaussians.
