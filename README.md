# Slow-moving pattern interfaces in general directions for a two-dimensional Swift--Hohenberg-type equation

This repository contains the supplementary code files related to the article "Slow-moving pattern interfaces in general directions for a two-dimensional Swift--Hohenberg-type equation" by Bastian Hilder and Jonas Jansen. The preprint can be found at [arXiv:tbd](https://arxiv.org/abs/tbd).

### Abstract

We rigorously prove the bifurcation of slow-moving pattern interfaces with general direction in a two-dimensional Swift–Hohenberg-type model close to a Turing instability for a large class of nonlinearities. These interfaces describe the invasion of stripe and hexagonal patterns into the spatially homogeneous state and model a possible mechanism for pattern formation, as observed in a wide range of real-world applications. For this, we develop a rigorous framework to establish the existence of such solutions using spatial dynamics and non-standard centre manifold theory. Our approach exploits geometric and algebraic structures generic to $\mathrm{O}(2)$-symmetric pattern-forming systems near a Turing instability, and addresses fundamental technical challenges due to a non-uniform spectral gap around the imaginary axis, quadratic resonances induced by the hexagonal structure, and the high-dimensional phase space of the reduced equations.

### Selected spreading speed

The calculations for the formal prediction of the spreading speed obtained in Section 1.2 using marginal stability analysis can be found in the Mathematica file "selected-speed.nb" in the folder "Mathematica". It can be viewed using the [Wolfram Player](https://www.wolfram.com/player).

### Numerical computation of heteroclinic orbtis

The numerical implementation to find heteroclinic orbits for the leading-order reduced system on the centre manifold can be found in the folder "Orbit search". The results in the paper can be reproduced by running "main.jl".
