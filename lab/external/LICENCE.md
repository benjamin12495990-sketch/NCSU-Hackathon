# lab/external/boxcrete_data.csv

Copied unchanged from the BOxCrete repository, `data/boxcrete_data.csv`:
https://github.com/facebookresearch/SustainableConcrete (facebookresearch/SustainableConcrete).

- Licence: MIT (copyright Meta Platforms, Inc. and affiliates), as stated in that repository.
- Cite: Baten B., Iqbal M.A., Ament S., Kusuma J., Garg N. (2026). BOxCrete. arXiv:2603.21525.
- 670 rows, 149 mixes, ages 1, 3, 5, 14 and 28 days, strengths in psi. It does not overlap the UCI data.
- Used only as pre-training rows (`lab/physics/boxcrete.mjs`): concrete rows (coarse aggregate > 0)
  cured at 22 C, mapped to the nine UCI features, psi / 145.04 = MPa. Nothing beyond 28 days.
