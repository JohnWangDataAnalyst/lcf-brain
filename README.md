# lcf-brain
# lcf-brain — Linear Control Framework for Brain Networks
**Repo:** https://github.com/JohnWangDataAnalyst/lcf-brain

Two-sided linear control for SC→FC alignment and energetic biomarkers (Ec, Es).
This repo reproduces figures for the methodology paper and provides pipelines for HCP (Schaefer-200) and LSD.

## Highlights
- Step-1 drift: **MOU (time series)** or **FC-eigenspace Lyapunov (FC-only)**
- Step-2: cortical stabilizer **Kc** via ridge (residual modes)
- Step-3: subcortical shaper **Ks** via continuous-time ARE (LQR)
- Biomarkers: **Ec = ||Kc||²_F**, **Es = ||Ks||²_F**
- Fair scoring in **correlation space** (lower triangle)

## Data pointers (Google Drive folders)
- **HCP SC:** `1Cff7Ns1aHoeU6tAfLgVfkK4g3ifBzKjy`
- **HCP FC:** `1FPYJygqiM7kU6hY6KCbgI-BJmUtRATeS`
