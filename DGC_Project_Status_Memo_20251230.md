# Project Memo: Deep Global Clustering (DGC) - Current Status

**Date:** December 30, 2025  
**Project:** Deep Global Clustering for Hyperspectral Image Segmentation  
**Status:** ArXiv submission completed, GitHub publication pending

---

## 1. COMPLETED WORK

### 1.1 Technical Report (ArXiv)
- ✅ **Full paper written** (10 pages, 4 figures)
- ✅ **Title finalized:** "Deep Global Clustering for Hyperspectral Image Segmentation: Concepts, Applications, and Open Challenges"
- ✅ **Submitted to ArXiv** (cs.CV primary, cs.LG cross-list)
- ✅ **Final PDF** ready for upload to project shared space

**Key sections completed:**
1. Introduction (HSI properties, foundation model gap, problem motivation)
2. Design Philosophy (navigable granularity, sparse activation, overlapping grids)
3. Method (architecture, loss formulation, implementation details)
4. Application (leaf disease dataset, sync/async results)
5. Discussion (failure mode analysis: loss balancing, sampling, termination)
6. Conclusion (positioning as intellectual scaffolding)
7. Acknowledgments (Xiu-Rui Lin for data collection)
8. Author Contributions (CRediT format)

### 1.2 Conference Materials
- ✅ ACPA 2025 conference paper presented (October 2025)
- ✅ Conference presentation slides available
- ✅ Extended abstract published in conference proceedings

---

## 2. GITHUB REPOSITORY STATUS

**Repository URL:** https://github.com/b05611038/HSI_global_clustering

### 2.1 Current State
- ✅ Code implemented (sync and async variants)
- ✅ Default hyperparameters match reported results
- ✅ Repository exists and is functional
- ⚠️ **README.md needs substantial improvement**

### 2.2 README Requirements (NOT YET DONE)

The README should provide sufficient information for readers to:
1. Understand what DGC is and does
2. Understand the project structure
3. Run the code (installation, dependencies, basic usage)
4. Interpret results
5. Understand limitations and known issues

**Minimum sections needed:**
- Project overview and motivation
- Installation instructions
- Dataset setup
- Basic usage examples
- File structure explanation
- Known issues / stability warnings
- Citation information (ArXiv link once available)
- License information

---

## 3. PROJECT POSITIONING & KEY MESSAGES

### 3.1 What DGC Is
- **Conceptual framework** for memory-efficient HSI clustering
- Learns global structure from local patch observations
- Operates on 64×64 patches to handle 1000×1000 HSI cubes
- Trains in <30 minutes on consumer GPU (RTX 4080, 10GB VRAM)

### 3.2 What DGC Achieves
- **Sync version:** Background-tissue separation (mean IoU 0.925)
- **DGC-4:** Unsupervised disease detection (lesions form coherent clusters)
- **Efficiency:** Constant memory usage regardless of image size
- **Navigable granularity:** K=2 (coarse) vs K=4 (fine semantic distinctions)

### 3.3 What DGC Struggles With
- **Optimization instability** (core problem: multi-objective loss balancing)
- **Async version:** "Firework" behavior (patterns emerge briefly then collapse)
- **Hyperparameter sensitivity:** Works only in narrow parameter ranges
- **Not production-ready:** Requires human-in-the-loop for stability

### 3.4 Core Contribution
- **Design philosophy:** Overlapping grids for global consistency
- **Conceptual insights:** Sparse cluster activation, navigable granularity
- **Honest diagnosis:** Loss balancing is the bottleneck (not architecture)
- **Intellectual scaffolding:** Framework valuable even if implementation unstable

---

## 4. DEVELOPMENT TIMELINE

### Phase 1 (Week 1)
- Concept → code → working sync version
- Code development assisted by Codex/Claude
- Design decisions made by Yu-Tang Chang

### Phase 2 (Week 2)  
- Async variant implementation
- Discovery of "firework" instability
- Diagnosis of failure modes

### Phase 3 (Post-Conference)
- Conference concluded (ACPA 2025)
- Technical report writing (extended from conference paper)
- ArXiv submission preparation

### Phase 4 (Current - December 30, 2025)
- ✅ ArXiv submission completed
- ⏳ GitHub README documentation pending
- ⏳ Final PDF upload to shared space pending

---

## 5. NEXT STEPS (GITHUB PUBLICATION)

### 5.1 Essential Tasks
1. **Write comprehensive README.md**
   - Installation guide (dependencies, environment setup)
   - Dataset preparation instructions
   - Usage examples (training, inference)
   - Project structure explanation
   - Known issues and limitations

2. **Verify code accessibility**
   - Ensure all referenced files exist
   - Check that default hyperparameters are documented
   - Confirm data paths are clearly explained

3. **Add citation information**
   - ArXiv link (once available)
   - BibTeX entry
   - Reference to ACPA 2025 paper

4. **License file**
   - Add appropriate open-source license (MIT, Apache 2.0, etc.)

### 5.2 Optional Enhancements
- Requirements.txt or environment.yml
- Example outputs (sample pseudo-segmentations)
- Training logs or convergence curves
- Comparison table (sync vs async characteristics)
- Troubleshooting guide

---

## 6. DESIGN PHILOSOPHY SUMMARY (FOR README)

**Core Concepts to Communicate:**

1. **Overlapping Grids:**
   - Two patches (G₁, G₂) sampled from same HSI cube
   - Overlapping pixels → consistency constraint
   - Non-overlapping pixels → independent learning
   - Enforces global structure through local observations

2. **Navigable Granularity:**
   - Not fixed K (like traditional K-means)
   - Start with more clusters (e.g., K=16)
   - User merges clusters to desired semantic level
   - Reverse refinement: coarse-to-fine is user-controlled

3. **Sparse Activation:**
   - Not all K clusters active in every scene
   - Healthy leaves: only background + tissue clusters
   - Infected leaves: lesion cluster activates
   - Bottom-up concept learning (like human cognition)

4. **Efficiency through Patches:**
   - 64×64 patches from 1000×1000 cubes (250× reduction)
   - Load once, sample many patches (10-100× iterations per load)
   - Addresses I/O bottleneck (disk → RAM → VRAM)
   - Constant memory regardless of image size

---

## 7. KNOWN ISSUES TO DOCUMENT

### 7.1 Sync Version
- ⚠️ Hyperparameter sensitive (narrow working range)
- ⚠️ Results are "cherry-picked" from specific settings
- ⚠️ Centroid initialization affects convergence
- ✓ Stable enough for reported results when properly configured

### 7.2 Async Version
- ❌ **Severe instability** ("firework" behavior)
- ❌ No quantitative results reported
- ❌ Training phases: inactive → ignite → afterglow → smoldering → aftermath
- ❌ No reliable stopping criterion for "ignite" phase
- ⚠️ Included in repository for research/diagnostic purposes only

### 7.3 Root Cause Analysis
**Primary bottleneck:** Multi-objective loss balancing
- Four loss terms with contradictory objectives
- Fixed λ weights insufficient for dynamic equilibrium
- Uniform assignment breaks collapse but destroys boundaries
- Over-merging in feature space after initial good separation

**Not the bottleneck:**
- ✓ Architecture is reasonable
- ✓ CNN learns spectral features quickly
- ✓ Concept is sound (brief ignite phase proves it works)

---

## 8. FILE ORGANIZATION (TO VERIFY)

Expected structure in GitHub repository:

```
HSI_global_clustering/
├── README.md                 # ⚠️ NEEDS WRITING
├── LICENSE                   # ⚠️ ADD LICENSE FILE
├── requirements.txt          # Dependencies (optional but recommended)
├── hsi_clustering.py         # Main model class
├── clustering.py             # UnrolledMeanShift module
├── train_sync.py            # Synchronous training script (assumed)
├── train_async.py           # Asynchronous training script (assumed)
├── utils/                    # Helper functions (assumed)
├── configs/                  # Default hyperparameters (assumed)
├── data/                     # Dataset loading (assumed)
└── outputs/                  # Results directory (assumed)
```

**Action item:** Verify actual repository structure and document it in README.

---

## 9. ARXIV SUBMISSION DETAILS

**Submission date:** December 30, 2025  
**Primary category:** cs.CV (Computer Vision and Pattern Recognition)  
**Cross-list:** cs.LG (Machine Learning)  
**ACM classes:** I.4.6; I.5.3; I.2.6  
**MSC classes:** 68T05; 68T10; 62H30

**Abstract word count:** 196 words  
**Paper length:** 10 pages  
**Number of figures:** 4  
**Number of references:** 10

**Once ArXiv assigns ID:**
- Update GitHub README with ArXiv link
- Add BibTeX citation
- Update project shared space documentation

---

## 10. SHARED SPACE UPLOADS NEEDED

**Files to upload to project shared space:**

1. ✅ **Final PDF:** `HSI_DGC_extend_report.pdf`
2. ✅ **This memo:** `DGC_Project_Status_Memo_20251230.md`
3. ⏳ **Conference materials:** (if not already uploaded)
   - ACPA 2025 presentation slides
   - ACPA 2025 extended abstract
4. ⏳ **Updated README draft** (once written)

---

## 11. CONTACT & COLLABORATION

**Primary author:** Yu-Tang Chang (b05611038@ntu.edu.tw)  
**Co-authors:** Pin-Wei Chen, Shih-Fang Chen  
**Institution:** Department of Biomechatronics Engineering, National Taiwan University

**Data collection acknowledgment:**  
Xiu-Rui Lin (Agricultural Chemicals Research Institute, Ministry of Agriculture)

**Current project status:** Active documentation phase  
**Future work:** README completion for public GitHub release

---

## 12. SUMMARY

**What's done:**
- ✅ Research completed (2 weeks implementation + analysis)
- ✅ Conference presentation delivered
- ✅ Technical report written and submitted to ArXiv
- ✅ Code functional and in repository

**What's next:**
- ⏳ Write comprehensive GitHub README
- ⏳ Add LICENSE file
- ⏳ Verify code accessibility and documentation
- ⏳ Upload final PDF to shared space
- ⏳ Update README with ArXiv link (once assigned)

**Timeline estimate for GitHub publication:**
- README writing: 2-4 hours
- Code verification: 1-2 hours
- Final review: 1 hour
- **Total:** Can be completed in one focused work session

---

**End of Memo**

*Generated: December 30, 2025*  
*Next update: After GitHub README completion*
