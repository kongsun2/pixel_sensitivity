# Research validation final verification — 2026-11-11

- All7newtrainingjobs completed; mainstudy9models (3datasets ×seeds11/22/33). OriginalGC10/NEUseed11weights reused, correctedPCBseed11newlytrained.
- CorrectedPCB numericgroup split train3295/val728/test639, all8classes andzero numericID crosssplitoverlap. NumericIDs are proxies, not confirmedphysicalunitIDs.
- Mainmatrix27completedmodel/conditionjobs: seed11sevenmatchedconditions per dataset; seed22/33bright25only. All8×8grid,10masks,FP32batch16, frozen threshold/evaluator rules.
- Valid main imageevaluations7286247;17307predictioncompletionmarkers;7274880paired-imagerows. Eagerpartial/prototype/controlcomputation excluded from these studycounts.
- Fullreport regenerated with --verify-prediction-sha, exit0; all17307prediction files rehashed (report_meta.prediction_sha_rechecked_by_report=true). Model/source/analysisoutput hashes independentlychecked againstrecordedmetadata.
- Report output contracts passed:27conditionrows,270patternrows,9cleanbaselines,3seedsummaryrows,9spatialcorrelationrows,21matchedcontrasts,2PCBcohortrows.
- OriginalGC10/NEUseed11bright25 full640case metrics reproduced with maxabsdifference0.0.
- GuardedcompiledNMS preserves originalsort/tieorder, usesoriginalconfidence andfallsbacknearIoU/large/invalidcases. Originalbatch16kept; largerbatchnumericaldrift rejected. Globalfixedcontrols3models×130validationimages×8conditions exact. Every maincheckpoint also17valimages×8condition exactcontrols. Finitevalidation isnot mathematicalproof forallinputs.
- Existing and correctedPCB class/size/hit descriptive diagnostics plusjoint group×mask simultaneouspositionbands completed. Dynamiccohort/group caveats verified. No exactpvalue/intrinsicsensorcausality claim.
- Final automatedtests:79passed, oneexternalanyio pytest deprecationwarning; nofailures/skips.
- Condition andseedcomparisonfigures visually inspected. All27states complete and final COMPLETE.json matchescurrentmanifestSHA.

Interpretation distinguishes single-seed RQ3comparisons from3seed bright25replication, changedPCBtestcohorts, andnative-resolution/dose differences. PhysicalRAW/ISP/DPC and factoryshipment metrics remainoutside thedeclared experiment.
