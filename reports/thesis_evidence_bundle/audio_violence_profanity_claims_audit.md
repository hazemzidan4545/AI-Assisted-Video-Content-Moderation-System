# Audio, Violence, Profanity Claims Audit

## Is there a trained audio profanity model in the project with saved checkpoint and metrics?

NO

- Checkpoint path if YES: NOT_FOUND
- Metrics file path if YES: NOT_FOUND
- Notebook/script evidence: `Dataset/Cursing/TAPAD/README.md` exists as a profanity audio dataset reference, but no project-trained profanity checkpoint and metrics file were found in `Implimentation/checkpoints/` or `Implimentation/eval_logs/`.
- Conclusion: This must be removed from final thesis claims.

## Is there a trained violence detection model in the project with saved checkpoint and metrics?

NO

- Checkpoint path if YES: NOT_FOUND
- Metrics file path if YES: NOT_FOUND
- Notebook/script evidence: `Dataset/XDViolence/` exists as dataset files, but no trained violence model checkpoint and metrics file were found in `Implimentation/checkpoints/` or `Implimentation/eval_logs/`.
- Conclusion: This must be removed from final thesis claims.

## Is there a final accepted multimodal audio-visual fusion model with saved checkpoint and metrics?

NO

- Checkpoint path if YES: NOT_FOUND for an accepted final model. Existing rejected exploratory artifacts: `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/checkpoints/audio_fusion_binary_roi_model.joblib`, `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/checkpoints/audio_late_fusion_model.joblib`.
- Metrics file path if YES: NOT_FOUND for accepted final. Existing rejected exploratory metrics: `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/eval_logs/audio_fusion_binary_roi_summary.json`, `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/eval_logs/audio_late_fusion_summary.json`.
- Notebook/script evidence: audio binary ROI summary decision is `REJECT_AUDIO` with reason `audio did not improve macro F1 by +0.02 and did not safely improve unsafe recall`. Late-fusion summary interpretation: `Audio late fusion did not meaningfully improve over visual-only. Audio may be mostly neutral/noisy for Dataset 2.`.
- Conclusion: This must be removed from final thesis claims, or moved to future work / exploratory negative result.

## Is audio muting implemented and validated in the final censorship output?

NO

- Checkpoint path if YES: NOT_FOUND
- Metrics file path if YES: NOT_FOUND
- Notebook/script evidence: `Implimentation/censorship_module/run_censorship.py` warns that `MUTE_UNSAFE_AUDIO=True` is accepted but segment-level audio muting is not implemented. Censorship summaries set `mute_unsafe_audio=false`.
- Conclusion: This must be removed from final thesis claims.
