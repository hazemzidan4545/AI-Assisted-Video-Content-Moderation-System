# Final Thesis Claims Audit

## A. Claims supported by evidence

- A ConvNeXt-Small 3-class image classifier was trained/evaluated for Normal/Suggestive/Unsafe visual classification. Evidence: `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/eval_logs/convnext_sexynude_20260426_012841/run_summary.json`, `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/eval_logs/backbone_hardneg_comparison.json`.
- Hard-negative normal augmentation was evaluated and reduced false unsafe predictions on 813 hard-negative normal images from `66.91266912669127`% to `3.8130381303813037`%. Evidence: `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/eval_logs/hard_negative_bias_summary.json`, `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/eval_logs/hard_negative_bias_summary_after.json`, `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/eval_logs/hard_negative_bias_before_after_delta.json`.
- The hard-negative image backbone improved validation macro-F1 from `0.87839166184111` to `0.8824166375141688` and reduced validation false-unsafe normal errors by `-91`. Evidence: `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/eval_logs/backbone_hardneg_comparison.json`.
- A binary ROI temporal model for Dataset 2 Medium+Extreme merge was accepted as the final visual temporal model with macro-F1 `0.7505793132504739`, safe recall `0.6326530612244898`, and unsafe recall `0.8585858585858586`. Evidence: `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/eval_logs/binary_temporal_roi_summary.json`.
- ROI-aware temporal inference improved over the non-ROI binary baseline by macro-F1 delta `0.0649793132504739` and unsafe recall delta `0.06058585858585852`. Evidence: `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/eval_logs/binary_temporal_roi_summary.json`.
- A standalone censorship prototype exists using temporal-only full-frame censorship during detected unsafe segments. Evidence: `Implimentation/censorship_module/`, `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/outputs/censored/Boobs_and_Heavenly_Areolas_Vol.50_20260602_032027_summary.json`.
- Audio preservation/remux was demonstrated in an existing censorship output with `audio_remux_success=True`. Evidence: `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/outputs/censored/Boobs_and_Heavenly_Areolas_Vol.50_20260602_032027_summary.json`.
- A Streamlit local demo app exists. Evidence: `Implimentation/app.py`.

## B. Claims that must be removed or moved to future work

- A trained profanity audio model exists with saved checkpoint and metrics. NOT supported; this must be removed from final thesis claims.
- A trained violence detection model exists with saved checkpoint and metrics. NOT supported; this must be removed from final thesis claims.
- A final accepted multimodal audio-visual fusion model improves over the visual model. NOT supported; audio fusion summary decision is `REJECT_AUDIO`. This must be removed or presented as rejected exploratory work.
- Audio muting of unsafe segments is implemented and validated. NOT supported; source code explicitly warns this is not implemented. This must be removed from final thesis claims.
- Precise nude-region segmentation/localization is implemented. NOT supported; final censorship mode is full-frame temporal censorship, not segmentation.
- Dataset 3-only ordinal clip training generalizes to Dataset 2. NOT supported by Dataset2 evaluation summary; keep as exploratory negative/diagnostic result only.
