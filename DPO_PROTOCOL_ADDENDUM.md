# Additional retention measurement

Added after DPO training, before inspection of either generated comparison arm.
The original protocol is preserved unchanged. Evaluate both selected adapters on
the exact same 17 tokenized SFT validation conversations using reply-only,
token-weighted NLL, FP16 and the native EOS. This is a secondary forgetting check,
not a checkpoint-selection criterion. No private dialogue is decoded or read.
