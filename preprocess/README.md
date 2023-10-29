# Preprocessing

Follow these preprocessing steps:

1. Run `quickdraw_array_to_pil.py` to convert the QuickDraw dataset into a format compatible with the SVOL model.
2. Execute `extract_sketch_vit_features.sh`:
	- This script involves running two components:
		* `sketch_vit_feature_extractor.py`: Extract features from the sketches.
		* `sketch_vit_finetune.py`: Fine-tune the SketchVIT (Vision Transformer) model.
