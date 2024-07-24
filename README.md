PyTorch dataset for CUB-200-2011 (http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

- CUB20 subset: use first 20 species only as a toy dataset
- CubCap: With Scott Reed descriptions, as fixed by [Franz Rieger](https://github.com/riegerfr)) (as well as images)
- CubFeats: With CUB attribute values (as well as images)

Data Instructions
- Create a data/ directory
- within data/ create a softlink to the no_oov_descriptions.json file created by the reigerfr repo.
` no_oov_descriptions.json@	 --> /home/stella/datasets/reedscott_birds/cub_updated_descriptions/updated_dataset/no_oov_descriptions.json`


