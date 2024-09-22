# PhonATe
This repository contains code accompanying the paper [Deas et al., 2024, "PhonATe: Impact of Type-Written Phonological Features of African American Language on Generative Language Modeling Tasks"](https://openreview.net/pdf?id=rXEwxmnGQs) to be presented at the 2024 Conference on Language Modeling.

# Setup
1. To apply augmentations to text using PhonATe, clone the repository and install the necessary dependencies in a virtual environment:
  ```
    virtualenv phonate_env
    source phonate_env/bin/activate
    pip install -r requirements.txt
  ```

2. The Phoneme-to-Grapheme model based on the model released by [Zhu et al., 2022](https://arxiv.org/abs/2204.03067) and finetuned on AAL tweets is included in [byt5-aal-p2g.tar.gz](byt5-aal-p2g.tar.gz). Untar the archive to use with PhonATe:
  ```
    tar -xzf byt5-aal-p2g.tar.gz
  ```

3. Components of PhonATe also rely on the spacy `en_core_web_md` model and the nltk english lexicon, which can be installed with:
  ```
    python -m spacy download en_core_web_md
    python -c "import nltk; nltk.download('words')"
  ```

# Applying Phonological Augmentations
Example code for creating a PhonATe instance and applying augmentations for use in evaluating model robustness is included in the [Example Notebook](phonate_example.ipynb). A PhonATe instance can be created using a configuration file. A default configuration is included in [`phonate/default_config.json`](phonate/default_config.json) setting all augmentation probabilities to .2.

Using the default configuration, PhonATe can be constructed with 
```
aal_phonate = AALPhonate(config = 'phonate/default_config.json')
```
and then used to augment texts with
```
  phon_trans, phon_aug, paug_out, clean_out = aal_phonate.full_phon_aug(<list_of_texts>)
```
`clean_out` contains the augmented text, while `phon_trans`, `phon_aug`, and `paug_out` contain the original phoneme transcription, augmented phoneme sequence, and raw decoded phonemes respectively. Augmentations can then be filtered with `phonate_filter.filter_transforms(<original_text>, <augmented_text>)`.
| Original Text                                                                                                                             | Clean PhonATe Result                                                                                                                   | Filtered Text                                                                                                                          |
|:------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| Hellloooo? I'm done with this....If I want information I'll just go the source or Encarta.                                                | Hellloooo? I'm done wiff this....If I want infermation I'll jus go deh source or Encarta.                                              | Hellloooo? I'm done wif this....If I want infermation I'll jus go deh source or Encarta.                                               |
| Or at least review the timing of Moreschi's obscene haste and agree that I had no way of seeing it before he acted.                       | Or at lease review deh taming of Moreskis obscene hast and gree dat I hat no wa of seein it before he acted.                           | Or at least review deh taming of Moreskis obscene hast and agree dat I had no way of seein it before he acted.                         |

# Cite
If you use PhonATe in your work, please cite the paper:
```
@inproceedings{deas-phonate,
  title={PhonATe: Impact of Type-Written Phonological Features of African American Language on Generative Language Modeling Tasks},
  author={Deas, Nicholas and Grieser, Jessica A and Hou, Xinmeng and Kleiner, Shana and Martin, Tajh and Nandanampati, Sreya and Patton, Desmond U and McKeown, Kathleen},
  booktitle={First Conference on Language Modeling},
  year={2024},
}
```

# Contact
For questions, please email [ndeas@cs.columbia.edu](mailto:ndeas@cs.columbia.edu).
