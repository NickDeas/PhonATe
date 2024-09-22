import torch
import numpy as np

from transformers import T5ForConditionalGeneration, ByT5Tokenizer

import re, regex
from ast import literal_eval
import string
import json

from typing import List, Set, Dict, Tuple, Optional, Union
import warnings


class AALPhonate:
    '''
        Apply rule-based phonological transformations to a given WME text to generate text with synthetic phonologial features of African American Language
        
        Texts are first transcribed into phoneme transcriptions using a grapheme-to-phoneme model, and phonological transformations are applied 
            at the phoneme level. Finally, a phoneme-to-grapheme model transforms the modified phonemes back into text.
    '''
    
    # --- Regex Patterns ---
    tok_regex          = r'(\s*[' + string.punctuation.replace('\'', '') + r']+\s+)|(\s+)|(\s+\'+\s+)'
    val_regex          = r'[\ˈ\w&&\D][[\w\'\ˈ\ˌ\`\u0361\u032F]&&[^\d\_]]{1,10}$' # At least one letter, then other possibilities
    val_regex_nl       = r'[\ˈ\w&&\D][[\w\'\ˈ\ˌ\`\u0361\u032F]&&[^\d\_]]+$' # val regex without length limit
    graph_val_regex    = r'[\ˈ\w&&\D][[a-zA-Z\'\ˈ\ˌ\`\u0361\u032F]&&[^\d\_]]{1,10}$' # At least one letter, then other possibilities
    graph_val_regex_nl = r'[\ˈ\w&&\D][[a-zA-Z\'\ˈ\ˌ\`\u0361\u032F]&&[^\d\_]]+$' # At least one letter, then other possibilities without length limit
    
    # --- Special Tokens ---
    END_TOKEN          = '<END_SENT>' # For post-processing batch predictions
    banned_chars =  [[227], [211], [219], [220], [228]] # Bans cyrillic and other scripts from being generated

    # --- Phoneme Lists ---
    vowels = ['i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u', 'ɪ', 'ʏ', 'ʊ', 'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o', 'ə', 'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ', 'æ', 'ɐ', 'a', 'ɶ', 'ɑ', 'ɒ']
    dipthongs = ['eɪ', 'oʊ', 'aʊ', 'ɪə', 'eə', 'ɔɪ', 'aɪ', 'ʊə']
    consonants = ['m̥', 'm', 'ɱ', 'n̥', 'n', 'ɳ̊', 'ɳ', 'ɲ̟̊', 'ɲ̟', 'ɲ̊', 'ɲ', 'ŋ̊', 'ŋ', 'ɴ̥', 'ɴ', 'p', 'b', 'p̪', 'b̪', 't̪', 'd̪', 't', 'd', 'ʈ', 'ɖ', 'c', 'ɟ', 'k', 'ɡ', 'q', 'ɢ', 'ʡ', 'ʔ', 't͡s', 'd͡z', 't͡ʃ', 'd͡ʒ', 'ʈ͡ʂ', 'ɖ͡ʐ', 't͡ɕ', 'd͡ʑ', 'p͡ɸ', 'b͡β', 'p̪͡f', 'b̪͡v', 't͡θ', 'd͡ð', 'c͡ç', 'ɟ͡ʝ', 'k͡x', 'ɡ͡ɣ', 'q͡χ', 'ɢ͡ʁ', 'ʡ͡ħ', 'ʡ͡ʕ', 'ʔ͡h', 't͡ɬ', 'd͡ɮ', 'ʈ͡ɭ̊˔', 'c͡ʎ̥˔', 'k͡ʟ̝̊', 'ɡ͡ʟ̝', 's', 'z', 'ʃ', 'ʒ', 'ʂ', 'ʐ', 'ɕ', 'ʑ', 'ɸ', 'β', 'f', 'v', 'θ', 'ð', 'ç', 'ʝ', 'x', 'ɣ', 'χ', 'ʁ', 'ħ', 'ʕ', 'h', 'ɦ', 'ɬ', 'ɮ', 'ɭ̊', 'ʎ̥', 'ʎ̝', 'ʟ̝̊', 'ʟ̝', 'ʋ̥', 'ʋ', 'ɹ̥', 'ɹ', 'ɻ̊', 'ɻ', 'j̊', 'j', 'ɰ̊', 'ɰ', 'l̥', 'l', 'ɭ̊', 'ɭ', 'ʎ̥', 'ʎ', 'ʟ̥', 'ʟ', 'ʟ̠', 'ⱱ̟', 'ⱱ', 'ɾ̥', 'ɾ', 'ɽ̊', 'ɽ', 'ɢ̆', 'ʡ̮', 'ɺ', 'ɭ̆', 'ʎ̮', 'ʟ̆', 'ʙ', 'r̥', 'r', 'ɽ͡r̥', 'ɽ͡r', 'ʀ̥', 'ʀ', 'ʜ', 'ʢ', 'ʘ', 'ǀ', 'ǂ', 'ǁ', 'ɓ', 'ɗ', 'ʄ', 'ɠ', 'ʛ', 'pʼ', 'tʼ', 'ʈ', 'cʼ', 'kʼ', 'qʼ', 'ʡʼ', 'f', 'θʼ', 'sʼ', 'ʃʼ', 'ʂʼ', 'xʼ', 'χ', 'ɬ']
    rhotics = ['r', 'ɾ', 'ɹ', 'ɻ', 'ʀ', 'ʁ', 'ɽ', 'ɺ']
    rcolored_vowels = ['ɚ', 'ɝ']
    rounded_vowels = ['y', 'ʉ', 'u', 'ʏ', 'ʊ', 'ø', 'ɵ', 'o', 'œ', 'ɞ', 'ɔ', 'ɶ', 'ɒ']
    labials = ['p', 'b', 'm', 'f', 'v']
    vcd_pairs = {'d': 't', 'ɡ': 'k', 'z': 's', 'b': 'p', 'v': 'f', 'ʒ': 'ʃ'} # Relevant voiced/unvoiced pairs
    voiced    = ['d', 'g', 'z', 'b', 'v', 'ʒ']
    unvoiced  = ['t', 'k', 's', 'p', 'f', 'ʃ']
    
    # Probability of applying a phonological transformation
    probs = {'dpt_simp':       0.2,
             'th_front':       0.2,
             'non_rhot':       0.2,
             'str_back':       0.2,
             'l_del':          0.2,
             'fin_dvc':        0.2,
             'g_drop':         0.2,
             'hap':            0.2,
             'cons_red':       0.2,
             'stress_drop':    0.2,
            }
    
    
    # List of all phonological feature functions (instantiated in constructor)
    all_augs = None
            
    
    def __init__(self, 
                 p2g_model: Union[str, T5ForConditionalGeneration] = None, 
                 g2p_model: Union[str, T5ForConditionalGeneration] = None, 
                 tok: Union[str, ByT5Tokenizer] = None, 
                 device = None, 
                 probs: Union[float, Dict[str,float]] = None, augs: Tuple[str] = None,
                 config: str = None):
        '''
            Constructor for AAL phonological augmentations
            
            Parameters:
                -p2g_model
                    A byT5 model or path to model for converting phonemes into graphemes. Must be passed with g2p_model and tok
                -g2p_model
                    A byT5 model or path to model for converting graphemes into phonemes. Must be passed with p2g_model and tok
                -tok
                    A T5 tokenizer or path to tokenizer. Must be passed with g2p_model and p2g_model
                -device
                    Device to use with p2g and g2p model
                -probs
                    Dictionary of probabilities for each phonological feature. Should contain keys
                -augs
                    List of augmentation names to consider. Equivalent to setting non-listed augs to probability 0
                -config
                    Config file with the above parameters to load. Takes priority over manually passed parameters
                    
        '''
        
        # Check and load configuration file
        if config is not None:
            print(f'Loading phonate from configuration: {config}')
            with open(config, 'r') as f:
                config = json.load(f)
                
            p2g_model = config['p2g_model']
            g2p_model = config['g2p_model']
            tok       = config['tok']
            device    = config['device']
            probs     = config['probs']
            augs      = config['augs']
                
                
        # Implemented augmentations 
        self.cand_augs = {
             'dpt_simp': self.dpt_simp,  'th_front':    self.th_front, 'other_non_rhot':    self.non_rhot,
             'str_back': self.str_back,  'l_del':       self.l_del,    'fin_dvc':  self.fin_dvc,   
             'g_drop':      self.g_drop,   'hap':      self.haplology, 'stress_drop': self.stress_drop,
             'cons_red':    self.cons_red, 'non_rhot': self.other_non_rhot,
        }
        
        # Check arguments are valid
        self.check_args(p2g_model, g2p_model, tok, device, probs, augs)

        # Set Device
        self.device = device if device is not None else torch.device('cpu')
        
        # Load models and tokenizer
        self.load_components(p2g_model, g2p_model, tok)

        # Configure augmentation probabilities
        self.update_probs(probs)

        # Set to Default
        if augs is None:
            self.all_augs = (self.dpt_simp, self.th_front, self.non_rhot, self.str_back, self.l_del, self.fin_dvc, self.g_drop, self.haplology, self.stress_drop, self.cons_red, self.other_non_rhot)
        else:
            self.all_augs = tuple([self.cand_augs[name] for name in augs])
                
    
# ---------- Setup Functions ----------

    def check_args(self, p2g_model, g2p_model, tok, device, probs, augs):
        '''
            Validate arguments passed to Phonate constructor
        '''
        
        # Check that all models are passed
        if (p2g_model is None or 
                g2p_model is None or
                tok is None):
            raise Exception('Invalid Arguments', 'You must pass either "p2g_model", "g2p_model", and "tok" or "config"')
        
        # Check that models and tokenizers are valid
        if not isinstance(p2g_model, str) and not isinstance(p2g_model, T5ForConditionalGeneration):
            raise Exception('Invalid Arguments', 'Passed p2g_model is not a checkpoint path, huggingface model name, or T5ForConditionalGeneration model. Other models are not yet implemented.') 
        if not isinstance(g2p_model, str) and not isinstance(g2p_model, T5ForConditionalGeneration):
            raise Exception('Invalid Arguments', 'Passed g2p_model is not a checkpoint path, huggingface model name, or T5ForConditionalGeneration model. Other models are not yet implemented.') 
        if not isinstance(tok, str) and not isinstance(tok, ByT5Tokenizer):
            raise Exception('Invalid Arguments', 'Passed tok is not a checkpoint path, huggingface model name, or ByT5Tokenizer object. Other tokenizers are not yet implemented and non-byte tokenizers will break functionality.')
            
        # Check passed probabilities
        if isinstance(probs, float):
            if probs <= 0:
                raise Exception('Invalid Arguments', f'Passed probs ({probs}) is invalid. If a numeric value is passed, it must be great than 0 and less than or equal to 1')
        elif isinstance(probs, dict):
            valid = False
            for prob in probs.values():
                if prob > 0:
                    valid = True
            if not valid:
                raise Exception('Invalid Arguments', f'Passed probs is invalid. If a dictionary is passed, at least one augmentation must have probability greater than 0 and less than or equal to 1.')
                
        # Check passed augmentations
        bad_augs = []
        for aug_name in augs:
            if aug_name not in self.cand_augs.keys():
                bad_augs.append(aug_name)
        if len(bad_augs) > 0:
            raise ValueError(f'Provided augmentation functions not implemented: {", ".join(bad_augs)}. Select from: {", ".join(self.cand_augs.keys())}')

            self.all_augs = (aug_func for aug_name, aug_func in self.cand_augs.items() if aug_name in augs)
    
    def load_components(self, p2g_model, g2p_model, tok):
        '''
            Load the components of phonate, including the Phoneme-to-Grapheme model, Grapheme-to-Phoneme model, and tokenizer. Passed arguments should be either strings representing checkpoint locations on huggingface or local file storage, or loaded instances of each
        '''
        
        print('Loading models and tokenizer...', end = '\r')
        # Load Grapheme to Phoneme Model
        if isinstance(g2p_model, str):
            self.g2p_model = T5ForConditionalGeneration.from_pretrained(g2p_model).to(self.device)
        else:
            self.g2p_model = g2p_model

        # Load Phoneme to Grapheme Model
        if isinstance(p2g_model, str):
            self.p2g_model = T5ForConditionalGeneration.from_pretrained(p2g_model).to(self.device)
        else:
            self.p2g_model = p2g_model

        self.g2p_model.eval()
        self.p2g_model.eval()

        # Load Tokenizer
        if isinstance(tok, str):
            self.tok = ByT5Tokenizer.from_pretrained(tok)
        else:
            self.tok = tok
        
        print('Finished loading models and tokenizer')
            
    def update_probs(self, probs):
        '''
            Update and validate the provided feature probabilities
            
            Parameters:
                -probs
                    If a float, will set all feature probabilities to the same given value
                    If a dict, will set each feature probability individually
        '''
        
        if probs is not None:                
            if isinstance(probs, float):
                if probs < 0:
                    raise Exception
                for key in self.probs.keys():
                    self.probs[key] = probs
            else:
                for feat in self.probs.keys():
                    if feat not in probs.keys():
                        probs[feat] = 0.
                        warnings.warn(f'Missing Feature Probabilities: {feat} not found in `probs` keys. Probability will be set to 0.')
                    if probs[feat] < 0 or probs[feat] > 1:
                        if probs[feat] < 0:
                            fixed_prob = 0
                        elif probs[feat] > 1:
                            fixed_prob = 1
                        warnings.warn('Invalid Feature Probabilities', 
                                        f'If "probs" is passed, all probabilities must be between 0 and 1. {feat} was passed with probability {probs[feat]}, but will be set to {fixed_prob}')
                        probs[feat] = fixed_prob
                            
                self.probs = probs

            
# ---------- Text Processing Functions ----------
    def tok_and_valid(self, word_or_phons_l, end_tok = True, graphs = False, length_lim = True):
        '''
            Tokenize and validate each token for whether augmentations can be applied (i.e. contains no numbers, punctuation, etc)
        '''
        
        # Split sentences into tokens for individual feed into model
        if isinstance(word_or_phons_l, str):
            toks = [tok for tok in re.split(self.tok_regex, word_or_phons_l) if tok is not None]
            # Modify single string
        elif isinstance(word_or_phons_l, list) and isinstance(word_or_phons_l[0], str):
            # Modify set of strings
            toks = [tok for word_or_phon_s in word_or_phons_l for tok in (re.split(self.tok_regex, word_or_phon_s) + [self.END_TOKEN]) if tok is not None]
        else:
            # Invalid word_or_phons_l type
            raise ValueError(f'Expected a str or list of strs for word_or_phons_l, got: {type(word_or_phons_l)}')
            
        # Identify which tokens should be converted
        # Set the regex depending on if graphs are being read and 
        #     if the terms transcribed should be length limited
        val_reg = self.val_regex
        if graphs:
            val_reg = self.graph_val_regex
            if not length_lim:
                val_reg = self.graph_val_regex_nl
        elif not length_lim:
            val_reg = self.val_regex_nl
            
        valid     = [bool(regex.match(val_reg, tok, regex.VERSION1)) for tok in toks]
    
        return toks, valid
    
    def text_to_phones(self, text_l, length_lim = True):
        '''
            Convert text to phoneme transcriptions
        '''
        
        # Tokenize and validate texts
        word_toks, valid = self.tok_and_valid(text_l, graphs = True, length_lim = length_lim)

        # Prepare model input texts
        words = [f'<eng_us>: {tok}' for tok, val in zip(word_toks, valid) if val]

        # Generate
        enc = self.tok(words, return_tensors = 'pt', add_special_tokens = False, padding = True).to(self.device)
        outputs = self.g2p_model.generate(enc['input_ids'], attention_mask = enc['attention_mask'], max_length = 32)
        phone_l = self.tok.batch_decode(outputs, skip_special_tokens = True)

        # Reform final sentences, conserving punctuation and spacing
        final_sents = []
        cur_sent = []
        for word_tok, val in zip(word_toks, valid):
            if word_tok == self.END_TOKEN:
                final_sents.append(''.join(cur_sent))
                cur_sent = []
            elif val:
                cur_sent.append(phone_l.pop(0))
            else:
                cur_sent.append(word_tok)

        return final_sents            
    
    def phones_to_text(self, phon_l, orig_text, orig_phons = None):
        '''
            Convert phoneme transcripts back into synthetic AAL text
            
            Only decode tokens that were modified by phonological augmentations
        '''
        
        # Tokenize and validate 
        phon_toks, valid      = self.tok_and_valid(phon_l)
        orig_toks, orig_valid = self.tok_and_valid(orig_text, graphs = True)
        
        # Check for changed phoneme tokens
        if orig_phons is not None:
            orig_phon_toks, orig_phon_valid = self.tok_and_valid(orig_phons)
            changed = [not (orig_p == p) for orig_p, p in zip(orig_phon_toks, phon_toks)]
        else:
            changed = [True] * len(phon_toks)
        
        # Create model input using valid and augmented tokens
        phons = [tok for tok, val, ch in zip(phon_toks, valid, changed) if (val and ch)]
        
        if len(phons) > 0:
            
            # Decode specified tokens
            enc = self.tok(phons, return_tensors = 'pt', padding = True).to(self.device)
            outputs = self.p2g_model.generate(enc['input_ids'], attention_mask = enc['attention_mask'],
                                             num_beams = 3, max_length = enc['input_ids'].shape[1] + 10,
                                             bad_words_ids = self.banned_chars) # Avoids Sanskrit and Cyrillic scripts
            words = self.tok.batch_decode(outputs, skip_special_tokens = True)

            # Reform final sentences, conserving punctuation and spacing and reintroducing original tokens
            final_sents = []
            cur_sent = []
            for word_tok, val, ch in zip(orig_toks, valid, changed):
                if word_tok == self.END_TOKEN:
                    final_sents.append(''.join(cur_sent))
                    cur_sent = []
                elif val and ch:
                    cur_sent.append(words.pop(0))
                else:
                    cur_sent.append(word_tok)

            return final_sents
        else:
            # If none are valid or changed, return the original text
            return orig_text
    
    def clean_capitals(self, new_text, old_text):
        '''
            Recapitalizes first letters and full words to conserve formatting after augmentations
        '''
        
        # Tokenize and validate new and old texts
        new_toks, _ = self.tok_and_valid(new_text, graphs = True)
        old_toks, _ = self.tok_and_valid(old_text, graphs = True)

        # Copy capitalization of old text in new text
        clean_str = []
        for ntok, otok in zip(new_toks, old_toks):
            if otok.isupper():
                clean_str.append(ntok.upper())
            elif len(otok) > 0 and otok[0].isupper():
                clean_str.append(ntok[0].upper() + ntok[1:])
            else:
                clean_str.append(ntok)

        return ''.join(clean_str)
    
    # AUGMENTATION FUNCTIONS
    def should_aug(self, feature):
        '''
            Randomly choose whether to augment or not depending on given probabilities
        '''
        
        return np.random.choice(a=[False, True], 
                                p = [1 - self.probs[feature], self.probs[feature]])
    
    
    def dpt_simp(self, phons):
        '''
            Augmentation for dipthong simplification
            Simplifies a dipthong to its first vowel sound, omitting the second
        '''
        
        init_len = len(phons)
        for match in re.finditer(f'{"|".join(self.dipthongs)}', phons):
            if self.should_aug('dpt_simp'):
                start = match.start()
                offset = len(phons) - init_len
                if (match.start() + offset >= 1 and phons[match.start() - 1 + offset] not in (' ', 'ˈ')):
                    phons = phons[:start + 1 + offset] + phons[start + 2 + offset:]
        return phons

    
    def th_front(self, phons):
        '''
            Augmentation for th- fronting/stopping
            Converts a th- sound (θ or ð) into d (if voiced and word initial), t (if voiceless and word initial), v (if voiced and not initial, ð) or f (if voiceless and not initial, θ) sounds depending on position in the term
        '''
        
        init_len = len(phons)
        
        # Word initials
        for match in re.finditer(r'(?<=\b)[θð]|(?<=\b[ˈˌ])[θð]', phons):
            if self.should_aug('th_front'):
                start = match.start()
                offset = len(phons) - init_len
                
                if match.group() == 'θ':
                    repl = 't'
                elif match.group() == 'ð':
                    repl = 'd'
                    
                phons = phons[:start + offset] + 'd' + phons[start + 1 + offset:]
        
        # Word medials and finals
        for match in re.finditer(r'(?<![\bˈˌ])[θð]', phons):
            if self.should_aug('th_front'):
                start = match.start()
                offset = len(phons) - init_len
                
                if match.group() == 'θ':
                    phons = phons[:start + offset] + 'f' + phons[start + 1 + offset:]
                elif match.group() == 'ð':
                    phons = phons[:start + offset] + 'v' + phons[start + 1 + offset:]
        return phons
    
    def non_rhot(self, phons):
        '''
            Phoneme augmentation for non-rhoticity (for cases where rhotics are preceded by vowels and followed by consonants in word-final positions)
        '''
        
        # Pre-consonantal, post-vocalic deletion
        init_len = len(phons)
        for match in re.finditer(fr'(?<=[{"".join(self.vowels)}])[{"".join(self.rhotics)}](?=[{"".join(self.consonants)}]\b)', phons):
            if self.should_aug('non_rhot'):
                start = match.start()
                offset = len(phons) - init_len
                
                phons = phons[:start + offset] + phons[start+1 + offset:]
        return phons
    
    def other_non_rhot(self, phons):
        '''
            Phoneme augmentation for non-rhoticity (for cases where rhotics are preceded by consonants but not followed by vowels)
        '''
        
        init_len = len(phons)
        for match in re.finditer(f'(?<=[{"".join(self.consonants)}])[{"".join(self.rhotics)}](?![{"".join(self.vowels)}])', phons):
            if self.should_aug('non_rhot'):
                start = match.start()
                offset = len(phons) - init_len
                
                valid_aug = False
                if start == init_len - 1:
                    valid_aug = True
                elif phons[start + 1 + offset] not in self.vowels:
                    valid_aug = True
                
                phons = phons[:start + offset - 1] + 'ə' + phons[start + 1 + offset:]
        
        # R-colored vowels
        init_len = len(phons)
        for match in re.finditer(f'(?<=[{"".join(self.consonants)}])[{"".join(self.rcolored_vowels)}](?![{"".join(self.vowels)}])', phons):
            if self.should_aug('non_rhot'):
                start = match.start()
                offset = len(phons) - init_len
                                
                valid_aug = False
                if start == init_len - 1:
                    valid_aug = True
                elif phons[start + 1 + offset] not in self.vowels:
                    valid_aug = True
                    
                phons = phons[:start + offset] + 'ə' + phons[start + 1 + offset:]

        return phons
    
    def str_back(self, phons):
        '''
            Phoneme augmentation for str-backing
        '''
        
        init_len = len(phons)
        for match in re.finditer(f'st[{"".join(self.rhotics)}]', phons):
            if self.should_aug('str_back'):
                start = match.start()
                offset = len(phons) - init_len
                phons = phons[:start+1 + offset] + 'k' + phons[start+2 + offset:]
        
        return phons
    
    def l_del(self, phons):
        '''
            Phoneme augmentation for l-lessness
        '''
        
        init_len = len(phons)
        for match in re.finditer(f'(?<=[{"".join(self.rounded_vowels)}])[lɫ](?=[^{"".join(self.vowels)}])', phons):
            if self.should_aug('l_del'):
                start = match.start()
                offset = len(phons) - init_len
                phons = phons[:start + offset] + phons[start + 1 + offset:]
        
        return phons
    
    def fin_dvc(self, phons):
        '''
            Phoneme augmentation for word-final devoicing
        '''
        
        init_len = len(phons)
        for match in re.finditer(r'(?<=[' + "".join(self.vowels) + r'])(' + "|".join(self.vcd_pairs.keys()) + r')\b', phons):
            if self.should_aug('fin_dvc'):
                start, end = match.start(), match.end()
                offset = len(phons) - init_len
                if phons[start + offset:end + offset] in self.vcd_pairs.keys():
                    phons = phons[:start + offset] + self.vcd_pairs[phons[start + offset:end + offset]] + phons[end + offset:]
                    
        return phons
    
    def haplology(self, phons):
        '''
            Phoneme augmentation for haplology
        '''
        
        new_phons = []
        phon_toks, valid = self.tok_and_valid(phons)
        
        for word, val in zip(phon_toks, valid):
            if val:
                for i, phon in enumerate(word):
                    if i < len(word) - 4:
                        if word[i:i+2] == word[i+2:i+4]:
                            if self.should_aug('hap'):
                                word = word[:i] + word[i+2:]
                        elif word[i] == word[i+2] and word[i+1] in self.vowels and word[i+3] in self.vowels:
                            if self.should_aug('hap'):
                                word = word[:i] + word[i+2:]
                for i, phon in enumerate(word):
                    if i < len(word) - 6:
                        if word[i:i+3] == word[i+3:i+6]:
                            if self.should_aug('hap'):
                                word = word[:i] + word[i+3:]
            
            new_phons.append(word)
        
        return ''.join(new_phons)
    
    def cons_red(self, phons):
        '''
            Phoneme augmentation for consonant cluster reduction
        '''
        
        init_len = len(phons)
        for match in re.finditer(fr'({"|".join(self.consonants)})' + r'{2}\b', phons):
            # Only when agree in voicing
            if (match.group()[0] in self.voiced and match.group()[1] in self.voiced) or (match.group()[0] in self.unvoiced and match.group()[1] in self.unvoiced):
                if (match.group()[1] not in ['s', 'z']):
                    if self.should_aug('cons_red'):
                        start, end = match.start(), match.end()
                        offset = len(phons) - init_len
                        rem    = np.random.choice(range(start + offset, end + offset))
                        phons = phons[:end + offset - 1] + phons[end + offset:]
                
                    
        return phons
    
    def g_drop(self, phons):
        '''
            Phoneme augmentation for g-dropping
        '''
        
        init_len = len(phons)
        for match in re.finditer(r'(?<=ɪ)ŋ\b', phons):
            if self.should_aug('g_drop'):
                start, end = match.start(), match.end()
                offset = len(phons) - init_len
                phons = phons[:start + offset] + 'n' + phons[end + offset:]
                    
        return phons
    
    def stress_drop(self, phons):
        '''
            Phoneme augmentation for stress dropping
        '''
        
        init_len = len(phons)
        for match in re.finditer(r'\b[^\s]ˈ', phons):
            if self.should_aug('stress_drop'):
                start, end = match.start(), match.end()
                offset = len(phons) - init_len
                
                if phons[start + offset] in self.rcolored_vowels and len(match.group()) == 1:
                    phons = phons[:start + offset] + 'r' + phons[end + offset - 1:]
                else:
                    phons = phons[:start + offset] + phons[end + offset - 1:]
                    
        return phons
    
    def all_aug(self, phons: str, augs: str = None):
        '''
            Apply all augmentations
        '''
        
        if augs is None:
            augs = self.all_augs
        else:
            new_augs = []
            for aug_name in augs:
                new_augs.append(self.cand_augs[aug_name])
            augs = new_augs
            
        orig_phons, _ = self.tok_and_valid(phons)
        changed = [0] * len(orig_phons)
        for aug_func in augs:
            phons = aug_func(phons)
            phon_toks, valid = self.tok_and_valid(phons)
            new_phons = []
            for i, p in enumerate(phon_toks):
                if p != orig_phons[i] and changed[i] == 0 and valid[i]:
                    changed[i] = 1
                    orig_phons[i] = p
                    new_phons.append(p)
                else:
                    new_phons.append(orig_phons[i])

            phons = ''.join(new_phons)

        return phons
    
    def random_aug(self, phons, num_ins = 1, num_dels = 1, num_subs = 1):
        '''
            Randomly augment a given phoneme sequence with the given number of insertions, deletions, and substitutions
        '''
        
        aug_phons = phons
        for i in range(num_dels):
            valid = False
            while not valid:
                valid = True
                r = np.random.randint(0, len(aug_phons)-1)
                
                if aug_phons[r] not in (self.vowels + self.consonants):
                    valid = False
                if aug_phons[r+1] in (' ', 'ˈ'):
                    valid = False
                if r > 0:
                    if aug_phons[r-1] in (' ', 'ˈ'):
                        valid = False
                
            aug_phons = aug_phons[:r] + aug_phons[r+1:]
            
        for i in range(num_subs):
            r = np.random.randint(0, len(aug_phons))
            while aug_phons[r] not in (self.vowels + self.consonants):
                r = np.random.randint(0, len(aug_phons))
            phon = aug_phons[r]
            if phon in self.consonants:
                aug_phons = aug_phons[:r] + np.random.choice(self.consonants) + aug_phons[r+1:]
            elif phon in self.vowels:
                aug_phons = aug_phons[:r] + np.random.choice(self.vowels) + aug_phons[r+1:]
            else:
                aug_phons = aug_phons[:r] + np.random.choice(self.vowels + self.consonants) + aug_phons[r+1:]
                
        for i in range(num_ins):
            r = np.random.randint(0, len(aug_phons))
            while aug_phons[r] not in (self.vowels + self.consonants):
                r = np.random.randint(0, len(aug_phons))
            aug_phons = aug_phons[:r] + np.random.choice(self.consonants + self.vowels) + aug_phons[r+1:]
            
        return aug_phons
    
    # Full Transformation Process
    def full_phon_aug(self, texts: list, augs: list = None):
        '''
            Perform full augmentation process, applying phonological transformations to input text
        '''
        
        phon_trans = self.text_to_phones(texts)
        phon_aug   = [self.all_aug(phout, augs = augs) for phout in phon_trans]
        phonate_out   = self.phones_to_text(phon_aug, orig_phons = phon_trans, orig_text = texts)
        clean_out  = [self.clean_capitals(o, texts[i]) for i, o in enumerate(phonate_out)]
        return phon_trans, phon_aug, phonate_out, clean_out
    
    def full_random_aug(self, texts: list, phons: list = None, num_ins: list = [0], num_dels: list = [0], num_subs: list = [0]):
        '''
            Perform full augmentation process, applying random phonological transformations to input text
        '''
        
        if phons is None:
            phons = self.text_to_phones(texts)
        phon_aug   = [self.random_aug(phon_s, num_ins[i], num_dels[i], num_subs[i]) for i, phon_s in enumerate(phons)]
        phonate_out   = self.phones_to_text(phon_aug, orig_phons = phons, orig_text = texts)
        clean_out  = [self.clean_capitals(o, texts[i]) for i, o in enumerate(phonate_out)]
        return phons, phon_aug, phonate_out, clean_out
