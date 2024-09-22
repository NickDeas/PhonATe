import re
import spacy
import nltk
import string

tok_regex = r'(\s*[' + string.punctuation.replace('\'', '') + r']+\s+)|(\s+)|(\s+\'+\s+)'
nlp = spacy.load('en_core_web_md')
english_vocab = set(w.lower() for w in nltk.corpus.words.words())


def filter_transforms(orig_text, aug_text):
    '''
        Filter undesirable transforms (e.g. augmentations to other existing words and changes to grammatical role.
        
        Parameters:
            -orig_text
                Original text before augmentations were applied
            -aug_text
                The augmented text to be filtered
    '''
    
    orig_doc = nlp(orig_text)
    aug_doc  = nlp(aug_text)
    
    # Get original and new parts of speech
    poss = {tok.text:tok.pos_ for tok in orig_doc}
    poss2 = {tok.text:tok.pos_ for tok in aug_doc}

    # Extract entities
    ents = ' '.join([e.text for e in orig_doc.ents]).split(' ')
    
    new_text = ''
    for t, t2 in zip(re.split(tok_regex, orig_text), re.split(tok_regex, aug_text)):
        
        if t2 is None:
            continue
        
        
        # Filter added letters (e.g. thiss)
        if len(t2) > 2:
            if t2[-1] == t2[-2] and t[-1] != t[-2] and t != t2:
                t2 = t2[:-1]
        
        if t2 == ' ':
            # Ignore spaces
            new_text += t2
        elif t2 in english_vocab and t != t2:
            # Revert modified words
            new_text += t
        elif t in ents:
            # Revert entities
            new_text += t
        elif poss.get(t, -1) != poss2.get(t2, -1) and t in poss.keys() and t2 in poss.keys():
            # Revert changes to parts of speech
            new_text += t
        else:
            # Otherwise maintain the transformation
            new_text += t2

    return new_text.strip()