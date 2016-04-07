import re
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet

punct_matcher = re.compile('[%s]+' % re.escape(string.punctuation))
stop_words = set(stopwords.words('english'))

def get_senses(stems):
    # collect synsets for each stem
    default_synsets = []
    synset_lists = []
    synset_sigs = {}
    for stem in stems:
        if stem in stop_words:
            synset_lists.append([])
            default_synsets.append(None)
            # get synsets from WordNet, calculate signatures and
            # save them, and set default synset to the first one
        else:
            synsets = wordnet.synsets(stem)
            synset_lists.append(synsets)
            default_synsets.append(synsets[0])
            for synset in synsets:
                # the signature is set of all words in the
                # definitions and examples, skipping stopwords
                signature = set()
                text_lists = synset.definition(), synset.examples()
                for text_list in text_lists:
                    for text in text_list:
                        text = punct_matcher.sub('', text)
                        signature.update(text.lower().split())
                synset_sigs[synset] = signature - stop_words
    # fill in the senses incrementally
    for i, synset_list in enumerate(synset_lists):
        # combine the signatures of all other synsets currently
        # still being considered for the words
        other_sig = set()
        for other_list in synset_lists[:i] + synset_lists[i + 1:]:
            for synset in other_list:
                other_sig.update(synset_sigs[synset])
        # for each synset of the word, count the overlapping
        # words between its signature and the combined signature
        overlap_counts = {}
        for synset in synset_list:
            overlaps = synset_sigs[synset] & other_sig
            overlap_counts[synset] = len(overlaps)
        # select the synset with the greatest overlap, or if no
        # synsets had any overlap, use the most frequent sense
        if synset_list:
            max_synset = max(synset_list, key=overlap_counts.get)
            if overlap_counts[max_synset] == 0:
                max_synset = default_synsets[i]
            synset_lists[i] = [max_synset]
    # return the selected synsets (or None for stopwords)
    return [synset_list[0] if synset_list else None
            for synset_list in synset_lists]
   

def main():
    list = get_senses(['time','flies','like','an','arrow'])
    #print list
    for sense in list:
        print sense

if __name__ == '__main__':
    main()
