from vector_space_models_with_subwords import *

class TestSet:

    def __init__(self,bsc):
        self.neighbors = list()
        self.same_doc = list()
        self.other_doc = list()
        self.bsc = bsc

        for i, bisegmentation in enumerate(bsc):
            for j, bisegment in enumerate(bisegmentation):
                change_side = random.randint(0,1)
                change_side_start = bisegment.bisegment[change_side][0]
                change_side_length = bisegment.bisegment[change_side][1]
                if change_side_length > 1:
                    change_type = random.choice(['reduce','extend'])
                else:
                    change_type = 'extend'
                if change_type == 'extend':
                    if change_side_start == 1:
                        change_direction = 'right'
                    elif (change_side_start + change_side_length) > len(bisegment.segmentations[bsc.langs[change_side]]):
                        change_direction = 'left'
                    else:
                        change_direction = random.choice(['left','right'])
                elif change_type == 'reduce':
                    change_direction = random.choice(['left','right'])

                new_bisegment = [[0,0],[0,0]]
                new_bisegment[1-change_side] = bisegment.bisegment[1-change_side]
                if change_type == 'extend' and change_direction == 'left':
                    new_bisegment[change_side] = [change_side_start-1,change_side_length+1]
                elif change_type == 'extend' and change_direction == 'right':
                    new_bisegment[change_side] = [change_side_start,change_side_length+1]
                elif change_type == 'reduce' and change_direction == 'left':
                    new_bisegment[change_side] = [change_side_start+1,change_side_length-1]
                elif change_type == 'reduce' and change_direction == 'right':
                    new_bisegment[change_side] = [change_side_start,change_side_length-1]
                
                new_bisegment = ((new_bisegment[0][0],new_bisegment[0][1]),(new_bisegment[1][0],new_bisegment[1][1]))
                self.neighbors.append({'doc': i, 'bisegment data': new_bisegment})

                targets = list(range(len(bisegmentation)))
                targets.pop(j)
                self.same_doc.append({'doc':i, 'source bisegment': j, 'target bisegment': random.choice(targets)})

                targets = list(range(len(bsc)))
                targets.pop(i)
                other_doc = random.choice(targets)
                self.other_doc.append({'target doc': other_doc, 'target bisegment': random.randint(0,len(bsc[other_doc])-1)})

def tokens_to_ids(token_list,bvsm,lang):
    if isinstance(bvsm,BilingualVectorSpaceModelWithSubwords):
        return [bvsm.vocab[lang].v_ids[t] for t in token_list if t in bvsm.vocab[lang].v_ids and 
                (bvsm.filtered_vocab[lang][bvsm.vocab[lang].v_ids[t]] or bvsm.vocab[lang].v_ids[t] in bvsm.word_subwords[lang])]

    if bvsm.vocab[lang].lemmatize:
        return [bvsm.vocab[lang].form_to_lemma[t] for t in token_list if t in bvsm.vocab[lang].form_to_lemma 
                                and bvsm.filtered_vocab[lang][bvsm.vocab[lang].form_to_lemma[t]]]
    else:
        return [bvsm.vocab[lang].v_ids[t] for t in token_list if t in bvsm.vocab[lang].v_ids and 
                bvsm.filtered_vocab[lang][bvsm.vocab[lang].v_ids[t]]]

def cosine_sim(a,b):
    return np.dot(a,b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

def sort_swaps(nums):
    result = list(nums).copy()
    num_swaps = 0
    for i in range(len(nums)-1):
        for j in range(len(nums)-1-i):
            if result[j] > result[j+1]:
                result[j], result[j+1] = result[j+1], result[j]
                num_swaps += 1
    return num_swaps

class ProcessedTestSet:
    def __init__(self,testset,bvsm,downweighting=0.001):
        self.testset = testset
        self.bvsm = bvsm
        if isinstance(bvsm,BilingualVectorSpaceModelWithSubwords):
            self.subwords = True
        elif isinstance(bvsm,BilingualVectorSpaceModel):
            self.subwords = False
        else:
            print('Error')
            print(bvsm.summary())
            
        self.token_freq = (np.array(bvsm.vocab[0].counts) / sum(bvsm.vocab[0].counts),np.array(bvsm.vocab[1].counts) / sum(bvsm.vocab[1].counts))
        self.downweighting = downweighting
        self.source_token_ids = list()
        self.target_token_ids = list()
        self.same_doc_token_ids = list()
        self.other_doc_token_ids = list()
        self.neighbors_en_token_ids = list()
        self.neighbors_hu_token_ids = list()
        
        self.bisegment_index = {}
        index = 0
        for i in range(len(testset.bsc)):
            for j in range(len(testset.bsc[i])):
                self.bisegment_index[(i,j)] = index
                index += 1
        
        for i in range(len(testset.same_doc)):
            test_item = testset.same_doc[i]

            source_tokens = testset.bsc[test_item['doc']][test_item['source bisegment']].get_tokens('en',lower=self.bvsm.vocab[0].lower)
            self.source_token_ids.append(tokens_to_ids(source_tokens,bvsm,0))
            
            target_tokens = testset.bsc[test_item['doc']][test_item['source bisegment']].get_tokens('hu',lower=self.bvsm.vocab[1].lower)
            self.target_token_ids.append(tokens_to_ids(target_tokens,bvsm,1))
            
            same_doc_tokens = testset.bsc[test_item['doc']][test_item['target bisegment']].get_tokens('hu',lower=self.bvsm.vocab[1].lower)
            self.same_doc_token_ids.append(tokens_to_ids(same_doc_tokens,bvsm,1))
            
            test_item = testset.other_doc[i]
            other_doc_tokens = testset.bsc[test_item['target doc']][test_item['target bisegment']].get_tokens('hu',lower=self.bvsm.vocab[1].lower)
            self.other_doc_token_ids.append(tokens_to_ids(other_doc_tokens,bvsm,1))
            
            test_item = testset.neighbors[i]
            bs = Bisegment(test_item['bisegment data'],[testset.bsc[test_item['doc']].segmentations['en'],testset.bsc[test_item['doc']].segmentations['hu']])
            self.neighbors_en_token_ids.append(tokens_to_ids(bs.get_tokens('en',lower=self.bvsm.vocab[0].lower),bvsm,0))
            self.neighbors_hu_token_ids.append(tokens_to_ids(bs.get_tokens('hu',lower=self.bvsm.vocab[1].lower),bvsm,1))
            
    def sum_sentence_vector(self,token_id_list,lang,what = 'U'):
        if lang == 0:
            U = self.bvsm.U1
            V = self.bvsm.V2
        else:
            U = self.bvsm.U2
            V = self.bvsm.V1
        if self.subwords:
            if what == 'U':
                word_rows = np.zeros((len(token_id_list),self.bvsm.vsm_dim))
                for i in range(len(token_id_list)):
                    word_rows[i] = self.bvsm.sum_subwords(lang,self.bvsm.vocab[lang].v[token_id_list[i]])
            elif what == 'V':
                word_rows = np.zeros((len(token_id_list),self.bvsm.vsm_dim))
                for i in range(len(token_id_list)):
                    if self.bvsm.filtered_vocab[lang][token_id_list[i]]:
                        word_rows[i] = V[self.bvsm.full_to_filtered_id[lang][token_id_list[i]]]
            elif what == 'U+V':
                word_rows = np.zeros((len(token_id_list),self.bvsm.vsm_dim))
                for i in range(len(token_id_list)):
                    word_rows[i] = self.bvsm.sum_subwords(lang,self.bvsm.vocab[lang].v[token_id_list[i]])
                    if self.bvsm.filtered_vocab[lang][token_id_list[i]]:
                        word_rows[i] += V[self.bvsm.full_to_filtered_id[lang][token_id_list[i]]]
                
#            word_rows = np.zeros((len(token_id_list),self.bvsm.vsm_dim))
 #           for i in range(len(token_id_list)):
  #              tid = token_id_list[i]
   #             parts = 0
    #            if self.bvsm.filtered_vocab[lang][tid]:
     #               word_rows[i] = summed_U[lang][[self.bvsm.full_to_filtered_id[lang][tid]]]
      #              parts = 1
       #         if tid in self.bvsm.word_subwords[lang]:
        #            if self.bvsm.weight_subwords:
         #               part_weights = self.bvsm.downweight_subwords(lang,self.bvsm.word_subwords[lang][tid],self.bvsm.vocab[lang].subsample_threshold)
          #              word_rows[i] += (U[self.bvsm.word_subwords[lang][tid]] * np.array([part_weights]).T).sum(0)
           #         else:
            #            word_rows[i] += U[self.bvsm.word_subwords[lang][tid]].sum(0)
             #           parts += self.bvsm.word_subwords[lang][tid]
              #          word_rows[i] /= parts
        else:
            if what == 'U':
                word_rows = U[token_id_list]
            elif what == 'V':
                word_rows = V[token_id_list]
            elif what == 'U+V':
                word_rows = U[token_id_list]+V[token_id_list]
        return np.dot(np.array([self.token_weights(token_id_list,lang)]),word_rows) / len(token_id_list)

    def evaluate_test_case(self,case_id,method='U+V'):
        if method == 'U-V':
            en_what = 'U'
            hu_what = 'V'
        elif method == 'U+V':
            en_what = 'U+V'
            hu_what = 'U+V'
        
        source_vector = self.sum_sentence_vector(self.source_token_ids[case_id],0,en_what)
        target_vector = self.sum_sentence_vector(self.target_token_ids[case_id],1,hu_what)
        same_doc_vector = self.sum_sentence_vector(self.same_doc_token_ids[case_id],1,hu_what)
        other_doc_vector = self.sum_sentence_vector(self.other_doc_token_ids[case_id],1,hu_what)
        neighbor_en_vector = self.sum_sentence_vector(self.neighbors_en_token_ids[case_id],0,en_what)
        neighbor_hu_vector = self.sum_sentence_vector(self.neighbors_hu_token_ids[case_id],1,hu_what)
        
        similarities = [
            cosine_sim(source_vector,target_vector),
            cosine_sim(neighbor_en_vector,neighbor_hu_vector),
            cosine_sim(source_vector,same_doc_vector),
            cosine_sim(source_vector,other_doc_vector),
        ]
        return similarities
        
    def token_weights(self,token_id_list,lang,a=0):
        if a == 0:
            a = self.downweighting
        return [a / (a + self.token_freq[lang][t]) for t in token_id_list]
    
    def evaluate_test_set(self,method='U+V'):
        scores = np.zeros((len(self.testset.same_doc),4))
        for i in range(len(self.testset.same_doc)):
            scores[i] = self.evaluate_test_case(i,method)
        scores_clean = scores[~np.isnan(scores).any(axis=1)]
        print('Valid test cases:',len(scores_clean))
        scores_ranks = np.argsort(-scores_clean, axis=1)
        badness = np.zeros(len(scores_ranks))
        for i in range(len(scores_ranks)):
            badness[i] = sort_swaps(scores_ranks[i])
        print(scores_clean.mean(0),'mean badness:',badness.mean())
