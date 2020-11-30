from vector_space_models import *

class VectorSpaceModelWithSubwords(VectorSpaceModel):
    def __init__(self,vocab,vsm_dim = 300,min_count = 1,subsample_exponent = 0.75,subword_method='fastText',ngram_min_length = 4,ngram_max_length = 6, left_boundary = '<', right_boundary = '>', min_subword_frequency = 3, weight_subwords = True):
        super().__init__(vocab,vsm_dim,min_count,subsample_exponent)
        self.filtered_indices = np.where(self.filtered_vocab)[0]
        self.full_to_filtered_id = {full_id: filtered_id for filtered_id, full_id in enumerate(self.filtered_indices)}

        self.ngram_min_length = ngram_min_length
        self.ngram_max_length = ngram_max_length
        assert(subword_method == 'fastText' or subword_method == 'entropy_peak')
        self.subword_method = subword_method
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.min_subword_frequency = min_subword_frequency
        self.weight_subwords = weight_subwords
        subword_candidate_counts = defaultdict(int)
        self.word_subwords = {}
        self.subword_type_token_counts = {}
        self.vocabulary_token_count = sum(vocab.counts)

        if subword_method == 'entropy_peak':
            self.root = Trie()
            self.reverse_root = Trie()
            for w in self.vocab.v:
                bounded_w = self.left_boundary + w + self.right_boundary
                self.root.add(bounded_w)
                self.reverse_root.add(bounded_w[::-1])

        for w in self.vocab.v:
            bounded_w = self.left_boundary + w + self.right_boundary
            if subword_method == 'fastText':
                candidates = list(VectorSpaceModelWithSubwords.decompose_word_fasttext(bounded_w,self.ngram_min_length,self.ngram_max_length))
            else:
                candidates = VectorSpaceModelWithSubwords.decompose_word_entropy_peak(bounded_w, self.vocab, self.root, self.reverse_root)
            if len(candidates):
                self.word_subwords[self.vocab.v_ids[w]] = candidates
                for c in candidates:
                    subword_candidate_counts[c] += 1
        
        filtered_subwords = [sw for sw, c in subword_candidate_counts.items() if c >= min_subword_frequency]
        self.subword_ids = {sw: i + len(self.filtered_indices) for i, sw in enumerate(sorted(filtered_subwords))}
        for wid in range(len(self.vocab.v)):
            if wid in self.word_subwords:
                candidates = list()
                for sw in self.word_subwords[wid]:
                    if sw in self.subword_ids:
                        swid = self.subword_ids[sw]
                        candidates.append(swid)
                        if swid in self.subword_type_token_counts:
                            self.subword_type_token_counts[swid][0] += 1
                            self.subword_type_token_counts[swid][1] += self.vocab.counts[wid]
                        else:
                            self.subword_type_token_counts[swid] = [1,self.vocab.counts[wid]]
                if len(candidates):
                    self.word_subwords[wid] = candidates
                else:
                    self.word_subwords.pop(wid)
                        
        self.U = np.random.randn(len(self.filtered_indices)+len(self.subword_ids),vsm_dim)
        self.V = np.random.randn(len(self.filtered_indices),vsm_dim)
        self.normalize()
        self.summed_U = np.random.randn(len(self.filtered_indices),vsm_dim)

    def sum_U(self):
        for i, wid in enumerate(self.filtered_indices):
            self.summed_U[i] = self.U[i]
            if wid in self.word_subwords:
                if self.weight_subwords:
                    self.summed_U[i] += (self.U[self.word_subwords[wid]] * np.array([self.downweight_subwords(self.word_subwords[wid],self.vocab.subsample_threshold)]).T).sum(0)
                else:
#                    self.summed_U[i] += self.U[self.word_subwords[wid]].sum(0) / len(self.word_subwords[wid])
#                    self.summed_U[i] /= len(self.word_subwords[wid]) + 1
                    self.summed_U[i] += self.U[self.word_subwords[wid]].sum(0)

    def check_vector_norms(self):
        for name, matrix in [['U',self.U],['V',self.V]]:
            norms = np.linalg.norm(matrix[:len(self.filtered_indices)],2,1)
            print(name, 'mean:', np.mean(norms), '; max:', np.max(norms), '; min:', np.min(norms), '; median:', np.median(norms))
        sw_norms = np.linalg.norm(self.U[len(self.filtered_indices):],2,1)
        print('subwords mean:', np.mean(sw_norms), '; max:', np.max(sw_norms), '; min:', np.min(sw_norms), '; median:', np.median(sw_norms))
            
    def decompose_word_fasttext(word,min_ngram,max_ngram):
        for ngram_length in range(min_ngram,max_ngram+1):
            if ngram_length >= len(word):
                break
            for i in range(len(word)-ngram_length+1):
                yield word[i:i+ngram_length]

    def decompose_word_entropy_peak(w,vocabulary,trie_root,reverse_trie_root,min_word_length=4,split_boundary='#',verbose=False):
        stripped_w = w[1:-1]
        if len(stripped_w) < min_word_length:
            return []
        else:
            forward = trie_root.all_entropies(w)[1:]
            back = reverse_trie_root.all_entropies((w)[::-1])[::-1][0:-1]
            forward_boundaries = list()
            back_boundaries = list()
            for i in range(2,len(forward)-1):
                if forward[i] > forward[i-1] and (i == len(forward) - 1 or forward[i+1] < forward[i]):
                    forward_boundaries.append(i)
                if back[-(i+1)] > back[-i] and (i == len(forward) - 1 or back[-(i+2)] < back[-(i+1)]):
                    back_boundaries.append(len(forward)-(i+1))
            back_boundaries = back_boundaries[::-1]
            forward_segmented = insert_in_positions(w,forward_boundaries,split_boundary)
            back_segmented = insert_in_positions(w,back_boundaries,split_boundary)
            if verbose:
                print('    '.join(list(stripped_w)))
                print(' '+''.join(['%1.2f '%abs(f) for f in forward[1:-1]]))
                print(' '+''.join(['%1.2f '%abs(b) for b in back[1:-1]]))
                print(forward_segmented)
                print(back_segmented)
        subword_candidates = [stripped_w]
        for bd in [forward_boundaries,back_boundaries]:
            for i in bd:
                if len(stripped_w[:i]) > 2 and stripped_w[:i] in vocabulary.v_ids:
                    if verbose:
                        print(stripped_w[:i], 'is a word')
                    subword_candidates.append(w[:i+1])
                    subword_candidates.append(w[1:i+1])
                if len(stripped_w[i:]) > 2 and stripped_w[i:] in vocabulary.v_ids:
                    if verbose:    
                        print(stripped_w[i:], 'is a word')
                    subword_candidates.append(w[i+1:])
                    subword_candidates.append(w[i+1:-1])
        for s in [forward_segmented,back_segmented]:
            splits = s.split(split_boundary)
            for si in range(len(splits)):
                if si < len(splits) - 1 and len(splits[si]+splits[si+1]) > 2:
                    subword_candidates.append(splits[si] + splits[si+1])
                if len(splits[si]) > 2 or si == len(splits) - 1:
                    subword_candidates.append(splits[si])
        return list(set(subword_candidates)-set([w]))

    def downweight_subwords(self, subword_id_list, a=0.001):
        if a <= 0 or a >= 1:
            sys.exit('Invalid weighting constant', a)
        return [a / (a + self.subword_type_token_counts[swid][1] / self.vocabulary_token_count) for swid in subword_id_list]

    def sum_subwords(self,word,downsample = None):
        if downsample is None:
            downsample = self.weight_subwords

        if word in self.vocab.v_ids:
            wid = self.vocab.v_ids[word]
            if self.filtered_vocab[wid]:
                return(self.summed_U[self.full_to_filtered_id[wid]])
            elif wid in self.word_subwords:
                subword_ids = self.word_subwords[wid]
                if downsample:
                    return (self.U[subword_ids] * np.array([self.downweight_subwords(subword_ids,self.vocab.subsample_threshold)]).T).sum(0)
                else:
                    return self.U[subword_ids].sum(0)
        else:
            bounded_word = self.left_boundary + word + self.right_boundary
            if self.subword_method == 'fastText':
                subwords = VectorSpaceModelWithSubwords.decompose_word_fasttext(bounded_word,self.ngram_min_length,self.ngram_max_length)
            else:
                subwords = VectorSpaceModelWithSubwords.decompose_word_entropy_peak(bounded_word,self.root,self.reverse_root)

            subword_ids = [self.subword_ids[sw] for sw in subwords if sw in self.subword_ids]
            if len(subword_ids) == 0:
                return []
            if downsample:
                return (self.U[subword_ids] * np.array([self.downweight_subwords(subword_ids,self.vocab.subsample_threshold)]).T).sum(0)
            else:
                return self.U[subword_ids].sum(0)
                
    def save_data(self):
        data = {
            'id': self.id,
            'vocab_id': self.vocab.id,
            'min_count': self.min_count,
            'subsample_exponent': self.subsample_exponent,
            'trained': self.trained,
            'ngram_min_length': self.ngram_min_length,
            'ngram_max_length': self.ngram_max_length,
            'subword_method': self.subword_method,
            'left_boundary': self.left_boundary,
            'right_boundary': self.right_boundary,
            'min_subword_frequency': self.min_subword_frequency,
            'weight_subwords': self.weight_subwords
        }
        
        self.vocab.save_data()
        print(f'Saving {self.vocab.lang} vector space model {self.id}')
        np.savez_compressed(self.id+'.'+self.vocab.lang+'.vsm_sw', U=self.U, V=self.V)
        with open(self.id+'.'+self.vocab.lang+'.vsm_sw.json','w',encoding='utf-8') as outfile:
            outfile.write(json.dumps(data))

    def load_data(self,identifier):
        with open(identifier+'.'+self.vocab.lang+'.vsm_sw.json',encoding='utf-8') as infile:
            data = json.loads(infile.read())
        new_vocab = Vocabulary(self.vocab.lang)
        new_vocab.load_data(data['vocab_id'])
        loaded_data = np.load(identifier+'.'+self.vocab.lang+'.vsm_sw.npz')

        self.__init__(new_vocab,loaded_data['U'].shape[1], data['min_count'], data['subsample_exponent'],
                      data['subword_method'], data['ngram_min_length'], data['ngram_max_length'],
                      data['left_boundary'], data['right_boundary'], data['min_subword_frequency'],
                     data['weight_subwords'])

        self.id = data['id']
        self.trained = data['trained']

        assert(self.U.shape == loaded_data['U'].shape)
        assert(self.V.shape == loaded_data['V'].shape)

        self.U = loaded_data['U']
        self.V = loaded_data['V']
        self.sum_U()

        print("Loading vector space model complete")
        print(self.summary())

    def normalize(self):
        self.U = normalize_matrix(self.U)
        self.V = normalize_matrix(self.V)
    
    def update_vocabulary(self):
        print('The vocabulary for a subword VSM cannot be updated in this implementation, generate a new VSM')

    def update_min_count(self,min_count):
        print('The minimum word count for a subword VSM cannot be updated in this implementation, generate a new VSM')

    def compare_vectors(self,w1,w2,method='U-U',use_subwords=True):
        u_vector = []
        v_vector = []
        if use_subwords:
            oov = 0
            for w in (w1,w2):
                vector = self.sum_subwords(w)
                if len(vector) == 0:
                    print(w, "is not in the filtered vocabulary and does not have any listed subwords")
                    oov = 1
                    break
                u_vector.append(vector)
                if w in self.vocab.v_ids:
                    if self.filtered_vocab[self.vocab.v_ids[w]]:
                        v_vector.append(self.V[self.full_to_filtered_id[self.vocab.v_ids[w]]])
                    else:
                        print(w, "is not in the filtered vocabulary, approximating using subwords")
                        v_vector.append(np.zeros(self.vsm_dim,dtype=float))
            if oov:
                sys.exit("OOV error")
        else:
            oov = 0
            for w in (w1,w2):
                if w not in self.vocab.v_ids or not self.filtered_vocab[self.vocab.v_ids[w]]:
                    oov += 1
                    print(w, "is not in the filtered dictionary")
                    break
                else:
                    wid = self.full_to_filtered_id[self.vocab.v_ids[w]]
                    u_vector.append(self.summed_U[wid])
                    v_vector.append(self.V[wid])
            if oov:
                sys.exit("OOV error")
        
        if method == 'U-U':
            return np.dot(normalize_matrix(u_vector[0]),normalize_matrix(u_vector[1]))
        elif method == 'U-V':
            if np.sum(v_vector[1]) == 0:
                sys.exit(f"Cannot compare since {w2} is not in the filtered vocabulary")
            else:
                return np.dot(normalize_matrix(u_vector[0]),normalize_matrix(v_vector[1]))
        elif method == 'U+V':
            return np.dot( normalize_matrix(u_vector[0]+v_vector[0]) ,
                           normalize_matrix(u_vector[1]+v_vector[1]) )

    def compare_vector_to_matrix(self,w,method='U-U',n=0,use_subwords=True):
        if use_subwords:
            u_vector = self.sum_subwords(w)
            if len(u_vector) == 0:
                sys.exit(w, "is not in the filtered vocabulary and does not have any listed subwords")
            if w in self.vocab.v_ids and self.filtered_vocab[self.vocab.v_ids[w]]:
                v_vector = self.V[self.full_to_filtered_id[self.vocab.v_ids[w]]]
            else:
                v_vector = np.zeros(self.vsm_dim,dtype=float)
        else:
            if w not in self.vocab.v_ids or not self.filtered_vocab[self.vocab.v_ids[w]]:
                sys.exit(w, "is not in the filtered dictionary")
            else:
                u_vector = self.U[self.full_to_filtered_id[self.vocab.v_ids[w]]]
                v_vector = self.V[self.full_to_filtered_id[self.vocab.v_ids[w]]]

        if method == 'U-U':
            filtered_similarities = np.matmul(normalize_matrix(u_vector),normalize_matrix(self.summed_U[:len(self.filtered_indices)]).T)
        elif method == 'U-V':
            filtered_similarities = np.matmul(normalize_matrix(u_vector),normalize_matrix(self.V).T)
        elif method == 'U+V':
            added_word_vector = normalize_matrix(u_vector+v_vector)
            added_vsm_matrices = normalize_matrix(self.summed_U[:len(self.filtered_indices)]+self.V)
            filtered_similarities = np.matmul(added_word_vector,added_vsm_matrices.T)
        sorted_similarities = sorted(zip(filtered_similarities,self.filtered_indices),reverse=True)
        if n == 0:
            n = len(filtered_similarities)
        return [[sorted_similarities[i][0],self.vocab.v[sorted_similarities[i][1]]] for i in range(n)]

    def compare_ranks(self,w1,w2,return_method=None):
        for w in [w1,w2]:
            if w not in self.vocab.v_ids or not self.filtered_vocab[self.vocab.v_ids[w]]:
                sys.exit(w, 'is not in the filtered dictionary, cannot rank similarity')
        return super().compare_ranks(w1,w2,return_method)
    
    def train(self,corpus,epochs = 1, num_neg_samples = 10, lr = 0.1, lr_decay = True, l2_lambda = 0, normalize_vectors = '', shuffled = 0, epoch_callback=None):
        filtered_cumprob = np.cumsum(self.subsampled_counts[self.filtered_indices]) / sum(self.subsampled_counts[self.filtered_indices])
        warning_count = 0
        initial_lr = lr

        self.vocab.set_subsample_threshold(self.vocab.subsample_threshold)
        for e in range(epochs):
            start_time = time()
            epoch_loss = 0
            epoch_tokens = 0
            epoch_center_tokens = 0
            
            doc_index = list(range(len(corpus)))
            if shuffled:
                shuffle(doc_index)
            
            for di in range(len(corpus)):
                raw_doc_length = len(corpus[doc_index[di]])
                drop_threshold = [self.vocab.subsample_probs[wid] for wid in corpus[doc_index[di]]]
                doc = corpus.doc_list[doc_index[di]][np.random.random(len(corpus[doc_index[di]])) > drop_threshold]
                if len(doc) == 1:
                    continue

                windows = np.random.randint(low=1,high=6,size=len(doc))

                center_word_index = list(range(len(doc))) 
                if shuffled > 1:
                    shuffle(center_word_index)

                if normalize_vectors == 'document':
                    updated_U = list()
                    updated_V = list()

                for ti in range(len(doc)):
                    i = center_word_index[ti]
                    center_id = doc[i]
                    if not self.filtered_vocab[center_id] and center_id not in self.word_subwords:
                        continue
                    U_update_ids = list()
                    U_update_weights = list()
                    if self.filtered_vocab[center_id]:
                        center_vector = self.U[[self.full_to_filtered_id[center_id]]]
                        U_update_ids.append(self.full_to_filtered_id[center_id])
                        U_update_weights.append(1)
                    else:
                        center_vector = np.zeros((1,self.vsm_dim),dtype=float)
                    if center_id in self.word_subwords:
                        if self.weight_subwords:
                            center_weights = self.downweight_subwords(self.word_subwords[center_id],self.vocab.subsample_threshold)
                            U_update_weights += center_weights
                            center_vector += (self.U[self.word_subwords[center_id]] * np.array([center_weights]).T).sum(0)
                        else:
#                            U_update_weights += [1/len(self.word_subwords[center_id])] * len(self.word_subwords[center_id])
#                            center_vector += self.U[self.word_subwords[center_id]].sum(0) / len(self.word_subwords[center_id])
                            center_vector += self.U[self.word_subwords[center_id]].sum(0)
                        U_update_ids += self.word_subwords[center_id]
                    if not self.weight_subwords:
                        center_vector /= len(U_update_ids)
                    c_words = [self.full_to_filtered_id[doc[k]] for k in range(max(i-windows[i],0),min(i+windows[i]+1,len(doc))) if k != i and self.filtered_vocab[doc[k]]]
                    if len(c_words) == 0:
                        continue
                    epoch_center_tokens += 1
                    neg_rand = np.random.rand(len(c_words) * num_neg_samples)
                    neg_samples = [cumulative_binary_search(filtered_cumprob,r) for r in neg_rand]
                    neg_samples_hash = defaultdict(int)
                    for n in neg_samples:
                        neg_samples_hash[n] += 1
                    neg_sample_ids = [*neg_samples_hash.keys()]

                    cns = c_words+neg_sample_ids
                    y = np.array([1]*len(c_words)+[0]*len(neg_sample_ids))
                    yhat = scipy.special.expit(np.dot(center_vector,self.V[cns].T))
                    update = (yhat - y) * lr / num_neg_samples * ([1] * len(c_words) + list(neg_samples_hash.values()))
                    U_update = np.dot(update, self.V[cns])
#                    print('update',update.shape)
#                    print('self.V[cns]',self.V[cns].shape)
#                    print('center_vector',center_vector.shape)
#                    print('np.dot(update.T, center_vector)',np.dot(update.T, center_vector).shape)
#                    print('weights shape',np.array([[1]*len(c_words)+list(neg_samples_hash.values())]).T.shape)
                    self.V[cns] = self.V[cns] * (1 - l2_lambda) - np.dot(update.T, center_vector) * np.array([[1]*len(c_words)+list(neg_samples_hash.values())]).T
#                        print(U_update_weights)
#                        print(np.array([U_update_weights]).shape)
                    if self.weight_subwords:
                        self.U[U_update_ids] = self.U[U_update_ids] * (1 - l2_lambda) - np.dot(np.array([U_update_weights]).T,U_update)
                    else:
                        self.U[U_update_ids] = self.U[U_update_ids] * (1 - l2_lambda) - U_update
#                        self.U[U_update_ids] = self.U[U_update_ids] * (1 - l2_lambda) - U_update / len(U_update_ids)

                    with warnings.catch_warnings():
                        try:
                            epoch_loss -= np.sum(y * np.log(yhat) + (1-y) * np.log(1-yhat))
                            epoch_tokens += len(cns)
                        except Warning:
                            warning_count += 1
                            if warning_count < 6:
                                print('Warning caught')

                    if normalize_vectors == 'document':
                        updated_U += U_update_ids                                                            
                        updated_V += cns
                if (di + 1) % 100 == 0:
                    print('%d / %d done, avg loss: %f, lr: %f'%(di+1, len(corpus.doc_list), epoch_loss / epoch_tokens, lr))
                if normalize_vectors == 'document':
                    updated_U = list(set(updated_U))
                    updated_V = list(set(updated_V))
                    U[updated_U] = normalize_matrix(U[updated_U])
                    V[updated_V] = normalize_matrix(V[updated_V])
                if lr_decay:
                    lr -= initial_lr * raw_doc_length / (corpus.token_count * epochs)
            if normalize_vectors == 'epoch': 
                self.normalize()
            if epoch_callback:
                self.sum_U()
                epoch_callback(self)
            print("Completed in %d s"%(time()-start_time))
            print("Trained center words: %d out of %d"%(epoch_center_tokens,corpus.token_count))
            print("Avg loss for epoch %d:"%e, epoch_loss / epoch_tokens)
        self.trained = True
        if normalize_vectors == 'end':
            self.normalize()
        self.sum_U()
        print('Total warnings:', warning_count)

    def summary(self):
        return {'id': self.id, 'vector space dimension': self.vsm_dim, 'vocabulary info': self.vocab.summary(),
                'rare word minimum threshold': self.min_count, 'filtered vocabulary size': sum(self.filtered_vocab),
                'context smoothing exponent': self.subsample_exponent, 'trained': self.trained, 'subword method': self.subword_method,
                'minimum subword length': self.ngram_min_length, 'maximum subword length': self.ngram_max_length,
               'minimum subword frequency': self.min_subword_frequency, 'number of subwords': len(self.subword_ids),
               'weighted subwords': self.weight_subwords}
    
class VSM_Subwords_Corpus(VSM_Corpus):
    def __init__(self,vsm):
        assert isinstance(vsm, VectorSpaceModelWithSubwords)
        super().__init__(vsm)

    def add_docs(self,file_list,segment_file=False,boundary_symbol='¤',verbose = False):
        if self.vsm_min_count != self.vsm.min_count:
            sys.exit('Error: The minimum token count threshold of the VSM was changed, the corpus should be recompiled')

        for i, fn in enumerate(file_list):
            with open(fn,encoding='utf-8') as f:
                f_text = f.read()
                
            if segment_file:
                docs = f_text.split(boundary_symbol)
            else:
                docs = [f_text]

            for d in docs:
                tokens_filtered = [t for t in re.split('(\W)',d) if len(t) and not t.isspace()]
                if self.vsm.vocab.lower:
                    tokens_filtered = [t.lower() for t in tokens_filtered]
                tokens_filtered = [self.vsm.vocab.v_ids[t] for t in tokens_filtered 
                                    if t in self.vsm.vocab.v_ids
                                    and (self.vsm.filtered_vocab[self.vsm.vocab.v_ids[t]] or self.vsm.vocab.v_ids[t] in self.vsm.word_subwords)
                                    ]
                
                self.doc_list.append(np.array(tokens_filtered))
                self.token_count += len(tokens_filtered)
            if verbose and ((i + 1) % 1000 == 0):
                print('Added %d / %d files'%(i+1,len(file_list)))

    def load_data(self,identifier):
        with open(identifier+'.'+self.vsm.vocab.lang+'.corpus.json') as infile:
            data = json.loads(infile.read())
        with open(identifier+'.'+self.vsm.vocab.lang+'.corpus.pickle','rb') as infile:
            self.doc_list = pickle.load(infile)
        self.id = data['id']
        self.vsm_min_count = data['vsm_min_count']
        self.token_count = data['token_count']
        new_vsm = VectorSpaceModelWithSubwords(Vocabulary(self.vsm.vocab.lang))
        new_vsm.load_data(data['vsm_id'])
        
        if new_vsm.min_count != self.vsm_min_count:
            sys.exit(f'Error, cannot load corpus. VSM minimum count threshold is {new_vsm.min_count}, the corpus was compiled using a threshold of {self.vsm_min_count}.')
        
        self.vsm = new_vsm
        
        print("Loading corpus complete")
        print(self.summary())

class Trie:
    def __init__(self):
        self.children = {}
        self.total = 0
    def add(self,word_string):
        if len(word_string):
            if word_string[0] in self.children:
                self.children[word_string[0]][1] += 1
            else:
                self.children[word_string[0]] = [Trie(),1]
            self.children[word_string[0]][0].add(word_string[1:])
            self.total += 1
    def prob(self,child_char):
        return self.children[child_char][1] / self.total
    def entropy(self):
        probs = np.array([self.prob(c) for c in self.children])
        return -np.sum(probs * np.log(probs))
    def all_entropies(self,word_string,accumulator = None):
        if accumulator is None:
            accumulator = list()
        if len(word_string):
            accumulator.append(self.entropy())
            return self.children[word_string[0]][0].all_entropies(word_string[1:],accumulator)
        else:
            return accumulator
    def get_node(self,word_string):
        if len(word_string) == 0:
            return self
        elif word_string[0] in self.children:
            return self.children[word_string[0]][0].get_node(word_string[1:])
        else:
            return None
    def __str__(self):
        return f'Total: {self.total}; ' + str([(k, v[1]) for k, v in self.children.items()])

def insert_in_positions(w,pos_list,char='#'):
    substrings = list()
    previous_pos = 0
    for i in pos_list:
        substrings.append(w[previous_pos:(i+1)])
        previous_pos = i+1
    substrings.append(w[previous_pos:])
    return char.join(substrings)
    
class BilingualVectorSpaceModelWithSubwords(BilingualVectorSpaceModel):
    
    def __init__(self,vocab1,vocab2,vsm_dim = 300,min_count = 1, min_count1 = 0,min_count2 = 0,subsample_exponent = 0.75,subword_method='fastText',ngram_min_length = 4,ngram_max_length = 6, left_boundary = '<', right_boundary = '>', min_subword_frequency = 3, weight_subwords = True):
        super().__init__(vocab1,vocab2,vsm_dim,min_count,min_count1,min_count2,subsample_exponent)

        self.filtered_indices = (np.where(self.filtered_vocab[0])[0],np.where(self.filtered_vocab[1])[0])
        self.full_to_filtered_id = ({full_id: filtered_id for filtered_id, full_id in enumerate(self.filtered_indices[0])},
                                   {full_id: filtered_id for filtered_id, full_id in enumerate(self.filtered_indices[1])})
        self.ngram_min_length = ngram_min_length
        self.ngram_max_length = ngram_max_length
        assert(subword_method == 'fastText' or subword_method == 'entropy_peak')
        self.subword_method = subword_method
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.min_subword_frequency = min_subword_frequency
        self.weight_subwords = weight_subwords

        self.word_subwords = ({},{})
        self.subword_type_token_counts = ({},{})
        self.vocabulary_token_count = (sum(vocab1.counts),sum(vocab2.counts))
        self.root = (Trie(),Trie())
        self.reverse_root = (Trie(),Trie())
        self.subword_ids = [{},{}]
        U = [0,0]
        V = [0,0]

        for language in (0,1):
            subword_candidate_counts = defaultdict(int)
            if subword_method == 'entropy_peak':
                for w in self.vocab[language].v:
                    bounded_w = self.left_boundary + w + self.right_boundary
                    self.root[language].add(bounded_w)
                    self.reverse_root[language].add(bounded_w[::-1])

            for w in self.vocab[language].v:
                bounded_w = self.left_boundary + w + self.right_boundary
                if subword_method == 'fastText':
                    candidates = list(VectorSpaceModelWithSubwords.decompose_word_fasttext(bounded_w,self.ngram_min_length,self.ngram_max_length))
                else:
                    candidates = VectorSpaceModelWithSubwords.decompose_word_entropy_peak(bounded_w, self.vocab[language], self.root[language], self.reverse_root[language])
                if len(candidates):
                    self.word_subwords[language][self.vocab[language].v_ids[w]] = candidates
                    for c in candidates:
                        subword_candidate_counts[c] += 1

            filtered_subwords = [sw for sw, c in subword_candidate_counts.items() if c >= min_subword_frequency]
            self.subword_ids[language] = {sw: i + len(self.filtered_indices[language]) for i, sw in enumerate(filtered_subwords)}
            for wid in range(len(self.vocab[language].v)):
                if wid in self.word_subwords[language]:
                    candidates = list()
                    for sw in self.word_subwords[language][wid]:
                        if sw in self.subword_ids[language]:
                            swid = self.subword_ids[language][sw]
                            candidates.append(swid)
                            if swid in self.subword_type_token_counts[language]:
                                self.subword_type_token_counts[language][swid][0] += 1
                                self.subword_type_token_counts[language][swid][1] += self.vocab[language].counts[wid]
                            else:
                                self.subword_type_token_counts[language][swid] = [1,self.vocab[language].counts[wid]]
                    if len(candidates):
                        self.word_subwords[language][wid] = candidates
                    else:
                        self.word_subwords[language].pop(wid)

            U[language] = np.random.randn(len(self.filtered_indices[language])+len(self.subword_ids[language]),vsm_dim)
            V[language] = np.random.randn(len(self.filtered_indices[language]),vsm_dim)

        self.U1 = U[0]
        self.V2 = V[0]
        self.U2 = U[1]
        self.V1 = V[1]
        
        self.summed_U = (np.zeros((len(self.filtered_indices[0]),vsm_dim)),
                        np.zeros((len(self.filtered_indices[1]),vsm_dim)))
        self.normalize()
        
    def normalize(self):
        for matrix in (self.U1,self.U2,self.V1,self.V2):
            matrix[:] = normalize_matrix(matrix[:])

    def sum_U(self):
        for language,U in ([0,self.U1],[1,self.U2]):
            for i, wid in enumerate(self.filtered_indices[language]):
                self.summed_U[language][i] = U[i]
                if wid in self.word_subwords[language]:
                    if self.weight_subwords:
                        self.summed_U[language][i] += (U[self.word_subwords[language][wid]] * np.array([self.downweight_subwords(language,self.word_subwords[language][wid],self.vocab[language].subsample_threshold)]).T).sum(0)
                    else:
                        self.summed_U[language][i] += U[self.word_subwords[language][wid]].sum(0)

    def check_vector_norms(self):
        for language, Uname, Vname, U, V  in [[0,'U1','V2',self.U1,self.V2],[1,'U2','V1',self.U2,self.V1]]:
            for matrix_name,matrix in [(Uname,U),(Vname,V)]:
                norms = np.linalg.norm(matrix[:len(self.filtered_indices[language])],2,1)
                print(matrix_name, 'mean:', np.mean(norms), '; max:', np.max(norms), '; min:', np.min(norms), '; median:', np.median(norms))
            sw_norms = np.linalg.norm(U[len(self.filtered_indices[language]):],2,1)
            print(self.vocab[language].lang,'subwords mean:', np.mean(sw_norms), '; max:', np.max(sw_norms), '; min:', np.min(sw_norms), '; median:', np.median(sw_norms))
            print()

    def downweight_subwords(self, language, subword_id_list, a=0.001):
        if a <= 0 or a >= 1:
            sys.exit('Invalid weighting constant', a)
        return [a / (a + self.subword_type_token_counts[language][swid][1] / self.vocabulary_token_count[language]) for swid in subword_id_list]

    def sum_subwords(self,language,word,downsample = None):
        if language == 0:
            U = self.U1
        else:
            U = self.U2
        if downsample is None:
            downsample = self.weight_subwords

        if word in self.vocab[language].v_ids:
            wid = self.vocab[language].v_ids[word]
            if self.filtered_vocab[language][wid]:
                return(self.summed_U[language][self.full_to_filtered_id[language][wid]])
            elif wid in self.word_subwords[language]:
                subword_ids = self.word_subwords[language][wid]
                if downsample:
                    return (U[subword_ids] * np.array([self.downweight_subwords(language,subword_ids,self.vocab[language].subsample_threshold)]).T).sum(0)
                else:
                    return U[subword_ids].sum(0)
        else:
            bounded_word = self.left_boundary + word + self.right_boundary
            if self.subword_method == 'fastText':
                subwords = VectorSpaceModelWithSubwords.decompose_word_fasttext(bounded_word,self.ngram_min_length,self.ngram_max_length)
            else:
                subwords = VectorSpaceModelWithSubwords.decompose_word_entropy_peak(bounded_word,self.root[language],self.reverse_root[language])

            subword_ids = [self.subword_ids[language][sw] for sw in subwords if sw in self.subword_ids[language]]
            if len(subword_ids) == 0:
                return []
            if downsample:
                return (U[subword_ids] * np.array([self.downweight_subwords(language,subword_ids,self.vocab[language].subsample_threshold)]).T).sum(0)
            else:
                return U[subword_ids].sum(0)

    def save_data(self):
        data = {
            'id': self.id,
            'vocab_id_1': self.vocab[0].id,
            'vocab_id_2': self.vocab[1].id,
            'min_count_1': self.min_count[0],
            'min_count_2': self.min_count[1],
            'subsample_exponent': self.subsample_exponent,
            'trained': self.trained,
            'ngram_min_length': self.ngram_min_length,
            'ngram_max_length': self.ngram_max_length,
            'subword_method': self.subword_method,
            'left_boundary': self.left_boundary,
            'right_boundary': self.right_boundary,
            'min_subword_frequency': self.min_subword_frequency,
            'weight_subwords': self.weight_subwords
        }

        self.vocab[0].save_data()
        self.vocab[1].save_data()
        print(f'Saving {self.vocab[0].lang}-{self.vocab[1].lang} bilingual vector space model {self.id}')
        np.savez_compressed(self.id+'.'+self.vocab[0].lang+'-'+self.vocab[1].lang+'.bvsm_sw', U1=self.U1, V1=self.V1, U2=self.U2, V2=self.V2)
        with open(self.id+'.'+self.vocab[0].lang+'-'+self.vocab[1].lang+'.bvsm_sw.json','w',encoding='utf-8') as outfile:
            outfile.write(json.dumps(data))

    def load_data(self,identifier):
        lang1 = self.vocab[0].lang
        lang2 = self.vocab[1].lang
        with open(identifier+'.'+lang1+'-'+lang2+'.bvsm_sw.json',encoding='utf-8') as infile:
            data = json.loads(infile.read())

        new_vocab1 = Vocabulary(lang1)
        new_vocab1.load_data(data['vocab_id_1'])

        new_vocab2 = Vocabulary(lang2)
        new_vocab2.load_data(data['vocab_id_2'])

        loaded_data = np.load(identifier+'.'+lang1+'-'+lang2+'.bvsm_sw.npz')

        self.__init__(new_vocab1,new_vocab2,loaded_data['U1'].shape[1],
                0,data['min_count_1'],data['min_count_2'],data['subsample_exponent'],
                data['subword_method'], data['ngram_min_length'], data['ngram_max_length'],
                data['left_boundary'], data['right_boundary'], data['min_subword_frequency'],
                data['weight_subwords'])

        self.id = data['id']
        self.trained = data['trained']
        
        assert(self.U1.shape == loaded_data['U1'].shape)
        assert(self.V1.shape == loaded_data['V1'].shape)
        assert(self.U2.shape == loaded_data['U2'].shape)
        assert(self.V2.shape == loaded_data['V2'].shape)

        self.U1 = loaded_data['U1']
        self.U2 = loaded_data['U2']
        self.V1 = loaded_data['V1']
        self.V2 = loaded_data['V2']
        self.sum_U()

        print("Loading bilingual vector space model complete")
        print(self.summary())

    def update_vocabulary(self):
        print('The vocabulary for a subword VSM cannot be updated in this implementation, generate a new VSM')

    def update_min_count(self,min_count):
        print('The minimum word count for a subword VSM cannot be updated in this implementation, generate a new VSM')
        
    def initialize_from_VSMs(self,vsms):
        if len(vsms) != 2 or not isinstance(vsms[0],VectorSpaceModelWithSubwords) or not isinstance(vsms[1],VectorSpaceModelWithSubwords):
            sys.exit('Requires a list of two vector space model objects')

        for i, v in enumerate(vsms):
            if self.vsm_dim != v.vsm_dim:
                sys.exit(f'Dimension mismatch error. Bilingual VSM: {self.vsm_dim}, {v.vocab.lang} VSM: {v.vsm_dim}')
            if not v.trained:
                sys.exit(f'Error: {v.vocab.lang} VSM is untrained')
            if self.vocab[i].lang != v.vocab.lang:
                sys.exit(f'Language mismatch error: Bilingual VSM: ({self.vocab[0].lang},{self.vocab[1].lang}); input VSMs: ({vsms[0].vocab.lang},{vsms[1].vocab.lang})')
            if self.vocab[i].lower != v.vocab.lower:
                sys.exit(f'Vocabulary mismatch error: Bilingual VSM {self.vocab[i].lang} lower {self.vocab[i].lower}; input {v.vocab.lang} VSM {v.vocab.lower}')
            if self.vocab[i].lemmatize != v.vocab.lemmatize:
                sys.exit(f'Vocabulary mismatch error: Bilingual VSM {self.vocab[i].lang} lemmatized {self.vocab[i].lemmatize}; input {v.vocab.lang} VSM {v.vocab.lemmatize}')
        
        trained_words = [0,0]
        untrained_words = [0,0]
        for language, U, V in ([0,self.U1,self.V2],[1,self.U2,self.V1]):
            for w in np.array(self.vocab[language].v)[self.filtered_vocab[language]]:
                if w in vsms[language].vocab.v_ids and vsms[language].filtered_vocab[vsms[language].vocab.v_ids[w]]:
                    this_wid = self.full_to_filtered_id[language][self.vocab[language].v_ids[w]]
                    other_wid = vsms[language].full_to_filtered_id[vsms[language].vocab.v_ids[w]]
                    U[this_wid] = vsms[language].U[other_wid]
                    V[this_wid] = vsms[language].V[other_wid]
                    trained_words[language] += 1
                else:
                    untrained_words[language] += 1

        result_hash = {self.vocab[language].lang: {'trained': trained_words[language], 'untrained': untrained_words[language]} for language in (0,1) }

        trained_subwords = [0,0]
        untrained_subwords = [0,0]

        for language, U in ([0,self.U1],[1,self.U2]):
            result_hash[self.vocab[language].lang]['trained subwords'] = 0
            result_hash[self.vocab[language].lang]['untrained subwords'] = 0
            for sw, i in self.subword_ids[language].items():
                if sw in vsms[language].subword_ids:
                    U[i] = vsms[language].U[vsms[language].subword_ids[sw]]
                    result_hash[self.vocab[language].lang]['trained subwords'] += 1
                else:
                    result_hash[self.vocab[language].lang]['untrained subwords'] += 1
            
        return result_hash

    def initialize_from_bilingual_VSM(self,bvsm):

        if not isinstance(bvsm,BilingualVectorSpaceModelWithSubwords):
            sys.exit('Requires a bilingual vector space model object')
        if self.vsm_dim != bvsm.vsm_dim:
            sys.exit(f'Dimension mismatch error. This VSM: {self.vsm_dim}, input VSM: {bvsm.vsm_dim}')
        if not bvsm.trained:
            sys.exit(f'Error: input VSM is untrained')
        for i in range(2):
            if self.vocab[i].lang != bvsm.vocab[i].lang:
                sys.exit(f'Language mismatch error: This VSM: ({self.vocab[0].lang},{self.vocab[1].lang}); input VSMs: ({bvsm.vocab[0].lang},{bvsm.vocab[1].lang})')
            if self.vocab[i].lower != bvsm.vocab[i].lower:
                sys.exit(f'Vocabulary mismatch error: Bilingual VSM {self.vocab[i].lang} lower {self.vocab[i].lower}; input VSM {bvsm.vocab[i].lower}')
            if self.vocab[i].lemmatize != bvsm.vocab[i].lemmatize:
                sys.exit(f'Vocabulary mismatch error: Bilingual VSM {self.vocab[i].lang} lemmatized {self.vocab[i].lemmatize}; input VSM {bvsm.vocab[i].lemmatize}')
            if self.subword_method != bvsm.subword_method:
                sys.exit(f'VSM mismatch error: Bilingual VSM subword method {self.subword_method}; input VSM {bvsm.subword_method}')
        
        trained_words = [0,0]
        untrained_words = [0,0]

        for language, this_U, this_V, other_U, other_V in ([0,self.U1,self.V2,bvsm.U1,bvsm.V2],[1,self.U2,self.V1,bvsm.U2,bvsm.V1]):
            for i, w in enumerate(np.array(self.vocab[language].v)[self.filtered_vocab[language]]):
                if w in bvsm.vocab[language].v_ids and bvsm.filtered_vocab[language][bvsm.vocab[language].v_ids[w]]:
                    other_wid = bvsm.full_to_filtered_id[language][bvsm.vocab[language].v_ids[w]]
                    this_U[i] = other_U[other_wid]
                    this_V[i] = other_V[other_wid]
                    trained_words[language] += 1
                else:
                    untrained_words[language] += 1
        
        result_hash = {self.vocab[language].lang: {'trained': trained_words[language], 'untrained': untrained_words[language]} for language in (0,1) }

        trained_subwords = [0,0]
        untrained_subwords = [0,0]

        for language, this_U, other_U in ([0,self.U1,bvsm.U1],[1,self.U2,bvsm.U2]):
            result_hash[self.vocab[language].lang]['trained subwords'] = 0
            result_hash[self.vocab[language].lang]['untrained subwords'] = 0
            for sw, i in self.subword_ids[language].items():
                if sw in bvsm.subword_ids[language]:
                    this_U[i] = other_U[bvsm.subword_ids[language][sw]]
                    result_hash[self.vocab[language].lang]['trained subwords'] += 1
                else:
                    result_hash[self.vocab[language].lang]['untrained subwords'] += 1
            
        return result_hash

    def copy_identical(self,what='all',list_only=True):
        if list_only:
            print('Only listing identical items, not copying!')
        copied = {}
        if what == 'all' or what == 'words':
            copied['words'] = list()
            for i, w in enumerate(self.vocab[0].v):
                if self.filtered_vocab[0][i] and w in self.vocab[1].v_ids and self.filtered_vocab[1][self.vocab[1].v_ids[w]]:
                    if list_only == False:
                        source_wid = self.full_to_filtered_id[0][self.vocab[0].v_ids[w]]
                        target_wid = self.full_to_filtered_id[1][self.vocab[1].v_ids[w]]
                        self.U2[target_wid] = self.U1[source_wid]
                        self.V1[target_wid] = self.V2[source_wid]
                    copied['words'].append(w)
            print(f"Copied {len(copied['words'])} word vectors from {self.vocab[0].lang} to {self.vocab[1].lang}")
        if what == 'all' or what == 'subwords':
            copied['subwords'] = list()
            for sw, i in self.subword_ids[0].items():
                if sw in self.subword_ids[1]:
                    if list_only == False:
                        self.U2[self.subword_ids[1][sw]] = self.U1[i]
                    copied['subwords'].append(sw)
            print(f"Copied {len(copied['subwords'])} subword vectors from {self.vocab[0].lang} to {self.vocab[1].lang}")
            
        return copied

    def compare_vectors(self,w1,w2,lang1,lang2=None,method='U-U',use_subwords=True):
        if not use_subwords:
            return super().compare_vectors(w1,w2,lang1,lang2,method)

        if lang2 == None:
            lang2 = lang1

        lang_ids = [-1,-1]
        
        for i,l in enumerate([lang1,lang2]):
            if l == self.vocab[0].lang:
                lang_ids[i] = 0
            elif l == self.vocab[1].lang:
                lang_ids[i] = 1
            else:
                sys.exit(f'Wrong language selected: {l}. Available languages: {self.vocab[0].lang}, {self.vocab[1].lang}')

        V = [self.V2,self.V1]
                  
        u_vector = []
        v_vector = []

        oov = 0
        for language,w in ([lang_ids[0],w1],[lang_ids[1],w2]):
            vector = self.sum_subwords(language,w)
            if len(vector) == 0:
                print(f"{w} is not in the filtered {self.vocab[language].lang} vocabulary and does not have any listed subwords")
                oov = 1
                break
            u_vector.append(vector)
            if w in self.vocab[language].v_ids:
                if self.filtered_vocab[language][self.vocab[language].v_ids[w]]:
                    v_vector.append(V[language][self.full_to_filtered_id[language][self.vocab[language].v_ids[w]]])
                else:
                    print(f"{w} is not in the filtered {self.vocab[language].lang} vocabulary, approximating using subwords")
                    v_vector.append(np.zeros(self.vsm_dim,dtype=float))
        if oov:
            sys.exit("OOV error")

        if method == 'U-U':
            return np.dot(normalize_matrix(u_vector[0]),normalize_matrix(u_vector[1]))
        elif method == 'U-V':
            if np.sum(v_vector[1]) == 0:
                sys.exit(f"Cannot compare since {w2} is not in the filtered vocabulary")
            else:
                return np.dot(normalize_matrix(u_vector[0]),normalize_matrix(v_vector[1]))
        elif method == 'V-V':
            if np.sum(v_vector[0]) == 0:
                sys.exit(f"Cannot compare since {w1} is not in the filtered vocabulary")
            if np.sum(v_vector[1]) == 0:
                sys.exit(f"Cannot compare since {w2} is not in the filtered vocabulary")
            return np.dot(normalize_matrix(v_vector[0]),normalize_matrix(v_vector[1]))
        elif method == 'V-U':
            if np.sum(v_vector[1]) == 0:
                sys.exit(f"Cannot compare since {w1} is not in the filtered vocabulary")
            else:
                return np.dot(normalize_matrix(v_vector[1]),normalize_matrix(u_vector[1]))
        elif method == 'U+V':
            return np.dot( normalize_matrix(u_vector[0]+v_vector[0]) ,
                           normalize_matrix(u_vector[1]+v_vector[1]) )

    def compare_vector_to_matrix(self,w,lang1,lang2=None,method='U-U',n=0,use_subwords=True):
        if not use_subwords:
            return super().compare_vector_to_matrix(w,lang1,lang2,method,n)

        if lang2 == None:
            lang2 = lang1

        lang_ids = [-1,-1]
        
        for i,l in enumerate([lang1,lang2]):
            if l == self.vocab[0].lang:
                lang_ids[i] = 0
            elif l == self.vocab[1].lang:
                lang_ids[i] = 1
            else:
                sys.exit(f'Wrong language selected: {l}. Available languages: {self.vocab[0].lang}, {self.vocab[1].lang}')

        V = [self.V2,self.V1]
                  
        u_vector = self.sum_subwords(lang_ids[0],w)
        if len(u_vector) == 0:
            sys.exit(w, "is not in the filtered vocabulary and does not have any listed subwords")
        
        if w in self.vocab[lang_ids[0]].v_ids and self.filtered_vocab[lang_ids[0]][self.vocab[lang_ids[0]].v_ids[w]]:
            v_vector = V[lang_ids[0]][self.full_to_filtered_id[lang_ids[0]][self.vocab[lang_ids[0]].v_ids[w]]]
        else:
            v_vector = np.zeros(self.vsm_dim)

        if method == 'U-U':
            filtered_similarities = np.matmul(normalize_matrix(u_vector),normalize_matrix(self.summed_U[lang_ids[1]][:len(self.filtered_indices[lang_ids[0]])]).T)
        elif method == 'U-V':
            filtered_similarities = np.matmul(normalize_matrix(u_vector),normalize_matrix(V[lang_ids[1]]).T)
        elif method == 'V-U':
            if np.sum(v_vector) == 0:
                sys.exit(f"Cannot compare since {w} is not in the filtered vocabulary")
            filtered_similarities = np.matmul(normalize_matrix(v_vector),normalize_matrix(self.summed_U[lang_ids[1]][:len(self.filtered_indices[lang_ids[0]])]).T)
        elif method == 'U+V':
            added_word_vector = normalize_matrix(u_vector+v_vector)
            added_vsm_matrices = normalize_matrix(self.summed_U[lang_ids[1]][:len(self.filtered_indices[lang_ids[1]])]+V[lang_ids[1]])
            filtered_similarities = np.matmul(added_word_vector,added_vsm_matrices.T)
        sorted_similarities = sorted(zip(filtered_similarities,self.filtered_indices[lang_ids[1]]),reverse=True)
        if n == 0:
            n = len(filtered_similarities)
        return [[sorted_similarities[i][0],self.vocab[lang_ids[1]].v[sorted_similarities[i][1]]] for i in range(n)]

                  
    def compare_ranks(self,w1,w2,lang,return_results=False,methods=None):
        if lang == self.vocab[0].lang:
            lang_id = 0
        elif lang == self.vocab[1].lang:
            lang_id = 1
        else:
            sys.exit(f'Wrong language selected: {lang}. Available languages: {self.vocab[0].lang}, {self.vocab[1].lang}')

        for w in [w1,w2]:
            if w not in self.vocab[lang_id].v_ids or not self.filtered_vocab[lang_id][self.vocab[lang_id].v_ids[w]]:
                sys.exit(w, 'is not in the filtered dictionary, cannot rank similarity')
        return super().compare_ranks(w1,w2,lang,return_results,methods)

    def bilingual_compare_ranks(self,w1,w2,lang1='.',lang2='.',return_results=False,methods=None):
        if lang1 == '.':
            lang1 = self.vocab[0].lang
        if lang2 == '.':
            lang2 = self.vocab[1].lang
        lang_to_id = {self.vocab[0].lang: 0, self.vocab[1].lang: 1}
        lang_ids = [lang_to_id[lang1],lang_to_id[lang2]]
        
        for lang_id, w in [[lang_ids[0],w1],[lang_ids[1],w2]]:
            if w not in self.vocab[lang_id].v_ids or not self.filtered_vocab[lang_id][self.vocab[lang_id].v_ids[w]]:
                sys.exit(w, 'is not in the filtered dictionary, cannot rank similarity')
        return super().bilingual_compare_ranks(w1,w2,lang1,lang2,return_results,methods)
                  
    def train(self,corpus,target_center_method = 'linear',window_size = 20, dynamic_window = False, epochs = 1, lr_decay = True, num_neg_samples = 5, lr = 0.05, l2_lambda = 0, normalize_vectors = '', shuffled = 0, epoch_callback=None, report_state=500):
        if corpus.vsm != self:
            sys.exit('The training corpus was not set up for this vector space model')
        if corpus.vsm_min_count != self.min_count:
            sys.exit('Error: The minimum token count threshold of the VSM was changed, the corpus should be recompiled')

        initial_lr = lr
        warning_count = 0

        self.vocab[0].set_subsample_threshold(self.vocab[0].subsample_threshold)
        self.vocab[1].set_subsample_threshold(self.vocab[1].subsample_threshold)

        filtered_cumprob = (np.cumsum(self.subsampled_counts[0][self.filtered_indices[0]]) / sum(self.subsampled_counts[0][self.filtered_indices[0]]),
                            np.cumsum(self.subsampled_counts[1][self.filtered_indices[1]]) / sum(self.subsampled_counts[1][self.filtered_indices[1]]))

        total_filtered_tokens = [0,0]
        for doc_pair in corpus:
            for i in range(2):
                total_filtered_tokens[i] += len(doc_pair[i])
        
        for e in range(epochs):
            start_time = time()
            epoch_loss = [0,0]
            epoch_tokens = [0,0]
            epoch_center_tokens = [0,0]
                  
            doc_index = list(range(len(corpus)))
            if shuffled:
                shuffle(doc_index)

            if dynamic_window:
                windows = [np.random.randint(low=1,high=window_size,size=total_filtered_tokens[0]),
                        np.random.randint(low=1,high=window_size,size=total_filtered_tokens[1])]
                
            for di in range(len(corpus)):
                drop_threshold = list()
                doc = list()
                for l in range(2):
                    drop_threshold.append([self.vocab[l].subsample_probs[wid] for wid in corpus[doc_index[di]][l]])
                    doc.append(corpus[doc_index[di]][l][np.random.random(len(corpus[doc_index[di]][l])) > drop_threshold[l]])
                if len(doc[0]) < 2 or len(doc[1]) < 2:
                    continue
                raw_doc_length = len(corpus[doc_index[di]][0])

                for direction in range(2):
                    if direction == 0:
                        U = self.U1
                        V = self.V1
                        source_lang = 0
                        source_doc = doc[0]
                        target_doc = doc[1]
                        lang_pair_string = f'{self.vocab[0].lang}->{self.vocab[1].lang}'
                    else:
                        U = self.U2
                        V = self.V2
                        source_lang = 1
                        source_doc = doc[1]
                        target_doc = doc[0]
                        lang_pair_string = f'{self.vocab[1].lang}->{self.vocab[0].lang}'

                    source_word_index = list(range(len(source_doc))) 
                    if shuffled > 1:
                        shuffle(source_word_index)

                    if len(target_doc) <= 2 * window_size + 1:
                        long_target = False
                        target_words = target_doc
                    elif target_center_method == 'linear':
                        long_target = True
                        target_centers = np.floor(np.arange(len(source_doc)) * len(target_doc) / len(source_doc)).astype(np.int)
                    else:
                        long_target = True
                        pass # TODO

                    if normalize_vectors == 'document': 
                        updated_U = list()
                        updated_V = list()

                    for ti in range(len(source_doc)):
                        i = source_word_index[ti]
                        source_id = source_doc[i]
                        if long_target:
                            if dynamic_window:
                                target_word_indices = np.arange(max(0,target_centers[i]-windows[i]),min(len(target_doc),target_centers[i]+windows[i]))
                            else:
                                target_word_indices = np.arange(max(0,target_centers[i]-window_size),min(len(target_doc),target_centers[i]+window_size))
                            target_words = target_doc[target_word_indices]

                        target_words = [self.full_to_filtered_id[1-source_lang][wid] for wid in target_words if self.filtered_vocab[1-source_lang][wid]]
                        if len(target_words) == 0:
                            continue
                  
                        if not self.filtered_vocab[source_lang][source_id] and source_id not in self.word_subwords[source_lang]:
                            continue
                  
                        epoch_center_tokens[source_lang] += 1
                        U_update_ids = list()
                        U_update_weights = list()
                        if self.filtered_vocab[source_lang][source_id]:
                            source_vector = U[[self.full_to_filtered_id[source_lang][source_id]]]
                            U_update_ids.append(self.full_to_filtered_id[source_lang][source_id])
                            U_update_weights.append(1)
                        else:
                            source_vector = np.zeros((1,self.vsm_dim),dtype=float)
     
                        if source_id in self.word_subwords[source_lang]:
                            if self.weight_subwords:
                                source_weights = self.downweight_subwords(source_lang,self.word_subwords[source_lang][source_id],self.vocab[source_lang].subsample_threshold)
                                U_update_weights += source_weights
                                source_vector += (U[self.word_subwords[source_lang][source_id]] * np.array([source_weights]).T).sum(0)
                            else:
                                source_vector += U[self.word_subwords[source_lang][source_id]].sum(0)
                            U_update_ids += self.word_subwords[source_lang][source_id]
                        if not self.weight_subwords:
                            source_vector /= len(U_update_ids)

                        neg_rand = np.random.rand(len(target_words) * num_neg_samples)
                        neg_samples = [cumulative_binary_search(filtered_cumprob[1-source_lang],r) for r in neg_rand]
                        neg_samples_hash = defaultdict(int)
                        for n in neg_samples:
                            neg_samples_hash[n] += 1
                        neg_sample_ids = [*neg_samples_hash.keys()]

                        cns = list(target_words)+neg_sample_ids
                        y = np.array([1]*len(target_words)+[0]*len(neg_sample_ids))

                        yhat = scipy.special.expit(np.dot(source_vector,V[cns].T))
                        update = (yhat - np.array([1]*len(target_words)+[0]*len(neg_sample_ids))) * lr / num_neg_samples * ([1] * len(target_words) + list(neg_samples_hash.values()))
             
                        U_update = np.dot(update, V[cns])
                        V[cns] = V[cns] * (1 - l2_lambda) - np.dot(update.T, source_vector) * np.array([[1]*len(target_words)+list(neg_samples_hash.values())]).T

                        if self.weight_subwords:
                            U[U_update_ids] = U[U_update_ids] * (1 - l2_lambda) - np.dot(np.array([U_update_weights]).T,U_update)
                        else:
                            U[U_update_ids] = U[U_update_ids] * (1 - l2_lambda) - U_update

                        with warnings.catch_warnings():
                            try:
                                epoch_loss[direction] -= np.sum(y * np.log(yhat) + (1-y) * np.log(1-yhat))
                                epoch_tokens[direction] += len(cns)
                            except Warning:
                                warning_count += 1
                                if warning_count < 6:
                                    print('Warning caught')

                        if normalize_vectors == 'document':
                            updated_U += U_update_ids                                                            
                            updated_V += cns
                  
                    if (di + 1) % report_state == 0:
                        print('%d / %d done, avg loss: %f, learning rate: %f'%(di+1, len(corpus), epoch_loss[direction] / epoch_tokens[direction], lr))
                    if normalize_vectors == 'document':
                        updated_U = list(set(updated_U))
                        updated_V = list(set(updated_V))
                        U[updated_U] = normalize_matrix(U[updated_U])
                        V[updated_V] = normalize_matrix(V[updated_V])

                if lr_decay:
                    lr -= initial_lr * raw_doc_length / (corpus.token_count[0] * epochs)

            if normalize_vectors == 'epoch': 
                self.normalize()
            if epoch_callback:
                self.sum_U()
                epoch_callback(self)
            print("Completed in %d s"%(time()-start_time))
            print(f"Trained center words {self.vocab[0].lang}: {epoch_center_tokens[0]} out of {corpus.token_count[0]}")
            print(f"Trained center words {self.vocab[1].lang}: {epoch_center_tokens[1]} out of {corpus.token_count[1]}")
            print("Avg losses for epoch %d:"%e)
            print(f'{self.vocab[0].lang}->{self.vocab[1].lang}: {epoch_loss[0] / epoch_tokens[0]}')
            print(f'{self.vocab[1].lang}->{self.vocab[0].lang}: {epoch_loss[1] / epoch_tokens[1]}')

        self.trained = True
        if normalize_vectors == 'end':
            self.normalize()
        self.sum_U()
        print('Total warnings:', warning_count)
    
    def summary(self):
        return {'id': self.id, 'vector space dimension': self.vsm_dim, 'vocabulary 1 info': self.vocab[0].summary(),
                'vocabulary 2 info': self.vocab[1].summary(), 'rare word minimum thresholds': self.min_count,
                'filtered vocabulary sizes': (sum(self.filtered_vocab[0]),sum(self.filtered_vocab[1])),
                'context smoothing exponent': self.subsample_exponent, 'trained': self.trained, 'subword method': self.subword_method, 
                'minimum subword length': self.ngram_min_length, 'maximum subword length': self.ngram_max_length,
                'minimum subword frequency': self.min_subword_frequency, 'number of subwords': (len(self.subword_ids[0]),len(self.subword_ids[1])),
                'weighted subwords': self.weight_subwords}

class BilingualVSM_Subwords_Corpus(BilingualVSM_Corpus):
    def __init__(self,vsm):
        assert isinstance(vsm, BilingualVectorSpaceModelWithSubwords)
        super().__init__(vsm)

                  
    def load_data(self,identifier):
        lang1 = self.vsm.vocab[0].lang
        lang2 = self.vsm.vocab[1].lang
        with open(identifier+'.'+lang1+'-'+lang2+'.corpus.json') as infile:
            data = json.loads(infile.read())
        with open(identifier+'.'+lang1+'-'+lang2+'.corpus.pickle','rb') as infile:
            self.doc_pairs_list = pickle.load(infile)
        self.id = data['id']
        self.vsm_min_count = (data['vsm_min_count'][0],data['vsm_min_count'][1])
        self.token_count = data['token_count']
        self.file_list = data['file_list']
        new_vsm = BilingualVectorSpaceModelWithSubwords(Vocabulary(lang1),Vocabulary(lang2))
        new_vsm.load_data(data['vsm_id'])
        
        if new_vsm.min_count != self.vsm_min_count:
            sys.exit(f'Error, cannot load corpus. VSM minimum count threshold is {new_vsm.min_count}, the corpus was compiled using a threshold of {self.vsm_min_count}.')
        
        self.vsm = new_vsm

        print("Loading bilingual corpus complete")
        print(self.summary())

    def add_bisegmentations(self,bisegmentations,verbose=False):
        if self.vsm_min_count != self.vsm.min_count:
            sys.exit('Error: The minimum token count threshold of the VSM was changed, the corpus should be recompiled')
        for n, bs in enumerate(bisegmentations):
            if bs.segmentations[bs.langs[0]].file != None:
                new_files = [bs.segmentations[bs.langs[0]].file,bs.segmentations[bs.langs[1]].file]
                if new_files in self.file_list:
                    print(new_files,'already included in corpus, skipping')
                else:
                    self.file_list.append(new_files)
            for bisegment in bs:
                if min(bisegment.get_link_type()) > 0:
                    filtered_tokens = [list(),list()]
                    for i, l in ((0,self.vsm.vocab[0].lang),(1,self.vsm.vocab[1].lang)):
                        filtered_tokens[i] = [self.vsm.vocab[i].v_ids[t] for t in bisegment.get_tokens(l,lower=self.vsm.vocab[i].lower) if t in self.vsm.vocab[i].v_ids and (self.vsm.filtered_vocab[i][self.vsm.vocab[i].v_ids[t]] or self.vsm.vocab[i].v_ids[t] in self.vsm.word_subwords[i])]
                        self.token_count[i] += len(filtered_tokens[i])

                    self.doc_pairs_list.append((np.array(filtered_tokens[0]),np.array(filtered_tokens[1])))
            if verbose and ((n + 1) % 1000 == 0):
                print('Added %d / %d files'%(n+1,len(bisegmentations)))
