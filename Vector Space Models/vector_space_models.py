import numpy as np
import re, os, sys, string, pickle, scipy.special, bisect, json, warnings
from collections import defaultdict
from random import randrange, choices, shuffle
from time import time
sys.path.append('/home/pgergo/prog/msc')
from classic_alignment import *

def cumulative_binary_search(cumulative_list,rand_num):
    return bisect.bisect(cumulative_list,rand_num)

def normalize_matrix(row_matrix):
    if len(row_matrix.shape) == 1:
        return row_matrix / np.linalg.norm(row_matrix)
    else:
        return row_matrix / np.linalg.norm(row_matrix,2,1).reshape((len(row_matrix),1))

#import spacy
#import en_core_web_sm
#import hu_core_ud_lg
#spacy_en = en_core_web_sm.load()
#spacy_hu = hu_core_ud_lg.load()
#spacy.info()

#sys.path.append('/home/pgergo/prog/emmorphpy')
#from emmorphpy import EmMorphPy
#hu_morph = EmMorphPy()

#warnings.filterwarnings('error')
#np.seterr(all='warn')

class Vocabulary:
    def __init__(self,lang,lower=False,lemmatize=False,subsample_threshold=1e-5):
        self.lang = lang
        self.v_ids = {}
        self.counts = list()
        self.v = list()
        self.lower = lower
        self.lemmatize = lemmatize
        self.language_model_meta = ''
        self.id = ''.join(choices(string.ascii_letters + string.digits, k=8))
        self.subsample_threshold = subsample_threshold
        self.subsample_probs = list()
        self.form_to_lemma = {}
        
    def process_docs(self,doc_list,language_model=None, hu_lemmatizer=False, verbose = False):
        for i, f in enumerate(doc_list):
            if verbose:
                print('%d/%d %s'%(i,len(doc_list),f))
            with open(f,encoding='utf-8') as infile:
                txt = infile.read()
                self.process_text(txt,language_model=language_model,hu_lemmatizer = hu_lemmatizer)
        freq = np.array(self.counts) / sum(self.counts)
        self.subsample_probs = [max(0,n) for n in list(1-np.sqrt(self.subsample_threshold / freq))]
        
    def fit_to_bilingual_corpus(lang,biling_corpus):
        if lang == biling_corpus.vsm.vocab[0].lang:
            lang_index = 0
        elif lang == biling_corpus.vsm.vocab[1].lang:
            lang_index = 1
        else:
            sys.exit(f'Language mismatch error: Cannot fit {lang} vocabulary to {biling_corpus.vsm.vocab[0].lang}-{biling_corpus.vsm.vocab[1].lang} corpus')

        counts_dd = defaultdict(int)
        new_vocab = Vocabulary(lang)

        for doc in biling_corpus.doc_pairs_list:
            for t in doc[lang_index]:
                counts_dd[t] += 1
        
        counts = list()
        v = list()
        v_ids = {}
        for wid, wc in counts_dd.items():
            v.append(biling_corpus.vsm.vocab[lang_index].v[wid])
            v_ids[biling_corpus.vsm.vocab[lang_index].v[wid]] = len(counts)
            counts.append(wc)
        
        form_to_lemma = {}
        
        for form, lemma in biling_corpus.vsm.vocab[lang_index].form_to_lemma.items():
            if lemma in counts_dd:
                form_to_lemma[form] = v_ids[biling_corpus.vsm.vocab[lang_index].v[lemma]]
        
        new_vocab.counts = counts
        new_vocab.v = v
        new_vocab.v_ids = v_ids
        new_vocab.lower = biling_corpus.vsm.vocab[lang_index].lower
        new_vocab.lemmatize = biling_corpus.vsm.vocab[lang_index].lemmatize
        new_vocab.language_model_meta = biling_corpus.vsm.vocab[lang_index].language_model_meta
        new_vocab.subsample_threshold = biling_corpus.vsm.vocab[lang_index].subsample_threshold
        new_vocab.form_to_lemma = form_to_lemma
        
        freq = np.array(counts) / sum(counts)
        new_vocab.subsample_probs = [max(0,n) for n in list(1-np.sqrt(new_vocab.subsample_threshold / freq))]
        
        return new_vocab
        
    def save_data(self):
        data = {
            'id': self.id,
            'lang': self.lang,
            'lower': self.lower,
            'lemmatize': self.lemmatize,
            'lm_meta': self.language_model_meta,
            'counts': self.counts,
            'v': self.v,
            'subsample_threshold': self.subsample_threshold,
            'forms': self.form_to_lemma
        }
        print(f'Saving {self.lang} vocabulary {self.id}')
        with open(self.id+'.'+self.lang+'.vocab.json','w',encoding='utf-8') as outfile:
            outfile.write(json.dumps(data))
    
    def load_data(self,identifier):
        with open(identifier+'.'+self.lang+'.vocab.json',encoding='utf-8') as infile:
            data = json.loads(infile.read())
        self.id = data['id']
        self.lower = data['lower']
        self.lemmatize = data['lemmatize']
        self.language_model_meta = data['lm_meta']
        self.counts = data['counts']
        self.set_subsample_threshold(data['subsample_threshold'])
        self.v = data['v']
        if len(self.counts) != len(self.v):
            sys.exit('Failed to load vocabulary, input is corrupted')
        self.v_ids = {w: i for (i,w) in enumerate(self.v)}
        self.form_to_lemma = data['forms']
        print("Loading vocabulary complete")
        print(self.summary())
    
    def set_subsample_threshold(self,th):
        self.subsample_threshold = th
        freq = np.array(self.counts) / sum(self.counts)
        self.subsample_probs = [max(0,n) for n in list(1-np.sqrt(self.subsample_threshold / freq))]
    
    def to_lemma(self,token,method='lm',model=None):
        if method == 'lm':
            return model(token)[0].lemma_
        else:
            lemmas = hu_morph.stem(token)
            if len(lemmas) == 0:
                return token
            else:
                return lemmas[0][0]
    
    def process_text(self,txt,language_model=None,hu_lemmatizer=False):
        tokens_filtered = [t for t in re.split('(\W)',txt) if len(t) and not t.isspace()]

        if self.lower:
            tokens_filtered = [t.lower() for t in tokens_filtered]

        if self.lemmatize:
            if hu_lemmatizer:
                if self.lang != 'hu':
                    sys.exit('Error: Hungarian lemmatizer can only be used for Hungarian')
                elif self.language_model_meta == '':
                    self.language_model_meta = 'EmMorph'
                elif self.language_model_meta != 'EmMorph':
                    sys.exit('Error: Earlier lemmatization of this vocabulary was carried out using a different language model')
                lemmatize_method = 'emmorph'
            else:
                if language_model == None:
                    sys.exit('Error: Lemmatized vocabulary requires SpaCy language model for processing')
                if self.lang != language_model.meta['lang']:
                    sys.exit('Error: Wrong language model (%s) provided for %s language vocabulary'%(language_model.meta['lang'],self.lang))
                lm_meta = ' '.join([language_model.meta['name'],language_model.meta['version']])

                if self.language_model_meta == '':
                    self.language_model_meta = lm_meta
                elif lm_meta != self.language_model_meta:
                    sys.exit('Error: Earlier lemmatization of this vocabulary was carried out using a different language model')
                lemmatize_method = 'lm'
            
            new_forms = list(set([t for t in tokens_filtered if t not in self.form_to_lemma]))
            new_alpha = list()
            for n in new_forms:
                if n.isalpha():
                    new_alpha.append(n)
                else:
                    if n not in self.v_ids:
                        self.v_ids[n] = len(self.counts)
                        self.v.append(n)
                        self.counts.append(0)
                    self.form_to_lemma[n] = self.v_ids[n]
            if lemmatize_method == 'lm':
                new_lemmas = [t.lemma_ for t in language_model(' '.join(new_alpha))]
            elif lemmatize_method == 'emmorph':
                new_lemmas = [self.to_lemma(t,method='emmorph') for t in new_alpha]

            if len(new_alpha) == len(new_lemmas):
                for i in range(len(new_lemmas)):
                    if new_lemmas[i] not in self.v_ids:
                        self.v_ids[new_lemmas[i]] = len(self.counts)
                        self.v.append(new_lemmas[i])
                        self.counts.append(0)
                    self.form_to_lemma[new_alpha[i]] = self.v_ids[new_lemmas[i]]
                tokens_filtered = [self.v[self.form_to_lemma[t]] for t in tokens_filtered]
            else:
                print(new_alpha)
                print(new_lemmas)
                tokens_filtered = [t.lemma_ for t in language_model(' '.join(tokens_filtered))]
   
        for t in tokens_filtered:
            if t not in self.v_ids:
                self.v_ids[t] = len(self.counts)
                self.v.append(t)
                self.counts.append(1)
            else:
                self.counts[self.v_ids[t]] += 1

    def id_to_tokens(self,id_array):
        return [self.v[t] for t in id_array]
    
    def summary(self):
        self.set_subsample_threshold(self.subsample_threshold)
        return {'id': self.id, 'language': self.lang, 'language model': self.language_model_meta,
                'word types': len(self.v_ids), 'word tokens': sum(self.counts),
                'lower-cased': self.lower, 'lemmatized': self.lemmatize, 'lemmatized forms': len(self.form_to_lemma),
                'subsample threshold': self.subsample_threshold,
                'subsampled frequent words': sum(np.array(self.subsample_probs) > 0)}

class VectorSpaceModel:
    
    SHUFFLE_DOCUMENTS = 1
    SHUFFLE_WORDS = 2
    
    def __init__(self,vocab,vsm_dim = 300,min_count = 1,subsample_exponent = 0.75):
        self.vocab = vocab
        self.min_count = min_count
        self.subsample_exponent = subsample_exponent
        self.filtered_vocab = np.zeros(len(vocab.counts), dtype=np.bool)

        for c in range(len(vocab.counts)):
            if vocab.counts[c] >= min_count:
                self.filtered_vocab[c] = True

        self.subsampled_counts = np.array(vocab.counts)**subsample_exponent
        self.vsm_dim = vsm_dim

        self.U = np.random.randn(len(vocab.counts),vsm_dim)
        self.V = np.random.randn(len(vocab.counts),vsm_dim)
        self.normalize()
        self.id = ''.join(choices(string.ascii_letters + string.digits, k=8))
        self.trained = False
        
    def save_data(self):
        data = {
            'id': self.id,
            'vocab_id': self.vocab.id,
            'min_count': self.min_count,
            'subsample_exponent': self.subsample_exponent,
            'trained': self.trained
        }
        self.vocab.save_data()
        print(f'Saving {self.vocab.lang} vector space model {self.id}')
        np.savez_compressed(self.id+'.'+self.vocab.lang+'.vsm', U=self.U, V=self.V)
        with open(self.id+'.'+self.vocab.lang+'.vsm.json','w',encoding='utf-8') as outfile:
            outfile.write(json.dumps(data))

    def load_data(self,identifier):
        with open(identifier+'.'+self.vocab.lang+'.vsm.json',encoding='utf-8') as infile:
            data = json.loads(infile.read())

        new_vocab = Vocabulary(self.vocab.lang)
        new_vocab.load_data(data['vocab_id'])

        loaded_data = np.load(identifier+'.'+self.vocab.lang+'.vsm.npz')

        self.__init__(new_vocab,loaded_data['U'].shape[1],data['min_count'],data['subsample_exponent'])
        self.id = data['id']
        self.trained = data['trained']
        self.U = loaded_data['U']
        self.V = loaded_data['V']
        
        if len(self.U) != len(self.V) or len(self.U) != len(self.vocab.counts) or len(self.U) != len(self.filtered_vocab):
            sys.exit('Failed to load VSM, input is corrupted')

        print("Loading vector space model complete")
        print(self.summary())
            
    def update_vocabulary(self):
        if len(self.vocab.v) == len(self.filtered_vocab):
            print("No new words")
        else:
            new_rows_U = normalize_matrix(np.random.randn(len(self.vocab.v)-len(self.filtered_vocab),self.vsm_dim))
            new_rows_V = normalize_matrix(np.random.randn(len(self.vocab.v)-len(self.filtered_vocab),self.vsm_dim))
            self.U = np.concatenate((self.U,new_rows_U))
            self.V = np.concatenate((self.V,new_rows_V))

            self.filtered_vocab = np.zeros(len(self.vocab.counts), dtype=np.bool)

            for c in range(len(self.vocab.counts)):
                if self.vocab.counts[c] >= self.min_count:
                    self.filtered_vocab[c] = True

            self.subsampled_counts = np.array(self.vocab.counts)**self.subsample_exponent

    def update_min_count(self,min_count):
        if min_count == self.min_count:
            print("No change")
        else:
            self.min_count = min_count
            self.filtered_vocab = np.zeros(len(self.filtered_vocab), dtype=np.bool)

            for c in range(len(self.filtered_vocab)):
                if self.vocab.counts[c] >= min_count:
                    self.filtered_vocab[c] = True
                    
    def normalize(self,filtered=True):
        if filtered:
            self.U[self.filtered_vocab] = normalize_matrix(self.U[self.filtered_vocab])
            self.V[self.filtered_vocab] = normalize_matrix(self.V[self.filtered_vocab])
        else:
            self.U = normalize_matrix(self.U)
            self.V = normalize_matrix(self.V)
    
    def check_vector_norms(self):
        for name, matrix in [['U',self.U],['V',self.V]]:
            norms = np.linalg.norm(matrix[self.filtered_vocab],2,1)
            print(name, 'mean:', np.mean(norms), '; max:', np.max(norms), '; min:', np.min(norms), '; median:', np.median(norms))
        
    def compare(self,w1,w2):
        print('U <-> U: %1.5f ; U <-> V: %1.5f ; U + V: %1.5f'%( self.compare_vectors(w1,w2,method='U-U'),
                                                    self.compare_vectors(w1,w2,method='U-V'),
                                                    self.compare_vectors(w1,w2,method='U+V') ))
    
    def compare_vectors(self,w1,w2,method='U-U'):
        w_index = []
        oov = 0
        for w in (w1,w2):
            if w not in self.vocab.v_ids or not self.filtered_vocab[self.vocab.v_ids[w]]:
                oov += 1
                print(w, "is not in the filtered dictionary")
                break
            else:
                w_index.append(self.vocab.v_ids[w])
        if oov:
            sys.exit("OOV error")

        if method == 'U-U':
            return np.dot(normalize_matrix(self.U[w_index[0]]),normalize_matrix(self.U[w_index[1]]))
        elif method == 'U-V':
            return np.dot(normalize_matrix(self.U[w_index[0]]),normalize_matrix(self.V[w_index[1]]))
        elif method == 'U+V':
            return np.dot( normalize_matrix(self.U[w_index[0]]+self.V[w_index[0]]) ,
                           normalize_matrix(self.U[w_index[1]]+self.V[w_index[1]]) )

    def compare_vector_to_matrix(self,w,method='U-U',n=0):
        if w not in self.vocab.v_ids or not self.filtered_vocab[self.vocab.v_ids[w]]:
            sys.exit(w, "is not in the filtered dictionary")
        else:
            word_index = self.vocab.v_ids[w]
        if method == 'U-U':
            filtered_similarities = np.matmul(normalize_matrix(self.U[word_index]),normalize_matrix(self.U[self.filtered_vocab]).T)
        elif method == 'U-V':
            filtered_similarities = np.matmul(normalize_matrix(self.U[word_index]),normalize_matrix(self.V[self.filtered_vocab]).T)
        elif method == 'U+V':
            added_word_vector = normalize_matrix(self.U[word_index]+self.V[word_index])
            added_vsm_matrices = normalize_matrix(self.U[self.filtered_vocab]+self.V[self.filtered_vocab])
            filtered_similarities = np.matmul(added_word_vector,added_vsm_matrices.T)
        sorted_similarities = sorted(zip(filtered_similarities,np.where(self.filtered_vocab)[0]),reverse=True)
        if n == 0:
            n = len(filtered_similarities)
        return [[sorted_similarities[i][0],self.vocab.v[sorted_similarities[i][1]]] for i in range(n)]
        
    def compare_ranks(self,w1,w2,return_method=None):
        similarities = {}
        ranks_w1 = {}
        ranks_w2 = {}
        for method in ['U-U','U-V','U+V']:
            similarities[method] = self.compare_vectors(w1,w2,method)
            sorted_similarities = self.compare_vector_to_matrix(w1,method)
            for i in range(len(sorted_similarities)):
                if sorted_similarities[i][1] == w2:
                    ranks_w1[method] = i + 1
                    break
            sorted_similarities = self.compare_vector_to_matrix(w2,method)
            for i in range(len(sorted_similarities)):
                if sorted_similarities[i][1] == w1:
                    ranks_w2[method] = i + 1
                    break
        if return_method:
            return [similarities[return_method],ranks_w1[return_method],ranks_w2[return_method]]
        else:
            for m in ['U-U','U-V','U+V']:
                print('%s: similarity %1.5f, rank of %s for %s %d, rank of %s for %s %d'%(m,similarities[m],w2,w1,ranks_w1[m],w1,w2,ranks_w2[m]))
    
    def most_similar(self,w,n=10):
        results_UU = self.compare_vector_to_matrix(w,'U-U',n)
        results_UV = self.compare_vector_to_matrix(w,'U-V',n)
        results_UplusV = self.compare_vector_to_matrix(w,'U+V',n)
        print('%-25s%-25s%-25s'%('        U <-> U', '         U <-> V', '         U + V'))
        for i in range(n):
            print(' %2.5f %14s ; %2.5f %14s ; %2.5f %14s'%(results_UU[i][0], results_UU[i][1],
                                                          results_UV[i][0], results_UV[i][1],
                                                          results_UplusV[i][0], results_UplusV[i][1] ))
            
    def train(self,corpus,epochs = 1, num_neg_samples = 10, lr = 0.1, lr_decay = True, l2_lambda = 0, normalize_vectors = '', shuffled = 0, epoch_callback=None):
        if corpus.vsm_min_count != self.min_count:
            sys.exit('Error: The minimum token count threshold of the VSM was changed, the corpus should be recompiled')
        warning_count = 0
        initial_lr = lr
        
        filtered_indices = np.where(self.filtered_vocab)[0]
        filtered_cumprob = np.cumsum(self.subsampled_counts[filtered_indices]) / sum(self.subsampled_counts[filtered_indices])

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
                doc = corpus[doc_index[di]][np.random.random(len(corpus[doc_index[di]])) > drop_threshold]
                if len(doc) == 1:
                    continue
                epoch_center_tokens += len(doc)

                windows = np.random.randint(low=1,high=6,size=len(doc))

                center_word_index = list(range(len(doc))) 
                if shuffled > 1:
                    shuffle(center_word_index)

                if normalize_vectors == 'document': 
                    updated_V = list()

                for ti in range(len(doc)):
                    i = center_word_index[ti]
                    c_words = [doc[k] for k in range(max(i-windows[i],0),min(i+windows[i]+1,len(doc))) if k != i]
                    neg_rand = np.random.rand(len(c_words) * num_neg_samples)
                    neg_samples = [filtered_indices[cumulative_binary_search(filtered_cumprob,r)] for r in neg_rand]
                    neg_samples_hash = defaultdict(int)
                    for n in neg_samples:
                        neg_samples_hash[n] += 1
                    neg_sample_ids = [*neg_samples_hash.keys()]
                    center_id = doc[i]
                    #yhat = scipy.special.expit(np.dot(self.U[[center_id]],self.V[c_words].T))
                    #context_error = yhat - 1
                    #Unew = self.U[center_id] - np.matmul(context_error,self.V[c_words,:]) * lr
                    #self.V[c_words,:] -= np.matmul(context_error.T,self.U[center_id,:].reshape((1,self.vsm_dim))) * lr
                    #epoch_loss -= np.sum(np.log(yhat))
                
                    #yhat = scipy.special.expit(np.dot(self.U[center_id,:].reshape((1,self.vsm_dim)),self.V[list(neg_samples_hash.keys()),:].T))
                    #Unew -= np.matmul(yhat * list(neg_samples_hash.values()),self.V[list(neg_samples_hash.keys()),:]) * lr
                    #self.V[list(neg_samples_hash.keys()),:] -= np.matmul((yhat * list(neg_samples_hash.values())).T,self.U[center_id,:].reshape((1,self.vsm_dim))) * lr
                    #epoch_loss -= np.sum(np.log(1-yhat))
                    
                    #self.U[center_id] = Unew
                    cns = c_words+neg_sample_ids
                    y = np.array([1]*len(c_words)+[0]*len(neg_sample_ids))
                    yhat = scipy.special.expit(np.dot(self.U[[center_id]],self.V[cns].T))
                    update = (yhat - y) * lr / num_neg_samples * ([1] * len(c_words) + list(neg_samples_hash.values()))
                    Unew = self.U[[center_id]] * (1 - l2_lambda) - np.dot(update, self.V[cns])
                    self.V[cns] = self.V[cns] * (1 - l2_lambda) - np.dot(update.T, self.U[[center_id]]) * np.array([[1]*len(c_words)+list(neg_samples_hash.values())]).T
                    self.U[[center_id]] = Unew

                    with warnings.catch_warnings():
                        try:
                            epoch_loss -= np.sum(y * np.log(yhat) + (1-y) * np.log(1-yhat))
                            epoch_tokens += len(cns)
                        except Warning:
                            warning_count += 1
                            if warning_count < 6:
                                print('Warning caught')

                    if normalize_vectors == 'document': 
                        updated_V += cns
                
                if (di + 1) % 100 == 0:
                    print('%d / %d done, avg loss: %f, lr: %f'%(di+1, len(corpus), epoch_loss / epoch_tokens, lr))
                if normalize_vectors == 'document':
                    updated_U = list(set(doc))
                    updated_V = list(set(updated_V))
                    self.U[updated_U] = normalize_matrix(self.U[updated_U])
                    self.V[updated_V] = normalize_matrix(self.V[updated_V])
                if lr_decay:
                    lr -= initial_lr * raw_doc_length / (corpus.token_count * epochs)
            if normalize_vectors == 'epoch': 
                self.normalize()
            if epoch_callback:
                epoch_callback(self)
            print("Completed in %d s"%(time()-start_time))
            print("Trained center words: %d out of %d"%(epoch_center_tokens,corpus.token_count))
            print("Avg loss for epoch %d:"%e, epoch_loss / epoch_tokens)
        self.trained = True
        if normalize_vectors == 'end':
            self.normalize()
        print('Total warnings:', warning_count)

    def summary(self):
        return {'id': self.id, 'vector space dimension': self.vsm_dim, 'vocabulary info': self.vocab.summary(),
                'rare word minimum threshold': self.min_count, 'filtered vocabulary size': sum(self.filtered_vocab),
                'context smoothing exponent': self.subsample_exponent, 'trained': self.trained}        
            
class VSM_Corpus:
    def __init__(self,vsm):
        self.doc_list = list()
        self.vsm = vsm
        self.vsm_min_count = vsm.min_count
        self.token_count = 0
        self.id = ''.join(choices(string.ascii_letters + string.digits, k=8))
    
    def __len__(self):
        return len(self.doc_list)

    def __getitem__(self,key):
        return self.doc_list[key]
    
    def __iter__(self):
        for d in self.doc_list:
            yield d

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
                if self.vsm.vocab.lemmatize:
                    tokens_filtered = [self.vsm.vocab.form_to_lemma[t] for t in tokens_filtered if self.vsm.filtered_vocab[self.vsm.vocab.form_to_lemma[t]]]
                else:
                    tokens_filtered = [self.vsm.vocab.v_ids[t] for t in tokens_filtered if self.vsm.filtered_vocab[self.vsm.vocab.v_ids[t]]]
                
                self.doc_list.append(np.array(tokens_filtered))
                self.token_count += len(tokens_filtered)
            if verbose and ((i + 1) % 1000 == 0):
                print('Added %d / %d files'%(i+1,len(file_list)))

    def save_data(self):
        data = {
            'id': self.id,
            'vsm_id': self.vsm.id,
            'vsm_min_count': self.vsm_min_count,
            'token_count': self.token_count
        }
        self.vsm.save_data()
        print(f'Saving {self.vsm.vocab.lang} corpus {self.id}')
        with open(self.id+'.'+self.vsm.vocab.lang+'.corpus.json','w') as outfile:
            outfile.write(json.dumps(data))
        with open(self.id+'.'+self.vsm.vocab.lang+'.corpus.pickle','wb') as outfile:
            pickle.dump(self.doc_list,outfile)

    def load_data(self,identifier):
        with open(identifier+'.'+self.vsm.vocab.lang+'.corpus.json') as infile:
            data = json.loads(infile.read())
        with open(identifier+'.'+self.vsm.vocab.lang+'.corpus.pickle','rb') as infile:
            self.doc_list = pickle.load(infile)
        self.id = data['id']
        self.vsm_min_count = data['vsm_min_count']
        self.token_count = data['token_count']
        new_vsm = VectorSpaceModel(Vocabulary(self.vsm.vocab.lang))
        new_vsm.load_data(data['vsm_id'])
        
        if new_vsm.min_count != self.vsm_min_count:
            sys.exit(f'Error, cannot load corpus. VSM minimum count threshold is {new_vsm.min_count}, the corpus was compiled using a threshold of {self.vsm_min_count}.')
        
        self.vsm = new_vsm
        
        print("Loading corpus complete")
        print(self.summary())

    def summary(self):
        return {'id': self.id, 'vector space model info': self.vsm.summary(),
                'minimum count threshold of corpus': self.vsm_min_count,
                'number of documents': len(self.doc_list), 'number of tokens': self.token_count}


class BilingualVectorSpaceModel:

    SHUFFLE_DOCUMENTS = 1
    SHUFFLE_WORDS = 2
    
    def __init__(self,vocab1,vocab2,vsm_dim = 300,min_count = 1, min_count1 = 0,min_count2 = 0,subsample_exponent = 0.75):
        self.vocab = [vocab1, vocab2]
        min_c1 = min_count
        min_c2 = min_count
        if min_count1 != 0:
            min_c1 = min_count1
        if min_count2 != 0:
            min_c2 = min_count2
        self.min_count = (min_c1,min_c2)
            
        self.subsample_exponent = subsample_exponent
        self.filtered_vocab = [np.zeros(len(vocab1.counts), dtype=np.bool),np.zeros(len(vocab2.counts), dtype=np.bool)]

        for l in range(2):
            for c in range(len(self.vocab[l].counts)):
                if self.vocab[l].counts[c] >= self.min_count[l]:
                    self.filtered_vocab[l][c] = True

        self.subsampled_counts = [np.array(vocab1.counts)**subsample_exponent,
                                  np.array(vocab2.counts)**subsample_exponent]
        self.vsm_dim = vsm_dim

        self.U1 = np.random.randn(len(self.filtered_vocab[0]),vsm_dim)
        self.V1 = np.random.randn(len(self.filtered_vocab[1]),vsm_dim)
        self.U2 = np.random.randn(len(self.filtered_vocab[1]),vsm_dim)
        self.V2 = np.random.randn(len(self.filtered_vocab[0]),vsm_dim)

        self.normalize()
        self.id = ''.join(choices(string.ascii_letters + string.digits, k=8))
        self.trained = False
        
    def normalize(self,filtered=True):
        for l, matrix in ([0,self.U1],[1,self.U2],[1,self.V1],[0,self.V2]):
            if filtered:
                matrix[self.filtered_vocab[l]] = normalize_matrix(matrix[self.filtered_vocab[l]])
            else:
                matrix[:] = normalize_matrix(matrix[:])
 
 #           self.U1[self.filtered_vocab[0]] = normalize_matrix(self.U1[self.filtered_vocab[0]])
 #           self.V1[self.filtered_vocab[1]] = normalize_matrix(self.V1[self.filtered_vocab[1]])
 #           self.U2[self.filtered_vocab[1]] = normalize_matrix(self.U2[self.filtered_vocab[1]])
 #           self.V2[self.filtered_vocab[0]] = normalize_matrix(self.V2[self.filtered_vocab[0]])
 #       else:
 #           self.U1 = normalize_matrix(self.U1)
 #           self.V1 = normalize_matrix(self.V1)
 #           self.U2 = normalize_matrix(self.U2)
 #           self.V2 = normalize_matrix(self.V2)

    def check_vector_norms(self):
        for name, fv, matrix in [['U1',self.filtered_vocab[0],self.U1],['V1',self.filtered_vocab[1],self.V1],['U2',self.filtered_vocab[1],self.U2],['V2',self.filtered_vocab[0],self.V2]]:
            norms = np.linalg.norm(matrix[fv],2,1)
            print(name, 'mean:', np.mean(norms), '; max:', np.max(norms), '; min:', np.min(norms), '; median:', np.median(norms))
        
    def initialize_from_VSMs(self,vsms):

        if len(vsms) != 2 or not isinstance(vsms[0],VectorSpaceModel) or not isinstance(vsms[1],VectorSpaceModel):
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
                    this_wid = self.vocab[language].v_ids[w]
                    other_wid = vsms[language].vocab.v_ids[w]
                    U[this_wid] = vsms[language].U[other_wid]
                    V[this_wid] = vsms[language].V[other_wid]
                    trained_words[language] += 1
                else:
                    untrained_words[language] += 1

        return {self.vocab[language].lang: {'trained': trained_words[language], 'untrained': untrained_words[language]} for language in (0,1) }

    def initialize_from_bilingual_VSM(self,bvsm):

        if not isinstance(bvsm,BilingualVectorSpaceModel):
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
        
        trained_words = [0,0]
        untrained_words = [0,0]

        for language, this_U, this_V, other_U, other_V in ([0,self.U1,self.V2,bvsm.U1,bvsm.V2],[1,self.U2,self.V1,bvsm.U2,bvsm.V1]):
            for w in np.array(self.vocab[language].v)[self.filtered_vocab[language]]:
                if w in bvsm.vocab[language].v_ids and bvsm.filtered_vocab[language][bvsm.vocab[language].v_ids[w]]:
                    this_wid = self.vocab[language].v_ids[w]
                    other_wid = bvsm.vocab[language].v_ids[w]
                    this_U[this_wid] = other_U[other_wid]
                    this_V[this_wid] = other_V[other_wid]
                    trained_words[language] += 1
                else:
                    untrained_words[language] += 1
        return {self.vocab[language].lang: {'trained': trained_words[language], 'untrained': untrained_words[language]} for language in (0,1) }

    def copy_identical(self):
        copied = list()
        for i, w in enumerate(self.vocab[0].v):
            if self.filtered_vocab[0][i] and w in self.vocab[1].v_ids:
                self.U2[[self.vocab[1].v_ids[w]]] = self.U1[[i]]
                self.V1[[self.vocab[1].v_ids[w]]] = self.V2[[i]]
                copied.append(w)
        print(f"Copied {len(copied)} out of {sum(self.filtered_vocab[0])} word vectors from {self.vocab[0].lang} to {self.vocab[1].lang}")
        return copied
    
    def save_data(self):
        data = {
            'id': self.id,
            'vocab_id_1': self.vocab[0].id,
            'vocab_id_2': self.vocab[1].id,
            'min_count_1': self.min_count[0],
            'min_count_2': self.min_count[1],
            'subsample_exponent': self.subsample_exponent,
            'trained': self.trained
        }
        self.vocab[0].save_data()
        self.vocab[1].save_data()
        print(f'Saving {self.vocab[0].lang}-{self.vocab[1].lang} bilingual vector space model {self.id}')
        np.savez_compressed(self.id+'.'+self.vocab[0].lang+'-'+self.vocab[1].lang+'.bvsm', U1=self.U1, V1=self.V1, U2=self.U2, V2=self.V2)
        with open(self.id+'.'+self.vocab[0].lang+'-'+self.vocab[1].lang+'.bvsm.json','w',encoding='utf-8') as outfile:
            outfile.write(json.dumps(data))

    def load_data(self,identifier):
        lang1 = self.vocab[0].lang
        lang2 = self.vocab[1].lang
        with open(identifier+'.'+lang1+'-'+lang2+'.bvsm.json',encoding='utf-8') as infile:
            data = json.loads(infile.read())

        new_vocab1 = Vocabulary(lang1)
        new_vocab1.load_data(data['vocab_id_1'])

        new_vocab2 = Vocabulary(lang2)
        new_vocab2.load_data(data['vocab_id_2'])

        loaded_data = np.load(identifier+'.'+lang1+'-'+lang2+'.bvsm.npz')
        self.__init__(new_vocab1,new_vocab2,loaded_data['U1'].shape[1],0,data['min_count_1'],data['min_count_2'],data['subsample_exponent'])

        self.id = data['id']
        self.trained = data['trained']
        
        self.U1 = loaded_data['U1']
        self.U2 = loaded_data['U2']
        self.V1 = loaded_data['V1']
        self.V2 = loaded_data['V2']
        
        if self.U1.shape[1] != self.V2.shape[1] or len(self.U1) != len(self.vocab[0].counts) or self.U2.shape[1] != self.V1.shape[1] or len(self.U2) != len(self.vocab[1].counts) or self.U1.shape[1] != self.U2.shape[1]:
            sys.exit('Failed to load VSM, input is corrupted')

        print("Loading bilingual vector space model complete")
        print(self.summary())

    def update_vocabulary(self):
        if len(self.vocab[0].v) == len(self.filtered_vocab[0]) and len(self.vocab[1].v) == len(self.filtered_vocab[1]):
            print("No new words")
        else:
            new_rows_U1 = np.random.randn(len(self.vocab[0].v)-len(self.filtered_vocab[0]),self.vsm_dim)
            new_rows_U1 /= np.linalg.norm(new_rows_U1,2,1).reshape((len(new_rows_U1),1))
            self.U1 = np.concatenate((self.U1,new_rows_U1))
            
            new_rows_V1 = np.random.randn(len(self.vocab[1].v)-len(self.filtered_vocab[1]),self.vsm_dim)
            new_rows_V1 /= np.linalg.norm(new_rows_V1,2,1).reshape((len(new_rows_V1),1))
            self.V1 = np.concatenate((self.V1,new_rows_V1))

            new_rows_U2 = np.random.randn(len(self.vocab[1].v)-len(self.filtered_vocab[1]),self.vsm_dim)
            new_rows_U2 /= np.linalg.norm(new_rows_U2,2,1).reshape((len(new_rows_U2),1))
            self.U2 = np.concatenate((self.U2,new_rows_U2))
            
            new_rows_V2 = np.random.randn(len(self.vocab[0].v)-len(self.filtered_vocab[0]),self.vsm_dim)
            new_rows_V2 /= np.linalg.norm(new_rows_V2,2,1).reshape((len(new_rows_V2),1))
            self.V2 = np.concatenate((self.V2,new_rows_V2))

            self.filtered_vocab = [np.zeros(len(self.vocab[0].counts), dtype=np.bool),
                                  np.zeros(len(self.vocab[1].counts), dtype=np.bool)]

            for l in range(2):
                for c in range(len(self.vocab[l].counts)):
                    if self.vocab[l].counts[c] >= self.min_count[l]:
                        self.filtered_vocab[l][c] = True

            self.subsampled_counts = [np.array(self.vocab[0].counts)**subsample_exponent,
                                      np.array(self.vocab[1].counts)**subsample_exponent]

    def update_min_count(self,min_count):
        if len(min_count) != 2:
            print("New value must be a 2-tuple")
        elif min_count[0] == self.min_count[0] and min_count[1] == self.min_count[1]:
            print("No change")
        else:
            self.min_count = (min_count[0],min_count[1])
            self.filtered_vocab = [np.zeros(len(self.vocab[0].counts), dtype=np.bool),
                                  np.zeros(len(self.vocab[1].counts), dtype=np.bool)]

            for l in range(2):
                for c in range(len(self.vocab[l].counts)):
                    if self.vocab[l].counts[c] >= self.min_count[l]:
                        self.filtered_vocab[l][c] = True
                        
    def compare_vectors(self,w1,w2,lang1,lang2=None,method='U-U'):
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

        lU = [self.U1,self.U2]
        lV = [self.V2,self.V1]

        U = [lU[lang_ids[0]],lU[lang_ids[1]]]
        V = [lV[lang_ids[0]],lV[lang_ids[1]]]
        v_ids = [self.vocab[lang_ids[0]].v_ids,self.vocab[lang_ids[1]].v_ids]
        filtered = [self.filtered_vocab[lang_ids[0]],self.filtered_vocab[lang_ids[1]]]

        w_index = []
        oov = 0
        w = [w1,w2]
        for i in range(2):
            if w[i] not in v_ids[i] or not filtered[i][v_ids[i][w[i]]]:
                oov += 1
                print(f"{w[i]} is not in the filtered {[lang1,lang2][i]} dictionary")
                break
            else:
                w_index.append(v_ids[i][w[i]])
        if oov:
            sys.exit("OOV error")

        if method == 'U+V':
            return np.dot( normalize_matrix(U[0][w_index[0]]+V[0][w_index[0]]) ,
                           normalize_matrix(U[1][w_index[1]]+V[1][w_index[1]]) )
        else:
            matrix = [None,None]
            for i in range(2):
                if method.replace('-','')[i] == 'U':
                    matrix[i] = U[i]
                elif method.replace('-','')[i] == 'V':
                    matrix[i] = V[i]
                else:
                    sys.exit(f'Invalid method {method}')
                    
        return np.dot(normalize_matrix(matrix[0][w_index[0]]),normalize_matrix(matrix[1][w_index[1]]))
    
    def compare(self,w1,w2,lang,return_scores = False):
        if return_scores:
            return({method: self.compare_vectors(w1,w2,lang,method) for method in ['U-U','U+V','U-V']})
        print('U <-> U: %1.5f ; U <-> V: %1.5f ; U + V: %1.5f'%( self.compare_vectors(w1,w2,lang,method='U-U'),
                                                    self.compare_vectors(w1,w2,lang,method='U-V'),
                                                    self.compare_vectors(w1,w2,lang,method='U+V') ))

    def bilingual_compare(self,w1,w2,lang1='.',lang2='.',full=False, return_scores = False):
        if lang1 == '.':
            lang1 = self.vocab[0].lang
        if lang2 == '.':
            lang2 = self.vocab[1].lang
        if return_scores:
            return({method: self.compare_vectors(w1,w2,lang1,lang2,method) for method in ['U-V','V-U','U+V','U-U','V-V']})
        elif full:
            pattern1 = 'l1 source <-> l2 target: %1.5f ; l1 target <-> l2 source: %1.5f ; l1 sum <-> l2 sum: %1.5f ;'.replace('l1',lang1).replace('l2',lang2)
            pattern2 = 'l1 source <-> l2 source: %1.5f ; l1 target <-> l2 target: %1.5f'.replace('l1',lang1).replace('l2',lang2)
            print(pattern1%( self.compare_vectors(w1,w2,lang1,lang2,method='U-V'),
                        self.compare_vectors(w1,w2,lang1,lang2,method='V-U'),
                        self.compare_vectors(w1,w2,lang1,lang2,method='U+V') ))
            print(pattern2%( self.compare_vectors(w1,w2,lang1,lang2,method='U-U'),
                            self.compare_vectors(w1,w2,lang1,lang2,method='V-V') ))
        else:
            print('U <-> V: %1.5f ; sum %s <-> sum %s: %1.5f'%( self.compare_vectors(w1,w2,lang1,lang2,method='U-V'),
                                                    lang1, lang2,
                                                    self.compare_vectors(w1,w2,lang1,lang2,method='U+V') ))

    def compare_vector_to_matrix(self,w,lang1,lang2=None,method='U-U',n=0):
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

        lU = [self.U1,self.U2]
        lV = [self.V2,self.V1]

        U = [lU[lang_ids[0]],lU[lang_ids[1]]]
        V = [lV[lang_ids[0]],lV[lang_ids[1]]]
        v = self.vocab[lang_ids[1]].v
        v_ids = self.vocab[lang_ids[0]].v_ids
        filtered = [self.filtered_vocab[lang_ids[0]],self.filtered_vocab[lang_ids[1]]]

        if w not in v_ids or not filtered[0][v_ids[w]]:
            sys.exit(f"{w} is not in the {lang1} filtered dictionary")
        else:
            word_index = v_ids[w]
        
        if method == 'U+V':
            added_word_vector = normalize_matrix(U[0][word_index]+V[0][word_index])
            added_vsm_matrices = normalize_matrix(U[1][filtered[1]]+V[1][filtered[1]])
            filtered_similarities = np.matmul(added_word_vector,added_vsm_matrices.T)
        else:
            matrix = [None,None]
            for i in range(2):
                if method.replace('-','')[i] == 'U':
                    matrix[i] = U[i]
                elif method.replace('-','')[i] == 'V':
                    matrix[i] = V[i]
                else:
                    sys.exit(f'Invalid method {method}')
            filtered_similarities = np.matmul(normalize_matrix(matrix[0][word_index]),normalize_matrix(matrix[1][filtered[1]]).T)
        
        sorted_similarities = sorted(zip(filtered_similarities,np.where(filtered[1])[0]),reverse=True)
        if n == 0:
            n = len(filtered_similarities)
        return [[sorted_similarities[i][0],v[sorted_similarities[i][1]]] for i in range(n)]
        
    def most_similar(self,w,lang,n=10,return_results=False,methods=None):
        results = {}
        if methods is None:
            methods = ['U-U','U-V','U+V']
        for m in methods:
            results[m] = self.compare_vector_to_matrix(w,lang,method=m,n=n)
        if return_results:
            return results
        else:
            print(''.join(['%-30s'%(f'           {m}') for m in methods]) )
            for i in range(n):
                print(';'.join([' %2.5f %20s '%(results[m][i][0],results[m][i][1]) for m in methods]))

    def bilingual_most_similar(self,w,lang,n=10,return_results=False,methods=None):
        if lang == self.vocab[0].lang:
            lang2 = self.vocab[1].lang
        elif lang == self.vocab[1].lang:
            lang2 = self.vocab[0].lang
        else:
            sys.exit(f'Wrong language selected: {lang}. Available languages: {self.vocab[0].lang}, {self.vocab[1].lang}')
        
        method_labels = {'U-V': 'l1 source <-> l2 target'.replace('l1',lang).replace('l2',lang2),
                        'V-U': 'l1 target <-> l2 source'.replace('l1',lang).replace('l2',lang2),
                        'U+V': 'l1 sum <-> l2 sum'.replace('l1',lang).replace('l2',lang2),
                        'U-U': 'l1 source <-> l2 source'.replace('l1',lang).replace('l2',lang2),
                        'V-V': 'l1 target <-> l2 target'.replace('l1',lang).replace('l2',lang2)}

        if return_results:
            if methods is None:
                methods = method_labels.keys()
            return({m: self.compare_vector_to_matrix(w,lang,lang2,method=m,n=n) for m in methods})
        else:
            if methods is None:
                methods = ['U-V','V-U','U+V']
            results = {m: self.compare_vector_to_matrix(w,lang,lang2,method=m,n=n) for m in methods}
        
            print(''.join([' %-30s'%method_labels[m] for m in methods]))
            for i in range(n):
                print(';'.join([' %2.5f %20s '%(results[m][i][0],results[m][i][1]) for m in methods]))
    
    def compare_ranks(self,w1,w2,lang,return_results=False,methods=None):
        similarities = {}
        ranks_w1 = {}
        ranks_w2 = {}
        if methods is None:
            methods = ['U-U','U-V','U+V']
        for method in methods:
            similarities[method] = self.compare_vectors(w1,w2,lang,method = method)
            sorted_similarities = self.compare_vector_to_matrix(w1,lang,method = method)
            for i in range(len(sorted_similarities)):
                if sorted_similarities[i][1] == w2:
                    ranks_w1[method] = i + 1
                    break
            sorted_similarities = self.compare_vector_to_matrix(w2,lang,method = method)
            for i in range(len(sorted_similarities)):
                if sorted_similarities[i][1] == w1:
                    ranks_w2[method] = i + 1
                    break
        if return_results:
            return {m: [similarities[m],ranks_w1[m],ranks_w2[m]] for m in methods}
        else:
            for m in methods:
                print('%s: similarity %1.5f, rank of %s for %s %d, rank of %s for %s %d'%(m,similarities[m],w2,w1,ranks_w1[m],w1,w2,ranks_w2[m]))

    def bilingual_compare_ranks(self,w1,w2,lang1='.',lang2='.',return_results=False,methods=None):
        if lang1 == '.':
            lang1 = self.vocab[0].lang
        if lang2 == '.':
            lang2 = self.vocab[1].lang

        similarities = {}
        ranks_w1 = {}
        ranks_w2 = {}
        if methods is None:
            methods = ['U-V','V-U','U+V']
        for method in methods:
            similarities[method] = self.compare_vectors(w1,w2,lang1,lang2,method = method)
            sorted_similarities = self.compare_vector_to_matrix(w1,lang1,lang2,method = method)
            for i in range(len(sorted_similarities)):
                if sorted_similarities[i][1] == w2:
                    ranks_w1[method] = i + 1
                    break
            sorted_similarities = self.compare_vector_to_matrix(w2,lang2,lang1,method = method)
            for i in range(len(sorted_similarities)):
                if sorted_similarities[i][1] == w1:
                    ranks_w2[method] = i + 1
                    break
        if return_results:
            return {m: [similarities[m],ranks_w1[m],ranks_w2[m]] for m in methods}
        else:
            for m in methods:
                print('%s: similarity %1.5f, rank of %s for %s %d, rank of %s for %s %d'%(m,similarities[m],w2,w1,ranks_w1[m],w1,w2,ranks_w2[m]))

    def train(self,corpus,target_center_method = 'linear',window_size = 20, dynamic_window = False, epochs = 1, lr_decay = True, num_neg_samples = 5, lr = 0.05, l2_lambda = 0, normalize_vectors = '', shuffled = 0, epoch_callback=None):
        if corpus.vsm != self:
            sys.exit('The training corpus was not set up for this vector space model')
        if corpus.vsm_min_count != self.min_count:
            sys.exit('Error: The minimum token count threshold of the VSM was changed, the corpus should be recompiled')

        initial_lr = lr
        warning_count = 0

        self.vocab[0].set_subsample_threshold(self.vocab[0].subsample_threshold)
        self.vocab[1].set_subsample_threshold(self.vocab[1].subsample_threshold)

        filtered_indices = (np.where(self.filtered_vocab[0])[0], np.where(self.filtered_vocab[1])[0])
        filtered_cumprob = (np.cumsum(self.subsampled_counts[0][filtered_indices[0]]) / sum(self.subsampled_counts[0][filtered_indices[0]]),
                            np.cumsum(self.subsampled_counts[1][filtered_indices[1]]) / sum(self.subsampled_counts[1][filtered_indices[1]]))

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
                epoch_center_tokens[0] += len(doc[0])
                epoch_center_tokens[1] += len(doc[1])
                
                for direction in range(2):
                    if direction == 0:
                        U = self.U1
                        V = self.V1
                        source_doc = doc[0]
                        target_doc = doc[1]
                        lang_pair_string = f'{self.vocab[0].lang}->{self.vocab[1].lang}'
                    else:
                        U = self.U2
                        V = self.V2
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
                        updated_V = list()

                    for ti in range(len(source_doc)):
                        i = source_word_index[ti]
                        if long_target:
                            if dynamic_window:
                                target_word_indices = np.arange(max(0,target_centers[i]-windows[i]),min(len(target_doc),target_centers[i]+windows[i]))
                            else:
                                target_word_indices = np.arange(max(0,target_centers[i]-window_size),min(len(target_doc),target_centers[i]+window_size))
                            target_words = target_doc[target_word_indices]
                        neg_rand = np.random.rand(len(target_words) * num_neg_samples)
                        neg_samples = [filtered_indices[1-direction][cumulative_binary_search(filtered_cumprob[1-direction],r)] for r in neg_rand]
                        neg_samples_hash = defaultdict(int)
                        for n in neg_samples:
                            neg_samples_hash[n] += 1
                        neg_sample_ids = [*neg_samples_hash.keys()]
                        source_id = source_doc[i]

                        cns = list(target_words)+neg_sample_ids
                        y = np.array([1]*len(target_words)+[0]*len(neg_sample_ids))

                        yhat = scipy.special.expit(np.dot(U[[source_id]],V[cns].T))
                        update = (yhat - np.array([1]*len(target_words)+[0]*len(neg_sample_ids))) * lr / num_neg_samples * ([1] * len(target_words) + list(neg_samples_hash.values()))
                        Unew = U[[source_id]] * (1 - l2_lambda) - np.dot(update, V[cns])
                        V[cns] = V[cns] * (1 - l2_lambda) - np.dot(update.T, U[[source_id]]) * np.array([[1]*len(target_words)+list(neg_samples_hash.values())]).T
#                        Unew = U[[source_id]] * (1 - l2_lambda) - np.dot(update, V[cns])
#                        V[cns] -= np.dot(update.T, U[[source_id]])
                        U[[source_id]] = Unew

                        with warnings.catch_warnings():
                            try:
                                epoch_loss[direction] -= np.sum(y * np.log(yhat) + (1-y) * np.log(1-yhat))
                                epoch_tokens[direction] += len(cns)
                            except Warning:
                                warning_count += 1
                                if warning_count < 6:
                                    warning_data = {
                                        'source_doc': source_doc,
                                        'target_words': target_words,
                                        'i': i,
                                        'neg_sample_ids': neg_sample_ids,
                                        'yhat': yhat,
                                        'y': y
                                    }
                                    with open(f'warning{warning_count}.pickle','wb') as outfile:
                                        pickle.dump(warning_data,outfile)
                                    print('Warning caught')
                                    print('y:',y)
                                    print('yhat:',yhat)
                        
                        if normalize_vectors == 'document':
                            updated_V += cns

                    if (di + 1) % 1000 == 0:
                        print(f'{di+1} / {len(corpus.doc_pairs_list)} done, {lang_pair_string} avg loss: {epoch_loss[direction] / epoch_tokens[direction]}, learning rate: {lr}')
                    if normalize_vectors == 'document':
                        updated_U = list(set(source_doc))
                        updated_V = list(set(updated_V))
                        U[updated_U] = normalize_matrix(U[updated_U])
                        V[updated_V] = normalize_matrix(V[updated_V])
                if lr_decay:
                    lr -= initial_lr * raw_doc_length / (corpus.token_count[0] * epochs)
            if normalize_vectors == 'epoch': 
                self.normalize()
            if epoch_callback:
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
        print('Total warnings:', warning_count)

    def summary(self):
        return {'id': self.id, 'vector space dimension': self.vsm_dim, 'vocabulary 1 info': self.vocab[0].summary(),
                'vocabulary 2 info': self.vocab[1].summary(), 'rare word minimum thresholds': self.min_count,
                'filtered vocabulary sizes': (sum(self.filtered_vocab[0]),sum(self.filtered_vocab[1])),
                'context smoothing exponent': self.subsample_exponent, 'trained': self.trained}
                
class BilingualVSM_Corpus:
    def __init__(self,vsm):
        self.doc_pairs_list = list()
        self.file_list = list()
        self.vsm = vsm
        self.vsm_min_count = vsm.min_count
        self.token_count = [0,0]
        self.id = ''.join(choices(string.ascii_letters + string.digits, k=8))
        
    def __len__(self):
        return len(self.doc_pairs_list)

    def __getitem__(self,key):
        return self.doc_pairs_list[key]
    
    def __iter__(self):
        for dp in self.doc_pairs_list:
            yield dp

    def save_data(self):
        data = {
            'id': self.id,
            'vsm_id': self.vsm.id,
            'vsm_min_count': self.vsm_min_count,
            'file_list': self.file_list,
            'token_count': self.token_count
        }
        self.vsm.save_data()
        print(f'Saving {self.vsm.vocab[0].lang}-{self.vsm.vocab[1].lang} bilingual corpus {self.id}')
        with open(self.id+'.'+self.vsm.vocab[0].lang+'-'+self.vsm.vocab[1].lang+'.corpus.json','w') as outfile:
            outfile.write(json.dumps(data))
        with open(self.id+'.'+self.vsm.vocab[0].lang+'-'+self.vsm.vocab[1].lang+'.corpus.pickle','wb') as outfile:
            pickle.dump(self.doc_pairs_list,outfile)

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
        new_vsm = BilingualVectorSpaceModel(Vocabulary(lang1),Vocabulary(lang2))
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
                        if self.vsm.vocab[i].lemmatize:
                            filtered_tokens[i] = [self.vsm.vocab[i].form_to_lemma[t] for t in bisegment.get_tokens(l,lower=self.vsm.vocab[i].lower) if t in self.vsm.vocab[i].form_to_lemma[t] and self.vsm.filtered_vocab[i][self.vsm.vocab[i].form_to_lemma[t]]]
                        else:
                            filtered_tokens[i] = [self.vsm.vocab[i].v_ids[t] for t in bisegment.get_tokens(l,lower=self.vsm.vocab[i].lower) if t in self.vsm.vocab[i].v_ids and self.vsm.filtered_vocab[i][self.vsm.vocab[i].v_ids[t]]]
#                        filtered_tokens[i] = [self.vsm.vocab[i].v_ids[t] for t in bisegment.get_tokens(l,lower=self.vsm.vocab[i].lower) if self.vsm.filtered_vocab[i][self.vsm.vocab[i].v_ids[t]]]
                        self.token_count[i] += len(filtered_tokens[i])

                    self.doc_pairs_list.append((np.array(filtered_tokens[0]),np.array(filtered_tokens[1])))
            if verbose and ((n + 1) % 1000 == 0):
                print('Added %d / %d files'%(n+1,len(bisegmentations)))
            
    def get_langs(self):
        return (self.vsm.vocab[0].lang,self.vsm.vocab[1].lang)
    
    def get_tokens(self,doc_nr,lang=-1):
        if lang in [0,1]:
            lang_index = lang
        elif lang == -1:
            return [self.get_tokens(doc_nr,lang=0),self.get_tokens(doc_nr,lang=1)]
        elif lang in self.get_langs():
            lang_index = self.get_langs().index(lang)
        else:
            raise Exception('Invalid language')
        return [self.vsm.vocab[lang_index].v[wid] for wid in self.doc_pairs_list[doc_nr][lang_index]]
    
    def summary(self):
        return {'id': self.id, 'vector space model info': self.vsm.summary(),
                'number of files': len(self.file_list), 'number of bisegments': len(self.doc_pairs_list),
                'number of tokens': {self.vsm.vocab[0].lang: self.token_count[0], self.vsm.vocab[1].lang: self.token_count[1]}}
