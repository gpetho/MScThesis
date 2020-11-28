import numpy as np, re, scipy.stats, pickle

def calculate_num_diff_costs(list_of_standard_bisegmentations,upper_limit):
    num_differences = [bs.number_numbering_diff() for b in list_of_standard_bisegmentations for bs in b]
    num_diff_counts = np.array([[numdiff,num_differences.count(numdiff)] for numdiff in set(num_differences)]).transpose()
    popt, pcov = curve_fit(scipy.stats.nbinom.pmf, num_diff_counts[0], num_diff_counts[1] / len(num_differences), p0=[0.5,0.9])
    print(popt)
    for i in range(num_diff_counts[0].max()+1):
        if i in num_diff_counts[0]:
            print(i, scipy.stats.nbinom.pmf(i,popt[0],popt[1]) * len(num_differences), num_diff_counts[1][np.where(num_diff_counts[0] == i)][0])
            # print(scipy.stats.poisson.pmf(i, sum(num_diff_counts[0]*num_diff_counts[1]) / sum(num_diff_counts[1])) * len(num_differences))
    return [-np.log(scipy.stats.nbinom.pmf(i,popt[0],popt[1])) for i in range(upper_limit)]

def num_diff_cost(num_diff):
    if num_diff < 6:
        return [0, 3.435, 5.032, 6.383, 7.631, 8.82][num_diff]
    else:
        return num_diff + 4
            
STEP_FREQ = {
    (1,1): 1,
    (0,1): 1/85000,
    (1,0): 1/85000,
    (1,2): 0.0118,
    (2,1): 0.0058,
    (2,2): 0.001,
    (3,1): 2/8500,
    (1,3): 2/85000,
    (3,2): 1/8500,
    (2,3): 1/85000
}

STEP_COSTS = {}
for s,f in STEP_FREQ.items():
    STEP_COSTS[s] = -np.log(f)
    
STEPS = [k for k, _ in sorted(STEP_COSTS.items(), key=lambda item: item[1])]

NUMBERING_SET = set('bcdfghjklmnqr').union(set(['ii','iii','iv','v','vi','vii','viii','ix','x']))

class Segmentation:
    def __init__(self, lang, txt = None, file = None, boundary_symbol = '¤'):
        if txt == None:
            with open(file,encoding='utf8') as f:
                txt = f.read()
        self.boundary_symbol = boundary_symbol
        self.segments = [Segment(t) for t in txt.split(boundary_symbol)]
        self.lang = lang
        self.file = file

    def __len__(self):
        return len(self.segments)

    def __str__(self):
        return '\n'.join(["%d: %s"%(i,str(s).strip()) for i, s in enumerate(self.segments)])
    
    def __getitem__(self,key):
        return self.segments[key]
    
    def __iter__(self):
        for i in self.segments:
            yield i
    
    def get_text(self, boundaries = False):
        if boundaries:
            bd_symbol = self.boundary_symbol
        else:
            bd_symbol = ''
        return bd_symbol.join([str(s) for s in self.segments])
    
    def get_text_length(self, boundaries = False):
        bds = len(self.segments) * int(boundaries)
        return sum([len(str(s)) for s in self.segments]) + bds

    def get_combined_lengths(self, max_combined_length = 3):
        combined_lengths = np.zeros([len(self.segments),max_combined_length+1],dtype=int)
        for j in range(len(combined_lengths)):
            length_sum = 0 
            for k in range(1,max_combined_length+1):
                if j + k - 1 == len(combined_lengths):
                    break
                length_sum += len(self.segments[j+k-1])
                combined_lengths[j][k] = length_sum
        return combined_lengths

    def align(self,other,d_model='gc',use_anchors=False,exact_anchor_segments = None, anchor_pattern_list = None):
        return Alignment(self,other,d_model,use_anchors,exact_anchor_segments,anchor_pattern_list)

class shifted_array:
    def __init__(self,m,n,m_shift,n_shift,m_dense,n_dense,fill_value = np.inf,dtype = np.float64):
        self.hidden = np.full((m,n), fill_value, dtype)
        self.m = m
        self.n = n
        self.m_shift = m_shift
        self.n_shift = n_shift
        self.m_dense = m_dense
        self.n_dense = n_dense
        self.fill_value = fill_value
        self.dtype = dtype

    def __getitem__(self,key):
        if key[1]+self.n_shift-key[0] >= 0 and key[1]+self.n_shift-key[0] < self.n:
            return self.hidden[key[0],key[1]+self.n_shift-key[0]]
        else:
            return np.full((1,),fill_value = self.fill_value, dtype = self.dtype)[0]
    
    def __setitem__(self,key,value):
        self.hidden[key[0],key[1]+self.n_shift-key[0]] = value
    
    def has_key(key):
        return key[1]+self.n_shift-key[0] > 0 and key[1]+self.n_shift-key[0] < self.n
    
    def to_numpy(self):
        return_array = np.full((self.m_dense,self.n_dense),fill_value=self.fill_value,dtype = self.dtype)
        for i in range(self.m_dense):
            for j in range(self.n_dense):
                return_array[i,j] = self[i,j]
        return return_array
    
class Alignment:
    def __init__(self, sl1, sl2, d_model = 'gc_numbering',use_anchors = True, exact_anchor_segments = None, anchor_pattern_list = None):
        combined_lengths = [sl1.get_combined_lengths(), sl2.get_combined_lengths()]
        langs = [sl1.lang,sl2.lang]
        self.segmentations = {sl1.lang: sl1, sl2.lang: sl2}
        self.anchors = list()
        self.number_of_anchors = 0
        if use_anchors:
            anchors_temp = list()
            anchor_candidates = {langs[0]:{},langs[1]:{}}
            for l in langs:
                for i in range(len(self.segmentations[l])):
                    if len(str(self.segmentations[l][i])) < 70:
                        t = str(self.segmentations[l][i]).strip().lower()
                        if t in anchor_candidates[l]:
                            anchor_candidates[l][t] = None
                        else:
                            anchor_candidates[l][t] = i
            l2_previous = 0
            for k, v in anchor_candidates[langs[0]].items():
                if v == None:
                    continue
                if k in exact_anchor_segments:
                    l2_index = -1
                    for l2_anchor in exact_anchor_segments[k]:
                        if l2_anchor in anchor_candidates[langs[1]]:
                            if l2_index >= 0 or anchor_candidates[langs[1]][l2_anchor] == None or anchor_candidates[langs[1]][l2_anchor] < l2_previous:
                                l2_index = -1
                                break
                            else:
                                l2_index = anchor_candidates[langs[1]][l2_anchor]
                    if l2_index >= 0:
                        anchors_temp.append((v,l2_index))
                        l2_previous = l2_index
            #                print('(%d,%d): %s / %s'%(v, hu_index, str(b.segmentations['en'][v]), str(b.segmentations['hu'][hu_index])))
                else:
                    for item in anchor_pattern_list:
                        if len(k) < item['minl']:
                            break
                        if len(k) <= item['maxl']:
                            match = item['l1_pattern'].fullmatch(k)
                            if match == None:
                                continue
                            match_groups = [match[i] for i in item['match_indices']]
                            l2_anchor = item['l2_format'].format(*match_groups)
                            l2_index = -1
                            if l2_anchor in anchor_candidates[langs[1]] and anchor_candidates[langs[1]][l2_anchor] != None and anchor_candidates[langs[1]][l2_anchor] > l2_previous:
                                l2_index = anchor_candidates[langs[1]][l2_anchor]
                            elif k in anchor_candidates[langs[1]] and anchor_candidates[langs[1]][k] != None and anchor_candidates[langs[1]][k] > l2_previous:
                                l2_index = anchor_candidates[langs[1]][k]
                            if l2_index >= 0:
                                anchors_temp.append((v,l2_index))
                                l2_previous = l2_index
            #                       print('(%d,%d): %s / %s'%(v, hu_index, str(b.segmentations['en'][v]), str(b.segmentations['hu'][hu_index])))
                            break
 #           print(anchors_temp)
 #           print(len(anchors_temp), 'found, en:', len(sl1), 'hu:', len(sl2))
            anchors_amended = list()
            self.number_of_anchors = len(anchors_temp)
            omit = False
            for i in range(1,len(anchors_temp)):
                if omit:
                    omit = False
                    continue
                if anchors_temp[i-1][0] == anchors_temp[i][0]-1 and anchors_temp[i-1][1] != anchors_temp[i][1]-1:
                    omit = True
                    self.number_of_anchors -= 2
#                    print('Problem:', anchors_temp[i-1], anchors_temp[i])
                elif anchors_temp[i-1][0] != anchors_temp[i][0]-1 and anchors_temp[i-1][1] == anchors_temp[i][1]-1:
                    omit = True
                    self.number_of_anchors -= 2
#                    print('Problem:', anchors_temp[i-1], anchors_temp[i])
                elif anchors_temp[i-1][0] == anchors_temp[i][0]-2 and anchors_temp[i-1][1] == anchors_temp[i][1]-2:
                    anchors_amended.append(anchors_temp[i-1])
                    anchors_amended.append((anchors_temp[i][0]-1,anchors_temp[i][1]-1))
                else:
                    anchors_amended.append(anchors_temp[i-1])
            if len(anchors_temp) and not omit:
                anchors_amended.append(anchors_temp[-1])
            self.anchors = anchors_amended
 #           print('Amended:', len(anchors_amended), anchors_amended)

        self.anchors.insert(0,(-1,-1))
        self.anchors.append((len(sl1),len(sl2)))

        self.bisegments = list()
        self.partial_alignments = {}
        for i in range(len(self.anchors)-1):
            if i != 0:
                self.bisegments.append(((self.anchors[i][0]+1,1),(self.anchors[i][1]+1,1)))
            if self.anchors[i + 1][0] - self.anchors[i][0] > 1:
#                print('align((%d,%d),(%d,%d))'%(self.anchors[i][0]+1, self.anchors[i][1]+1, self.anchors[i+1][0], self.anchors[i+1][1]))
                a = PartialAlignment(sl1,sl2,combined_lengths,start_segments = (self.anchors[i][0]+1,self.anchors[i][1]+1), end_segments = (self.anchors[i+1][0],self.anchors[i+1][1]), d_model = d_model)
                self.bisegments += a.bisegments
                self.partial_alignments[((self.anchors[i][0]+1,self.anchors[i][1]+1),(self.anchors[i+1][0]-1,self.anchors[i+1][1]-1))] = a
        self.anchors.pop(0)
        self.anchors.pop(-1)

    def get_bisegmentation(self):
        langs = list(self.segmentations.keys())
        return Bisegmentation([self.segmentations[langs[0]],self.segmentations[langs[1]]],self.bisegments)
    
    def get_languages(self):
        return list(self.segmentations.keys())
    
class PartialAlignment:
    def __init__(self,sl1,sl2,combined_lengths,start_segments = (0,0), end_segments = (-1,-1), d_model = 'gc'):
        self.segmentations = [sl1,sl2]
        if end_segments == (-1,-1):
            end_segments = (len(sl1),len(sl2))

        section_lengths = (end_segments[0]-start_segments[0],end_segments[1]-start_segments[1])
        
        margin = int(np.log(section_lengths[0]) * 2)
        window_l2 = max(0,section_lengths[1] - section_lengths[0]) + margin + 1
        window_l1 = max(0,section_lengths[0] - section_lengths[1]) + margin + 1
        
        if window_l1 + window_l2 >= section_lengths[1]+1:
            self.best_previous = np.zeros([section_lengths[0]+1,section_lengths[1]+1],dtype=(int,2))
            self.D = np.full([section_lengths[0]+1,section_lengths[1]+1],fill_value = np.inf,dtype=np.float64)
#            print('dense array')
        else:
            self.best_previous = shifted_array(section_lengths[0]+1,window_l1+window_l2,0,window_l1,section_lengths[0]+1,section_lengths[1]+1,fill_value = 0,dtype=(np.int,2))
            self.D = shifted_array(section_lengths[0]+1,window_l1+window_l2,0,window_l1,section_lengths[0]+1,section_lengths[1]+1)
#            print('shifted array')
        self.D[0,0] = 0

        for l1_index in range(start_segments[0],end_segments[0]+1):
            for l2_index in range(max(start_segments[1],start_segments[1]+(l1_index-start_segments[0])-window_l1),min(end_segments[1]+1,start_segments[1]+(l1_index-start_segments[0])+window_l2)):
                if l1_index == start_segments[0] and l2_index == start_segments[1]:
                    continue
#                print('(%d,%d)'%(l1_index,l2_index))
                for step in STEPS:
                    si = l1_index - step[0]
                    sj = l2_index - step[1]
                    
                    if si - start_segments[0] < 0 or sj -start_segments[1] < 0:
                        continue

                    if self.D[si-start_segments[0],sj-start_segments[1]] + STEP_COSTS[step] >= self.D[l1_index-start_segments[0],l2_index-start_segments[1]]:
                        continue
                    if min(step[0],step[1]) == 0:
                        self.D[l1_index-start_segments[0],l2_index-start_segments[1]] = self.D[si-start_segments[0],sj-start_segments[1]] + STEP_COSTS[step]
                        self.best_previous[l1_index-start_segments[0],l2_index-start_segments[1]] = (si,sj)
                    else:
                        d = PartialAlignment.calculate_d(si,sj,step,sl1,sl2,combined_lengths,d_model)
                        if d == np.inf:
                            continue
                        if self.D[si-start_segments[0],sj-start_segments[1]] + d < self.D[l1_index-start_segments[0],l2_index-start_segments[1]]:
                            self.D[l1_index-start_segments[0],l2_index-start_segments[1]] = self.D[si-start_segments[0],sj-start_segments[1]] + d
                            self.best_previous[l1_index-start_segments[0],l2_index-start_segments[1]] = (si,sj)
        
        self.best_path = list()
        self.bisegments = list()
        (i,j) = (end_segments[0],end_segments[1])
#        print(self.D.to_numpy())
#        print(self.best_previous.to_numpy())
        while (i-start_segments[0],j-start_segments[1]) != (0,0):
#            print(i,j)
#            print(start_segments[0],start_segments[1])
 #           print(i-start_segments[0],j-start_segments[1])
            self.best_path.append((i,j))
            (si,sj) = self.best_previous[i-start_segments[0],j-start_segments[1]]
 #           print('(%d,%d)'%(si,sj))
            l1_length = i-si
            l2_length = j-sj
            if l1_length == 0:
                l1_side = None
            else:
                l1_side = (i+1-l1_length,l1_length)
            if l2_length == 0:
                l2_side = None
            else:
                l2_side = (j+1-l2_length,l2_length)
            self.bisegments.append((l1_side,l2_side))
            (i,j) = (si,sj)
        self.best_path.reverse()
        self.bisegments.reverse()

    def negative_log_probability_by_length(en_length,hu_length):
        C = 1.0753
        hu_en_avg_length = (hu_length + en_length) / 2
        delta = (hu_length - en_length * C) / np.sqrt(max(hu_en_avg_length, (6.31 * hu_en_avg_length - 195)))
        if abs(delta) > 8.28: # stats.norm.cdf returns 1
            return np.inf
        phi_delta = 1 - scipy.stats.norm.cdf(abs(delta))
        return -np.log(phi_delta * 2)

    def calculate_d(si,sj,step,sl1,sl2,combined_lengths,model='gc'):
        d_gc = PartialAlignment.negative_log_probability_by_length(combined_lengths[0][si][step[0]],combined_lengths[1][sj][step[1]])
        if d_gc == np.inf:
            return d_gc
    #    print('GC cost:', d_gc)
        if model == 'gc_numbering':
            si_set = set()
            for i in range(si,si+step[0]):
                si_set.update(sl1[i].get_token_set('num_numbering'))
            sj_set = set()
            for j in range(sj,sj+step[1]):
                sj_set.update(sl2[j].get_token_set('num_numbering'))
            num_diff = len(si_set.symmetric_difference(sj_set))
            d_gc += num_diff_cost(num_diff)
    #        print('num diff cost for', num_diff, 'diffs:', num_diff_cost(num_diff))
        d_gc += STEP_COSTS[step]
    #    print('step cost:', STEP_COSTS[step])
    #    print('total cost:', d_gc)
        return d_gc

    def get_bisegments(self):
        return self.bisegments
    
    def get_languages(self):
        return [self.segmentations[0].lang, self.segmentations[1].lang]

class Segment:
    def __init__(self,text):
        self.text = text
        self.tokens = np.array([w for w in re.split('(\W)',text) if len(w) and not w.isspace()])
        num_num_set = (self.get_token_set('all',lower=True).intersection(NUMBERING_SET)).union(self.get_token_set('numeric'))
        all_token_list = np.array(self.tokens)
        mixed_set = set(all_token_list[np.char.isalnum(all_token_list)
                                      & ~np.char.isalpha(all_token_list)
                                      & ~np.char.isnumeric(all_token_list)])
        self.num_num_mixed_set = num_num_set.union(mixed_set)

    def __len__(self):
        return len(self.text)

    def __str__(self):
        return self.text

    def get_tokens(self, what = 'all', lower = False):
        if what == 'alpha':
            return_tokens = self.tokens[np.char.isalpha(self.tokens)]
        elif what == 'numeric':
            return_tokens = self.tokens[np.char.isnumeric(self.tokens)]
        else:
            return_tokens = self.tokens
        if lower:
            return_tokens = np.array([t.lower() for t in return_tokens])
        return return_tokens

    def get_token_set(self,what = 'alpha', lower = True):
        if what == 'num_numbering':
            return self.num_num_mixed_set
        else:
            return set(self.get_tokens(what,lower))

class Bisegment:
    def __init__(self,bisegment,segmentations):
        self.bisegment = bisegment
        self.segmentations = {s.lang: s for s in segmentations}
        self.langs = [s.lang for s in segmentations]
        self.start = {l: self.bisegment[i][0] - 1 for i, l in enumerate(self.langs) if self.bisegment[i] != None}
        self.end = {l: self.start[l] + self.bisegment[i][1] for i, l in enumerate(self.langs) if self.bisegment[i] != None}
        
    def get_segments(self,lang):
        if lang in self.start:
            return {i: self.segmentations[lang][i] for i in range(self.start[lang],self.end[lang])}
        else:
            return {}

    def get_combined_length(self,lang):
        return sum([len(s) for s in self.get_segments(lang).values()])
    
    def get_length_diff(self,l1,l2):
        return self.get_combined_length(l1) - self.get_combined_length(l2) 
    
    def get_link_type(self):
        return (len(self.get_segments(self.langs[0])),len(self.get_segments(self.langs[1])))

    def get_tokens(self,lang,what='all',lower=False):
        tokens = list()
        if lang in self.start:
            for i in range(self.start[lang],self.end[lang]):
#                tokens += self.segmentations[lang][i].get_tokens(what,lower) # a jobb oldalt be kell tenni egy list()-be, egyébként hibás, mert np.array
                tokens += list(self.segmentations[lang][i].get_tokens(what,lower))
        return tokens

    def get_token_set(self,lang,what='alpha',lower = True):
        token_set = set()
        if lang in self.start:
            for i in range(self.start[lang],self.end[lang]):
                token_set.update(self.segmentations[lang][i].get_token_set(what,lower))
        return token_set
    
    def number_numbering_diff(self):
        return len(self.get_token_set(self.langs[0],what = 'num_numbering').symmetric_difference(self.get_token_set(self.langs[1],what = 'num_numbering')))
        
    def __str__(self):
        return ('\n%d <'%self.get_combined_length(self.langs[0])+'-'*20+'> %d\n'%self.get_combined_length(self.langs[1])).join(['\n'.join(['<%s.%d>: %s'%(lang,index+1,str(segment_text).strip()) for index, segment_text in self.get_segments(lang).items()]) for lang in self.langs ])

class Bisegmentation:
    def __init__(self,segmentations,bisegments):
        self.segmentations = {s.lang: s for s in segmentations}
        self.langs = [s.lang for s in segmentations]
        self.bisegment_list = bisegments
        self.bisegment_objects = [Bisegment(bs,segmentations) for bs in bisegments]                
        self.bisegment_lengths = {l: [bs.get_combined_length(l) for bs in self.bisegment_objects] for l in self.langs}

        for l in range(2):
            filtered_segment_ranges = [b[l] for b in bisegments if b[l] != None]
            for i in range(1,len(filtered_segment_ranges)):
                if sum(filtered_segment_ranges[i-1]) != filtered_segment_ranges[i][0]:
                    print('Error in bisegmentation on %s side: %s followed by %s'%(self.langs[l],str(filtered_segment_ranges[i-1]),filtered_segment_ranges[i]))
            if filtered_segment_ranges[0][0] != 1:
                print('Error: Bisegmentation on %s side begins with segment %d'%(self.langs[l],filtered_segment_ranges[0][0]))
            if sum(filtered_segment_ranges[-1])-1 != len(segmentations[l]):
                print('Length mismatch in bisegmentation on %s side: segmentation length is %d but the last segment in the bisegmentation is %d'%(self.langs[l],len(segmentations[l]),sum(filtered_segment_ranges[-1])-1))

    def create_from_files(langs,files,bisegment_list):
        return Bisegmentation([Segmentation(langs[0],file=files[0]), Segmentation(langs[1],file=files[1])],bisegment_list)
    
    def __len__(self):
        return len(self.bisegment_objects)

    def __getitem__(self,key):
        return self.bisegment_objects[key]
    
    def __iter__(self):
        for o in self.bisegment_objects:
            yield o

    def __str__(self):
        return '\n\n'.join([str(bs) for bs in self.bisegment_objects])
    
    def get_segmentation(self,lang):
        return self.segmentations[lang]
    
    def count_link_types(self):
        link_type_list = [b.get_link_type() for b in self]
        link_types = set(link_type_list)
        link_type_counts = {lt: link_type_list.count(lt) for lt in link_types}
        return link_type_counts

class BisegmentationCorpus:
    def __init__(self,langs):
        self.bisegmentation_list = []
        self.langs = langs
    def add_bisegmentation(self,bs):
        if isinstance(bs,Bisegmentation) and bs.langs == self.langs:
            self.bisegmentation_list.append(bs)
    def save_data(self,pickle_file):
        data = {
                'langs': self.langs,
                'bisegmentations': [
                                    {'files': [b.segmentations[self.langs[0]].file,b.segmentations[self.langs[1]].file],
                                    'bisegment_list': b.bisegment_list}
                                    for b in self.bisegmentation_list
                                    ]
               }
        with open(pickle_file,'wb') as outfile:
            pickle.dump(data,outfile)
    def create_from_pickle(pickle_file):
        with open(pickle_file,'rb') as infile:
            data = pickle.load(infile)
        new_bsc = BisegmentationCorpus(data['langs'])
        for b in data['bisegmentations']:
            new_bsc.add_bisegmentation(Bisegmentation.create_from_files(new_bsc.langs,b['files'],b['bisegment_list']))
        return new_bsc
    
    def __len__(self):
        return len(self.bisegmentation_list)

    def __getitem__(self,key):
        return self.bisegmentation_list[key]
    
    def __iter__(self):
        for b in self.bisegmentation_list:
            yield b
            
    def total_segment_count(self):
        return sum([len(b) for b in self])
    
    def total_token_count(self):
        return {self.langs[0]: sum([sum([len(bisegment.get_tokens(self.langs[0])) for bisegment in b]) for b in self]),
                self.langs[1]: sum([sum([len(bisegment.get_tokens(self.langs[1])) for bisegment in b]) for b in self])}
    
def bisegments_to_individual_links(bisegments):
    individual_links = list()
    for b in bisegments:
        if b[0] == None:
            for l2_segment in range(b[1][0],b[1][0]+b[1][1]):
                individual_links.append((None,l2_segment))
        elif b[1] == None:
            for l1_segment in range(b[0][0],b[0][0]+b[0][1]):
                individual_links.append((l1_segment,None))
        else:
            for l1_segment in range(b[0][0],b[0][0]+b[0][1]):
                for l2_segment in range(b[1][0],b[1][0]+b[1][1]):
                    individual_links.append((l1_segment,l2_segment))
    return individual_links

exact_anchor_segments_jox_en_hu = {'whereas:': ['mivel:'],
 'parties': ['felek'],
 'the president': ['az elnök'],
 'for the commission': ['a bizottság részéről'],
 '(text with eea relevance)': ['(egt-vonatkozású szöveg)', '(egt vonatkozású szöveg)'],
 'has adopted this regulation:': ['elfogadta ezt a rendeletet:'],
 'the european commission,': ['az európai bizottság,'],
 'annex': ['melléklet'],
 'pleas in law and main arguments': ['jogalapok és fontosabb érvek', 'jogalapok és fontosabb érvek:'],
 'has adopted this decision:': ['elfogadta ezt a határozatot:'],
 'the european parliament,': ['az európai parlament,'],
 'parties to the main proceedings': ['az alapeljárás felei'],
 'language of the case: french': ['az eljárás nyelve: francia'],
 'member of the commission': ['a bizottság tagja'],
 'form of order sought': ['kereseti kérelmek', 'a fellebbező kérelmei', 'kérelmek', 'a felperes kérelmei', 'a fellebbezők kérelmei'],
 'referring court': ['a kérdést előterjesztő bíróság'],
 'for the council': ['a tanács részéről'],
 'language of the case: english': ['az eljárás nyelve: angol'],
 'language of the case: german': ['az eljárás nyelve: német'],
 'having regard to the treaty establishing the european community,': ['tekintettel az európai közösséget létrehozó szerződésre,'],
 'the commission of the european communities,': ['az európai közösségek bizottsága,'],
 'the council of the european union,': ['az európai unió tanácsa,'],
 'operative part of the judgment': ['az ítélet rendelkező része', 'rendelkező rész'],
 'josé manuel barroso': ['josé manuel barroso'],
 'jean-claude juncker': ['jean-claude juncker'],
 'belgique/belgië': ['belgique/belgië'],
 'prior notification of a concentration': ['összefonódás előzetes bejelentése'],
 'european commission': ['european commission', 'european commission (európai bizottság)', 'európai bizottság'],
 're:': ['tárgy', 'az ügy tárgya'],
 '1049 bruxelles/brussel': ['1049 bruxelles/brussel'],
 'language of the case: italian': ['az eljárás nyelve: olasz'],
 'merger registry': ['merger registry', 'merger registry (fúziós iktatási osztály)', 'fúziós iktatási osztály'],
 'directorate-general for competition': ['directorate-general for competition', 'directorate-general for competition (versenypolitikai főigazgatóság)', 'versenypolitikai főigazgatóság'],
 '2. the business activities of the undertakings concerned are:': ['2. az érintett vállalkozások üzleti tevékenysége a következő:',
  '2. az érintett vállalatok üzleti tevékenysége a következő:'],
 'commission decision': ['a bizottság határozata'],
 'the applicant claims that the court should:': ['a felperes azt kéri, hogy a törvényszék:',
  'a felperes keresetében azt kéri, hogy az elsőfokú bíróság:', 'a fellebbező azt kéri, hogy a bíróság:'],
 'has decided as follows:': ['a következőképpen határozott:'],
 'operative part of the order': ['a végzés rendelkező része', 'rendelkező rész'],
 'oj l 24, 29.1.2004, p. 1 (the "merger regulation").': ['hl l 24., 2004.1.29., 1. o. (az összefonódás-ellenőrzési rendelet).'],
 'definitions': ['fogalommeghatározások'],
 'mariann fischer boel': ['mariann fischer boel'],
 'for the european parliament': ['az európai parlament részéről'],
 'introduction': ['bevezetés'],
 'having regard to the proposal from the commission,': ['tekintettel a bizottság javaslatára,'],
 'council decision': ['a tanács határozata'],
 'where:': ['ahol:'],
 'having regard to the proposal from the european commission,': ['tekintettel az európai bizottság javaslatára,'],
 'vice-president': ['alelnök'],
 'the european parliament and the council of the european union,': ['az európai parlament és az európai unió tanácsa,'],
 '1. introduction': ['1. bevezetés'],
 'language of the case: dutch': ['az eljárás nyelve: holland'],
 'language in which the application was lodged: english': ['a keresetlevél nyelve: angol'],
 'candidate case for simplified procedure': ['egyszerűsített eljárás alá vont ügy'],
 'final provisions': ['záró rendelkezések'],
 'language of the case: spanish': ['az eljárás nyelve: spanyol'],
 'read:': ['helyesen:'],
 'general provisions': ['általános rendelkezések'],
 'for the eea joint committee': ['az egt vegyes bizottság részéről', 'az egt-vegyesbizottság részéről'],
 'defendant: european union intellectual property office (euipo)': ['alperes: az európai unió szellemi tulajdoni hivatala (euipo)'],
 'the eea joint committee,': ['az egt vegyes bizottság,', 'az egt-vegyesbizottság,'],
 '(ordinary legislative procedure: first reading)': ['(rendes jogalkotási eljárás: első olvasat)'],
 'annul the contested decision;': ['helyezze hatályon kívül a megtámadott határozatot;'],
 'plea in law': ['jogalap'],
 'has adopted this directive:': ['elfogadta ezt az irányelvet:'],
 'defendant: european commission': ['alperes: európai bizottság'],
 'no constitutional requirements indicated.': ['alkotmányos követelmények fennállását nem jelezték.'],
 'language of the case: portuguese': ['az eljárás nyelve: portugál'],
 'details of the proceedings before euipo': ['az euipo előtti eljárás adatai'],
 'have adopted this regulation:': ['elfogadta ezt a rendeletet:', 'elfogadták ezt a rendeletet:'],
 'scope': ['hatály'],
 'member states shall determine how such reference is to be made.': ['a hivatkozás módját a tagállamok határozzák meg.'],
 'cases where the commission raises no objections': ['olyan esetek, amelyekkel kapcsolatban a bizottság nem emel kifogást'],
 'joaquín almunia': ['joaquín almunia'],
 'defendant: commission of the european communities': ['alperes: az európai közösségek bizottsága', 'alperes: európai közösségek bizottsága'],
 'acting in accordance with the ordinary legislative procedure,': ['rendes jogalkotási eljárás keretében,'],
 'judgment of the court': ['a bíróság ítélete'],
 'language of the case: greek': ['az eljárás nyelve: görög'],
 'france': ['franciaország'],
 'germany': ['németország'],
 'questions referred': ['az előzetes döntéshozatalra előterjesztett kérdések'],
 'commission implementing decision': ['a bizottság végrehajtási határozata'],
 'the committee of the regions': ['a régiók bizottsága'],
 'postal address:': ['postai cím:'],
 'please use the contact details below:': ['az elérhetőségi adatok a következők:'],
 'this notification concerns the following undertakings:': ['e bejelentés az alábbi vállalkozásokat érinti:'],
 'or': ['vagy'],
 'implementation': ['végrehajtás'],
 'president': ['elnök'],
 '1. adopts its position at first reading hereinafter set out;': ['1. elfogadja első olvasatban az alábbi álláspontot;'],
 'the president of the european economic and social committee': ['az európai gazdasági és szociális bizottság elnöke'],
 'single document': ['egységes dokumentum'],
 'spain': ['spanyolország'],
 'markos kyprianou': ['markos kyprianou'],
 'other': ['egyéb'],
 'peter straub': ['peter straub'],
 'areas of union competence deriving from the treaty': ['a szerződésből eredő uniós hatáskör'],
 'competences and activities': ['hatáskör és tevékenységek'],
 'pleas in law': ['jogalapok'],
 'procedure before euipo: opposition proceedings': ['az euipo előtti eljárás: felszólalási eljárás'],
 'entry into force and application': ['hatálybalépés és alkalmazás'],
 'order euipo to pay the costs.': ['az euipo-t kötelezze a költségek viselésére.'],
 'italy': ['olaszország'],
 'tasks': ['feladatok'],
 'review': ['felülvizsgálat'],
 'date:': ['dátum:'],
 '1. approves the commission proposal as amended;': ['1. jóváhagyja a bizottság javaslatát, annak módosított formájában;'],
 'the following reference should always be specified:': ['az alábbi hivatkozási számot minden esetben fel kell tüntetni:'],
 'the concentration is accomplished by way of purchase of shares.': ['az összefonódásra részesedés vásárlása útján kerül sor.'],
 'commission opinion': ['a bizottság véleménye'],
 'mario draghi': ['mario draghi'],
 'the president of the ecb': ['az ekb elnöke'],
 'other party to the proceedings: european commission': ['a másik fél az eljárásban: európai bizottság'],
 '(only the french text is authentic)': ['(csak a francia nyelvű szöveg hiteles)', '(csak a francia nyelvű változat hiteles)'],
 'committee procedure': ['a bizottsági eljárás', 'bizottsági eljárás', 'a bizottság eljárása'],
 'exercise of the delegation': ['a felhatalmazás gyakorlása'],
 '(codification)': ['(kodifikált szöveg)'],
 'note:': ['megjegyzés:'],
 'general considerations': ['általános megfontolások'],
 'yes': ['igen'],
 'language of the case: czech': ['az eljárás nyelve: cseh'],
 'defendant: european parliament': ['alperes: európai parlament']}

anchor_segment_patterns_jox_en_hu = [
 [4,16,'\d{1,3}\.\d{1,3}\.(\d{1,3}\.)?(\d{1,3}\.)?', '{0}', [0]],
 [6, 7, 'part ([a-z0-9]{1,2})', '{0}. rész', [1]],
 [7, 9, 'table ([a-z0-9]{1,3})', '{0}. táblázat', [1]],
 [7, 10, 'title ([ivxcl]{1,4})', '{0}. cím', [1]],
 [8, 10, 'figure ([a-z0-9]{1,3})', '{0}. ábra', [1]],
 [7, 10, 'annex ([a-z0-9]{1,4})', '{0}. melléklet', [1]],
[9, 11, 'article ([a-z0-9]{1,3})', '{0}. cikk', [1]],
 [9, 11, 'recital ([a-z0-9]{1,3})', '({0}) preambulumbekezdés', [1]],
 [9, 11, 'section ([a-z0-9]{1,3})', '{0}. szakasz', [1]],
 [9, 12, 'chapter ([ivxcl]{1,4})', '{0}. fejezet', [1]],
 [10, 12, 'appendix ([a-z0-9]{1,3})', '{0}. függelék', [1]],
 [11, 12, 'amendment ([a-z0-9]{1,3})', '{0}. módosítás', [1]],
 [12, 15, '([ivxcl]{1,4}). procedure', '{0}. eljárás', [1]],
 [15, 17, 'see footnote ([0-9]{1,3})\.', 'lásd a {0}. lábjegyzetet.', [1]],
 [15, 15, '([ivxcl]{1,4})\. introduction', '{0}. bevezetés', [1]],
     [15, 22, 'fax +?[0-9 \-]{10,17}', '{0}', [0]],
 [19,70, 'https?://[a-z1-9_/\.]{12,}', '{0}', [0]],
    [19, 20, '([0-9]{1,2})\. general comments', '{0}. általános megjegyzések', [1]],
    [20, 21, '([0-9]{1,2})\. specific comments', '{0}. részletes megjegyzések', [1]],
     [23, 30, 'oj ([lc] [0-9]{1,3}), (\d{1,2})\.(\d{1,2})\.(\d{4}), p\. (\d{1,4})\.', 'hl {0}., {1}.{2}.{3}., {4}. o.', [1,4,3,2,5]],
 [25, 50, 'email: [a-z\-\.]{4,29}@ec\.europa\.eu', '{0}',[0]],
 [33, 36, 'council regulation \(ec\) no (\d{1,4}/\d{4})', 'a tanács {0}/ek rendelete', [1]],
 [33, 36, 'council regulation \(eu\) no (\d{1,4}/\d{4})', 'a tanács {0}/eu rendelete', [1]],
 [35, 36, '(\d{1,2})\. conclusions and recommendations', '{0}. következtetések és ajánlások', [1]],
    [36,39,'commission regulation \(eu\) no (\d{1,4}/\d{4})','a bizottság {0}/eu rendelete', [1]],
    [36,39,'commission regulation \(ec\) no (\d{1,4}/\d{4})','a bizottság {0}/ek rendelete', [1]],
 [49, 51, 'having regard to rule (\d{1,3}) of its rules of procedure,', 'tekintettel eljárási szabályzata {0}. cikkére,', [1]],
[49,52,'commission implementing regulation \(eu\) no (\d{1,4}/\d{4})','a bizottság {0}/eu végrehajtási rendelete', [1]]
]

anchor_pattern_list_jox_en_hu = [{'minl': i[0],'maxl': i[1], 'l1_pattern': re.compile(i[2]), 'l2_format': i[3], 'match_indices': i[4]} for i in anchor_segment_patterns_jox_en_hu]