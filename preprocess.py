import os
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from html.parser import HTMLParser
from sentencepiece import SentencePieceTrainer
from rapidfuzz.process import cdist
from pathlib import Path
import utilities

def main():
    dirname = 'test1' # output directory name
    vocab = 5000 # the vocabulary size for SentencePiece
    cc = 1 # the character coverage for SentencePiece
    no_kugiri = True # eliminate line separators
    only_original = True
    intermediate_size = 768 # the number of units in the intermediate layers of language models
    model = 'BERT' # or 'GPT' for GPT-2
    seed = 1 # the seed for the random number generator
    
    if True: # to speed up, you can switch it to False on the second run and later...
        volume_df = get_volume()
        database_df = generate_volume_and_database(dirname, volume_df, use_database_errors=True, replace_characters=True)
        chapterdict, notfound = generate_chapterdict_authorid(dirname, database_df)
        similarity, verse_df = generate_verse(dirname, volume_df, database_df, chapterdict, notfound, exclude_ido=False, similarity_threshold=80, id2_mode=False)
        
        figures = Path('figures')
        utilities.safe_mkdir(figures)
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.set_position([0.2, 0.18, 0.75, 0.8])
        ax.hist(np.array(similarity)/100, bins=25, range=(0, 1), color='white', ec='k')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Levenshtein ratio')
        ax.set_ylabel('Count')
        plt.savefig(figures/'fig_levenshtein.pdf')
    else:
        _, volume_df, verse_df, _ = utilities.load_all(dirname, None, None)
    
    generate_sentencepiece(dirname, volume_df, verse_df, vocab=vocab, cc=cc, no_kugiri=no_kugiri, only_original=only_original, intermediate_size=intermediate_size, model=model, seed=seed)
    
nichibun = 'nichibun'
imperial_anthologies = ['古今集',
                        '後撰集',
                        '拾遺集',
                        '後拾遺集',
                        '金葉集＿初度本',
                        '金葉集＿二度本',
                        '金葉集＿三奏本',
                        '詞花集',
                        '千載集',
                        '新古今集',
                        '新勅撰集',
                        '続後撰集',
                        '続古今集',
                        '続拾遺集',
                        '新後撰集',
                        '玉葉集',
                        '続千載集',
                        '続後拾遺集',
                        '風雅集',
                        '新千載集',
                        '新拾遺集',
                        '新後拾遺集',
                        '新続古今集']

def get_volume():
    index_era = 'index_era.html'
    path = Path(nichibun)/index_era
    soup = BeautifulSoup(open(path, 'r', encoding='euc-jp'), 'html.parser')
    title = []
    url = []
    year = []
    for x in soup.select('td'):
        a = x.find_all('a')
        text = x.get_text()
        if len(a)>0:
            theurl = a[0].get('href')
            if 'html' in theurl:
                title.append(text)
                url.append(theurl)
        else:
            if len(year)<len(title):
                m = re.search('[0-9]+', text)
                if m is None:
                    year.append(np.nan)
                else:
                    year.append(int(m.group()))
    code = [int(x[6:-5]) for x in url]
    type = ['E' if x in imperial_anthologies else '' for x in title]
    volume_df = pd.DataFrame({'code': code, 'title': title, 'chronology': year, 'type': type}).set_index('code')
    return volume_df
    
def s2s(s):
    if s is None:
        return ''
    else:
        return s
            
class MyHTMLParser(HTMLParser):
    def __init__(self, title, code, fp, line2line, lastindex, replace_characters):
        self.chapter = None
        self.h2mode = False
        self.tablemode = False
        self.maintext = False
        self.tablecounter = 0
        self.title = title
        self.code = code
        self.fp = fp
        self.line2line = line2line
        self.index = lastindex
        self.replace_characters = replace_characters
        super(MyHTMLParser, self).__init__(convert_charrefs=True)
        
    def handle_starttag(self, tag, attrs):
        if tag=='h2':
            self.h2mode = True
        elif tag=='table':
            self.tablemode = True
            self.tablecounter += 1
            self.tdcounter = 0
            self.id = None
            self.kotoba = None
            self.hito = None
            self.uta = []
            self.ido = None
        elif tag=='td':
            self.tdcounter += 1
            self.divcounter = 0
        elif tag=='div':
            self.divcounter += 1

    def handle_endtag(self, tag):
        if tag=='h2':
            self.h2mode = False
        elif tag=='table':
            self.tablemode = False
            if len(self.uta)>0 and self.tablecounter>=4:
                def eliminate_kanji(x):
                    if self.replace_characters:
                        for k, v in [('／', '−'), ('－', '−'), (' ', ''), ('0', ''), ('リ', 'り'), ('ヘ', 'へ'),
                                     ('ぐ', 'く'), ('ぷ', 'ふ'), ('ぢ', 'ち'), ('ペ', 'へ'), ('ご', 'こ')]:
                            x = x.replace(k, v)
                        m = re.search(r'[^ぁ-ん−хｘ□]', x)
                        if m is not None:
                            print(x)
                            for k, v in [('人', 'ひと'), ('木', 'き'), ('明', 'あ'), ('染', 'そ'), # ('月', 'つき'), 
                                         ('浅', 'あさ'), ('入り', 'いり'), ('入らは', 'いらは'), ('入', 'いり'),
                                         ('鳴', 'な'), ('夜', 'よ'), ('わきも子', 'わきもこ'), ('子', 'ね'), ('戸', 'と'),
                                         ('火', 'ひ'), ('契', 'ちき'), ('水', 'みつ'), ('枝', 'え'), ('逢', 'あ'),
                                         ('分', 'わ'), ('波', 'なみ'), ('森', 'もり'), ('千', 'ち'), ('羽', 'は'),
                                         ('名', 'な'), ('待', 'ま')]:
                                x = x.replace(k, v)
                            print(x)
                    return x

                if len(self.uta)==1:
                    self.uta = eliminate_kanji(self.uta[0])+','
                elif len(self.uta)==2:
                    self.uta = eliminate_kanji(self.uta[1])+','+self.uta[0]
                else:
                    print(self.title, self.code, self.id, self.kotoba, self.hito, self.uta, self.ido, self.chapter)
                    raise Exception
                com = ','.join([str(self.code), self.title, s2s(self.chapter), s2s(self.id), self.uta, s2s(self.hito), s2s(self.kotoba), s2s(self.ido)])
                if com in self.line2line:
                    oldcom, com = com, self.line2line[com]
                    print(f'{oldcom} was replaced by {com}.')
                    self.line2line[oldcom] = None
                self.fp.write(f'{self.index},{com}\n')
                self.index += 1
        
    def handle_data(self, data):
        #print("Encountered some data  :", data)
        data = data.strip()
        if self.h2mode:
            if data!='':
                self.chapter = data
        else:
            if self.tablemode:
                if len(data)>0:
                    if self.tdcounter==1 and self.tablecounter>=4:
                        s = re.search(r'[0-9]+', data)
                        if not data.startswith('ID') and s is None:
                            print(data)
                            raise Exception()
                        self.id = s.group()
                    elif self.tdcounter==2:
                        if self.divcounter==1:
                            self.kotoba = data
                        elif self.divcounter==2:
                            if data!='未入力　(xxx)':
                                self.hito = data
                        elif self.divcounter==3:
                            self.uta.append(data)
                        elif self.divcounter==4:
                            if not data.startswith('異同資料句番号：'):
                                print(data)
                                raise Exception()
                            s = re.search(r'[0-9]+', data)
                            self.ido = s.group()

def generate_volume_and_database(dirname, volume_df, use_database_errors, replace_characters):
    utilities.safe_mkdir(dirname)
    dir = Path(dirname)
    volume_df.to_csv(dir/'volume.csv')
    
    line2line = {}
    if use_database_errors:
        with open('database_errors.csv', 'r') as fp:
            lines = fp.readlines()
        if len(lines)%2==1:
            raise Exception('databas_errors.csv has an odd number of lines.')
        for i in range(len(lines)//2):
            line2line[lines[2*i].rstrip()] = lines[2*i+1].rstrip()
    
    index = 0
    with open(dir/'database.csv', 'w') as fp:
        fp.write('index,code,title,chapter,id,uta1,uta2,author,kotobagaki,id2\n')
        for l in volume_df.itertuples():
            html = open(Path('nichibun/')/(f'waka_i{l.Index:03}.html'), 'r', encoding='euc-jp').read()
            parser = MyHTMLParser(l.title, l.Index, fp, line2line, index, replace_characters)
            parser.feed(html)
            index = parser.index

    for k, v in line2line.items():
        if v is not None:
            print(f'{k.rstrip()} has not been replaced.')
    
    database_df = pd.read_csv(dir/'database.csv', dtype={'code': int, 'title': str, 'chapter': str, 'id': str, 'uta1': str, 'uta2': str, 'author': str, 'kotobagaki': str, 'id2': str})
    return database_df

def generate_chapterdict_authorid(dirname, database_df):
    chapterdict = {}
    id2author = {}
    for l in database_df.itertuples():
        c = int(l.code)
        if c not in chapterdict:
            chapterdict[c] = {}
        chapterdict[c][l.chapter] = l.chapter
        if type(l.author)==str:
            m = re.search(r'\((.+)\)', l.author)
            if m is None:
                raise Exception('Not found'+l.author)
            i = m.group(1)
            if i not in id2author:
                id2author[i] = []
            id2author[i].append(l.author)
    for c, chapters in chapterdict.items():
        if len(chapters)==0:
            continue
        if len(chapters)==1:
            continue
        n = np.min([len(x) for x in chapters.keys()])
        for i in range(n):
            same = True
            for chap1 in chapters.keys():
                for chap2 in chapters.keys():
                    if chap1[:i]!=chap2[:i]:
                        same = False
                        break
            if not same:
                maxi = i-1
                break
        else:
            maxi = n-1
        for k in chapterdict[c].keys():
            chapterdict[c][k] = k[maxi:]
    notfound = {'xxx': -1}
    notfoundid = 1000
    with open(Path(dirname)/'authorid.csv', 'w') as fp:
        for k, v in id2author.items():
            try:
                int(k)
            except ValueError:
                if k not in notfound:
                    notfound[k] = notfoundid
                    notfoundid += 1
                k = notfound[k]
            fp.write(f'{k},{sorted(v, key=len, reverse=True)[0]}\n')
    return chapterdict, notfound

def generate_verse(dirname, volume_df, database_df, chapterdict, notfound, exclude_ido, similarity_threshold, id2_mode):
    dir = Path(dirname)
    fp = open(dir/'verse.csv', 'w')
    fp.write('index,originalindex,dindex,code,chapter,uta1,author,validation\n')
    batchsize = 512
    counter = 0
    verses = {}
    deleted = set()
    lastcode = -1
    similarity = []
    for l0 in volume_df.itertuples():
        print(l0.title)
        df1 = database_df[database_df['code']==l0.Index]
        ind1, ind2 = np.min(df1.index), np.max(df1.index)
        ids = set()
        for start in range(0, ind2-ind1+1, batchsize):
            df2i = df1.iloc[start:min(start+batchsize, ind2-ind1+1)]
            similarity_matrix = cdist(df2i['uta1'], database_df.iloc[:ind2+1]['uta1'], workers=-1)
            for l, sm in zip(df2i.itertuples(), similarity_matrix):
                if l.uta1.count('−')!=4:
                    continue
                if exclude_ido and type(l.chapter)==str and '異同' in l.chapter:
                    deleted.add(l.uta1)
                    pass
                elif id2_mode and type(l.id2)==str and l.id2 in ids:
                    deleted.add(l.uta1)
                    pass
                elif l.uta1 not in deleted:
                    m = re.search(r'[^ぁ-ん−]', l.uta1)
                    if m is None:
                        if type(l.author)==str:
                            m = re.search(r'\((.+)\)', l.author)
                        else:
                            m = None
                        if m is None:
                            author = ''
                        else:
                            try:
                                author = int(m.group(1))
                            except ValueError:
                                author = notfound[m.group(1)]
                        chapname = chapterdict[l.code][l.chapter] if type(l.chapter)==str else ''
                        if l.Index>0:
                            roi = sm[:l.Index]
                            ind0 = np.argmax(roi)
                            similarity.append(roi[ind0])
                            oldverse = database_df.loc[ind0, 'uta1']
                            if oldverse in verses and roi[ind0]>=similarity_threshold:
                                verses[l.uta1] = verses[oldverse]
                        if l.uta1 not in verses:
                            verses[l.uta1] = counter
                        validation = np.isnan(volume_df.loc[int(l.code), 'chronology'])
                        fp.write(f'{counter},{verses[l.uta1]},{l.Index},{l0.Index},{chapname},{l.uta1},{author},{validation}\n')
                        counter += 1
                ids.add(l.id2)
    fp.close()
    verse_df = pd.read_csv(dir/'verse.csv', index_col=0)
    return similarity, verse_df

def generate_sentencepiece(dirname, volume_df, verse_df, vocab, cc, no_kugiri, only_original, intermediate_size=768, model='BERT', seed=1):
    dir = Path(dirname)
    setting_df = pd.DataFrame({'character_coverage': [cc],
                               'vocab_size': [vocab],
                               'no_kugiri': [no_kugiri],
                               'only_original': [only_original],
                               'model': [model],
                               'seed': [seed],
                               'intermediate_size': [intermediate_size]})
    setting_df.to_csv(dir/'setting.csv')
    cfile = 'tmp_corpus.txt'
    for code in list(volume_df[volume_df['type'].str.contains('E')].index)+[0]:
        utilities.generate_corpus(verse_df, cfile, no_kugiri, only_original, False, code)
        dir2 = dir/str(code)
        utilities.safe_mkdir(dir2)
        spname = dir2/'sentencepiece'
        SentencePieceTrainer.Train(f'--input={cfile}, --model_prefix={spname} --character_coverage={cc} --vocab_size={vocab} --pad_id=3 --add_dummy_prefix=False')

if __name__=='__main__':
    main()
