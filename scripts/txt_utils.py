#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:31:10 2017

@author: elliott
"""

# set this to your working directory
WORK_DIR = '/home/elliott/Dropbox/_Ash_Teaching/2018-09 - Bocconi - Text Data and ML/code'

from nltk.stem import SnowballStemmer
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from collections import Counter
import string
translator = str.maketrans('','',string.punctuation) 
stemmer = SnowballStemmer('english')

newstop = ['death','penalty','would']

def clean_document(doc):
    doc = doc.translate(translator)
    doc = [i for i in doc.lower().split() if i not in stop and len(i) < 10]
    doc = [stemmer.stem(t) for t in doc if t not in newstop]
    return doc

stopde = set(stopwords.words('german'))
stemmerde = SnowballStemmer('german')
def clean_german(doc):
    doc = doc.translate(translator)
    doc = [i for i in doc.lower().split() if i not in stopde and len(i) < 10]
    doc = [stemmerde.stem(t) for t in doc if t not in newstop]
    return doc

def process_document(text):    
    rawsents = sent_tokenize(text)    
    sentences = [word_tokenize(s) for s in rawsents]    
    return sentences

def get_sentences(doc):
    sentences = []
    
    for raw in sent_tokenize(doc):
        raw2 = [i for i in raw.translate(translator).lower().split() if i not in stop and len(i) < 10]
        raw3 = [stemmer.stem(t) for t in raw2]
        sentences.append(raw3)
    return sentences

def get_docfreqs(documents):
    docfreqs = Counter()
    for doc in documents:
        proc = doc.lower()
        proc = proc.translate(translator)
        tokens = proc.split()
        tokens = [x for x in set(tokens) if x not in stop]
        docfreqs.update(tokens)
    return docfreqs

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 07:37:50 2013

@author: elliott
"""

from collections import Counter, defaultdict
import re
import numpy as np
from unidecode import unidecode

romanNumeralMap = (('M',  1000),
                   ('CM', 900),
                   ('D',  500),
                   ('CD', 400),
                   ('C',  100),
                   ('XC', 90),
                   ('L',  50),
                   ('XL', 40),
                   ('X',  10),
                   ('IX', 9),
                   ('V',  5),
                   ('IV', 4),
                   ('I',  1))
class RomanError(Exception): pass
class OutOfRangeError(RomanError): pass
class NotIntegerError(RomanError): pass
class InvalidRomanNumeralError(RomanError): pass
romanNumeralPattern = re.compile("""
    ^                   # beginning of string
    M{0,4}              # thousands - 0 to 4 M's
    (CM|CD|D?C{0,3})    # hundreds - 900 (CM), 400 (CD), 0-300 (0 to 3 C's),
                        #            or 500-800 (D, followed by 0 to 3 C's)
    (XC|XL|L?X{0,3})    # tens - 90 (XC), 40 (XL), 0-30 (0 to 3 X's),
                        #        or 50-80 (L, followed by 0 to 3 X's)
    (IX|IV|V?I{0,3})    # ones - 9 (IX), 4 (IV), 0-3 (0 to 3 I's),
                        #        or 5-8 (V, followed by 0 to 3 I's)
    $                   # end of string
    """ ,re.VERBOSE)

def fromRoman(s):
    result = 0
    index = 0
    for numeral, integer in romanNumeralMap:
        while s[index:index+len(numeral)] == numeral:
            result += integer
            index += len(numeral)
    return result

def subber(marker, pattern, string):
    try:
        output = pattern.sub(r'<%s>\1</%s>' % (marker, marker), string)
    except Exception as e:
        print (str(e))
        print (pattern.pattern)
        print (string)
    return output


linenumRe = re.compile('^[0-9\*\]\s]+')

sectags = {
           'sec': '((Section)|(Sec\.)|(SS))',
           'chap': '((Chapter)|(Ch\.)|(Chap\.))',
           'art': '((Article)|(Art\.))',
           'title': '(Title)'
          }

secend = """
            (\s)+                # space between header and number
           ((no\.(\s)+)|)          # optional, 'No.'
           [0-9IXVLC\.\-]+       # section number, allows for decimals, dashes, and roman numerals
            )"""

sectionRe = {}
for item, tag in sectags.items():
    reg = '([^a-z]*' + tag + secend
    sectionRe[item] = re.compile(reg, re.VERBOSE | re.I)

# numbered clauses
clausetag = """
                    (^
                    \(
                    (
                    ([A-Z])        |
                    ([0-9]{1,3})   |
                    )
                    \)
                    )
                    """
clauseRe = re.compile(clausetag, re.VERBOSE | re.I)

endsentRe = re.compile('[a-z][a-z](\.|>)$')
hyphenRe = re.compile('[a-z][a-z]-$')

dateRe = re.compile("""((JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|
                        JAN|FEB|MAR|APR|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)
                        [,\s\.]+[0-9]{1,4}(((,|)(\s)*[0-9]{2,4})|))
                        """, re.I | re.VERBOSE)

is_ascii = lambda s: len(s) == len(s.encode())

def scraper(lines, allcaps_secs=False):
    
    lawlines = []
    actlength = 0
    numacts = 0
    done = 0
    
    linelist = []
    
    breakme = False
    scraping = False
    tagcount = 0
    
    index = 0
    numberlist = defaultdict(lambda: [0])
    prevsubnumber = defaultdict(int)
    prevlevel = ''
    prevnumber = defaultdict(str)
    level = ''
    number = 0
    subnumber = 0
    endsentence, prevend = True,True
    lawtext = ''
    sectionheaders = []
    lawlineslist = [] # this is the full document, a list of clauses
    endsentence, prevend = True,True
    
    headercounts = Counter()
    linelengths = Counter()

    secnum = 0
    for line in lines:
        
        line = unidecode(line)
               
        if secnum > 10:
            # stop at schedules if not in TOC
            if line.startswith('SCHEDULE'):
                break
            if line.startswith('APPENDIX'):
                break
        #if not is_ascii(line):
        #    continue
#    numcheck = 0
#    for linenum, line in enumerate(lines):
#        if '....' in line or '---' in line or '      ' in line:
#            continue
#        if not re.search('[a-zA-Z]', line):
#            continue
#        if linenumRe.match(line):
#            #print line
#            numcheck += 1
#    haslinenumbers = False
#    
#    if numcheck / len(lines) > .4:
#        #print numcheck / len(lines)
#        #print row['rawtext']
#        haslinenumbers = True

        # skip page numbers
        if line.isdigit():
            endsentence = True
            continue

        if not re.search('[a-zA-Z]', line):
            endsentence = True
            continue

        prevend = endsentence
        endsentence = False
        if endsentRe.search(line):
            endsentence = True
        if '....' in line or '---' in line or '      ' in line or '. . .' in line: # indicates index or budget item
            endsentence = True
            continue
        
#        if line[1] == '.' and (line[0].isupper() or line[0].isdigit()):
#            line = line[2:]

#        if haslinenumbers:
#            if linenumRe.match(line):
#                m = re.search("[a-zA-Z]", line)
#                firstchar = m.start()
#                #print line
#                line = line[firstchar:]
#                #print line
#                #return

        newsec = False
        
        for marker, pattern in sectionRe.items():
            if pattern.match(line):
                #if len(lawlines) < 2:
                    #print(lawlines)
                #    continue
                #else:
                #print(lawlines)
                scraping=True
                if len(lawlines) == 0:
                    continue
                if len(lawlines) <= 2 and all('.' not in L for L in lawlines):
                    continue
                
                newsec = True
                if lawlines == []:
                    break
                lawlineslist.append(lawlines)
                lawlines = []
                secnum += 1
                break
#                if line[0].isupper():
#
#                    words = line.upper().split() #re.findall(r"[-9a-zA-Z]+-?[0-9a-zA-Z]+", line[j:].upper())
#
#                    subnumber = ''
#                    level = marker
#
#                    if words[1] == 'NO':
#                        words.pop(1)
#                    rawnums = re.findall(r"[0-9a-zA-Z]+-?[0-9a-zA-Z]*", words[1])
#                    if len(rawnums) == 1:
#                        w = rawnums[0]
#                        if w.isdigit():
#                            number = int(w)
#                        elif romanNumeralPattern.match(w): # consider replacing l with I (common OCR error)
#                            number = fromRoman(w)
#                        else:
#                            number = w
#
#                    elif len(rawnums) > 1: # check for a decimal/hierarchical number
#
#                        subnumber = ''
#                        for w in rawnums[1:]:
#                            if w.isdigit():
#                                subnumber = subnumber + w + '.'
#                            elif romanNumeralPattern.match(w):
#                                subnumber = subnumber + str(fromRoman(w)) + '.'
#                        subnumber = subnumber[:-1]
#                    #else:
#                        #print('')
#                        #print(line)
#                        #print(rawnums)
#                        #return
#
#                    prevnum = numberlist[level][-1]
#                    if number != prevnum:
#
#                        headercounts[level] +=1
#                        if level == 'chap':
#                        #print('\t'.join([code, str(i), state,year,str(len(lawlines)),level,str(numberlist[level][-1]),str(number),str(prevsubnumber[level]),str(subnumber),'"'+line+'"']))
#                            if len(lawlines) > 10:
#                                lawlineslist.append(lawlines)
#                                lawlines = []
#
#                numberlist[level].append(number)
#                prevsubnumber[level] = subnumber

        # stash the section headers

        # skip all-upper-case lines
        if line.isupper():
            if allcaps_secs:
                scraping=True
                if len(lawlines) == 0:
                    continue
                if len(lawlines) <= 2 and all('.' not in L for L in lawlines):
                    continue
                newsec = True
                if lawlines == []:
                    break
                lawlineslist.append(lawlines)
                lawlines = []
                secnum += 1
            else:
                continue

        if newsec:
            sectionheaders.append(line)
            continue


        prevline = line
        if scraping:
            lawlines.append(line)
    return lawlineslist, sectionheaders

def process_contract(rawtext):
    

    lines = [x.strip() for x in rawtext.splitlines()]
    


    lawlineslist,sectionheaders = scraper(lines,allcaps_secs=False)
    
    if len(lawlineslist) < 5:
        lawlineslist,sectionheaders = scraper(lines,allcaps_secs=True)

    lawtextlist = []
    for lawlines in lawlineslist:
        lawtext = ''
        meanlength = np.mean([len(x) for x in lawlines if len(x) > 0])
        for line in lawlines:
            line = re.sub(r"\(.*\)","",line)
            if ':' in line:
                #pass
                line = line.replace(':','.')
                #TODO:
                # detect hierarchial clauses and include them as separate sentences.
            rawtokens = line.split()
            tokens = []
            for t in rawtokens:
                if '.' in t:
                    if t.isupper():
                        continue
                    if all(x.isdigit() for x in t if t != '.'):                        
                        continue
                tokens.append(t)
            line = ' '.join(tokens)
            if line == '':
                if len(lawtext) == 0:
                    continue
                if lawtext[-1] != '.':
                    lawtext = lawtext + '. '
                continue
    
            if hyphenRe.search(line):
                lawtext += line[:-1]
                continue
    
            if len(line) < .6 * meanlength:
                lawtext += line
                if lawtext[-1] != '.':
                    lawtext = lawtext + '. '
                continue
    
    
            if line[-1].isdigit() and len(line) < .8 * meanlength:
                lawtext += line
                if lawtext[-1] != '.':
                    lawtext = lawtext + '. '
                continue
    
            lawtext += line + ' '
    
        if len(lawtext) == 0:
            continue
        #print len(lawlines)
        lawtextlist.append(lawtext)
        
    return lawtextlist, sectionheaders
    

def recurse(*tokens):
    "Recurse through parse tree."
    children = []
    def add(tok):       
        sub = tok.children
        for item in sub:
            children.append(item)
            add(item)
    for token in tokens:
        add(token)    
    return children
    
def get_branch(t,sent,include_self=True):
    "Get branch associated with token."        
    branch = recurse(t)
    if include_self:
        branch += [t]
        
    branch = [w for w in sent if w in branch]

    lemmas = []
    tags = []
    
    for token in branch:
        lemma = token.lemma_.lower()
        if any([char.isdigit() for char in lemma]):
            lemma = "#" # replace number if "#"            
        if any(punc in lemma for punc in ['.',',',':',';', '-']):
            # exclude vocabulary
            continue
        lemmas.append(lemma)
        tags.append(token.tag_)
    
    #tags = [w.tag_ for w in sent if w in mods]
    return lemmas, tags
