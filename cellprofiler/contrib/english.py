# No CellProfiler copyright notice here.

# english - generic english tools

'''

Some generic English word-munging routines.

SYNOPSIS

    >>> import english ; from english import *

    >>> stuff = [ 'apple', 'box', 'card', 'donkey' ]

    >>> enumerate(stuff)
        'apple, box, card and donkey'

    >>> cardinal(1234567)
        'one million, two hundred thirty-four thousand,
         five hundred sixty-seven'

    >>> ordinal(61)
        'sixty-first'

    >>> conjugate('forsake', 'past participle')
        'forsaken'

AUTHOR

    Ed Halley (ed@halley.cc) 22 August 2004
    Distributed under the Artistic License.
    Copying in whole or in part, with author attribution, is expressly allowed.
    http://http://halley.cc/code/?python/english.py

'''

__all__ = [ 'cardinal', 'ordinal',
            'conjugate', 'enumerate' ]

#----------------------------------------------------------------------------

def enumerate(stuff, separator=',', conjunction='and'):
    '''Returns a phrase that recites a list of noun phrases naturally.'''
    text = ''
    n = len(stuff)
    for i in range(len(stuff)):
        text += str(stuff[i])
        if (i < n-2): text += separator
        if (i == n-2): text += ' ' + conjunction
        if (i < n-1): text += ' '
    return text

#----------------------------------------------------------------------------

# The subject verbs.
__irregular_present = {
      'do' : 'does',
      'go' : 'goes',
      'forego' : 'foregoes' }

# The subject is verbing.
__irregular_progressive = {
      'quit' : 'quitting', 'quip' : 'quipping', 'quiz' : 'quizzing' }

# The subject verbed.
__irregular_past = {
      'arise' : 'arose', 'beat' : 'beat', 'become' : 'became', 'begin'
      : 'began', 'behold' : 'beheld', 'bend' : 'bent', 'beset' :
      'beset', 'bet' : 'bet', 'bind' : 'bound', 'bite' : 'bit',
      'bleed' : 'bled', 'blow' : 'blew', 'break' : 'broke', 'breed' :
      'bred', 'bring' : 'brought', 'broadcast' : 'broadcast', 'build'
      : 'built', 'burn' : 'burnt', 'burst' : 'burst', 'buy' :
      'bought', 'cast' : 'cast', 'catch' : 'caught', 'choose' :
      'chose', 'cling' : 'clung', 'come' : 'came', 'cost' : 'cost',
      'creep' : 'crept', 'cut' : 'cut', 'deal' : 'delt', 'dig' :
      'dug', 'do' : 'did', 'draw' : 'drew', 'dream' : 'dreamt',
      'drink' : 'drank', 'drive' : 'drove', 'dwell' : 'dwelt', 'eat'
      : 'ate', 'fall' : 'fell', 'feed' : 'fed', 'feel' : 'felt',
      'fight' : 'fought', 'find' : 'found', 'flee' : 'fled', 'fling'
      : 'flung', 'fly' : 'flew', 'forbid' : 'forbade', 'foretell' :
      'foretold', 'forget' : 'forgot', 'forsake' : 'forsook', 'freeze'
      : 'froze', 'get' : 'got', 'give' : 'gave', 'go' : 'went',
      'grind' : 'ground', 'grow' : 'grew', 'hang' : 'hung', 'have' :
      'had', 'hear' : 'heard', 'hit' : 'hit', 'hold' : 'held', 'hurt'
      : 'hurt', 'keep' : 'kept', 'kneel' : 'knelt', 'knit' : 'knit',
      'know' : 'knew', 'lay' : 'laid', 'lead' : 'led', 'lean' :
      'leant', 'leap' : 'lept', 'learn' : 'learnt', 'leave' : 'left',
      'lend' : 'lent', 'let' : 'let', 'lie' : 'lay', 'light' : 'lit',
      'lose' : 'lost', 'make' : 'made', 'mean' : 'meant', 'meet' :
      'met', 'pay' : 'paid', 'put' : 'put', 'read' : 'read', 'rend' :
      'rent', 'ride' : 'rode', 'ring' : 'rang', 'rise' : 'rose', 'run'
      : 'ran', 'say' : 'said', 'seek' : 'sought', 'sell' : 'sold',
      'send' : 'sent', 'set' : 'set', 'shake' : 'shook', 'shed' :
      'shed', 'shine' : 'shone', 'shoot' : 'shot', 'show' : 'showed',
      'shrink' : 'shrank', 'shut' : 'shut', 'sing' : 'sang', 'sink' :
      'sank', 'sit' : 'sat', 'slay' : 'slew', 'sleep' : 'slept',
      'slide' : 'slid', 'sling' : 'slung', 'slink' : 'slunk', 'smell'
      : 'smelt', 'smite' : 'smote', 'sow' : 'sowed', 'speak' :
      'spoke', 'spell' : 'spelt', 'spend' : 'spent', 'spill' :
      'spilt', 'spin' : 'spun', 'spit' : 'spat', 'split' : 'split',
      'spoil' : 'spoilt', 'spread' : 'spread', 'spring' : 'sprang',
      'stand' : 'stood', 'steal' : 'stole', 'stick' : 'stuck', 'sting'
      : 'stung', 'stink' : 'stank', 'stride' : 'strode', 'strike' :
      'struck', 'strive' : 'strove', 'swear' : 'swore', 'sweep' :
      'swept', 'swim' : 'swam', 'swing' : 'swang', 'take' : 'took',
      'teach' : 'taught', 'tear' : 'tore', 'think' : 'thought',
      'throw' : 'threw', 'thrust' : 'thrust', 'tread' : 'trod',
      'understand' : 'understood', 'upset' : 'upset', 'wake' : 'woke',
      'wear' : 'wore', 'weep' : 'wept', 'wind' : 'wound', 'win' :
      'won', 'write' : 'wrote' }

# The subject had verbed.
__irregular_participle = {
      'arise' : 'arisen', 'beat' : 'beaten', 'become' : 'become',
      'begin' : 'begun', 'bite' : 'bitten', 'blow' : 'blown', 'break'
      : 'broken', 'choose' : 'chosen', 'come' : 'come', 'do' :
      'done', 'draw' : 'drawn', 'drink' : 'drunk', 'drive' : 'driven',
      'eat' : 'eaten', 'fall' : 'fallen', 'fly' : 'flown', 'forbid' :
      'forbidden', 'foretell' : 'fortold', 'forget' : 'forgotten',
      'forsake' : 'forsaken', 'freeze' : 'frozen', 'give' : 'given',
      'go' : 'gone', 'grow' : 'grown', 'know' : 'known', 'lie' :
      'lain', 'ride' : 'ridden', 'ring' : 'rung', 'rise' : 'risen',
      'run' : 'run', 'shake' : 'shaken', 'show' : 'shown', 'shrink' :
      'shrunk', 'sing' : 'sung', 'sink' : 'sunk', 'slay' : 'slain',
      'smite' : 'smitten', 'sow' : 'sown', 'speak' : 'spoken',
      'spring' : 'sprung', 'steal' : 'stolen', 'stink' : 'stunk',
      'stride' : 'stridden', 'strive' : 'striven', 'swear' : 'sworn',
      'swim' : 'swum', 'swing' : 'swung', 'take' : 'taken', 'tear' :
      'torn', 'throw' : 'thrown', 'tread' : 'trodden', 'wake' :
      'woken', 'wear' : 'worn', 'write' : 'written' }

import re ; from re import sub, match

def conjugate(word, tense='present'):

    '''Conjugates an English word (a verb) for a specific tense.
    Tenses known include:
        'present participle'
        'past participle'
        'past'
        'progressive'
        'present'
    This routine does not add companion words such as "has" or "will"
    to the verb phrase.  It only conjugates the word itself.
    '''

    if 'participle' in tense:
        if word in __irregular_participle:
            return __irregular_participle[word]
        if word in __irregular_past:
            return __irregular_past[word]
        return word + 'ed'

    if 'past' in tense:
        if word in __irregular_past:
            return __irregular_past[word]
        if re.search(r'[^aeiou][aeiou][^aeioulr]$', word):
            return re.sub(r'([^aeiou])$', r'\1\1ed', word)
        return word + 'ed'

    if 'progressive' in tense:
        if word in __irregular_progressive:
            return __irregular_progressive[word]
        if re.search(r'[^aeiou]ie$', word):
            return re.sub(r'ie$', r'ying', word)
        if re.search(r'[^aeiou]e$', word):
            return re.sub(r'e$', r'ing', word)
        if re.search(r'[^aeiou][aeiou][^aeioulr]$', word):
            return re.sub(r'([^aeiou])$', r'\1\1ing', word)
        return word + 'ing'

    if 'present' in tense:
        if word in __irregular_present:
            return __irregular_present[word]
        if re.search(r'ss$', word):
            return word + 'es'
        return word + 's'

    return word

#----------------------------------------------------------------------------

__score = [ 'zero', 'one', 'two', 'three', 'four', 'five', 'six',
            'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
            'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
            'eighteen', 'nineteen' ]

__decade = [ 'zero', 'ten', 'twenty', 'thirty', 'forty', 'fifty',
             'sixty', 'seventy', 'eighty', 'ninety', 'hundred' ]

__groups = [ 'zero', 'thousand', 'million', 'billion',
             # 'trillion',
             # 'quadrillion', 'quintillion', 'sextillion', 'septillion',
             # 'octillion', 'nonillion'
             ]

__groupvalues = [ 0, 1000, 1000000, 1000000000,
                  # 1000000000000,
                  # 1000000000000000,
                  # 1000000000000000000,
                  # 1000000000000000000000,
                  # ...
                  ]

def cardinal(number, style=None):
    '''Returns a phrase that spells out the cardinal form of a number.
    This routine does not currently try to understand "big integer"
    numbers using words like 'septillion'.
    '''
    if not number:
        return __score[0]
    text = ''
    if number < 0:
        text = 'negative '
        number = -number
    for group in reversed( range( len(__groups) ) ):
        if not group: continue
        if number >= __groupvalues[group]:
            multiple = int(number / __groupvalues[group])
            text += cardinal(multiple) + ' ' + __groups[group]
            number %= __groupvalues[group]
            if number:
                text += ', '
    if number >= 100:
        text += cardinal(int(number / 100)) + ' ' + __decade[10]
        number %= 100
        if number:
            text += ' '
    if number >= 20:
        text += __decade[int(number/10)]
        number %= 10
        if number:
            text += '-'
    if number > 0:
        text += __score[number]
    return text

__scoreth = [ 'zeroth', 'first', 'second', 'third', 'fourth', 'fifth',
              'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
              'eleventh', 'twelfth', 'thirteenth', 'fourteenth',
              'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth',
              'nineteenth' ]

__decadeth = [ 'zeroth', 'tenth', 'twentieth', 'thirtieth', 'fortieth',
               'fiftieth', 'sixtieth', 'seventieth', 'eightieth',
               'ninetieth', 'hundredth' ]

def ordinal(number, style=None):
    '''Returns a phrase that spells out the ordinal form of a number.'''
    if not number:
        return __scoreth[0]
    text = ''
    if number < 0:
        text = 'negative '
        number = -number
    if number >= 100:
        text += cardinal(number - (number % 100))
        if (number % 1000) > 0 and (number % 1000) - (number % 100) == 0:
            text += ','
        number %= 100
        if number:
            text += ' '
    if not number:
        return text + 'th'
    if number >= 20:
        spare = number % 10
        if not spare:
            text += __decadeth[int(number/10)]
        else:
            text += __decade[int(number/10)] + '-'
        number = spare
    if number > 0:
        text += __scoreth[number]
    return text

#----------------------------------------------------------------------------

def span(time, brevity=0):
    # turns a number of seconds into a phrase like
    # "six days, four hours and two minutes"
    pass

#----------------------------------------------------------------------------

def __test__():
    print 'Testing english module...'
    from testing import __ok__

    __ok__(enumerate( [] ), '')
    __ok__(enumerate( ['a'] ), 'a')
    __ok__(enumerate( ['a','b'] ), 'a and b')
    __ok__(enumerate( ['a','b','c'] ), 'a, b and c')
    __ok__(enumerate( ['a','b','c','d'] ), 'a, b, c and d')
    __ok__(enumerate( [1,2,3,4] ), '1, 2, 3 and 4')

    __ok__(cardinal( -1234 ),
           'negative one thousand, two hundred thirty-four')
    __ok__(cardinal( -1 ), 'negative one')
    __ok__(cardinal( 0 ), 'zero')
    __ok__(cardinal( 1 ), 'one')
    __ok__(cardinal( 8 ), 'eight')
    __ok__(cardinal( 18 ), 'eighteen')
    __ok__(cardinal( 28 ), 'twenty-eight')
    __ok__(cardinal( 280 ), 'two hundred eighty')
    __ok__(cardinal( 2800 ), 'two thousand, eight hundred')
    __ok__(cardinal( 280028 ), 'two hundred eighty thousand, twenty-eight')
    __ok__(cardinal( 2800000 ), 'two million, eight hundred thousand')

    __ok__(ordinal( -1234 ),
           'negative one thousand, two hundred thirty-fourth')
    __ok__(ordinal( -1 ), 'negative first')
    __ok__(ordinal( 0 ), 'zeroth')
    __ok__(ordinal( 1 ), 'first')
    __ok__(ordinal( 8 ), 'eighth')
    __ok__(ordinal( 18 ), 'eighteenth')
    __ok__(ordinal( 28 ), 'twenty-eighth')
    __ok__(ordinal( 280 ), 'two hundred eightieth')
    __ok__(ordinal( 2800 ), 'two thousand, eight hundredth')
    __ok__(ordinal( 280028 ), 'two hundred eighty thousand, twenty-eighth')
    __ok__(ordinal( 2800000 ), 'two million, eight hundred thousandth')

    __ok__(conjugate('sleep', 'present'), 'sleeps')
    __ok__(conjugate('sleep', 'past'), 'slept')
    __ok__(conjugate('sleep', 'past participle'), 'slept')
    __ok__(conjugate('sleep', 'progressive'), 'sleeping')
    __ok__(conjugate('dream', 'past'), 'dreamt')
    __ok__(conjugate('rise', 'present'), 'rises')
    __ok__(conjugate('rise', 'past'), 'rose')
    __ok__(conjugate('rise', 'progressive'), 'rising')
    __ok__(conjugate('rise', 'past participle'), 'risen')
    __ok__(conjugate('chop', 'past'), 'chopped')
    __ok__(conjugate('chop', 'progressive'), 'chopping')
    __ok__(conjugate('quiz', 'progressive'), 'quizzing')
    __ok__(conjugate('meet', 'past'), 'met')
    __ok__(conjugate('meet', 'progressive'), 'meeting')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    raise Exception, \
        'This module is not a stand-alone script.  Import it in a program.'

