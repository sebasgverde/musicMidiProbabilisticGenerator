
https://golang.org/doc/codewalk/markov/
https://en.wikipedia.org/wiki/Markov_chain
https://en.wikipedia.org/wiki/N-gram#n-gram_models


```python
import nltk
import random

# create a dict with the ngram model, it receives a list with the samples as string
ngram_dict = {}

def create_ngram_dict(corpus):
    n = 3
    ngrams = nltk.ngrams(corpus, n)
    
    for grams in ngrams:
        dict_key = grams[:-1][0] + " " + grams[:-1][1]
        if dict_key in ngram_dict:
            ngram_dict[dict_key].append(grams[-1])
        else:
            ngram_dict[dict_key] = []
            ngram_dict[dict_key].append(grams[-1])
```


```python
def generate(seed, samples = 15):
    output = seed  
    for i in range(samples):
        # When it reaches the last prefix, there is no suffix, so end
        try:
            new_sample = random.choice(ngram_dict[seed])
        except:
            return output
        output += " " + new_sample
        seed = seed.split(" ")[1] + " " + new_sample

    return output
```


```python
import pickle

midi_dataset = pickle.load(open("validation_DB12_final_cleaned.p", "rb"))
print("ready!!!")
```

    ready!!!



```python
corpus = midi_dataset["x"]
corpus_str=map(str, corpus)
```


```python
ngram_dict = {}
create_ngram_dict(corpus_str)
print("ready!!!")
```

    ready!!!



```python
generated_melody = generate("60 62", 40)
print(generated_melody)
```

    60 62 66 66 65 63 73 71 73 71 72 74 72 71 69 67 66 73 71 69 66 64 62 62 62 60 66 66 70 75 73 75 71 73 75 76 75 74 75 67 66 67



```python
from midiutil.MidiFile import MIDIFile   

# takes a list of integers representing midi notes and creates a .mid
# with a contant time of a quarter note for all the notes and 120 
# as tempo(taken from garcia, 2018)
def sequenceVector2midiMelody(seqVector, file_dir):
    MyMIDI = MIDIFile(1)
    track = 0 
    time = 0
    MyMIDI.addTrackName(track,time,"Sample Track") 
    MyMIDI.addTempo(track,time,120)
    time = 0
    for note in seqVector:
        # MyMIDI.addNote(track,channel,pitch,time,duration,volume)
        MyMIDI.addNote(0,0,note,time,1,100)
        time = time + 1

    binfile = open(file_dir, 'wb') 
    MyMIDI.writeFile(binfile) 
    binfile.close()
```


```python
sequenceVector2midiMelody(map(int,generated_melody.split(" ")), 'generated_melody.mid')
```

For the next part is necessary to have the musescore software installed, you can do it with the comman sudo apt-get install musescore, and then replace the uri in the 2 lines of environment, it will usually be: "/usr/bin/musescore"


```python
from music21 import midi
import music21
music21.environment.set("musicxmlPath", "/usr/bin/musescore")
music21.environment.set('musescoreDirectPNGPath', '/usr/bin/musescore')

mid_file = midi.MidiFile()
mid_file.open("generated_melody.mid")
mid_file.read()
mid_file.close()
mid_stream = midi.translate.midiFileToStream(mid_file)
mid_stream.show()

mid_stream.show("midi")
sp = midi.realtime.StreamPlayer(mid_stream)
sp.play()
```

    music21: Certain music21 functions might need these optional packages: matplotlib, numpy, scipy;
                       if you run into errors, install them by following the instructions at
                       http://mit.edu/music21/doc/installing/installAdditional.html
     
    Music21 v.4 is the last version that will support Python 2.
    Please start using Python 3 instead.
    
    Set music21.environment.UserSettings()['warnings'] = 0
    to disable this message.
    



![png](output_10_1.png)




                <div id='midiPlayerDiv1467'></div>
                <link rel="stylesheet" href="http://artusi.xyz/music21j/css/m21.css"
                    type="text/css" />
                <script>
                require.config({
                    paths: {'music21': 'http://artusi.xyz/music21j/src/music21'}
                });
                require(['music21'], function() {
                               mp = new music21.miditools.MidiPlayer();
                               mp.addPlayer('#midiPlayerDiv1467');
                               mp.base64Load('data:audio/midi;base64,TVRoZAAAAAYAAQABBABNVHJrAAABjwD/AwAA4ABAAP9RAwehIIgAkDxkiACAPAAAkD5kiACAPgAAkEJkiACAQgAAkEJkiACAQgAAkEFkiACAQQAAkD9kiACAPwAAkElkiACASQAAkEdkiACARwAAkElkiACASQAAkEdkiACARwAAkEhkiACASAAAkEpkiACASgAAkEhkiACASAAAkEdkiACARwAAkEVkiACARQAAkENkiACAQwAAkEJkiACAQgAAkElkiACASQAAkEdkiACARwAAkEVkiACARQAAkEJkiACAQgAAkEBkiACAQAAAkD5kiACAPgAAkD5kiACAPgAAkD5kiACAPgAAkDxkiACAPAAAkEJkiACAQgAAkEJkiACAQgAAkEZkiACARgAAkEtkiACASwAAkElkiACASQAAkEtkiACASwAAkEdkiACARwAAkElkiACASQAAkEtkiACASwAAkExkiACATAAAkEtkiACASwAAkEpkiACASgAAkEtkiACASwAAkENkiACAQwAAkEJkiACAQgAAkENkiACAQwCIAP8vAA==');
                        });
                </script>


# Text


```python
word_corpus = 'am I a gram or am I a markov chain ... maybe I am both'
ngram_dict = {}
create_ngram_dict(word_corpus.split(" "))
ngram_dict
```


```python
print generate("am I", 100)
```
