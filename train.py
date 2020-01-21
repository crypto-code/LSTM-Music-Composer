import argparse
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import os

src = ""
name = ""
epoch = 10000

#--------------------------------------------------------------------------------------------------------------

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

#--------------------------------------------------------------------------------------------------------------
    
def get_notes():
    """ Get all the notes and chords from the midi files in the input directory """
    notes = []

    for file in glob.glob(src+"/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes_'+name, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes
#--------------------------------------------------------------------------------------------------------------

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)
#--------------------------------------------------------------------------------------------------------------

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model
#--------------------------------------------------------------------------------------------------------------

def train(model, network_input, network_output):
    """ Train the Neural Network """
    filepath = "weights_"+name+"/weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    if os.path.exists('weights_'+name+'/') and len(os.listdir('weights_'+name+'/') ) != 0:
        files = []
        for each in os.listdir('weights_'+name+'/'):
            files.append(os.path.join('weights_'+name+'/',each))
        files.sort(key=lambda x:float(x[-11:-5]))
        model.load_weights(files[0])
        print("Weight ", files[0]," Loaded...............")
    elif not os.path.exists('weights_'+name+'/'):
        os.mkdir('weights_'+name+'/')
            
    model.fit(network_input, network_output, epochs=epoch, batch_size=128, callbacks=callbacks_list)
#--------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Music Generator')
    parser.add_argument('--input',type=str,required=True,help='Directory containing input music samples. eg: ./data')
    parser.add_argument('--name',type=str,required=True,help='Name of the Music')
    parser.add_argument('--epoch',type=int,default=10000,help='Number of training epochs')
    args = parser.parse_args()
    src=args.input
    name=args.name
    epoch=args.epoch
    train_network()
#--------------------------------------------------------------------------------------------------------------
