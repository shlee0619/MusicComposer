from music21 import converter, instrument, note, chord, stream
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. MIDI 파일에서 음표 추출
midi_data = converter.parse('./JimiHendrix/VoodooChild(SlightReturn)(Woodstock).mid')

notes = []

for element in midi_data.flat.notes:
    if isinstance(element, note.Note):
        notes.append(str(element.pitch))
    elif isinstance(element, chord.Chord):
        notes.append('.'.join(str(pitch.midi) for pitch in element.pitches))

# 2. 고유한 음표 목록 생성
pitchnames = sorted(set(notes))
n_vocab = len(pitchnames)

# 3. 음표를 정수로 매핑 및 시퀀스 생성
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

sequence_length = 100
network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[note] for note in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)

# 4. 입력 데이터 정규화 및 형태 변환
X = np.reshape(network_input, (n_patterns, sequence_length, 1))
X = X / float(n_vocab)

y = to_categorical(network_output, num_classes=n_vocab)

# 5. 모델 구성
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# 6. 모델 학습
model.fit(X, y, epochs=100, batch_size=64)

# 7. 음악 생성
start = np.random.randint(0, len(network_input)-1)
seed_sequence = network_input[start]
result = []

generate_length = 500

for i in range(generate_length):
    input_data = np.reshape(seed_sequence, (1, len(seed_sequence), 1))
    input_data = input_data / float(n_vocab)

    prediction = model.predict(input_data, verbose=0)
    index = np.argmax(prediction)
    result.append(index)

    seed_sequence.append(index)
    seed_sequence = seed_sequence[1:]

# 8. 결과를 MIDI 파일로 저장
offset = 0
output_notes = []

int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

for index in result:
    pattern = int_to_note[index]
    
    if ('.' in pattern) or pattern.isdigit():
        # 화음 처리
        notes_in_chord = [int(n) for n in pattern.split('.')]
        new_chord = chord.Chord(notes_in_chord)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        # 음표 처리
        new_note = note.Note(pattern)
        new_note.offset = offset
        output_notes.append(new_note)
    
    offset += 0.5  # 음표 간격 조절

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='generated_music.mid')
