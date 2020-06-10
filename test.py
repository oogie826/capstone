import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from konlpy.tag import Okt

print(tf.__version__)
print(np.__version__)

train_data = [
['조재춘 교수님 전화번호 어떻게 돼?',  '조재춘 교수님 전화번호는 010 입니다.'],
['조재춘 교수님 전화번호 알려줘',  '조재춘 교수님 전화번호는 010 입니다.'],
['조재춘 교수님 전화번호 좀',  '조재춘 교수님 전화번호는 010 입니다.'],
['조재춘 교수님 전화번호', '조재춘 교수님 전화번호는 010 입니다.'],
['조재춘 교수님 번호 뭐야?' ,  '조재춘 교수님 전화번호는 010 입니다.'],
['조재춘 교수님 번호 어떻게 돼?',  '조재춘 교수님 전화번호는 010 입니다.'],
['조재춘 교수님 번호 알려줘',  '조재춘 교수님 전화번호는 010 입니다.'],
['조재춘 교수님 번호 좀',  '조재춘 교수님 전화번호는 010 입니다.'],
['조재춘 교수님 번호', '조재춘 교수님 전화번호는 010 입니다.'],
# ['조재춘 교수님 핸드폰 번호 뭐야?' ,  '조재춘 교수님 핸드폰번호는 010 입니다.'],
# ['조재춘 교수님 핸드폰 번호 어떻게 돼?',  '조재춘 교수님 핸드폰번호는 010 입니다.'],
# ['조재춘 교수님 핸드폰 번호 알려줘',  '조재춘 교수님 핸드폰번호는 010 입니다.'],
# ['조재춘 교수님 핸드폰 번호 좀',  '조재춘 교수님 핸드폰번호는 010 입니다.'],
# ['조재춘 교수님 핸드폰 번호', '조재춘 교수님 핸드폰번호는 010 입니다.'],
# ['조재춘 교수님 연구실이 어디야?', '조재춘 교수님 연구실은 60주년 기념관 4층 입니다'],
# ['조재춘 교수님 연구실 위치가 어떻게 돼?', '조재춘 교수님 연구실은 60주년 기념관 4층 입니다'],
# ['조재춘 교수님 연구실 위치가 어디쯤이야?', '조재춘 교수님 연구실은 60주년 기념관 4층 입니다'],
# ['조재춘 교수님 연구실 위치 알려줘', '조재춘 교수님 연구실은 60주년 기념관 4층 입니다'],
# ['조재춘 교수님 연구실 위치 좀', '조재춘 교수님 연구실은 60주년 기념관 4층 입니다'],
# ['조재춘 교수님 연구실 좀', '조재춘 교수님 연구실은 60주년 기념관 4층 입니다'],
# ['조재춘 교수님 연구실', '조재춘 교수님 연구실은 60주년 기념관 4층 입니다'],
# ['조재춘 교수님 이메일이 어떻게 돼?', '조재춘 교수님 이메일은 @hs.ac.kr 입니다.'],
# ['조재춘 교수님 이메일 뭐야?', '조재춘 교수님 이메일은 @hs.ac.kr 입니다.'],
# ['조재춘 교수님 이메일 알려줘', '조재춘 교수님 이메일은 @hs.ac.kr 입니다.'],
# ['조재춘 교수님 이메일이 좀 알려줘?', '조재춘 교수님 이메일은 @hs.ac.kr 입니다.'],
# ['조재춘 교수님 이메일이 좀?', '조재춘 교수님 이메일은 @hs.ac.kr 입니다.'],
# ['조재춘 교수님 이메일', '조재춘 교수님 이메일은 @hs.ac.kr 입니다.'],
# ['컴퓨터 공학부 전화번호 뭐야?' ,  '컴퓨터 공학부 전화번호는 031 입니다.'],
# ['컴퓨터 공학부 전화번호 어떻게 돼?',  '컴퓨터 공학부 전화번호는 031 입니다.'],
# ['컴퓨터 공학부 전화번호 알려줘',  '컴퓨터 공학부 전화번호는 031 입니다.'],
# ['컴퓨터 공학부 전화번호 좀',  '컴퓨터 공학부 전화번호는 031 입니다.'],
# ['컴퓨터 공학부 전화번호', '컴퓨터 공학부 전화번호는 031 입니다.'],
# ['컴퓨터 공학부 번호 뭐야?' ,  '컴퓨터 공학부 전화번호는 031 입니다.'],
# ['컴퓨터 공학부 번호 어떻게 돼?',  '컴퓨터 공학부 전화번호는 031 입니다.'],
# ['컴퓨터 공학부 번호 알려줘',  '컴퓨터 공학부 전화번호는 031 입니다.'],
# ['컴퓨터 공학부 번호 좀',  '컴퓨터 공학부 전화번호는 031 입니다.'],
# ['컴퓨터 공학부 번호', '컴퓨터 공학부 전화번호는 031 입니다.'],
# ['컴퓨터 공학부 어디야?', '컴퓨터 공학부 위치는 60주년 기념관 4층 입니다'],
# ['컴퓨터 공학부 위치가 어떻게 돼?', '컴퓨터 공학부 위치는 60주년 기념관 4층 입니다'],
# ['컴퓨터 공학부 위치가 어디쯤이야?', '컴퓨터 공학부 위치는 60주년 기념관 4층 입니다'],
# ['컴퓨터 공학부 위치 알려줘', '컴퓨터 공학부 위치는 60주년 기념관 4층 입니다'],
# ['컴퓨터 공학부 위치 좀', '컴퓨터 공학부 위치는 60주년 기념관 4층 입니다'],
# ['컴퓨터 공학부 위치', '컴퓨터 공학부 위치는 60주년 기념관 4층 입니다'],
# ['컴퓨터 공학부 이메일이 어떻게 돼?', '컴퓨터 공학부 이메일은 computer@hs.ac.kr 입니다.'],
# ['컴퓨터 공학부 이메일 뭐야?', '컴퓨터 공학부 이메일은 computer@hs.ac.kr 입니다.'],
# ['컴퓨터 공학부 이메일 알려줘', '컴퓨터 공학부 이메일은 computer@hs.ac.kr 입니다.'],
# ['컴퓨터 공학부 이메일 좀 알려줘?', '컴퓨터 공학부 이메일은 computer@hs.ac.kr 입니다.'],
# ['컴퓨터 공학부 이메일 좀?', '컴퓨터 공학부 이메일은 computer@hs.ac.kr 입니다.'],
# ['컴퓨터 공학부 이메일', '컴퓨터 공학부 이메일은 computer@hs.ac.kr 입니다.'],
# ['컴퓨터 공학부 홈페이지주소가 어떻게 돼?', '컴퓨터 공학부 홈페이지주소는 sce.hs.ac.kr 입니다.'],
# ['컴퓨터 공학부 홈페이지주소 뭐야?', '컴퓨터 공학부 홈페이지주소는 sce.hs.ac.kr 입니다.'],
# ['컴퓨터 공학부 홈페이지주소 알려줘', '컴퓨터 공학부 홈페이지주소는 sce.hs.ac.kr 입니다.'],
# ['컴퓨터 공학부 홈페이지주소 좀 알려줘?', '컴퓨터 공학부 홈페이지주소는 sce.hs.ac.kr 입니다.'],
# ['컴퓨터 공학부 홈페이지주소 좀?', '컴퓨터 공학부 홈페이지주소는 sce.hs.ac.kr 입니다.'],
# ['컴퓨터 공학부 홈페이지주소', '컴퓨터 공학부 홈페이지주소는 sce.hs.ac.kr 입니다.'],
# ['캡스톤 디자인1 A반 교수님 누구야?', '캡스톤 디자인1 A반 교수님은 조재춘 교수님 입니다'],
# ['캡스톤 디자인1 A반 교수님 어떻게 돼?', '캡스톤 디자인1 A반 교수님은 조재춘 교수님 입니다'],
# ['캡스톤 디자인1 A반 교수님 누구셔?', '캡스톤 디자인1 A반 교수님은 조재춘 교수님 입니다'],
# ['캡스톤 디자인1 A반 교수님 좀 알려줘?', '캡스톤 디자인1 A반 교수님은 조재춘 교수님 입니다'],
# ['캡스톤 디자인1 A반 교수님 알려줘?', '캡스톤 디자인1 A반 교수님은 조재춘 교수님 입니다'],
# ['컴퓨터 공학부 졸업 이수조건이 어떻게 돼?', '컴퓨터 공학부 졸업 이수조건은 ~입니다.'],
# ['컴퓨터 공학부 졸업 이수조건 뭐야?', '컴퓨터 공학부 졸업 이수조건은 ~입니다.'],
# ['컴퓨터 공학부 졸업 이수조건 알려줘','컴퓨터 공학부 졸업 이수조건은 ~입니다.'],
# ['컴퓨터 공학부 졸업 이수조건 좀','컴퓨터 공학부 졸업 이수조건은 ~입니다.'],
# ['컴퓨터 공학부 졸업 이수학점이 어떻게 돼?','컴퓨터 공학부 졸업 이수학점은 ~입니다.'],
# ['컴퓨터 공학부 졸업 이수학점 뭐야?','컴퓨터 공학부 졸업 이수학점은 ~입니다.'],
# ['컴퓨터 공학부 졸업 이수학점 알려줘','컴퓨터 공학부 졸업 이수학점은 ~입니다.'],
# ['컴퓨터 공학부 졸업 이수학점 좀','컴퓨터 공학부 졸업 이수학점은 ~입니다.'],
# ['디비넷 위치 어디야', '컴퓨터 공학부 디비넷 랩실의 위치는 60주년 기념관 2층 입니다.'],
# ['디비넷 위치가 어떻게 돼', '컴퓨터 공학부 디비넷 랩실의 위치는 60주년 기념관 2층 입니다.'],
# ['디비넷 위치 어디쯤이야', '컴퓨터 공학부 디비넷 랩실의 위치는 60주년 기념관 2층 입니다.'],
# ['디비넷 위치 알려줘', '컴퓨터 공학부 디비넷 랩실의 위치는 60주년 기념관 2층 입니다.'],
# ['디비넷 위치 좀', '컴퓨터 공학부 디비넷 랩실의 위치는 60주년 기념관 2층 입니다.'],
# ['디비넷 위치 알려줘', '컴퓨터 공학부 디비넷 랩실의 위치는 60주년 기념관 2층 입니다.'],
# ['디비넷 위치', '컴퓨터 공학부 디비넷 랩실의 위치는 60주년 기념관 2층 입니다.'],
]

char_array = []
all_char = ''
for text in train_data:
    all_char += all_char + ''.join(text)
char_array = ['P', '[', ']'] + list(set(all_char))

max_input_text = max(len(string[0]) for string in train_data)
max_output_text = max(len(string[1]) for string in train_data)

num_dic = {n: i for i, n in enumerate(char_array)}
dic_len = len(num_dic)

print(u'char list:', str(num_dic))
print(u'char size:', str(dic_len))

def make_train_data(train_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in train_data:
        input = [num_dic[n] for n in seq[0]+'P' * (max_input_text - len(seq[0]))]# P는 Padding 값
        output = [num_dic[n] for n in ('[' + seq[1] + 'P' * (max_output_text - len(seq[1])))]
        target = [num_dic[n] for n in (seq[1] + 'P' * (max_output_text - len(seq[1])) + ']' )]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)
    return input_batch, output_batch, target_batch

file_path = './model'
def model_file(file_path, flag):
    if(flag):
        import os
        saver = tf.train.Saver(tf.global_variables())

        if(not os.path.exists(file_path)):
            os.makedirs(file_path)
        saver.save(sess, ''.join(file_path + "/.model"))
        print("Model Saved")
    else:
        import shutil
        try:
            shutil.rmtree(file_path)
            print("Model Deleted")
        except OSError as e:
            if e.errno == 2:
                # 파일이나 디렉토리가 없음!
                print ('No such file or directory to remove')
                pass
            else:
                raise

# 옵션 설정
learning_rate = 0.01
n_hidden = 128
total_epoch = 200
# one hot 위한 사이즈
n_class = n_input = dic_len

# 그래프 초기화 
tf.reset_default_graph()
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])

# 인코더
with tf.variable_scope("encoder"):
    enc_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

# 디코더
with tf.variable_scope("decoder"):
    dec_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)

model = tf.layers.dense(outputs, n_class, activation=None)

#onehot로 sparse사용 
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
input_batch, output_batch, target_batch = make_train_data(train_data)

# 최적화가 끝난 뒤, 변수를 저장합니다.
model_file(file_path, True)

def display_train():
    plot_X = []
    plot_Y = []
    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer, cost],
                           feed_dict={enc_input: input_batch,
                                      dec_input: output_batch,
                                      targets: target_batch})
        plot_X.append(epoch + 1)
        plot_Y.append(loss)
    # Graphic display
    plt.plot(plot_X, plot_Y, label='cost')
    plt.show()

display_train()

# 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
def predict(word):
    input_batch, output_batch, target_batch = make_train_data([word])
    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    # http://pythonkim.tistory.com/73
    prediction = tf.argmax(model, 2)
    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})
    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [char_array[i] for i in result[0]]
        
    if 'P' in decoded:
        end = decoded.index('P')
        decoded = decoded[:end]
    elif ']' in decoded:
        end = decoded.index(']')
        decoded = decoded[:end] 
    return decoded

print ("Q: 컴퓨터 공학부 전화번호")
print("A: " + ''.join(predict(['컴퓨터 공학부 전화번호',''])))
print ("Q: 컴퓨터 공학부 전화번호")
print("A: " + ''.join(predict(['컴퓨터 공학부 전화번호',''])))

model_file(file_path, False)