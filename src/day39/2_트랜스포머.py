# 감성분석 # python3.12
import warnings
warnings.filterwarnings(action='ignore')

from transformers import pipeline # transformers모듈 설치
classifier = pipeline('sentiment-analysis')
data = ["This is what a true masterpiece looks like",
     "brilliant film, hard to better",
     "Are you kidding me. A horrible movie about horrible people.",
     "the plot itself is also very boring"]
results = classifier(data)
for result in results:
    print(f"레이블: {result['label']}, score: {round(result['score'], 3)}")


print( classifier("나는 수학이 어렵다" ) )

#
# 다국어 모델 불러오기
classifier = pipeline('sentiment-analysis',
                      model="nlptown/bert-base-multilingual-uncased-sentiment")
classifier("나는 수학이 어럽다")

# 감성분석(다국어)
classifier = pipeline('sentiment-analysis',
                      model="nlptown/bert-base-multilingual-uncased-sentiment")

data = ["이 영화 최고",
     "너무 지루하다",
     "또 보고싶은 최고의 걸작이다.",
     "내 취향은 아니다."]

results = classifier(data)

for i,result in enumerate(results):
    print(f"문장: {data[i]}, 레이블: {result['label']}, score: {round(result['score'], 3)}")


# 질의응답
# https://en.wikipedia.org/wiki/TensorFlow 텐서플로 소개 글 중 일부
from transformers import pipeline
nlp = pipeline("question-answering")
data = r"""
TensorFlow is Google Brain's second-generation system. Version 1.0.0 was released on February 11, 2017.[14] While the reference implementation runs on single devices, TensorFlow can run on multiple CPUs and GPUs (with optional CUDA and SYCL extensions for general-purpose computing on graphics processing units).[15] TensorFlow is available on 64-bit Linux, macOS, Windows, and mobile computing platforms including Android and iOS.
Its flexible architecture allows for the easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices.
TensorFlow computations are expressed as stateful dataflow graphs. The name TensorFlow derives from the operations that such neural networks perform on multidimensional data arrays, which are referred to as tensors. During the Google I/O Conference in June 2016, Jeff Dean stated that 1,500 repositories on GitHub mentioned TensorFlow, of which only 5 were from Google.[16]
In December 2017, developers from Google, Cisco, RedHat, CoreOS, and CaiCloud introduced Kubeflow at a conference. Kubeflow allows operation and deployment of TensorFlow on Kubernetes.
In March 2018, Google announced TensorFlow.js version 1.0 for machine learning in JavaScript.[17]
In Jan 2019, Google announced TensorFlow 2.0.[18] It became officially available in Sep 2019.[19]
In May 2019, Google announced TensorFlow Graphics for deep learning in computer graphics.[20]
"""
q1 = "What is TensorFlow?"
result = nlp(question=q1, context=data)
print(f"질문: {q1}, 응답: '{result['answer']}', score: {round(result['score'], 3)}")

q2 = "When is TensorFlow 2.0 announced?"
result = nlp(question="When is TensorFlow 2.0 announced?", context=data)
print(f"질문: {q2}, 응답: '{result['answer']}', score: {round(result['score'], 3)}")


# 문장생성
from transformers import pipeline

text_generator = pipeline("text-generation")
data = "I love you, I will"
print(text_generator(data, max_length=10, do_sample=False))


#
# 문장요약
from transformers import pipeline
summarizer = pipeline("summarization")
data ="""
TensorFlow is Google Brain's second-generation system. Version 1.0.0 was released on February 11, 2017.[14] While the reference implementation runs on single devices, TensorFlow can run on multiple CPUs and GPUs (with optional CUDA and SYCL extensions for general-purpose computing on graphics processing units).[15] TensorFlow is available on 64-bit Linux, macOS, Windows, and mobile computing platforms including Android and iOS.
Its flexible architecture allows for the easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices.
TensorFlow computations are expressed as stateful dataflow graphs. The name TensorFlow derives from the operations that such neural networks perform on multidimensional data arrays, which are referred to as tensors. During the Google I/O Conference in June 2016, Jeff Dean stated that 1,500 repositories on GitHub mentioned TensorFlow, of which only 5 were from Google.[16]
In December 2017, developers from Google, Cisco, RedHat, CoreOS, and CaiCloud introduced Kubeflow at a conference. Kubeflow allows operation and deployment of TensorFlow on Kubernetes.
In March 2018, Google announced TensorFlow.js version 1.0 for machine learning in JavaScript.[17]
In Jan 2019, Google announced TensorFlow 2.0.[18] It became officially available in Sep 2019.[19]
In May 2019, Google announced TensorFlow Graphics for deep learning in computer graphics.[20]
"""

print(summarizer(data, max_length=50, min_length=10, do_sample=False))
