import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import binascii
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import streamlit as st
from io import BytesIO
import heapq

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Scarf object detection and colour identification")
st.text("Upload image (.jpg) of to find the most frequent colour ")

@st.cache(allow_output_mutation=True)
def load_model():
  # object detection model inkage.....
  # load model here with cache enabled =====> model will be loaded only once 
  pass



def get_color(im):

  NUM_CLUSTERS = 5
  # print('reading image')
  im = im.resize((150, 150))      # optional, to reduce time
  ar = np.asarray(im)
  shape = ar.shape
  ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

  # print('finding clusters')
  codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
  # print('cluster centres:\n', codes)

  vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
  counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
  index_max = scipy.argmax(counts)
  # print("index_max",index_max)
  # print("counts",counts,"codes",codes)
  rank=heapq.nlargest(2, range(len(counts)), key=counts.__getitem__)
  # print("rank",rank)
  peak = codes[rank[-1]]
  # print("peak",peak)
  colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
  # print('most frequent is %s (##%s)' % (peak, colour))
  return peak,colour



uploaded_file =st.file_uploader("upload an image", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None)

st.write("Identifying colour :")
if uploaded_file:
  with st.spinner('Identifying.....'):
    # im = Image.open('scarf-10539009.jpg')
    im = Image.open(uploaded_file)
    rbg,hexa=get_color(im)
    # label =np.argmax(model.predict(decode_img(content)),axis=1)
    # st.write( rbg,hexa)    
  # st.write("")
  # image = Image.open(BytesIO(im))
  st.image(im, caption='most frequent is rbg code: %s Hex Code:#%s' % (rbg, hexa), use_column_width=True)
    
